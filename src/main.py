from fastapi import FastAPI, Request, HTTPException, WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
gi.require_version('GstWebRTC', '1.0')
from gi.repository import GstWebRTC
gi.require_version('GstSdp', '1.0')
from gi.repository import GstSdp

from contextlib import asynccontextmanager
from multiprocessing import shared_memory
import onnxruntime as ort
import numpy as np
import multiprocessing
import threading
import asyncio
import json
import psutil
import os

# For measuring performance
from functools import wraps
from time import perf_counter, sleep



HEIGHT = 640
WIDTH = 640
FORMAT = 'BGR'
MODEL_PATH = 'yolov8n.onnx'
SERVER_CORE_AFFINITY = None
MODEL_CORE_AFFINITY = None


Gst.init(None)
websocket_connection = None


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = perf_counter()
        result = f(*args, **kw)
        te = perf_counter()
        elapsed_time = (te - ts) * 1000
        print(f"Function '{f.__name__}' executed in {elapsed_time:.2f} ms")
        return result
    return wrap


def on_negotiation_needed(*args):
    global websocket_connection
    assert websocket_connection, "on_negotiation was called before websocket_connection was created"
    print("ON_NEGOTIATION_NEEDED was called: ", args)
    _ = asyncio.run(websocket_connection.send_text('waitingForOffer'))

def send_ice_candidate_message(_, mlineindex, candidate):
    print(f'INITIATING SENDING OF ICE CANDIDATE \n')
    global websocket_connection
    assert websocket_connection, f"send_ice_candidate_message was called before websocket_connection was created! \n candidate: {candidate} \n mlineindex: {mlineindex}"
    icemsg = json.dumps({'ice': {'candidate': candidate, 'sdpMLineIndex': mlineindex}})

    print(f'ICE candidate being sent: {icemsg}')
    asyncio.run(websocket_connection.send_text(icemsg))


def on_answer_created(promise, _, webrtcbin):
    global websocket_connection
    print(f'On answer creation initiated... \n')
    promise.wait()
    msg = promise.get_reply()
    answer = msg.get_value('answer')
    print(f'ANSWER: \n {answer.sdp.as_text()}')
    webrtcbin.emit('set-local-description', answer, promise)
    promise.interrupt()
    print(f'ON ANSWER CREATION COMPLETE \n')

    reply = json.dumps({'sdp': {'type':'answer', 'sdp': answer.sdp.as_text()}})


    asyncio.run(websocket_connection.send_text(reply))





def on_decodebin_added(_, pad):
    print(f'ON_DECODEBIN_ADDED CALLED')
    global pipeline
    global appsink

    if not pad.has_current_caps():
        print(pad, 'has no caps, ignoring')
        return

    caps = pad.get_current_caps()
    print(f'INCOMING STREAM CAPABILITIES: {caps.to_string()}')
    name = caps.to_string()


    q = Gst.ElementFactory.make('queue')
    scale = Gst.ElementFactory.make('videoscale')
    conv = Gst.ElementFactory.make('videoconvert')
    capsfilter = Gst.ElementFactory.make('capsfilter')


    enforce_caps = Gst.Caps.from_string(f"video/x-raw,format={FORMAT},width={WIDTH},height={HEIGHT}")
    capsfilter.set_property('caps', enforce_caps)


    pipeline.add(q)
    pipeline.add(scale)
    pipeline.add(conv)
    pipeline.add(capsfilter)
    pipeline.add(appsink)
    pipeline.sync_children_states()
    pad.link(q.get_static_pad('sink'))
    q.link(scale)
    scale.link(conv)
    conv.link(capsfilter)
    capsfilter.link(appsink)

    pipeline.set_state(Gst.State.PLAYING)




def on_stream_added(_, pad):
    print(f'------ ON STREAM ADDED EVENT HAS BEEN CALLED ---------')
    print(f'------ STREAM IS NOW FLOWING  ---------')


    global pipeline
    global webrtcbin

    decodebin = Gst.ElementFactory.make('decodebin')
    decodebin.connect('pad-added', on_decodebin_added)
    pipeline.add(decodebin)
    decodebin.sync_state_with_parent()
    webrtcbin.link(decodebin)

    print(f'----- Everything connected, stream should be flowing')






webrtcbin = Gst.ElementFactory.make("webrtcbin", "webrtcbin")
appsink = Gst.ElementFactory.make("appsink", "appsink")

webrtcbin.set_property("stun-server", "stun://stun.kinesisvideo.eu-west-2.amazonaws.com:443")

appsink.set_property("emit-signals", True)
appsink.set_property("drop", True)
appsink.set_property("max-buffers", 1)
appsink.set_property("sync", True)


webrtcbin.connect('on-negotiation-needed', on_negotiation_needed)
webrtcbin.connect('on-ice-candidate', send_ice_candidate_message)
webrtcbin.connect('pad-added', on_stream_added)



pipeline = Gst.Pipeline.new("pipeline")
pipeline.add(webrtcbin)




def pull_samples(appsink, connection, lock, process):
    global pipeline
    print('--- pipe_samples initiated ---')

    try:
        while True:
            sample = appsink.emit("pull-sample")
            if sample is None:
                sleep(1)
                print(f'No samples in the appsink')
                continue

            caps = sample.get_caps()

            assert HEIGHT == caps.get_structure(0).get_value("height"), f'appsink receive height: {caps.get_structure(0).get_value("height")}, expected: {HEIGHT}'
            assert WIDTH == caps.get_structure(0).get_value("width"), f'appsink receive width: {caps.get_structure(0).get_value("width")}, expected: {WIDTH}'

            buffer = sample.get_buffer()
            success, map_info = buffer.map(Gst.MapFlags.READ)

            if not success:
                raise RuntimeError("Could not map buffer data!")

            frame = np.ndarray(
                shape=(HEIGHT, WIDTH, 3),
                dtype=np.uint8,
                buffer=map_info.data) / 255

            frame = np.transpose(frame, (2, 0, 1))

            frame_nbytes, frame_dtype, frame_shape = frame.nbytes, frame.dtype, frame.shape

            memory = shared_memory.SharedMemory(create=True, size=frame_nbytes)
            init_message = {
                'state': 'init',
                'shared_memory_name': memory.name,
                'dtype': frame_dtype,
                'shape': frame_shape
            }

            connection.send(init_message)
            break

        while process.is_alive():
            lock.acquire()
            sample = appsink.emit("pull-sample")
            if sample is None:
                lock.release()
                continue


            buffer = sample.get_buffer()
            _, map_info = buffer.map(Gst.MapFlags.READ)

            frame = np.ndarray(
                shape=(HEIGHT, WIDTH, 3),
                dtype=np.uint8,
                buffer=map_info.data) / 255

            frame = np.transpose(frame, (2, 0, 1))

            shared_array = np.ndarray(frame_shape, dtype=frame_dtype, buffer=memory.buf)
            np.copyto(shared_array, frame)
            lock.release()

            connection.send({'state': 'cls'})

    except Exception as e:
        print(f'Exception in pull_samples occured: {e}')




def classification_process(model_path, connection, lock, process_affinity = None):
    if process_affinity:
        cls_process = psutil.Process()  # Gets the current process
        cls_process.cpu_affinity(process_affinity)

    try:
        # onnx_options = ort.SessionOptions()
        # onnx_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # onnx_options.intra_op_num_threads = 0
        # onnx_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        model = ort.InferenceSession(model_path)
        memory, dtype, shape = None, None, None
        while True:
            message = connection.recv()

            if message['state'] == 'cls':
                lock.acquire()
                received_frame = np.ndarray(shape, dtype=dtype, buffer=memory.buf)

                ts = perf_counter()
                frame = np.expand_dims(received_frame, axis=0).astype(np.float32)
                input = {model.get_inputs()[0].name: frame}
                output = model.run(None, input)
                te = perf_counter()

                lock.release()

                elapsed_time = (te - ts) * 1000
                execution_time = f'Executed in {elapsed_time:.2f} ms'

                connection.send(execution_time)


            if message['state'] == 'init':
                shared_memory_name, dtype, shape = message['shared_memory_name'], message['dtype'], message['shape']
                memory = shared_memory.SharedMemory(name=shared_memory_name)
                continue

            if message['state']  == 'close':
                break

    except Exception as e:
        connection.send(e)


def pipe_listener(connection, process):
    print('--- pipe_listener initiated ---')

    while process.is_alive():
        message = connection.recv()
        print(message)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    global appsink

    if SERVER_CORE_AFFINITY:
        server_process = psutil.Process()
        server_process.cpu_affinity(SERVER_CORE_AFFINITY)

    load_model_path = os.path.join('models', MODEL_PATH)
    parent_connection, child_connection = multiprocessing.Pipe()
    lock = multiprocessing.Lock()

    process = multiprocessing.Process(target=classification_process, args=(load_model_path, child_connection, lock, MODEL_CORE_AFFINITY))
    process.start()

    listener_thread = threading.Thread(target=pipe_listener, args=(parent_connection, process), daemon=True)
    listener_thread.start()

    pulling_samples_thread = threading.Thread(target=pull_samples, args=(appsink, parent_connection, lock, process), daemon=True)
    pulling_samples_thread.start()

    yield


    pipeline.set_state(Gst.State.NULL)



app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="/home/ubuntu/src/templates")
app.mount("/static", StaticFiles(directory="/home/ubuntu/src/static"), name="static")

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket):
    global appsink
    global websocket_connection
    await websocket.accept()
    websocket_connection = websocket

    # asyncio.create_task(run_in_background(pull_samples(appsink=appsink)))


    while True:
        data = await websocket_connection.receive_text()

        if data == 'PING':
            print(f'Received a message: {data}')
            pipeline.set_state(Gst.State.PLAYING)


            continue

        msg = json.loads(data)

        if 'offer' in msg:
            sdp = msg['offer']['sdp']
            print(f'Received an offer: \n {sdp}')
            res, sdpmsg = GstSdp.SDPMessage.new()
            GstSdp.sdp_message_parse_buffer(bytes(sdp.encode()), sdpmsg)
            offer = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.OFFER, sdpmsg)
            promise = Gst.Promise.new()
            webrtcbin.emit('set-remote-description', offer, promise)
            promise.interrupt()

            print('Creating an answer ... \n')
            promise = Gst.Promise.new_with_change_func(on_answer_created, None, webrtcbin)
            webrtcbin.emit('create-answer', None, promise)



        elif 'ice' in msg:
            ice = msg['ice']
            print(f'Received an ICE candidate: {ice}')
            candidate = ice['candidate']
            sdpmlineindex = ice['sdpMLineIndex']
            webrtcbin.emit('add-ice-candidate', sdpmlineindex, candidate)









