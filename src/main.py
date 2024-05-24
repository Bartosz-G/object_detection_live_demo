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
import torch
import torch_neuron
import numpy as np
import multiprocessing
import threading
import asyncio
import json
import os
import sys

# For measuring performance
from functools import wraps
from time import perf_counter, sleep



HEIGHT = 480
WIDTH = 480
FORMAT = 'RGB'
MODEL_PATH = 'rtdetr480_neuron.pt'
SERVER_CORE_AFFINITY = None
MODEL_CORE_AFFINITY = None
TOPK = 100


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

# Gstreamer WebRTC VIDEO ===========================
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

# Gstreamer WebRTC VIDEO ===========================

# Gstreamer WebRTC DATA ============================
def on_data_channel_ready(channel):
    print("[DataChannel]: ready")
    global data_channel
    data_channel = channel
    # Send a message to the client once the data channel is ready
    send_data_channel_message(b"Hello, Data Channel!")

def on_data_channel_open(channel):
    print("[DataChannel]: open")
    channel.connect("on-data", on_data_channel_message)

def on_data_channel_message(channel, data):
    print("[DataChannel], message received:", data)

def send_data_channel_message(data):
    global data_channel
    # print(f"[DataChannel], dir: {dir(data_channel)}")
    data_channel.send_data(data)

def on_data_channel(_, data_channel):
    print("[DataChannel], created")
    data_channel.connect("notify::ready-state", on_data_channel_open)
    data_channel.connect("on-open", on_data_channel_ready)

# Gstreamer WebRTC DATA =================================




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
webrtcbin.connect('on-data-channel', on_data_channel)



pipeline = Gst.Pipeline.new("pipeline")
pipeline.add(webrtcbin)


#TODO: Implement a data channel for sending back predictions



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

            warmup_frame = np.ndarray(
                shape=(HEIGHT, WIDTH, 3),
                dtype=np.uint8,
                buffer=map_info.data) / 255

            warmup_frame = np.transpose(warmup_frame, (2, 0, 1))

            frame_nbytes, frame_dtype, frame_shape = warmup_frame.nbytes, warmup_frame.dtype, warmup_frame.shape

            memory = shared_memory.SharedMemory(create=True, size=frame_nbytes)

            with lock:
                send_array = np.ndarray(frame_shape, dtype=frame_dtype, buffer=memory.buf)
                np.copyto(send_array, warmup_frame)

            init_message = {
                'state': 'init',
                'shared_memory_name': memory.name,
                'dtype': frame_dtype,
                'shape': frame_shape,
                'nbytes': frame_nbytes
            }

            connection.send(init_message)
            break

        while process.is_alive():
            sample = appsink.emit("pull-sample")
            if sample is None:
                continue


            buffer = sample.get_buffer()
            _, map_info = buffer.map(Gst.MapFlags.READ)

            frame = np.ndarray(
                shape=(HEIGHT, WIDTH, 3),
                dtype=np.uint8,
                buffer=map_info.data) / 255

            frame = np.transpose(frame, (2, 0, 1))
            with lock:
                shared_array = np.ndarray(frame_shape, dtype=frame_dtype, buffer=memory.buf)
                np.copyto(shared_array, frame)


            connection.send({'state': 'cls'})

    except Exception as e:
        print(f'Exception in pull_samples occured: {e}')


def model_process(model_path, receive_conn, post_conn, receive_lock, send_lock):

    model = torch.jit.load(model_path)

    try:
        while True:
            msg = receive_conn.recv()

            if msg['state'] not in ('cls', 'init', 'close'):
                raise Exception(f'received an unrecognised message from receive connection: {msg}')

            if msg['state'] == 'log':
                post_conn.send(msg)
                continue

            if msg['state'] == 'init':
                post_conn.send({'state': 'log', 'log': 'starting_warmup_process'})
                receive_shm_name, dtype, shape = msg['shared_memory_name'], msg['dtype'], msg['shape']
                receive_shared_memory = multiprocessing.shared_memory.SharedMemory(name=receive_shm_name)

                post_conn.send({'state': 'log', 'log': 'pulling_warmup_frame'})
                with receive_lock:
                    warmup_frame = np.ndarray(shape, dtype=dtype, buffer=receive_shared_memory.buf)

                post_conn.send({'state':'log', 'log': 'warming-up model'})
                warmup_frame = torch.from_numpy(warmup_frame).unsqueeze(0).float()
                output = model(warmup_frame)
                post_conn.send({'state': 'log', 'log': 'warming-up complete'})

                logits, bboxes = output[0].numpy(), output[1].numpy()
                logits_nbytes, logits_dtype, logits_shape = logits.nbytes, logits.dtype, logits.shape
                bboxes_nbytes, bboxes_dtype, bboxes_shape = bboxes.nbytes, bboxes.dtype, bboxes.shape

                post_conn.send({'state':'log', 'log': 'creating shared memory for the postprocessor'})
                send_logits_shared_memory = multiprocessing.shared_memory.SharedMemory(create=True, size=logits_nbytes)
                send_bboxes_shared_memory = multiprocessing.shared_memory.SharedMemory(create=True, size=bboxes_nbytes)

                post_conn.send({'state': 'log', 'log': 'putting warmup logits and frames to memory'})
                with send_lock:
                    shared_logits = np.ndarray(logits_shape, dtype=logits_dtype, buffer=send_logits_shared_memory.buf)
                    np.copyto(shared_logits, logits)
                    shared_bboxes = np.ndarray(bboxes_shape, dtype=bboxes.dtype, buffer=send_bboxes_shared_memory.buf)
                    np.copyto(shared_bboxes, bboxes)

                init_message = {
                    'state': 'init',
                    'logits': {
                        'shared_memory_name': send_logits_shared_memory.name,
                        'dtype': logits_dtype,
                        'shape': logits_shape,
                        'nbytes': logits_nbytes
                    },
                    'bboxes': {
                        'shared_memory_name': send_bboxes_shared_memory.name,
                        'dtype': bboxes_dtype,
                        'shape': bboxes_shape,
                        'nbytes': bboxes_nbytes
                    }
                }

                post_conn.send(init_message)

                continue

            #TODO: Refactor to not have to wait for a message and be able to respond to messages
            if msg['state'] == 'cls':
                with receive_lock:
                    received_array = np.ndarray(shape, dtype=dtype, buffer=receive_shared_memory.buf)

                frame = torch.from_numpy(received_array).float().unsqueeze(0)

                ts = perf_counter()
                output = model(frame)
                te = perf_counter()
                elapsed_time = (te - ts) * 1000
                latency = elapsed_time

                # Unpacking bboxes and labels
                logits, bbox_pred = output[0].numpy(), output[1].numpy()

                with send_lock:
                    shared_logits = np.ndarray(logits_shape, dtype=logits_dtype, buffer=send_logits_shared_memory.buf)
                    np.copyto(shared_logits, logits)
                    shared_bboxes = np.ndarray(bboxes_shape, dtype=bboxes.dtype, buffer=send_bboxes_shared_memory.buf)
                    np.copyto(shared_bboxes, bboxes)

                post_conn.send({'state': 'cls', 'latency': latency})

    except Exception as e:
        post_conn.send(e)


class PostProcessor:
    def __init__(self, topk, lock):
        self.lock = lock
        self.topk = topk

    def set_topk(self, topk):
        with self.lock:
            self.topk = topk

    def __call__(self, logits, bbox_pred, *args, **kwargs):
        with self.lock:
            logits, bbox_pred = torch.from_numpy(logits), torch.from_numpy(bbox_pred)
            scores = torch.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.topk, axis=-1)
            labels = index % 80
            index = index // 80
            bboxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])).squeeze(0).numpy().astype(np.float32)
            labels = labels.flatten().numpy().astype(np.uint8)

            return labels, bboxes


def encode_predictions(latency, labels, bboxes):
    latency = np.uint8(latency)

    print(f'[encode_predictions], latency: {latency.dtype.byteorder}')
    print(f'[encode_predictions], labels: {labels.dtype.byteorder}')
    print(f'[encode_predictions], bboxes: {bboxes.dtype.byteorder}')
    print(f'[encode_predictions], bboxes: {sys.byteorder}')



    labels = labels.tobytes()
    bboxes = bboxes.tobytes()

    byteslength_labels, byteslength_bboxes = np.uint32(len(labels)), np.uint32(len(bboxes))
    # print(f'[encode_predictions]: byeslength_labels: {byteslength_labels}')
    # print(f'[encode_predictions]: byeslength_labels: {byteslength_bboxes}')
    header = latency.tobytes() + byteslength_labels.tobytes() + byteslength_bboxes.tobytes()

    return header + labels + bboxes




#TODO: Impement configurable post_processor
def prediction_listener(connection, process, lock, postprocessor):
    print('--- pipe_listener initiated ---')

    while process.is_alive():
        message = connection.recv()
        if message['state'] == 'log':
            print(f"[model]: {message['log']}")
            continue

        if message['state'] == 'init':
            print(f"[prediction_listener]: received init from the model")
            init_logits = message['logits']
            init_bboxes = message['bboxes']

            logits_shm_name, logits_nbytes, logits_dtype, logits_shape = (
                init_logits['shared_memory_name'], init_logits['nbytes'], init_logits['dtype'], init_logits['shape'])
            bboxes_shm_name, bboxes_nbytes, bboxes_dtype, bboxes_shape = (
                init_logits['shared_memory_name'], init_bboxes['nbytes'], init_bboxes['dtype'], init_bboxes['shape'])

            logits_shared_memory = multiprocessing.shared_memory.SharedMemory(name=logits_shm_name)
            bboxes_shared_memory = multiprocessing.shared_memory.SharedMemory(name=bboxes_shm_name)

            print(f"[prediction_listener]: trying to access logits and bboxes")
            with lock:
                logits = np.ndarray(logits_shape, dtype=logits_dtype, buffer=logits_shared_memory.buf)
                bboxes = np.ndarray(bboxes_shape, dtype=bboxes_dtype, buffer=bboxes_shared_memory.buf)

            print(f"[prediction_listener], logits: {logits.shape}, bboxes: {bboxes.shape}")
            print(f'[prediction_listener], init successful!')
            continue

        if message['state'] == 'cls':
            latency = message['latency']
            print(f"[model] latency: {latency}")
            with lock:
                logits = np.ndarray(logits_shape, dtype=logits_dtype, buffer=logits_shared_memory.buf)
                bboxes = np.ndarray(bboxes_shape, dtype=bboxes_dtype, buffer=bboxes_shared_memory.buf)
            labels, bboxes = postprocessor(logits, bboxes)
            # print(f"[prediction_listener], post_logits_shape: {labels.shape}, post_bboxes_shape: {bboxes.shape}")
            # print(f"[prediction_listener], post_logits_dtype: {labels.dtype}, post_bboxes_dtype: {bboxes.dtype}")
            predictions_bytearray = encode_predictions(latency=latency, labels=labels, bboxes=bboxes)
            glib_bytes = GLib.Bytes.new(predictions_bytearray)

            # print(f'[prediction_listener]: sending bytearrays!')
            send_data_channel_message(glib_bytes)

            continue



        #TODO: Implement time logging for testing
        #TODO: Implement postprocessing of frames
        #TODO: Implement pipe from main to pipe_listener to configure the postprocessor
        #TODO: Implement sending the frames back to the WebRTC datachannels


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    global appsink
    global data_channel

    #TODO: Implement rtdetr cls instead of yolo
    #TODO: Implement appsrc for sending frames to the frontend


    # if SERVER_CORE_AFFINITY:
    #     server_process = psutil.Process()
    #     server_process.cpu_affinity(SERVER_CORE_AFFINITY)


    model_path = os.path.join('models', MODEL_PATH)
    pull_samples_to_model, from_pull_samples = multiprocessing.Pipe()
    model_to_postprocessor, from_model = multiprocessing.Pipe()
    pull_samples_lock = multiprocessing.Lock()
    prediction_listener_lock = multiprocessing.Lock()
    post_processor_lock = multiprocessing.Lock()

    post_processor = PostProcessor(TOPK, post_processor_lock)



    prediction_process = multiprocessing.Process(target=model_process,
                                                 kwargs={'model_path':model_path,
                                                         'receive_conn':from_pull_samples,
                                                         'post_conn':model_to_postprocessor,
                                                         'receive_lock':pull_samples_lock,
                                                         'send_lock':prediction_listener_lock})
    prediction_process.start()



    sample_puller = threading.Thread(target=pull_samples,
                                     kwargs={'appsink':appsink,
                                             'connection':pull_samples_to_model,
                                             'lock':pull_samples_lock,
                                             'process':prediction_process},
                                     daemon=True)
    sample_puller.start()

    prediction_postprocessor = threading.Thread(target=prediction_listener,
                                                kwargs={'connection':from_model,
                                                        'process': prediction_process,
                                                        'lock': prediction_listener_lock,
                                                        'postprocessor': post_processor},
                                                daemon=True)
    prediction_postprocessor.start()

    yield

    prediction_process.join()
    sample_puller.join()
    prediction_postprocessor.join()

    #TODO: Implement properly closing all the threads and processes
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









