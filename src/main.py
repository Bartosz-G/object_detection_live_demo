from fastapi import FastAPI, Request, HTTPException, WebSocket, BackgroundTasks
from starlette.concurrency import run_in_threadpool
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
from torchvision.ops import box_convert
import numpy as np
import multiprocessing
import threading
import queue
import asyncio
import json
import os
import sys

# For measuring performance
from functools import wraps
from time import perf_counter, sleep

from services.webrtcclient import WebRTCClient



HEIGHT = 480
WIDTH = 480
FORMAT = 'RGB'
MODEL_PATH = 'rtdetr480_neuron.pt'
SERVER_CORE_AFFINITY = None
MODEL_CORE_AFFINITY = None
TOPK = 10



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




# def pull_samples(appsink, connection, lock, process):
#     # global pipeline
#     print('--- pipe_samples initiated ---')
#
#
#     try:
#         while True:
#             sample = appsink.emit("pull-sample") # Dereferencing a pointer
#             if sample is None:
#                 sleep(1)
#                 print(f'No samples in the appsink')
#                 continue
#
#             caps = sample.get_caps()
#             print(f'[pull_samples] APPSINK CAPABILITIES: {caps.to_string()}')
#
#             assert HEIGHT == caps.get_structure(0).get_value("height"), f'[pull_samples] appsink receive height: {caps.get_structure(0).get_value("height")}, expected: {HEIGHT}'
#             assert WIDTH == caps.get_structure(0).get_value("width"), f'[pull_samples] appsink receive width: {caps.get_structure(0).get_value("width")}, expected: {WIDTH}'
#
#             buffer = sample.get_buffer()
#             success, map_info = buffer.map(Gst.MapFlags.READ)
#
#             if not success:
#                 raise RuntimeError("Could not map buffer data!")
#
#             warmup_frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)) / 255
#
#             # warmup_frame = np.ndarray(
#             #     shape=(HEIGHT, WIDTH, 3),
#             #     dtype=np.uint8,
#             #     buffer=map_info.data) / 255
#
#             warmup_frame = np.transpose(warmup_frame, (2, 0, 1))
#             buffer.unmap(map_info)
#
#             frame_nbytes, frame_dtype, frame_shape = warmup_frame.nbytes, warmup_frame.dtype, warmup_frame.shape
#
#
#             memory = shared_memory.SharedMemory(create=True, size=frame_nbytes)
#
#             with lock:
#                 send_array = np.ndarray(frame_shape, dtype=frame_dtype, buffer=memory.buf)
#                 np.copyto(send_array, warmup_frame)
#
#             init_message = {
#                 'state': 'init',
#                 'shared_memory_name': memory.name,
#                 'dtype': frame_dtype,
#                 'shape': frame_shape,
#                 'nbytes': frame_nbytes
#             }
#
#             connection.send(init_message)
#             break
#
#         while process.is_alive():
#             sample = appsink.emit("pull-sample") # Dereferencing a pointer
#             if sample is None:
#                 continue
#
#
#             buffer = sample.get_buffer()
#             _, map_info = buffer.map(Gst.MapFlags.READ)
#
#             frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)) / 255
#             buffer.unmap(map_info)
#
#             # frame = np.ndarray(
#             #     shape=(HEIGHT, WIDTH, 3),
#             #     dtype=np.uint8,
#             #     buffer=map_info.data) / 255
#
#             frame = np.transpose(frame, (2, 0, 1))
#             print(f'[pull_samples] sending input: {frame[0, 0, :10]}')
#             with lock:
#                 shared_array = np.ndarray(frame_shape, dtype=frame_dtype, buffer=memory.buf)
#                 np.copyto(shared_array, frame)
#
#
#             connection.send({'state': 'cls'})
#
#     except Exception as e:
#         print(f'Exception in pull_samples occured: {e}')


# def model_process(model_path, receive_conn, post_conn, receive_lock, send_lock):
#
#     model = torch.jit.load(model_path)
#
#     try:
#         while True:
#             msg = receive_conn.recv()
#
#             if msg['state'] not in ('cls', 'init', 'close'):
#                 raise Exception(f'received an unrecognised message from receive connection: {msg}')
#
#             if msg['state'] == 'log':
#                 post_conn.send(msg)
#                 continue
#
#             if msg['state'] == 'init':
#                 post_conn.send({'state': 'log', 'log': 'starting_warmup_process'})
#                 receive_shm_name, dtype, shape = msg['shared_memory_name'], msg['dtype'], msg['shape']
#                 receive_shared_memory = multiprocessing.shared_memory.SharedMemory(name=receive_shm_name)
#
#                 post_conn.send({'state': 'log', 'log': 'pulling_warmup_frame'})
#                 with receive_lock:
#                     warmup_frame = np.ndarray(shape, dtype=dtype, buffer=receive_shared_memory.buf)
#
#                 post_conn.send({'state':'log', 'log': 'warming-up model'})
#                 warmup_frame = torch.from_numpy(warmup_frame).unsqueeze(0).float()
#                 output = model(warmup_frame)
#                 post_conn.send({'state': 'log', 'log': 'warming-up complete'})
#
#                 logits, bboxes = output[0].numpy(), output[1].numpy()
#                 logits_nbytes, logits_dtype, logits_shape = logits.nbytes, logits.dtype, logits.shape
#                 bboxes_nbytes, bboxes_dtype, bboxes_shape = bboxes.nbytes, bboxes.dtype, bboxes.shape
#
#                 post_conn.send({'state':'log', 'log': 'creating shared memory for the postprocessor'})
#                 send_logits_shared_memory = multiprocessing.shared_memory.SharedMemory(create=True, size=logits_nbytes)
#                 send_bboxes_shared_memory = multiprocessing.shared_memory.SharedMemory(create=True, size=bboxes_nbytes)
#
#                 post_conn.send({'state': 'log', 'log': 'putting warmup logits and frames to memory'})
#                 with send_lock:
#                     shared_logits = np.ndarray(logits_shape, dtype=logits_dtype, buffer=send_logits_shared_memory.buf)
#                     np.copyto(shared_logits, logits)
#                     shared_bboxes = np.ndarray(bboxes_shape, dtype=bboxes.dtype, buffer=send_bboxes_shared_memory.buf)
#                     np.copyto(shared_bboxes, bboxes)
#
#                 init_message = {
#                     'state': 'init',
#                     'logits': {
#                         'shared_memory_name': send_logits_shared_memory.name,
#                         'dtype': logits_dtype,
#                         'shape': logits_shape,
#                         'nbytes': logits_nbytes
#                     },
#                     'bboxes': {
#                         'shared_memory_name': send_bboxes_shared_memory.name,
#                         'dtype': bboxes_dtype,
#                         'shape': bboxes_shape,
#                         'nbytes': bboxes_nbytes
#                     }
#                 }
#
#                 post_conn.send(init_message)
#
#                 continue
#
#             #TODO: Refactor to not have to wait for a message and be able to respond to messages
#             if msg['state'] == 'cls':
#                 with receive_lock:
#                     received_array = np.ndarray(shape, dtype=dtype, buffer=receive_shared_memory.buf)
#                     # print(f'[model_process] taken input: {received_array[0, 0, :10]}')
#                 # post_conn.send({'state': 'log', 'log': {'received_array': received_array[:4, :4]}})
#                 frame = torch.from_numpy(received_array).float().unsqueeze(0)
#                 print(f'[model_process] received input: {frame[0, 0, 0, :10]}')
#
#
#                 ts = perf_counter()
#                 # post_conn.send({'state': 'log', 'log': {'frame ': frame[:4, :4]}})
#                 # post_conn.send({'state': 'log', 'log': {'frame_shape ': frame.shape}})
#                 output = model(frame)
#                 te = perf_counter()
#                 elapsed_time = (te - ts) * 1000
#                 latency = elapsed_time
#
#                 # Unpacking bboxes and labels
#                 logits, bbox_pred = output[0].numpy(), output[1].numpy()
#                 # post_conn.send({'state': 'log', 'log':{'logits': logits[:4], 'bboxes': bbox_pred[:4]}})
#                 # ============= Still correct =======================
#
#                 with send_lock:
#                     shared_logits = np.ndarray(logits_shape, dtype=logits_dtype, buffer=send_logits_shared_memory.buf)
#                     np.copyto(shared_logits, logits)
#                     shared_bboxes = np.ndarray(bboxes_shape, dtype=bboxes.dtype, buffer=send_bboxes_shared_memory.buf)
#                     np.copyto(shared_bboxes, bboxes)
#
#                 post_conn.send({'state': 'cls', 'latency': latency})
#
#     except Exception as e:
#         post_conn.send(e)


class PostProcessor:
    def __init__(self, topk):
        # self.lock = lock
        self.topk = topk

    # def set_topk(self, topk):
    #     with self.lock:
    #         self.topk = topk

    def __call__(self, logits, boxes, *args, **kwargs):
        # with self.lock:
        # print(f'[postprocessor] before conversion: {boxes[0, 0, :]}')
        bbox_pred = box_convert(boxes, in_fmt='cxcywh', out_fmt='xywh')
        print(f'[postprocessor] after conversion: {bbox_pred[0, :5, :]}')
        bbox_pred = torch.clamp(bbox_pred, min=0, max=1)

        scores = torch.sigmoid(logits)
        scores, index = torch.topk(scores.flatten(1), self.topk, axis=-1)
        labels = index % 80
        index = index // 80
        # print(f'[PostProcessor] step 2: {bbox_pred[:4]}')
        bboxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])).squeeze(0).numpy().astype(np.float32)
        labels = labels.flatten().numpy().astype(np.uint8)
        # print(f'[PostProcessor] step 3: {bbox_pred[:4]}')
        return labels, bboxes




# #TODO: Impement configurable post_processor
# def prediction_listener(connection, process, lock, postprocessor, webrtc):
#     print('--- pipe_listener initiated ---')
#
#     while process.is_alive():
#         message = connection.recv()
#         if message['state'] == 'log':
#             print(f"[model]: {message['log']}")
#             continue
#
#         if message['state'] == 'init':
#             print(f"[prediction_listener]: received init from the model")
#             init_logits = message['logits']
#             init_bboxes = message['bboxes']
#
#             logits_shm_name, logits_nbytes, logits_dtype, logits_shape = (
#                 init_logits['shared_memory_name'], init_logits['nbytes'], init_logits['dtype'], init_logits['shape'])
#             bboxes_shm_name, bboxes_nbytes, bboxes_dtype, bboxes_shape = (
#                 init_bboxes['shared_memory_name'], init_bboxes['nbytes'], init_bboxes['dtype'], init_bboxes['shape'])
#
#             logits_shared_memory = multiprocessing.shared_memory.SharedMemory(name=logits_shm_name)
#             bboxes_shared_memory = multiprocessing.shared_memory.SharedMemory(name=bboxes_shm_name)
#
#             print(f"[prediction_listener]: trying to access logits and bboxes")
#             with lock:
#                 logits = np.ndarray(logits_shape, dtype=logits_dtype, buffer=logits_shared_memory.buf)
#                 bboxes = np.ndarray(bboxes_shape, dtype=bboxes_dtype, buffer=bboxes_shared_memory.buf)
#
#             print(f"[prediction_listener], logits: {logits.shape}, bboxes: {bboxes.shape}")
#             print(f'[prediction_listener], init successful!')
#             continue
#
#         if message['state'] == 'cls':
#             latency = message['latency']
#             print(f"[model] latency: {latency}")
#             with lock:
#                 logits = np.ndarray(logits_shape, dtype=logits_dtype, buffer=logits_shared_memory.buf)
#                 bboxes = np.ndarray(bboxes_shape, dtype=bboxes_dtype, buffer=bboxes_shared_memory.buf)
#
#             # print(f'[prediction_listener] bboxes: {bboxes[:4]}')
#             labels, bboxes = postprocessor(logits, bboxes)
#             # print(f"[prediction_listener], post_logits_shape: {labels.shape}, post_bboxes_shape: {bboxes.shape}")
#             # print(f"[prediction_listener], post_logits_dtype: {labels.dtype}, post_bboxes_dtype: {bboxes.dtype}")
#             predictions_bytearray = encode_predictions(latency=latency, labels=labels, bboxes=bboxes)
#             glib_bytes = GLib.Bytes.new(predictions_bytearray)
#
#             # print(f'[prediction_listener]: sending bytearrays!')
#             webrtc.send_data_channel_message(glib_bytes)
#
#             continue

def encode_predictions(latency, labels, bboxes):
    latency = np.uint8(latency)

    # print(f'[encode_predictions], latency: {latency.dtype.byteorder}')
    # print(f'[encode_predictions], labels: {labels.dtype.byteorder}')
    # print(f'[encode_predictions], bboxes: {bboxes.dtype.byteorder}')
    # print(f'[encode_predictions], bboxes: {sys.byteorder}')


    # print(f'[encode_predictions]: {bboxes[:4]}')


    labels = labels.tobytes()
    bboxes = bboxes.tobytes()

    byteslength_labels, byteslength_bboxes = np.uint32(len(labels)), np.uint32(len(bboxes))
    # print(f'[encode_predictions]: byeslength_labels: {byteslength_labels}')
    # print(f'[encode_predictions]: byeslength_labels: {byteslength_bboxes}')
    header = latency.tobytes() + byteslength_labels.tobytes() + byteslength_bboxes.tobytes()

    return header + labels + bboxes



@asynccontextmanager
async def lifespan(app: FastAPI):
    global webrtc
    global prediction_task
    Gst.init(None)
    webrtc = WebRTCClient(1, 1, {})
    prediction_task = PredictionTask(config={})

    yield

    webrtc.pipeline.set_state(Gst.State.NULL)



app = FastAPI(lifespan=lifespan)
# templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="/home/ubuntu/src/templates")
app.mount("/static", StaticFiles(directory="/home/ubuntu/src/static"), name="static")





class PredictionTask:
    def __init__(self, config: dict):
        self.config = config
        self.height = config.get('height', 480)
        self.width = config.get('width', 480)
        self.model = torch.jit.load(os.path.join('models', config.get('model_path', 'rtdetr480_neuron.pt')))

    def __call__(self, appsink, queue, loop):
        print('--- pipe_samples initiated ---')


        while True:
            sample = appsink.emit("pull-sample")
            if sample is None:
                sleep(1)
                print(f'No samples in the appsink')
                continue

            caps = sample.get_caps()
            print(f'[pull_samples] APPSINK CAPABILITIES: {caps.to_string()}')

            assert self.height == caps.get_structure(0).get_value(
                "height"), f'[pull_samples] appsink receive height: {caps.get_structure(0).get_value("height")}, expected: {self.height}'
            assert self.width == caps.get_structure(0).get_value(
                "width"), f'[pull_samples] appsink receive width: {caps.get_structure(0).get_value("width")}, expected: {self.width}'

            buffer = sample.get_buffer()
            success, map_info = buffer.map(Gst.MapFlags.READ)

            if not success:
                raise RuntimeError("Could not map buffer data!")

            warmup_frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((self.height, self.width, 3)) / 255
            buffer.unmap(map_info)

            warmup_frame = np.transpose(warmup_frame, (2, 0, 1))
            warmup_frame = torch.from_numpy(warmup_frame).float().unsqueeze(0)
            print(f'[model_process] received input: {warmup_frame[0, 0, 0, :10]}')

            ts = perf_counter()
            output = self.model(warmup_frame)
            te = perf_counter()
            latency = (te - ts) * 1000

            msg = {'state':'cls',
                    'cls': {'latency': latency,
                            'logits': output[0],
                            'bbox_pred': output[1]}
                    }

            queue.put(msg)
            break


        while True:
            sample = appsink.emit("pull-sample")
            if sample is None:
                continue

            buffer = sample.get_buffer()
            _, map_info = buffer.map(Gst.MapFlags.READ)

            frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)) / 255
            buffer.unmap(map_info)

            frame = np.transpose(frame, (2, 0, 1))
            frame = torch.from_numpy(frame).float().unsqueeze(0)
            print(f'[model_process] received input: {frame[0, 0, 0, :10]}')

            ts = perf_counter()
            output = self.model(frame)
            te = perf_counter()
            latency = (te - ts) * 1000

            msg = {'state':'cls',
                    'cls': {'latency': latency,
                            'logits': output[0],
                            'bbox_pred': output[1]}
                    }

            queue.put(msg)



class PredictionSender:
    def __init__(self, postprocessor, webrtc, queue):
        self.postprocessor = postprocessor
        self.webrtc = webrtc
        self.queue = queue

    def __call__(self, *args, **kwargs):
        while True:
            msg = self.queue.get()
            state = msg['state']

            if state == 'cls':
                data = msg['cls']
                latency, logits, bbox_pred = data['latency'], data['logits'], data['bbox_pred']

                labels, bboxes = self.postprocessor(logits, bbox_pred)

                predictions_bytearray = encode_predictions(latency=latency, labels=labels, bboxes=bboxes)
                glib_bytes = GLib.Bytes.new(predictions_bytearray)

                self.webrtc.send_data_channel_message(glib_bytes)


def postprocessing_task(prediction_sender, *args, **kwargs):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(prediction_sender(*args, **kwargs))



@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket):
    global webrtc
    global prediction_task
    await websocket.accept()
    data_queue = queue.Queue()
    # loop = asyncio.get_event_loop()
    loop = None

    websocket_connection = websocket
    webrtc.set_websocket(websocket = websocket_connection)

    postprocessor = PostProcessor(topk=TOPK)
    prediction_sender = PredictionSender(postprocessor=postprocessor, webrtc=webrtc, queue=data_queue)

    threading.Thread(target=postprocessing_task, kwargs={'prediction_sender': prediction_sender}, daemon=True).start()

    appsink = webrtc.get_appsink()

    threading.Thread(target=prediction_task, kwargs={'appsink': appsink, 'queue': data_queue, 'loop': None}, daemon=True).start()

    # asyncio.create_task(run_in_threadpool(prediction_task, appsink, data_queue, loop))
    print('======= listening for connections now ====================')

    while True:
        data = await websocket_connection.receive_text()

        if data == 'PING':
            print(f'Received a message: {data}')
            webrtc.pipeline.set_state(Gst.State.PLAYING)

            continue

        msg = json.loads(data)
        webrtc.receive_message(msg)









