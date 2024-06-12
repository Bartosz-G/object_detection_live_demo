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
import torch
import torch_neuron
from torchvision.ops import box_convert
import numpy as np
import threading
import queue
import asyncio
import json
import os
import sys
import logging


from time import perf_counter, sleep

from services.webrtcclient import WebRTCClient



HEIGHT = 480
WIDTH = 480
FORMAT = 'RGB'
MODEL_PATH = 'rtdetr480_neuron.pt'
SERVER_CORE_AFFINITY = None
MODEL_CORE_AFFINITY = None
TOPK = 10




class PostProcessor:
    def __init__(self, topk):
        self.topk = topk


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
    global logger
    logger = logging.getLogger("uvicorn")
    def log_assertions(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, AssertionError):
            logger.critical("", exc_info=(exc_type, exc_value, exc_traceback))
        else:
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = log_assertions



    Gst.init(None)

    webrtc = WebRTCClient(1, 1, {})
    prediction_task = PredictionTask(config={})


    yield

    webrtc.pipeline.set_state(Gst.State.NULL)



app = FastAPI(lifespan=lifespan)


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









