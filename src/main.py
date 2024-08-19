from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse

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
from config import CONFIG


class PredictionTask:
    def __init__(self, config: dict, logger):
        self.logger = logger
        self.config = config
        self.height = config.get('height', 480)
        self.width = config.get('width', 480)
        self.model = torch.jit.load(config.get('model_path', 'models/rtdetr480_neuron.pt'))
        self.jitter_buffer = config.get('jitter_buffer', 10)

    def __call__(self, appsink, queue, stop_event, *args, **kwargs):
        self.logger.info(f'[PredictionTask]: initiated')


        while not stop_event.is_set():
            sample = appsink.emit("pull-sample")
            if sample is None:
                sleep(1)
                self.logger.debug(f'[PredictionTask]: no samples in the appsink')
                continue

            caps = sample.get_caps()
            self.logger.info(f'[PredictionTask]: received the first frame')
            self.logger.debug(f'[PredictionTask]: appsink capabilities: \n {caps.to_string()}')

            assert self.height == caps.get_structure(0).get_value(
                "height"), f'[pull_samples] appsink receive height: {caps.get_structure(0).get_value("height")}, expected: {self.height}'
            assert self.width == caps.get_structure(0).get_value(
                "width"), f'[pull_samples] appsink receive width: {caps.get_structure(0).get_value("width")}, expected: {self.width}'

            process_ts = perf_counter()

            buffer = sample.get_buffer()
            success, map_info = buffer.map(Gst.MapFlags.READ)

            if not success:
                self.logger.critical("[PredictionTask]: Could not map buffer data!")
                raise RuntimeError("[PredictionTask]: Could not map buffer data!")

            warmup_frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((self.height, self.width, 3)) / 255
            buffer.unmap(map_info)
            self.logger.debug(f'[PredictionTask]: mapped the warmup frame from the buffer')

            warmup_frame = np.transpose(warmup_frame, (2, 0, 1))
            warmup_frame = torch.from_numpy(warmup_frame).float().unsqueeze(0)
            self.logger.debug(f'[PredictionTask]: warmup frame.shape {warmup_frame.shape}')
            self.logger.debug(f'[PredictionTask]: warmup frame input \n {warmup_frame[0, 0, 0, :10]}')

            ts = perf_counter()
            output = self.model(warmup_frame)
            te = perf_counter()
            latency = (te - ts) * 1000
            self.logger.info(f'[PredictionTask] warmup frame latency: {latency}')
            latency += self.jitter_buffer

            msg = {'state':'cls',
                    'cls': {'latency': latency,
                            'logits': output[0],
                            'bbox_pred': output[1],
                            'time': process_ts}
                    }

            queue.put(msg)
            break

        self.logger.info(f'[PredictionTask]: starting continous detection process')
        while not stop_event.is_set():

            sample = appsink.emit("pull-sample")
            if sample is None:
                self.logger.warning(f'[PredictionTask]detection process: sample missing in the appsink')
                continue

            buffer = sample.get_buffer()
            _, map_info = buffer.map(Gst.MapFlags.READ)

            frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((self.height, self.width, 3)) / 255
            buffer.unmap(map_info)

            frame = np.transpose(frame, (2, 0, 1))
            frame = torch.from_numpy(frame).float().unsqueeze(0)
            self.logger.debug(f'[PredictionTask] detection process: frame input \n {frame[0, 0, 0, :10]}')

            ts = perf_counter()
            output = self.model(frame)
            te = perf_counter()
            latency = (te - ts) * 1000
            self.logger.info(f'[PredictionTask] detection process: latency {latency}ms')
            latency += self.jitter_buffer

            msg = {'state':'cls',
                    'cls': {'latency': latency,
                            'logits': output[0],
                            'bbox_pred': output[1]}
                    }

            queue.put(msg)



class PredictionSender:
    def __init__(self, postprocessor, webrtc, queue, logger):
        self.postprocessor = postprocessor
        self.webrtc = webrtc
        self.queue = queue
        self.logger = logger

    def __call__(self, stop_event, *args, **kwargs):
        while not stop_event.is_set():
            msg = self.queue.get()
            state = msg['state']

            if state == 'cls':
                data = msg['cls']
                latency, logits, bbox_pred, process_ts = data['latency'], data['logits'], data['bbox_pred'], data.get('time', None)

                if torch.isnan(logits).any().item():
                    self.logger.warning(f'[PredictionSender]: received NaN logits')
                    self.logger.debug(
                        f'[PredictionSender]: {torch.isnan(logits).sum().item() / logits.numel() * 100}% of logits are NaN')

                if torch.isnan(bbox_pred).any().item():
                    self.logger.warning(f'[PredictionSender]: received NaN bounding boxes')
                    self.logger.debug(
                        f'[PredictionSender]: {torch.isnan(bbox_pred).sum().item() / bbox_pred.numel() * 100}% of bboxes are NaN')

                labels, bboxes = self.postprocessor(logits, bbox_pred)

                predictions_bytearray = encode_predictions(latency=latency, labels=labels, bboxes=bboxes)
                glib_bytes = GLib.Bytes.new(predictions_bytearray)

                self.webrtc.send_data_channel_message(glib_bytes)
                if process_ts:
                    process_te = perf_counter()
                    process_total = (process_te - process_ts) * 1000
                    self.logger.info(f'[PredictionSender]: total process time {process_total}ms')



def postprocessing_task(prediction_sender, *args, **kwargs):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(prediction_sender(*args, **kwargs))



class PostProcessor:
    def __init__(self, config: dict, logger):
        self.topk = config.get('topk', 10)
        self.confidence = config.get('confidence', 0.25)
        self.logger = logger
        self.lock = threading.Lock()

    def set(self, msg):
        with self.lock:
            self.logger.debug(f'[PostProcessor]: received new configuration: {msg}')
            try:
                if 'topk' in msg:
                    new_top_k_value = int(msg['topk'])
                    assert 0 < new_top_k_value <= 300, f'Invalid change parameters: {new_top_k_value}'
                    self.topk = new_top_k_value
                    return None

                if 'confidence' in msg:
                    new_confidence = float(msg['confidence'])
                    assert 0 <= new_confidence <= 1, f'Invalid change parameters: {new_confidence}'
                    self.confidence = new_confidence
                    return None

            except Exception as e:
                self.logger.debug(f'[PostProcessor]: Error changing configuration: {e}')

    def __call__(self, logits, boxes, *args, **kwargs):
        with self.lock:
            with torch.no_grad():
                bbox_pred = box_convert(boxes, in_fmt='cxcywh', out_fmt='xywh')
                self.logger.debug(f'[PostProcessor]: after box conversion {bbox_pred[0, :3, :]}')
                bbox_pred = torch.clamp(bbox_pred, min=0, max=1)

                scores = torch.sigmoid(logits)

                if torch.isnan(scores).any().item():
                    self.logger.warning(f'[PostProcessor]: scores became NaN after sigmoid conversion!')
                scores, index = torch.topk(scores.flatten(1), self.topk, axis=-1)
                index = index[scores >= self.confidence]
                # self.logger.debug(f'[PostProcessor]: scores - {scores} ')
                # self.logger.debug(f'[PostProcessor]: scores - {scores.size()} ')
                # self.logger.debug(f'[PostProcessor]: index - {index.size()} ')
                labels = index % 80
                index = index // 80
                bboxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])).squeeze(0).numpy().astype(np.float32)
                labels = labels.flatten().numpy().astype(np.uint8)
                self.logger.debug(f'[PostProcessor]: sending labels {labels[:3]}')
                return labels, bboxes





def encode_predictions(latency, labels, bboxes):
    global logger
    latency = np.uint8(latency)

    labels = labels.tobytes()
    bboxes = bboxes.tobytes()

    byteslength_labels, byteslength_bboxes = np.uint32(len(labels)), np.uint32(len(bboxes))
    logger.debug(f'[encode_predictions]: byeslength_labels {byteslength_labels}')
    logger.debug(f'[encode_predictions]: byeslength_bboxes {byteslength_bboxes}')
    header = latency.tobytes() + byteslength_labels.tobytes() + byteslength_bboxes.tobytes()

    return header + labels + bboxes



@asynccontextmanager
async def lifespan(app: FastAPI):
    # global webrtc
    global prediction_task
    global logger
    logger = logging.getLogger("uvicorn")
    logger.setLevel(logging.DEBUG)

    def log_assertions(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, AssertionError):
            logger.critical("", exc_info=(exc_type, exc_value, exc_traceback))
        else:
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = log_assertions



    Gst.init(None)

    prediction_task = PredictionTask(config={}, logger=logger)


    yield

    # webrtc.pipeline.set_state(Gst.State.NULL)



app = FastAPI(lifespan=lifespan)


templates = Jinja2Templates(directory="/home/ubuntu/src/templates")
app.mount("/static", StaticFiles(directory="/home/ubuntu/src/static"), name="static")




@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    context = {
        'request': request,
        'stun': CONFIG.get('stun', 'stun://stun.kinesisvideo.eu-west-2.amazonaws.com:443'),
        'topk': CONFIG.get('topk', 10),
        'confidence': CONFIG.get('confidence', 0.25),
        'classifywidth': '{'+f' exact: {CONFIG.get("width", 480)}'+'} ',
        'classifyheight': '{' + f' exact: {CONFIG.get("height", 480)} ' + '}',
        'classifyframerate': '{' + f' exact: {CONFIG.get("frameRate", 20)} ' + '}'
    }
    return templates.TemplateResponse("index.html", context)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket):
    global logger
    global prediction_task
    await websocket.accept()
    try:
        data_queue = queue.Queue()

        webrtc = WebRTCClient(1, 1, CONFIG, logger=logger)
        websocket_connection = websocket
        webrtc.set_websocket(websocket = websocket_connection)

        stop_event = threading.Event()
        postprocessor = PostProcessor(config=CONFIG, logger=logger)
        prediction_sender = PredictionSender(postprocessor=postprocessor, webrtc=webrtc, queue=data_queue, logger=logger)
        appsink = webrtc.get_appsink()

        sender_thread = threading.Thread(target=prediction_sender, kwargs={'stop_event': stop_event})
        prediction_thread = threading.Thread(target=prediction_task, kwargs={'appsink': appsink,
                                                                             'queue': data_queue,
                                                                             'stop_event': stop_event})

        sender_thread.start()
        prediction_thread.start()

        logger.info(f'[websocket_endpoint]: listening for connections...')

        while True:
            data = await websocket_connection.receive_text()

            if data == 'PING':
                logger.info(f'[websocket_endpoint]: received a message: {data}')
                webrtc.pipeline.set_state(Gst.State.PLAYING)
                logger.debug(f'[websocket_endpoint]: gstreamer pipeline set to playing')
                continue

            msg = json.loads(data)

            if 'offer' in msg or 'ice' in msg:
                webrtc.receive_message(msg)

            if 'state' in msg:
                state = msg['state']
                if state == 'change':
                    change_config = msg.get('change', {})
                    postprocessor.set(change_config)



    except WebSocketDisconnect:
        logger.info(f'[websocket_endpoint]: user disconnected')
        stop_event.set()
        sender_thread.join()
        prediction_thread.join()
        webrtc.pipeline.set_state(Gst.State.NULL)
        logger.info(f'[websocket_endpoint]: disconnection cleanup complete')










