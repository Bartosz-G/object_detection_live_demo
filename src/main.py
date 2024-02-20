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
import asyncio
import json
import signal



HEIGHT = 640
WIDTH = 640


Gst.init(None)
websocket_connection = None


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
    capsfilter = Gst.ElementFactory.make('capsfilter')
    conv = Gst.ElementFactory.make('videoconvert')


    # capsfilter.set_property('caps', Gst.Caps.from_string(f"video/x-raw,width={WIDTH},height={HEIGHT},format=RGB"))

    enforce_caps = Gst.Caps.from_string(f"video/x-raw,width={WIDTH},height={HEIGHT}")
    capsfilter.set_property('caps', enforce_caps)


    pipeline.add(q)
    pipeline.add(scale)
    pipeline.add(capsfilter)
    pipeline.add(conv)
    pipeline.add(appsink)
    pipeline.sync_children_states()
    pad.link(q.get_static_pad('sink'))
    q.link(scale)
    scale.link(capsfilter)
    capsfilter.link(conv)
    conv.link(appsink)

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





async def pull_samples(appsink):
    print(f'--- PULL SAMPLE CALLED ----')
    global pipeline
    try:
        while True:
            await asyncio.sleep(6)

            state_return, state, pending_state = pipeline.get_state(Gst.CLOCK_TIME_NONE)
            if state == Gst.State.PAUSED:
                print(f'Pipeline has been paused, continuing')
                continue

            sample = appsink.emit("pull-sample")
            if sample is None:
                print("No samples in appsink")
                continue

            caps = sample.get_caps()
            print(f"Pulled a sample from appsink: \n {sample}")
            print(f"CAPS: \n {caps.to_string()}")

    except Exception as e:
        print(f'Exception occured when trying to pull from the appsink: {e}')


    # except asyncio.CancelledError:
    #     print(f'Gracefully stopping pulling samples...')


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pulling_task = asyncio.create_task(pull_samples(appsink=appsink))

    yield

    pulling_task.cancel()
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









