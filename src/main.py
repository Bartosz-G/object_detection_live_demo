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

import asyncio
import json







app = FastAPI()

templates = Jinja2Templates(directory="/home/ubuntu/src/templates")
app.mount("/static", StaticFiles(directory="/home/ubuntu/src/static"), name="static")


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


    _ = asyncio.run(websocket_connection.send_text(reply))



pipeline = Gst.parse_launch(
    "webrtcbin name=webrtcbin stun-server=stun://stun.l.google.com:19302 ! decodebin name=decoder ! queue name=queue ! videoscale name=videoscale ! "
    "capsfilter name=capsfilter ! videoconvert name=convert"
)
webrtcbin = pipeline.get_by_name("webrtcbin")
webrtcbin.connect('on-negotiation-needed', on_negotiation_needed)
webrtcbin.connect('on-ice-candidate', send_ice_candidate_message)


# class Offer(BaseModel):
#     type: str
#     offer: dict


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket):
    global websocket_connection
    await websocket.accept()
    websocket_connection = websocket


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


