import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
gi.require_version('GstWebRTC', '1.0')
from gi.repository import GstWebRTC
gi.require_version('GstSdp', '1.0')
from gi.repository import GstSdp
from fastapi import FastAPI, Request, HTTPException, WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json



app = FastAPI()

templates = Jinja2Templates(directory="/home/ubuntu/src/templates")
app.mount("/static", StaticFiles(directory="/home/ubuntu/src/static"), name="static")


Gst.init(None)
websocket_connection = None


def on_negotiation_needed(*args):
    assert websocket_connection, "on_negotiation was called before websocket_connection was created"
    print("on_negotiation_needed was called: ", args)
    websocket_connection.send_text('waitingForOffer')


def send_ice_candidate_message(self, _, mlineindex, candidate):
    assert websocket_connection, f"send_ice_candidate_message was called before websocket_connection was created! \n candidate: {candidate} \n mlineindex: {mlineindex}"
    icemsg = json.dumps({'ice': {'candidate': candidate, 'sdpMLineIndex': mlineindex}})


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
    await websocket.accept()
    websocket_connection = websocket



    while True:
        data = await websocket_connection.receive_text()

        if data == 'PING':
            print(f'Received a message: {data}')
            await websocket_connection.send_text('PONG')
            continue












# @app.post("/offer")
# async def receive_offer(offer: Offer):
#     print("Offer.json() :", offer.json())
#
#     try:
#         offer_sdp = offer.offer['sdp']
#
#         print(f'offer.offer["offer"]: {offer_sdp}')
#
#         res, message_from_back_end = GstSdp.SDPMessage.new()
#         GstSdp.sdp_message_parse_buffer(bytes(offer_sdp.encode()), message_from_back_end)
#         answer_sdp = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.ANSWER, message_from_back_end)
#
#         print('--------------------------------------------------')
#
#
#
#         print(f'Dir: {dir(answer_sdp.sdp.get_connection)}')
#
#         print(f'get_connection.get_attribute: {answer_sdp.sdp.get_connection.get_arguments()}')
#         print(f'Answer methods: {dir(answer_sdp.sdp)}')
#
#     except Exception as e:
#         print(e)
#         return {"status": 300}
#
#     return {"status": 200}







