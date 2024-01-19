import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
gi.require_version('GstWebRTC', '1.0')
from gi.repository import GstWebRTC
gi.require_version('GstSdp', '1.0')
from gi.repository import GstSdp
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json



app = FastAPI()

templates = Jinja2Templates(directory="/home/ubuntu/src/templates")
app.mount("/static", StaticFiles(directory="/home/ubuntu/src/static"), name="static")


Gst.init(None)

pipeline = Gst.parse_launch(
    "webrtcbin name=webrtcbin ! decodebin name=decoder ! queue name=queue ! videoscale name=videoscale ! "
    "capsfilter name=capsfilter ! videoconvert name=convert"
)
webrtcbin = pipeline.get_by_name("webrtcbin")

res, sdpmsg = GstSdp.SDPMessage.new()
print(f'res: {res}')
print(f'sdpmsg: {sdpmsg.as_text()}')


class Offer(BaseModel):
    type: str
    offer: dict

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/offer")
async def receive_offer(offer: Offer):
    print("Received offer:", offer.json())

    try:
        offer_sdp = offer.offer['offer']

        print(f'{offer_sdp}')

        res, message_from_back_end = GstSdp.SDPMessage.new()
        GstSdp.sdp_message_parse_buffer(bytes(offer_sdp.encode()), message_from_back_end)
        answer_sdp = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.ANSWER, message_from_back_end)

        print('--------------------------------------------------')
        print(f'Answer_sdp text: {answer_sdp.as_text()}')

    except Exception as e:
        print(e)
        return {"status": 300}

    return {"status": 200}











# def on_offer_created(promise, _, webrtcbin):
#     promise = promise.wait()
#     reply = promise.get_reply()
#     offer = reply.get_value('offer')
#
#     promise = Gst.Promise.new()
#     webrtcbin.emit('set-local-description', offer, promise)
#     promise.interrupt()
#
#     # Convert the offer to a string and send it to the client
#     sdp_str = offer.sdp.as_text()
#     print("SDP Offer:\n{}".format(sdp_str))
#     # Here you would typically send the SDP offer to the client via your signaling mechanism
#
# def on_negotiation_needed(element, _):
#     promise = Gst.Promise.new_with_change_func(on_offer_created, element, element)
#     element.emit('create-offer', None, promise)
#
#
#
#
# pipeline.set_state(Gst.State.PLAYING)
# print("Pipeline started")
# loop = GLib.MainLoop()
# try:
#     loop.run()
# except KeyboardInterrupt:
#     loop.quit()