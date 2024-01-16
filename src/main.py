import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
gi.require_version('GstWebRTC', '1.0')
from gi.repository import GstWebRTC
gi.require_version('GstSdp', '1.0')
from gi.repository import GstSdp
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles



app = FastAPI()

templates = Jinja2Templates(directory="/home/ubuntu/src/templates")
app.mount("/static", StaticFiles(directory="/home/ubuntu/src/static"), name="static")

def on_offer_created(promise, _, webrtcbin):
    promise = promise.wait()
    reply = promise.get_reply()
    offer = reply.get_value('offer')

    promise = Gst.Promise.new()
    webrtcbin.emit('set-local-description', offer, promise)
    promise.interrupt()

    # Convert the offer to a string and send it to the client
    sdp_str = offer.sdp.as_text()
    print("SDP Offer:\n{}".format(sdp_str))
    # Here you would typically send the SDP offer to the client via your signaling mechanism

def on_negotiation_needed(element, _):
    promise = Gst.Promise.new_with_change_func(on_offer_created, element, element)
    element.emit('create-offer', None, promise)

Gst.init(None)
print("GStreamer version:", Gst.version())

pipeline = Gst.parse_launch(
    "webrtcbin name=webrtcbin ! decodebin name=decoder ! queue name=queue ! videoscale name=videoscale ! "
    "capsfilter name=capsfilter ! videoconvert name=convert"
)
webrtcbin = pipeline.get_by_name("webrtcbin")
webrtcbin.connect("on-negotiation-needed", on_negotiation_needed)
webrtcbin.emit('create-offer', None, None)

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


