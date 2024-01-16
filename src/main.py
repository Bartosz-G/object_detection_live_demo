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

Gst.init(None)
print("GStreamer version:", Gst.version())

pipeline = Gst.parse_launch("webrtcbin name=webrtcbin ! vp8dec ! videoconvert ! autovideosink")


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


