import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from fastapi import FastAPI
from jinja2 import Environment, FileSystemLoader

print("GStreamer version:", Gst.version())


Gst.init(None)

app = FastAPI()
pipeline = Gst.parse_launch("webrtcbin name=webrtcbin ! vp8dec ! videoconvert ! autovideosink")


@app.get("/")
async def simple_response():
    return {"Hello": "from server"}

