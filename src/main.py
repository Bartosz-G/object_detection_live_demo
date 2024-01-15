import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles



app = FastAPI()

templates = Jinja2Templates(directory="/templates")
app.mount("/static", StaticFiles(directory="/static"), name="static")

Gst.init(None)
# pipeline = Gst.parse_launch("webrtcbin name=webrtcbin ! vp8dec ! videoconvert ! autovideosink")
print("GStreamer version:", Gst.version())


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


