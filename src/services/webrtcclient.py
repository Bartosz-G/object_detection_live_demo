import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
gi.require_version('GstWebRTC', '1.0')
from gi.repository import GstWebRTC
gi.require_version('GstSdp', '1.0')
from gi.repository import GstSdp

import asyncio
import json


class WebRTCClient:
    def __init__(self, id_: int, peer_id: str | int, config: dict):
        self.id_ = id_
        self.peer_id = peer_id
        self.websocket = None
        self.pipeline = None
        self.webrtcbin = None
        self.appsink = None
        self.datachannel = None
        self.bus = None

        self.height = config.get('height', 480)
        self.width = config.get('width', 480)
        self.format = config.get('format', 'RGB')
        self.stun = config.get('stun', 'stun://stun.kinesisvideo.eu-west-2.amazonaws.com:443')

        self.webrtcbin = Gst.ElementFactory.make("webrtcbin", "webrtcbin")
        self.appsink = Gst.ElementFactory.make("appsink", "appsink")

        self.webrtcbin.set_property("stun-server", "stun://stun.kinesisvideo.eu-west-2.amazonaws.com:443")

        self.appsink.set_property("emit-signals", True)
        self.appsink.set_property("drop", True)
        self.appsink.set_property("max-buffers", 1)
        self.appsink.set_property("sync", True)

        self.webrtcbin.connect('on-negotiation-needed', self.on_negotiation_needed)
        self.webrtcbin.connect('on-ice-candidate', self.send_ice_candidate_message)
        self.webrtcbin.connect('pad-added', self.on_stream_added)
        self.webrtcbin.connect('on-data-channel', self.on_data_channel)

        self.pipeline = Gst.Pipeline.new("pipeline")
        self.pipeline.add(self.webrtcbin)


    def get_webrtcbin(self):
        return self.webrtcbin

    def get_appsink(self):
        return self.appsink

    def set_websocket(self, websocket):
        self.websocket = websocket

    def get_datachannel(self):
        return self.datachannel

    def on_negotiation_needed(self, *args):
        assert self.websocket, "on_negotiation was called before websocket_connection was created"
        print("ON_NEGOTIATION_NEEDED was called: ", args)
        _ = asyncio.run(self.websocket.send_text('waitingForOffer'))

    def send_ice_candidate_message(self, _, mlineindex, candidate):
        print(f'INITIATING SENDING OF ICE CANDIDATE \n')
        assert self.websocket, f"send_ice_candidate_message was called before websocket_connection was created! \n candidate: {candidate} \n mlineindex: {mlineindex}"
        icemsg = json.dumps({'ice': {'candidate': candidate, 'sdpMLineIndex': mlineindex}})

        print(f'ICE candidate being sent: {icemsg}')
        asyncio.run(self.websocket.send_text(icemsg))

    def on_answer_created(self, promise, _):
        print(f'On answer creation initiated... \n')
        promise.wait()
        msg = promise.get_reply()
        answer = msg.get_value('answer')
        print(f'ANSWER: \n {answer.sdp.as_text()}')
        self.webrtcbin.emit('set-local-description', answer, promise)
        promise.interrupt()
        print(f'ON ANSWER CREATION COMPLETE \n')

        reply = json.dumps({'sdp': {'type': 'answer', 'sdp': answer.sdp.as_text()}})

        asyncio.run(self.websocket.send_text(reply))

    def on_decodebin_added(self, _, pad):
        print(f'[on_decodebin_added]: CALLED')

        if not pad.has_current_caps():
            print(pad, 'has no caps, ignoring')
            return

        caps = pad.get_current_caps()
        print(f'[on_decodebin_added]: INCOMING STREAM CAPABILITIES: {caps.to_string()}')
        name = caps.to_string()

        q = Gst.ElementFactory.make('queue')
        scale = Gst.ElementFactory.make('videoscale')
        conv = Gst.ElementFactory.make('videoconvert')
        capsfilter = Gst.ElementFactory.make('capsfilter')

        enforce_caps = Gst.Caps.from_string(f"video/x-raw,format={self.format},width={self.width},height={self.height}")
        capsfilter.set_property('caps', enforce_caps)

        self.pipeline.add(q)
        self.pipeline.add(scale)
        self.pipeline.add(conv)
        self.pipeline.add(capsfilter)
        self.pipeline.add(self.appsink)
        self.pipeline.sync_children_states()
        pad.link(q.get_static_pad('sink'))
        q.link(scale)
        scale.link(conv)
        conv.link(capsfilter)
        capsfilter.link(self.appsink)

        self.pipeline.set_state(Gst.State.PLAYING)

    def on_stream_added(self, _, pad):
        print(f'------ ON STREAM ADDED EVENT HAS BEEN CALLED ---------')
        print(f'------ STREAM IS NOW FLOWING  ---------')

        decodebin = Gst.ElementFactory.make('decodebin')
        decodebin.connect('pad-added', self.on_decodebin_added)
        self.pipeline.add(decodebin)
        decodebin.sync_state_with_parent()
        self.webrtcbin.link(decodebin)

        print(f'----- Everything connected, stream should be flowing')

#TODO: Fix the Datachannel
    def on_data_channel_ready(self, channel):
        print("[DataChannel]: ready")
        self.datachannel = channel
        # send_data_channel_message(b"Hello, Data Channel!")

    # def on_data_channel_open(self, *args, **kwargs):
    #     print("[DataChannel]: open")
    #     self.datachannel.connect("on-data", self.on_data_channel_message)
    #
    # def on_data_channel_message(self, data):
    #     print("[DataChannel], message received:", data)

    def send_data_channel_message(self, data):
        assert self.datachannel, f'[send_data_channel_message]: called but self.data_channels is {self.datachannel}'
        # print(f"[DataChannel], dir: {dir(data_channel)}")
        self.datachannel.send_data(data)

    def on_data_channel(self, _, data_channel):
        print("[DataChannel], created")
        # data_channel.connect("notify::ready-state", self.on_data_channel_open)
        data_channel.connect("on-open", self.on_data_channel_ready)


    def receive_message(self, msg):
        if 'offer' in msg:
            sdp = msg['offer']['sdp']
            print(f'Received an offer: \n {sdp}')
            res, sdpmsg = GstSdp.SDPMessage.new()
            GstSdp.sdp_message_parse_buffer(bytes(sdp.encode()), sdpmsg)
            offer = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.OFFER, sdpmsg)
            promise = Gst.Promise.new()
            self.webrtcbin.emit('set-remote-description', offer, promise)
            promise.interrupt()

            print('Creating an answer ... \n')
            promise = Gst.Promise.new_with_change_func(self.on_answer_created, None)
            self.webrtcbin.emit('create-answer', None, promise)



        elif 'ice' in msg:
            ice = msg['ice']
            print(f'Received an ICE candidate: {ice}')
            candidate = ice['candidate']
            sdpmlineindex = ice['sdpMLineIndex']
            self.webrtcbin.emit('add-ice-candidate', sdpmlineindex, candidate)




