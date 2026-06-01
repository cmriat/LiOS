import json
import os
import re
from typing import Optional

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstSdp, GstWebRTC  # type: ignore

from ..ws_signal.signal_client import SignalClient


class WebRTCReceiver:
    """
    Minimal WebRTC receiver that links webrtcbin src_%u to a user-supplied
    RTP consumer chain described via a gst description string.

    Keep logic small: passive O/A and trickle ICE only; no reconnection loops,
    no dynamic decoder building.
    """

    def __init__(
        self,
        room: Optional[str] = None,
        peer_id: Optional[str] = None,
        signal_url: Optional[str] = None,
        stun: Optional[str] = None,
        turn: Optional[str] = None,
        latency_ms: int = 20,
    ) -> None:
        self.room = room or os.environ.get("ROOM", "demo")
        self.peer_id = peer_id or os.environ.get("PEER_ID", f"receiver-{os.getpid()}")
        self.signal_url = signal_url or os.environ.get("SIGNAL_URL", "ws://127.0.0.1:18080/ws")
        self.stun = stun or os.environ.get("STUN", "stun://stun.example.com")
        self.turn = turn or os.environ.get(
            "TURN", "turn://USERNAME:PASSWORD@TURN_HOST:3478?transport=udp"
        )
        self.latency_ms = latency_ms

        # Runtime state
        self.signal: Optional[SignalClient] = None
        self.remote_id: Optional[str] = None
        self._remote_sdp_set = False
        self._pending_local_ice: list[tuple[int, str]] = []
        self._pending_remote_ice: list[tuple[int, str]] = []

        # Build pipeline + webrtcbin
        self.pipe = Gst.Pipeline.new("webrtc-recv-pipeline")
        self.webrtc = Gst.ElementFactory.make("webrtcbin", "recvonly")
        if not self.webrtc:
            raise RuntimeError("webrtcbin not available")
        self.webrtc.set_property("bundle-policy", "max-bundle")
        self.webrtc.set_property("stun-server", self.stun)
        self.webrtc.set_property("turn-server", self.turn)
        self.webrtc.set_property("latency", self.latency_ms)

        self.pipe.add(self.webrtc)
        self.rtp_sink_desc: Optional[str] = None

        # Hooks
        self.webrtc.connect("pad-added", self._on_pad_added)
        self.webrtc.connect("on-ice-candidate", self._on_ice_candidate)

        # Bus logging
        bus = self.pipe.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_bus_error)
        bus.connect("message::warning", self._on_bus_warning)

    # -------------------------- Public --------------------------
    async def run(self) -> None:
        """Join signaling once and handle SDP/candidates. Keep logic minimal."""
        async with SignalClient(self.signal_url, self.room, self.peer_id) as sig:
            self.signal = sig
            print(f"[signal] connected: {self.signal_url}, room={self.room}, me={self.peer_id}")
            sig.join(role="receiver")
            sig.ready()

            # Start pipeline
            self.pipe.set_state(Gst.State.PLAYING)
            self._remote_sdp_set = False
            self._pending_remote_ice.clear()

            async for env in sig:
                if env.type == "offer":
                    self._handle_offer(env)
                elif env.type == "candidate":
                    self._handle_remote_candidate(env)

    # ------------------------- Signaling ------------------------
    def _handle_offer(self, env) -> None:
        payload = env.data
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = None
        sdp_txt = (payload or {}).get("sdp") if isinstance(payload, dict) else None
        self.remote_id = env.from_
        if not sdp_txt:
            print("[signal] offer missing sdp; ignoring")
            return

        # Set remote, create and send answer
        self._set_remote_sdp(sdp_txt, is_offer=True)
        self._create_and_send_answer()

        # Flush remote ICE received before remote SDP was set
        if self._pending_remote_ice:
            for mline, cand in self._pending_remote_ice:
                self.webrtc.emit("add-ice-candidate", int(mline), cand)
            self._pending_remote_ice.clear()

    def _handle_remote_candidate(self, env) -> None:
        payload = env.data
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = None
        cand = (payload or {}).get("candidate") if isinstance(payload, dict) else None
        mline = int((payload or {}).get("sdpMLineIndex", 0)) if isinstance(payload, dict) else 0
        if cand is None:
            return
        if not self._remote_sdp_set:
            self._pending_remote_ice.append((mline, cand))
            return
        self.webrtc.emit("add-ice-candidate", int(mline), cand)

    # -------------------------- WebRTC --------------------------
    def _set_remote_sdp(self, sdp_text: str, is_offer: bool) -> None:
        _, sdp = GstSdp.SDPMessage.new()
        GstSdp.sdp_message_parse_buffer(sdp_text.encode(), sdp)
        sdp_type = GstWebRTC.WebRTCSDPType.OFFER if is_offer else GstWebRTC.WebRTCSDPType.ANSWER
        desc = GstWebRTC.WebRTCSessionDescription.new(sdp_type, sdp)
        self.webrtc.emit("set-remote-description", desc, Gst.Promise.new())
        self._remote_sdp_set = True
        print(f"[webrtc] set remote {('offer' if is_offer else 'answer')}")

    def _create_and_send_answer(self) -> None:
        def on_answer_created(promise: Gst.Promise, _user_data, __user_data2=None):
            promise.wait()
            reply = promise.get_reply()
            if not reply or not reply.has_field("answer"):
                print("[webrtc] create-answer failed: empty reply")
                return
            answer = reply.get_value("answer")
            if not answer or not getattr(answer, "sdp", None):
                print("[webrtc] create-answer failed: no SDP")
                return
            self.webrtc.emit("set-local-description", answer, Gst.Promise.new())
            if not self.remote_id or not self.signal:
                print("[signal] cannot send answer: remote_id or signal missing")
                return
            self.signal.answer(self.remote_id, answer.sdp.as_text())
            print("[webrtc] sent answer")
            # Flush local ICE gathered before we knew remote_id
            if self._pending_local_ice:
                for mline, cand in self._pending_local_ice:
                    self._send_local_ice(mline, cand)
                self._pending_local_ice.clear()

        self.webrtc.emit(
            "create-answer",
            None,
            Gst.Promise.new_with_change_func(on_answer_created, None, None),
        )

    def _on_ice_candidate(self, _webrtc, mlineindex, candidate) -> None:
        if not self.remote_id:
            self._pending_local_ice.append((int(mlineindex), candidate))
            return
        self._send_local_ice(int(mlineindex), candidate)

    def _send_local_ice(self, mlineindex: int, candidate: str) -> None:
        if not self.signal or not self.remote_id:
            return
        self.signal.candidate(self.remote_id, int(mlineindex), candidate)

    # ---------------------- Sink description --------------------
    def set_rtp_sink_desc(self, desc: str) -> None:
        self.rtp_sink_desc = desc

    # ------------------------ Pad linking -----------------------
    def _on_pad_added(self, _webrtc, pad: Gst.Pad) -> None:
        if pad.get_direction() != Gst.PadDirection.SRC:
            return
        if not self.rtp_sink_desc:
            print("[receiver] no sink bin yet; drop incoming pad")
            return

        # Resolve msid from the incoming pad and bake it into the appsink name
        # at parse time. This guarantees the appsink is born with its final name,
        # so external observers iterating the pipeline never see a placeholder.
        msid = None
        try:
            msid = pad.get_property("msid")
        except Exception:
            msid = None

        desc = self.rtp_sink_desc
        if msid:
            desc, n = re.subn(
                r'(appsink\s+name=)\S+', rf'\g<1>{msid}', desc, count=1
            )
            if n == 0:
                print(f"[receiver] WARN: no 'appsink name=' in desc; cannot bind msid {msid}")

        bin_ = Gst.parse_bin_from_description(desc, True)
        if not bin_:
            raise RuntimeError("failed to parse sink description")
        self.pipe.add(bin_)
        _sink_pad = bin_.get_static_pad("sink")
        if not _sink_pad:
            raise RuntimeError("sink bin has no static 'sink' pad")
        bin_.sync_state_with_parent()
        print("[receiver] sink bin added from description")
        if pad.link(_sink_pad) != Gst.PadLinkReturn.OK:
            print("[receiver] ERROR: failed to link webrtc src -> sink bin")
            return
        print("[receiver] linked webrtc src to sink bin")
        if msid:
            print(f"[receiver] incoming pad has msid: {msid}")
            print(f"[receiver] appsink named: {msid}")

    # --------------------------- Bus ----------------------------
    def _on_bus_error(self, _bus, msg) -> None:
        err, dbg = msg.parse_error()
        print(f"[GStreamer][ERROR] {err} debug:{dbg}")

    def _on_bus_warning(self, _bus, msg) -> None:
        wrn, dbg = msg.parse_warning()
        print(f"[GStreamer][WARN ] {wrn} debug:{dbg}")
