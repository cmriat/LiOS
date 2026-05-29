import os
import sys
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")
from gi.repository import Gst, GstSdp, GstWebRTC  # type: ignore

from ..ws_signal.signal_client import SignalClient


@dataclass
class SourceHandle:
    id: str
    bin: Gst.Bin
    sink_pad: Gst.Pad


class WebRTCSender:
    """
    Engineering-grade WebRTC sender built around GStreamer webrtcbin.

    Key features
    - Dynamic source add via GST description strings (gst-launch style).
    - Multi-stream: each source is linked to a distinct webrtcbin sink_%u pad.
    - Simple perfect-negotiation guard and signaling integration.
    - Minimal, clean surface for apps/CLI to use.
    """

    def __init__(
        self,
        room: str | None = None,
        peer_id: str | None = None,
        signal_url: str | None = None,
        stun: str | None = None,
        turn: str | None = None,
        latency_ms: int = 20,
    ) -> None:
        self.room = room or os.environ.get("ROOM", "demo")
        self.peer_id = peer_id or os.environ.get("PEER_ID", f"sender-{os.getpid()}")
        self.signal_url = signal_url or os.environ.get("SIGNAL_URL", "ws://127.0.0.1:18080/ws")
        self.stun = stun or os.environ.get("STUN", "stun://stun.example.com")
        self.turn = turn or os.environ.get(
            "TURN", "turn://USERNAME:PASSWORD@TURN_HOST:3478?transport=udp"
        )
        self.latency_ms = latency_ms

        self.pipe = Gst.Pipeline.new("webrtc-send-pipeline")
        self.webrtc = Gst.ElementFactory.make("webrtcbin", "sendonly")
        if not self.webrtc:
            print("[fatal] webrtcbin not found")
            sys.exit(2)

        self.webrtc.set_property("bundle-policy", "max-bundle")
        self.webrtc.set_property("stun-server", self.stun)
        self.webrtc.set_property("turn-server", self.turn)
        self.webrtc.set_property("latency", self.latency_ms)

        # Negotiation flags (perfect negotiation style)
        self._making_offer = False
        self._remote_id: Optional[str] = None

        self.pipe.add(self.webrtc)

        # Bus logging
        bus = self.pipe.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_bus_error)
        bus.connect("message::warning", self._on_bus_warning)

        # WebRTC hooks
        self.webrtc.connect("on-negotiation-needed", self._on_negotiation_needed)
        self.webrtc.connect("on-ice-candidate", self._on_ice_candidate)
        self.webrtc.connect("notify::connection-state", self._on_pc_state)
        self.webrtc.connect("notify::ice-connection-state", self._on_ice_state)

        # Sources bookkeeping
        self._sources: Dict[str, SourceHandle] = {}

        # Signaling
        self.signal: Optional[SignalClient] = None
        self._ice_disconnected_at: Optional[float] = None
        self._ice_restart_deadline: Optional[float] = None  # escalate to reset after this
        self._last_remote_left_ts: Optional[float] = None

    # --------------------- Public API ---------------------
    def add_video_source_desc(self, desc: str) -> SourceHandle:
        """
        Add a video source from a GST description string.

        Example desc:
            "videotestsrc is-live=true pattern=ball ! video/x-raw,framerate=30/1 ! queue ! x264enc tune=zerolatency ! h264parse config-interval=-1 ! rtph264pay pt=96"
        Requirement: the description should output `application/x-rtp` (e.g., via `rtph264pay`, `rtpvp8pay`). We link its `src` to a new webrtcbin `sink_%u` pad.
        """
        bin_ = Gst.parse_bin_from_description(desc, True)
        if not bin_:
            raise RuntimeError("Failed to parse GST description")

        self.pipe.add(bin_)
        sink_pad = self.webrtc.request_pad_simple("sink_%u")
        if not sink_pad:
            self.pipe.remove(bin_)
            raise RuntimeError("Failed to request webrtcbin sink pad")

        src_pad = bin_.get_static_pad("src")
        if not src_pad:
            self.webrtc.release_request_pad(sink_pad)
            self.pipe.remove(bin_)
            raise RuntimeError("Source bin has no static 'src' pad")

        if src_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
            self.webrtc.release_request_pad(sink_pad)
            self.pipe.remove(bin_)
            raise RuntimeError("Failed to link source->webrtcbin")

        bin_.sync_state_with_parent()
        hid = uuid.uuid4().hex[:8]
        handle = SourceHandle(id=hid, bin=bin_, sink_pad=sink_pad)
        self._sources[hid] = handle

        # Best-effort: label the webrtc sink pad's msid after the source bin's
        # `identity` element (e.g. cam0/cam1) so the receiver can tell streams
        # apart. Cosmetic only — must never break adding the source.
        try:
            label = self._source_label(bin_) or f"src-{hid}"
            sink_pad.set_property("msid", label)
            print(f"[sender] set msid on sink pad: {label}")
        except Exception as e:
            print(f"[sender] msid labeling skipped: {e}")

        print(f"[sender] added source id={hid}")
        return handle

    @staticmethod
    def _source_label(bin_: Gst.Bin) -> Optional[str]:
        """Name of the bin's `identity` element (e.g. cam0), or None."""
        it = bin_.iterate_recurse()
        while True:
            res, elem = it.next()
            if res == Gst.IteratorResult.OK:
                fac = elem.get_factory()
                if fac and fac.get_name() == "identity":
                    nm = elem.get_name()
                    if nm and nm != "identity0":
                        return nm
            elif res == Gst.IteratorResult.RESYNC:
                it.resync()
            else:  # DONE or ERROR
                return None

    def remove_source(self, source_id: str) -> bool:
        h = self._sources.pop(source_id, None)
        if not h:
            return False
        try:
            # Unlink and release request pad
            src_pad = h.bin.get_static_pad("src")
            if src_pad and src_pad.is_linked():
                src_pad.unlink(h.sink_pad)
            self.webrtc.release_request_pad(h.sink_pad)
            h.bin.set_state(Gst.State.NULL)
            self.pipe.remove(h.bin)
            print(f"[sender] removed source id={source_id}")
            # Trigger renegotiation on removal
            self._maybe_negotiate()
            return True
        except Exception as e:
            print(f"[sender] remove_source error: {e}")
            return False

    async def run(self) -> None:
        async with SignalClient(self.signal_url, self.room, self.peer_id) as sig:
            self.signal = sig
            sig.join(role="sender")
            # Discover a peer (first available)
            self._remote_id = await sig.discover_peer()
            if not self._remote_id:
                print("[signal] no peer discovered yet; waiting for peer-join…")
                # Fall back to reading until peer-join arrives
            self.pipe.set_state(Gst.State.PLAYING)

            async for env in sig:
                # Periodic escalation check (cheap)
                self.tick()
                # Fast peer presence tracking
                if env.type == "peer-leave":
                    if env.from_ and env.from_ == self._remote_id:
                        print(f"[signal] peer left: {self._remote_id}")
                        self._remote_id = None
                        self._on_remote_left()
                    continue
                if env.type == "peers" and self._remote_id:
                    # If our current remote is no longer in the list, treat as left.
                    lst = env.data or []
                    if isinstance(lst, list) and self._remote_id not in lst:
                        print(f"[signal] current peer disappeared: {self._remote_id}")
                        self._remote_id = None
                        self._on_remote_left()
                        # Do not 'continue' — we still allow below peers handler to select a new one
                if env.type == "peers" and not self._remote_id:
                    lst = env.data or []
                    if isinstance(lst, list) and lst:
                        self._remote_id = lst[0]
                        print(f"[signal] selected peer: {self._remote_id}")
                        self._maybe_negotiate()

                elif env.type == "peer-join":
                    # If we don't currently have a remote, adopt this joiner.
                    if not self._remote_id:
                        self._remote_id = env.from_
                        print(f"[signal] peer joined: {self._remote_id}")
                        self._maybe_negotiate()

                elif env.type == "answer":
                    payload = env.data or {}
                    sdp_txt = payload.get("sdp") if isinstance(payload, dict) else None
                    if sdp_txt:
                        _, sdp = GstSdp.SDPMessage.new()
                        GstSdp.sdp_message_parse_buffer(sdp_txt.encode(), sdp)
                        answer = GstWebRTC.WebRTCSessionDescription.new(
                            GstWebRTC.WebRTCSDPType.ANSWER, sdp
                        )
                        self.webrtc.emit(
                            "set-remote-description", answer, Gst.Promise.new()
                        )
                        print("[webrtc] set remote description (answer)")

                elif env.type == "candidate":
                    payload = env.data or {}
                    cand = (
                        payload.get("candidate") if isinstance(payload, dict) else None
                    )
                    mline = (
                        int(payload.get("sdpMLineIndex", 0))
                        if isinstance(payload, dict)
                        else 0
                    )
                    if cand:
                        self.webrtc.emit("add-ice-candidate", mline, cand)
                        # print("[webrtc] add ice candidate")

    # --------------------- Internals ----------------------
    def _maybe_negotiate(self) -> None:
        if not self.signal or not self._remote_id or self._making_offer or getattr(self, "_rebuilding", False):
            return
        # Proactively create an offer when we learn the remote id.
        if self._has_linked_sources():
            self._send_offer(label="offer")
        else:
            print("[webrtc] skip negotiate: no linked sources yet")

    def _send_offer(self, options: Optional[Gst.Structure] = None, *, label: str = "offer") -> None:
        """Create-offer → set-local → signal, guarded against re-entrancy."""
        if not self.signal or not self._remote_id or self._making_offer or getattr(self, "_rebuilding", False):
            return
        self._making_offer = True

        def on_offer_created(promise: Gst.Promise, _user_data, __user_data2=None):
            try:
                promise.wait()
                reply = promise.get_reply()
                if not reply or not reply.has_field("offer"):
                    print(f"[webrtc] create-offer failed ({label}): empty reply")
                    return
                offer = reply.get_value("offer")
                if not offer or not getattr(offer, "sdp", None):
                    print(f"[webrtc] create-offer failed ({label}): no SDP")
                    return
                self.webrtc.emit("set-local-description", offer, Gst.Promise.new())
                self.signal.offer(self._remote_id, offer.sdp.as_text())
                print(f"[webrtc] sent {label}")
            finally:
                self._making_offer = False

        self.webrtc.emit(
            "create-offer",
            options,
            Gst.Promise.new_with_change_func(on_offer_created, None, None),
        )

    def _has_linked_sources(self) -> bool:
        for h in self._sources.values():
            try:
                sp = h.bin.get_static_pad("src")
                if sp and sp.is_linked():
                    return True
            except Exception:
                pass
        return False

    def _on_negotiation_needed(self, *_args) -> None:
        if not self._has_linked_sources():
            # Wait until sources re-linked (e.g., right after a full reset)
            print("[webrtc] on-negotiation-needed but no linked sources; delaying")
            return
        self._send_offer(label="offer")

    def _on_ice_candidate(self, _webrtc, mlineindex, candidate) -> None:
        if not self.signal or not self._remote_id:
            return
        self.signal.candidate(self._remote_id, int(mlineindex), candidate)

    def _on_pc_state(self, _webrtc, _pspec) -> None:
        state = self.webrtc.get_property("connection-state")
        nick = getattr(state, "value_nick", str(state))
        print(f"[webrtc] connection state: {nick}")

    def _on_ice_state(self, _webrtc, _pspec) -> None:
        st = self.webrtc.get_property("ice-connection-state")
        nick = getattr(st, "value_nick", str(st))
        # Lightweight version of L0-L2 policy from restore.md
        if nick == "disconnected" and self._ice_disconnected_at is None:
            import time as _t

            self._ice_disconnected_at = _t.time()
        elif nick == "connected":
            self._ice_disconnected_at = None
            self._ice_restart_deadline = None
        elif nick == "failed":
            self._restart_ice()
        else:
            # If disconnected persists >3s, restart ICE
            if self._ice_disconnected_at is not None:
                import time as _t

                # Faster reaction: 1.0s
                if _t.time() - self._ice_disconnected_at > 1.0:
                    self._restart_ice()

    @staticmethod
    def _ice_restart_options() -> Optional[Gst.Structure]:
        # Structured options; fall back to string parsing across GI versions.
        try:
            opts = Gst.Structure.new_empty("webrtc-ice-options")
            try:
                opts.set_value("ice-restart", True)
            except Exception:
                pass
            return opts
        except Exception:
            try:
                ok, s = Gst.Structure.from_string("webrtc-ice-options,ice-restart=(boolean)true")
                return s if ok else None
            except Exception:
                return None

    def _restart_ice(self) -> None:
        if not self.signal or not self._remote_id or self._making_offer:
            return
        print("[webrtc] ICE restart: creating offer with ice-restart=true")
        self._send_offer(self._ice_restart_options(), label="ICE restart offer")
        # Start/reset escalation timer
        try:
            import time as _t

            self._ice_restart_deadline = _t.time() + 5.0
        except Exception:
            self._ice_restart_deadline = None

    def _on_bus_error(self, _bus, msg) -> None:
        err, dbg = msg.parse_error()
        print(f"[GStreamer][ERROR] {err} debug:{dbg}")

    def _on_bus_warning(self, _bus, msg) -> None:
        wrn, dbg = msg.parse_warning()
        print(f"[GStreamer][WARN ] {wrn} debug:{dbg}")

    # ---------------------- Resilience ----------------------
    def _on_remote_left(self) -> None:
        """
        Called when the remote peer leaves the room.

        We promptly reset the peer connection (webrtcbin) to clear RTP state
        and be ready for a brand new peer. Sources are kept intact and re-linked.
        """
        import time as _t

        self._last_remote_left_ts = _t.time()
        self._ice_disconnected_at = None
        self._ice_restart_deadline = None
        try:
            self._full_reset_webrtc()
        except Exception as e:
            print(f"[sender] full reset failed: {e}")

    def _full_reset_webrtc(self) -> None:
        """Tear down and recreate webrtcbin, re-link sources, keep pipeline running."""
        old = self.webrtc
        self._rebuilding = True  # block negotiation during rebuild
        # Try calling action 'close' if available (GStreamer 1.28+)
        try:
            if hasattr(old, "emit"):
                old.emit("close")  # type: ignore[misc]
        except Exception:
            pass
        # 1) Quiesce all sources to avoid NOT_LINKED errors while we rewire
        for _sid, h in self._sources.items():
            try:
                h.bin.set_state(Gst.State.PAUSED)
            except Exception:
                pass

        # 2) Unlink sources from OLD webrtc and release pads (carefully)
        for sid, handle in list(self._sources.items()):
            try:
                src_pad = handle.bin.get_static_pad("src")
                if src_pad and src_pad.is_linked():
                    src_pad.unlink(handle.sink_pad)
                # Release request pad only if it still belongs to old
                try:
                    parent = handle.sink_pad.get_parent_element() if handle.sink_pad else None
                    if parent == old:
                        old.release_request_pad(handle.sink_pad)
                except Exception:
                    pass
            except Exception as e:
                print(f"[sender] unlink source {sid} failed: {e}")

        # 3) Fully stop and remove OLD webrtc
        try:
            old.set_state(Gst.State.NULL)
        except Exception:
            pass
        try:
            if old.get_parent() is self.pipe:
                self.pipe.remove(old)
        except Exception:
            pass

        # 4) Prepare NEW webrtcbin and add to pipeline
        new_webrtc = Gst.ElementFactory.make("webrtcbin", "sendonly")
        if not new_webrtc:
            self._rebuilding = False
            raise RuntimeError("webrtcbin not available for rebuild")
        new_webrtc.set_property("bundle-policy", "max-bundle")
        new_webrtc.set_property("stun-server", self.stun)
        new_webrtc.set_property("turn-server", self.turn)
        new_webrtc.set_property("latency", self.latency_ms)

        # Hook callbacks
        new_webrtc.connect("on-negotiation-needed", self._on_negotiation_needed)
        new_webrtc.connect("on-ice-candidate", self._on_ice_candidate)
        new_webrtc.connect("notify::connection-state", self._on_pc_state)
        new_webrtc.connect("notify::ice-connection-state", self._on_ice_state)

        try:
            self.pipe.add(new_webrtc)
        except Exception as e:
            self._rebuilding = False
            raise e

        # 5) Re-link sources to the NEW webrtc
        for sid, handle in list(self._sources.items()):
            try:
                src_pad = handle.bin.get_static_pad("src")
                new_sink = new_webrtc.request_pad_simple("sink_%u")
                if not new_sink:
                    raise RuntimeError("request_pad_simple returned None")
                if not src_pad or src_pad.link(new_sink) != Gst.PadLinkReturn.OK:
                    raise RuntimeError("failed to link source -> new webrtc")
                handle.sink_pad = new_sink
                # Resume source
                handle.bin.sync_state_with_parent()
            except Exception as e:
                print(f"[sender] re-link source {sid} failed: {e}")

        # 6) Finalize
        self.webrtc = new_webrtc
        self.webrtc.sync_state_with_parent()
        self._rebuilding = False
        print("[sender] webrtcbin rebuilt and sources re-linked; waiting for new peer…")

    def tick(self) -> None:
        """
        Optional periodic call to escalate from ICE-restart to full reset.

        If called from an external loop, when the ICE restart deadline expires
        and we're not connected, do a full reset. This is a no-op if not needed.
        """
        try:
            import time as _t

            if self._ice_restart_deadline and (
                self.webrtc.get_property("connection-state").value_nick != "connected"
            ) and _t.time() > self._ice_restart_deadline:
                print("[webrtc] ICE restart did not recover; performing full reset")
                self._ice_restart_deadline = None
                self._full_reset_webrtc()
        except Exception:
            pass
