# WebRTC 断流恢复设计（通用策略）

> sender 侧的落地实现见 [`src/gst_webrtc/sender/core.py`](../../src/gst_webrtc/sender/core.py)（`_on_ice_state` / `_restart_ice` / `_full_reset_webrtc`）。

下面给你一套**在 GStreamer webrtcbin + 自建信令**环境下，针对“网络抖动 / 断流后重启失败”的**分层恢复策略 + 事件驱动流程**。内容包含：什么时候需要重新协商 (O/A)、什么时候需要 ICE 重启、on-ice-candidate 的使用、信令该传什么以及严格的时序顺序；并给出 Python(gst-python) 的事件处理骨架。

> 关键 API/信号名和状态名下面都标注了出处，你可以对照自己的版本实现。webrtcbin 的核心信号/属性（on‑negotiation‑needed / on‑ice‑candidate / add‑ice‑candidate / connection‑state / ice‑connection‑state / ice‑gathering‑state / close 等）见官方文档。([GStreamer][1])

---

## 一、先定一个“分层恢复策略”（从轻到重）

> 你可以把恢复分成 L0→L3 四层，状态机只往更“重”的层级升级，成功后回落。

**L0｜短暂抖动/丢包：不用协商，不用 ICE 重启**

* webrtc/rtp 的重传、FEC、jitterbuffer 自恢复。不做任何 O/A 或候选交互。
* 可监控 webrtcbin 的 `ice-connection-state` 是否从 `disconnected` 回到 `connected`。连接属性参考：`notify::ice-connection-state` / `notify::connection-state`。([GStreamer][1])

**L1｜媒体源临时停止/你手动“停推→再推”，PeerConnection 仍然活着：做**“（可选）轻量协商”，**不要重建 PC**

* 不销毁 webrtcbin，不做 ICE 重启。
* 如果你动态加/减流（多流场景常见），webrtcbin 会触发 `on-negotiation-needed`，按正常 O/A 走一轮即可。**尽量复用原 transceiver**，只改它的 `direction`（sendrecv ↔ sendonly/recvonly），避免重排 m-line。`GstWebRTCRTPTransceiver.direction/current-direction/mid` 见文档。([GStreamer][2])
* **强烈建议**把 webrtcbin 的 `reuse-source-pads=true`（1.26+），这样 transceiver 改方向或暂时没流时不强行发 EOS，后续协商恢复能直接“续上”同一个 pad。([GStreamer][1])

**L2｜ICE 真的断了：做 ICE 重启（ICE restart），仍然**不**重建 PC**

* 触发条件：`ice-connection-state=failed`（或 `disconnected` 持续几秒不恢复），这时发起 ICE 重启比等到 failed 更快恢复。ICE 重启会生成**新的 `ice-ufrag/pwd`** 并重新收集候选。([MDN Web Docs][3])
* WebRTC 语义：通过 `createOffer({ iceRestart: true })` 或 `restartIce()` 来实现（在 webrtcbin 里体现在 “create‑offer 的 options 启用 ice‑restart”，生成带新 ICE 凭据的 SDP）。**新 offer + answer 的 O/A 一定要走一遍**。([MDN Web Docs][4])

**L3｜DTLS/PeerConnection 已关闭、对端进程重启、或者多次 ICE 重启仍失败：完全重建**

* 调 webrtcbin 的 `close` action（1.28+），并把 pipeline 相关支路设为 `READY/NULL`，销毁 webrtcbin，重新创建+完整 O/A。([GStreamer][1])
* GStreamer 圈子的惯例就是**重建最干净**（很多人实战里这么做）。([Stack Overflow][5])

---

## 二、**什么时候需要“重新协商 (O/A)”**，什么时候**不用**

**需要重新协商（O/A）的典型情况**

1. 本地**新增/删除**音视频流（请求/释放 webrtcbin 的 `sink_%u` pad，或 `add-transceiver`）。会触发 `on-negotiation-needed`。([GStreamer][1])
2. **改变 transceiver.direction**（例如临时停推，把 `sendrecv→sendonly`），或更换轨（编码器/格式发生变化且需要 SDP 反映）。([GStreamer][2])
3. **ICE 重启**（见 L2）。这是“要协商但不是换 PC”。([MDN Web Docs][4])
4. 新建 **data channel** 且 `negotiated=false`。([GStreamer][1])

**不需要重新协商的情况**

* 普通码率自适应、回声消除参数、RTP 传输层的丢包恢复等。
* 仅仅“源短暂无帧”（L0），连通性良好，无需 O/A。

---

## 三、**什么时候需要重新发送 on‑ice‑candidate（候选）**

* **初次连接**或**每次 ICE 重启**时：本地收集到的每个候选，都会触发 `on-ice-candidate`；你都要通过信令转发给对端，直到收集结束。([GStreamer][1])
* **收集结束**可以发送\*\*“end‑of‑candidates”\*\*：对 webrtcbin 调 `add-ice-candidate(mline_index, "")`（空串/NULL）即表示此 m= 行候选结束。([GStreamer][1])
* **网卡切换/网络环境改变**可能产生新候选；继续转发即可。
* 注意 `mline_index`（webrtcbin 用这个），浏览器侧通常是 `sdpMLineIndex/sdpMid`，你的信令可以同时带上，便于两头适配。基础知识可参考 ICE 教程。([Stream][6])

---

## 四、**信令里该传什么** & **时序**

### 1) 初次连接（Trickle ICE，Offerer 在 A 侧举例）

1. **A**（收到 `on-negotiation-needed`）→ `create-offer()` → `set-local-description(offer)`。([Nirbheek Blog][7])
2. **A** → 信令 → 发送 `{type:"offer", sdp}` 给 **B**。
3. **A** trickle 候选：`on-ice-candidate(mline,cand)` → 信令 → `{candidate, mline_index}` 给 **B**。([GStreamer][1])
4. **B** 收到 offer → `set-remote-description(offer)` → `create-answer()` → `set-local-description(answer)`。
5. **B** → 信令 → 发回 `{type:"answer", sdp}`。
6. **B** 也 trickle 自己的候选给 A；A 调 `add-ice-candidate()` 喂给 webrtcbin。**两边都在喂候选**。([GStreamer][1])
7. 收集结束（可选）发 “end‑of‑candidates”。([GStreamer][1])

### 2) **仅重新协商（增删流/改方向），**不**做 ICE 重启**

* 序列与初次连接一致，但 ICE 凭据不变（不用重传历史候选，照常 trickle 新的即可）。触发是 `on-negotiation-needed`。([GStreamer][1])

### 3) **ICE 重启（连接失败或长期断开）**

1. **发起方**：`create-offer(options with iceRestart=true)` → `set-local-description(offer)`。该 SDP 的 **`ice-ufrag/pwd` 会改变**。([MDN Web Docs][4])
2. 发送新的 `{type:"offer", sdp}`。
3. **对端** `set-remote-description(offer)` → `create-answer()` → `set-local-description(answer)` 并回传。
4. **双方重新 trickle 全新的候选集合**（注意旧候选与旧 ufrag/pwd 不再有效）。([MDN Web Docs][4])

> 采用 **Perfect Negotiation** 模式处理并发 O/A（glare）：一端设为“polite”，另一端“impolite”。polite 端在冲突时 rollback 自己的本地变更，优先处理对端的 offer，避免死锁。MDN 有完整范式。([MDN Web Docs][8])

### 4) **完全重连（重建 webrtcbin）**

* 双方通过信令先行“挂断/关闭”，**本地调用** webrtcbin 的 `close` action → pipeline 到 `READY/NULL`，释放 webrtcbin，**重新创建**再走一次初始 1) 的流程。([GStreamer][1])

---

## 五、Python(gst‑python) 事件处理骨架（可直接套）

> 下面演示**关键回调**和**状态机**。具体链接/解码器你已写好，多流场景请按你的业务在 add/remove track 时维护 transceiver 与 pad 的映射。

```python
import gi, json, time
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GObject

Gst.init(None)

class PC:
    def __init__(self, send_signaling, polite=True):
        self.webrtc = Gst.ElementFactory.make("webrtcbin", "pc")
        # 建议：多流下开启 pad 复用，恢复更平滑（1.26+）
        try:
            self.webrtc.set_property("reuse-source-pads", True)
        except Exception:
            pass
        self.send_signaling = send_signaling
        self.polite = polite
        self.making_offer = False
        self.ice_failed_at = None

        # 信号
        self.webrtc.connect("on-negotiation-needed", self._on_negotiation_needed)
        self.webrtc.connect("on-ice-candidate", self._on_ice_candidate)
        self.webrtc.connect("notify::ice-connection-state", self._on_ice_state)
        self.webrtc.connect("notify::connection-state", self._on_pc_state)
        # 接收端多流：绑定 pad-added 去链接 depay/decoder
        self.webrtc.connect("pad-added", self._on_pad_added)  # 见 StackOverflow 示例

    # ---- 信令收发（你已有信令服务器，这里只演示消息格式） ----
    def handle_signaling(self, msg: dict):
        t = msg.get("type")
        if t in ("offer", "answer"):
            sdp = GstSdp.SDPMessage.new()
            GstSdp.sdp_message_parse_buffer(bytes(msg["sdp"], "utf8"), sdp)
            desc = GstWebRTC.WebRTCSessionDescription.new(
                GstWebRTC.WebRTCSDPType.OFFER if t=="offer" else GstWebRTC.WebRTCSDPType.ANSWER, sdp
            )
            # Perfect negotiation：简化处理
            self.webrtc.emit("set-remote-description", desc, None)
            if t == "offer":
                promise = Gst.Promise.new_with_change_func(self._on_answer_created, None, None)
                self.webrtc.emit("create-answer", None, promise)

        elif "candidate" in msg:
            # 兼容浏览器字段：优先 mline_index
            mline = msg.get("mline_index", msg.get("sdpMLineIndex", 0))
            cand = msg["candidate"]  # 为空串表示 end-of-candidates
            self.webrtc.emit("add-ice-candidate", mline, cand)

    # ---- 回调实现 ----
    def _on_negotiation_needed(self, *_):
        # 注意：如果是 ICE 重启，这里要把 options 里开启 iceRestart
        def on_offer_created(promise, _):
            reply = promise.get_reply()
            offer = reply.get_value("offer")
            self.webrtc.emit("set-local-description", offer, None)
            self.send_signaling({"type": "offer", "sdp": offer.sdp.as_text()})

        # making_offer 标记避免并发
        if self.making_offer: 
            return
        self.making_offer = True
        promise = Gst.Promise.new_with_change_func(lambda p, u: (on_offer_created(p,u), setattr(self, "making_offer", False)), None, None)
        self.webrtc.emit("create-offer", None, promise)  # ICE 重启时把 None 换成包含 ice-restart 的 options 结构

    def _on_ice_candidate(self, _, mline, candidate):
        self.send_signaling({"candidate": candidate, "mline_index": int(mline)})

    def _on_ice_state(self, obj, pspec):
        state = self.webrtc.get_property("ice-connection-state").value_nick
        # L0: disconnected 先观察几秒；L2: failed 触发 ICE 重启
        if state == "disconnected" and self.ice_failed_at is None:
            self.ice_failed_at = time.time()
        if state == "connected":
            self.ice_failed_at = None
        if state == "failed" or (state == "disconnected" and self.ice_failed_at and time.time()-self.ice_failed_at > 3.0):
            self._restart_ice()

    def _on_pc_state(self, obj, pspec):
        st = self.webrtc.get_property("connection-state").value_nick
        if st in ("closed",):
            self._full_reconnect()

    def _restart_ice(self):
        # 关键点：生成带新 ufrag/pwd 的 offer（ICE 重启）
        def on_offer_created(promise, _):
            offer = promise.get_reply().get_value("offer")
            self.webrtc.emit("set-local-description", offer, None)
            self.send_signaling({"type":"offer", "sdp": offer.sdp.as_text()})
        promise = Gst.Promise.new_with_change_func(on_offer_created, None, None)
        # 这里需要把 options 设为 iceRestart=true（等价于 WebRTC 的 createOffer({iceRestart:true}))
        # 具体 options 结构创建请按你的 GStreamer 版本/绑定写法填入：
        options = None  # ← 用包含 'ice-restart': True 的 Gst.Structure
        self.webrtc.emit("create-offer", options, promise)

    def _full_reconnect(self):
        # 1. 先 close（1.28+ 有 action） 2. pipeline → READY/NULL 3. 销毁重建 webrtcbin
        try:
            self.webrtc.emit("close", None)
        except Exception:
            pass
        # ...你的清理与重建逻辑...
        pass

    def _on_pad_added(self, webrtc, pad):
        # 根据 pad 的 caps(kind: audio/video) 链接你的 depay/decoder
        pass
```

> 说明：
>
> * 回调和属性名参照 webrtcbin 文档（`on-negotiation-needed`, `on-ice-candidate`, `add-ice-candidate`, `notify::ice-connection-state`, `notify::connection-state`）。([GStreamer][1])
> * **ICE 重启**处需要给 `create-offer` 传入带 `iceRestart=true` 的 options（等价于浏览器的 `createOffer({iceRestart:true})` / `restartIce()` 语义，SDP 会出现新的 `ice-ufrag/pwd`）。([MDN Web Docs][4])
> * 多流接收用 `pad-added` 绑定 depay/decoder，StackOverflow 示例亦如此。([Stack Overflow][9])

---

## 六、你关心的几件“坑位”与建议

1. **多流下保持 transceiver 与 m-line 顺序稳定**

   * 不要随意销毁并重建 transceiver；只改 `direction` 或替换 track，避免 m-line 乱序导致“无法相交方向”的 SDP 错误。`GstWebRTCRTPTransceiver.direction/current-direction/mid` 可用来管理。([GStreamer][2])

2. **开启 `reuse-source-pads`（1.26+）**

   * 这会让“临时停推/方向改为接收”时，pad 不被 EOS 销毁，后续恢复时更平滑；默认是 false。([GStreamer][1])

3. **状态监听必须用 GObject notify**

   * `notify::ice-connection-state` / `notify::connection-state` / `notify::ice-gathering-state`。官方及社区都推荐这种方式而非轮询。([GStreamer][1])

4. **Trickle 的“收尾”**

   * 结束时用 `add-ice-candidate(mline, "")` 表示 end‑of‑candidates，便于对端完成 ICE 状态转换。([GStreamer][1])

5. **重建（L3）要“干净”**

   * 先 `close`（1.28+），再将 pipeline→`NULL`，移除 webrtcbin，释放，再创建新的 webrtcbin；不要在旧的 webrtcbin 上重复 set\_remote/ set\_local 直到乱套。([GStreamer][1])

6. **Perfect Negotiation 防并发 O/A 死锁**

   * 你的信令可在会话级指定一端为 “polite”。实战上能显著降低“两个端同时发 offer”造成的失败率。([MDN Web Docs][8])

7. **TURN/STUN 与端口策略**

   * 多 NAT/移动网络必配 TURN（`turn-server` / `add-turn-server`），端口范围可从 1.20 起在 ICE 对象上限制，利于防火墙。([GStreamer][1])

---

## 七、快速决策树（落地版）

* **画面卡顿但 1–2s 内能恢复** → **L0**：不动作。
* **你“停推→再推”，PC 还连着** → **L1**：让 `on-negotiation-needed` 驱动一轮 O/A；优先用同一 transceiver；`reuse-source-pads=true`。([GStreamer][1])
* **`ice-connection-state=disconnected` 超过 \~3s** 或到 **failed** → **L2**：**ICE 重启**（新 O/A + 全新候选）。([MDN Web Docs][3])
* **DTLS 失败/connection-state=closed/对端重启** 或 **多次 ICE 重启仍失败** → **L3**：`close`+重建 webrtcbin，走完整初始时序。([GStreamer][1])

---

## 参考出处（关键点）

* webrtcbin 的信号与属性：`on-negotiation-needed`、`on-ice-candidate`、`add-ice-candidate`（空串标记结束）、`connection-state`、`ice-connection-state`、`ice-gathering-state`、`close` action、`reuse-source-pads` 属性。([GStreamer][1])
* Transceiver 的 `direction/current-direction/mid`。([GStreamer][2])
* ICE 重启语义（`createOffer({iceRestart:true})` / `restartIce()`；SDP 的 `ice-ufrag/pwd` 变化）。([MDN Web Docs][4])
* Perfect Negotiation 模式（避免 glare）。([MDN Web Docs][8])
* 多流接收的 `pad-added` 用法（示例）。([Stack Overflow][9])
* 版本特性：1.20/1.22/1.26/1.28 的相关变更。([GStreamer][10])

---


[1]: https://gstreamer.freedesktop.org/documentation/webrtc/index.html "webrtcbin"
[2]: https://gstreamer.freedesktop.org/documentation/webrtclib/gstwebrtc-transceiver.html?utm_source=chatgpt.com "GstWebRTCRTPTransceiver"
[3]: https://developer.mozilla.org/en-US/docs/Web/API/RTCPeerConnection/restartIce?utm_source=chatgpt.com "RTCPeerConnection: restartIce() method - Web APIs - MDN"
[4]: https://developer.mozilla.org/en-US/docs/Web/API/RTCPeerConnection/createOffer?utm_source=chatgpt.com "RTCPeerConnection: createOffer() method - Web APIs - MDN"
[5]: https://stackoverflow.com/questions/37254563/how-to-restart-a-pipeline-when-it-is-in-playing-state?utm_source=chatgpt.com "gstreamer - How to restart a pipeline when it is in playing state"
[6]: https://getstream.io/resources/projects/webrtc/basics/ice-candidates/?utm_source=chatgpt.com "ICE Candidate Tutorial - WebRTC Interactive Connectivity ..."
[7]: https://blog.nirbheek.in/2018/02/gstreamer-webrtc.html?utm_source=chatgpt.com "GStreamer has grown a WebRTC implementation"
[8]: https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API/Perfect_negotiation?utm_source=chatgpt.com "The WebRTC perfect negotiation pattern - Web APIs | MDN"
[9]: https://stackoverflow.com/questions/57430215/how-to-use-webrtcbin-create-offer-only-receive-video?utm_source=chatgpt.com "How to use webrtcbin create offer,only receive video"
[10]: https://gstreamer.freedesktop.org/releases/1.20/?utm_source=chatgpt.com "GStreamer 1.20 release notes - Freedesktop.org"

