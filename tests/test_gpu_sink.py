"""Unit tests for GpuFrameSink pure logic (no live GStreamer pipeline)."""

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")

from gst_webrtc.gpu_sink.core import GpuFrameSink  # noqa: E402


class _FakeMap:
    def __init__(self, data: bytes) -> None:
        self.data = data


@pytest.fixture
def sink() -> GpuFrameSink:
    return GpuFrameSink(name="t", output_format="RGBA")


def test_invalid_output_format_raises():
    with pytest.raises(ValueError):
        GpuFrameSink(output_format="YUV")


def test_channels_for_format(sink):
    assert sink._channels_for_format("RGBA") == 4
    assert sink._channels_for_format("rgb") == 3
    assert sink._channels_for_format("BGR") == 3
    assert sink._channels_for_format("GRAY8") == 1


def test_appsink_tail_desc_embeds_name_and_format(sink):
    desc = sink.appsink_tail_desc()
    assert "appsink name=t" in desc
    assert "video/x-raw,format=RGBA" in desc


def test_rtp_desc_starts_from_rtp_and_ends_with_tail(sink):
    desc = sink.rtp_h264_sink_desc()
    assert "application/x-rtp" in desc
    assert "rtph264depay" in desc
    assert desc.endswith(sink.appsink_tail_desc())


def test_map_to_numpy_rgba_no_padding(sink):
    h, w, ch = 4, 5, 4
    arr = sink._map_to_numpy(_FakeMap(bytes(h * w * ch)), w, h, "RGBA")
    assert arr.shape == (h, w, ch)


def test_map_to_numpy_strips_row_padding(sink):
    h, w, ch = 3, 4, 4
    stride = (w + 2) * ch  # padded rows
    arr = sink._map_to_numpy(_FakeMap(bytes(h * stride)), w, h, "RGBA")
    assert arr.shape == (h, w, ch)


def test_map_to_numpy_gray8():
    h, w = 3, 6
    g = GpuFrameSink(output_format="GRAY8")
    arr = g._map_to_numpy(_FakeMap(bytes(h * w)), w, h, "GRAY8")
    assert arr.shape == (h, w)


@pytest.mark.gpu
def test_torch_cuda_available():
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "no CUDA device visible to torch"
