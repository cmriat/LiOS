def load_env(path: str | None = None) -> str | None:
    """
    Load KEY=VALUE pairs from a .env file into os.environ.

    Existing environment variables win (shell/CLI overrides .env). Returns the
    path actually loaded, or None if no .env was found. Stdlib only.
    """
    import os

    if path is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for cand in (os.path.join(os.getcwd(), ".env"), os.path.join(root, ".env")):
            if os.path.isfile(cand):
                path = cand
                break
    if not path or not os.path.isfile(path):
        return None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    return path


def init_gst():
    """
    Initialize GStreamer.

    `gi` is imported here (not at module top level) so that GStreamer-free
    submodules such as `inference_buffer_v2` stay importable in environments
    without the native GStreamer stack.
    """
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstWebRTC", "1.0")
    gi.require_version("GstSdp", "1.0")
    from gi.repository import Gst

    Gst.init(None)
