from __future__ import annotations

"""
InferenceBufferV2 — self-contained, base64-pickled transport for inference inputs/outputs.

Goals
-----
- Self-contained pickle: receivers do not need this module to reconstruct the payload.
- CUDA IPC friendly: torch CUDA tensors are serialized using the Torch ForkingPickler,
  preserving zero-copy semantics across processes (lifetime constraints apply).
- Metadata included: arbitrary JSON-like metadata is embedded alongside tensors.
- Base64 everywhere: transport-safe ASCII string for easy embedding in JSON/proto/ws.

Usage
-----
- Use `InferenceBufferV2.pack_base64()` to obtain a base64 string.
- Use `unpack_base64()` (module-level) to reconstruct a plain `dict` without importing
  this module on the receiver side. Optionally, `InferenceBufferV2.from_base64()` to
  wrap that dict back into the dataclass for ergonomic access when available.

Notes on CUDA IPC
-----------------
The pickle produced here uses `torch.multiprocessing.reductions` registrations, such
that CUDA tensors are represented as CUDA IPC handles. The receiving process will map
the same device memory. The exporting process must keep the source tensor alive until
all consumers finish; otherwise access becomes undefined.
"""

import base64
import pickle
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing.reduction import ForkingPickler
from typing import Any, Dict, Mapping, MutableMapping, Optional

import posix_ipc
import torch
import torch.multiprocessing.reductions  # noqa: F401  (ensure reducers are registered)

# Version tag for the packed format. Increment when breaking the on-wire structure.
IFBUF_PKL_VERSION = 1


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def _now_ns() -> int:
    try:
        return time.time_ns()
    except AttributeError:  # Py<3.7 fallback (unlikely here)
        return int(time.time() * 1e9)


def _ensure_str_keys(d: Mapping[Any, Any]) -> Dict[str, Any]:
    return {str(k): v for k, v in d.items()}


@dataclass
class InferenceBufferV2:
    """
    Improved inference buffer with self-contained base64 pickle support.

    Fields are intentionally simple so the packed payload can be a plain dict
    of built-ins and torch tensors (no class references needed to unpickle).
    """

    images: Mapping[str, torch.Tensor] = field(default_factory=dict)
    states: Mapping[str, torch.Tensor] = field(default_factory=dict)
    prev_actions: Mapping[str, torch.Tensor] = field(default_factory=dict)
    # Arbitrary JSON-serializable metadata; e.g., frame_id, model, ts, etc.
    meta: MutableMapping[str, Any] = field(default_factory=dict)
    # Name of POSIX semaphores. Keep `sem_name` for backward compatibility and
    # treat it as the images (video) semaphore. Add a dedicated state semaphore.
    sem_name: Optional[str] = None              # legacy/images lock name
    sem_state_name: Optional[str] = None        # state lock name

    # Internal, non-serializable handles
    _sem_images: Optional[posix_ipc.Semaphore] = field(default=None, init=False, repr=False, compare=False)
    _sem_state: Optional[posix_ipc.Semaphore] = field(default=None, init=False, repr=False, compare=False)

    def to_plain_payload(self) -> Dict[str, Any]:
        """
        Return a pickle-friendly plain dict (no custom classes).

        - Uses only built-in container types and torch tensors.
        - Keys are coerced to `str` for safety when later embedding in JSON.
        - Adds a small header with versioning.
        """
        header = {
            "__ifbuf__": IFBUF_PKL_VERSION,
            "created_ns": _now_ns(),
        }
        payload: Dict[str, Any] = {
            "header": header,
            "meta": _ensure_str_keys(self.meta),
            "images": _ensure_str_keys(self.images),
            "states": _ensure_str_keys(self.states),
            "prev_actions": _ensure_str_keys(self.prev_actions),
            # Expose semaphore name so remote processes can open the same lock
            # to coordinate buffer access.
            # Expose both semaphore names. `sem_name` kept for compatibility
            "sem_name": self.sem_name,
            "sem_state_name": self.sem_state_name,
        }
        return payload

    def pack_bytes(self) -> bytes:
        """Serialize to bytes using ForkingPickler (CUDA IPC aware)."""
        return ForkingPickler.dumps(self.to_plain_payload(), protocol=pickle.HIGHEST_PROTOCOL)

    def pack_base64(self) -> str:
        """
        Serialize to a base64 ASCII string.

        This string is fully self-contained. The receiver can reconstruct a
        plain dict via `pickle.loads(base64.b64decode(s))` without importing
        this module. For convenience, use `unpack_base64` below.
        """
        return _b64e(self.pack_bytes())

    # Aliases for ergonomics
    to_base64 = pack_base64
    to_bytes = pack_bytes

    # ---------- Construction helpers ----------
    @classmethod
    def from_plain_payload(cls, data: Mapping[str, Any]) -> "InferenceBufferV2":
        images = data.get("images", {})
        states = data.get("states", {})
        prev_actions = data.get("prev_actions", {})
        meta = dict(data.get("meta", {}))
        sem_name = data.get("sem_name")
        sem_state_name = data.get("sem_state_name")
        return cls(
            images=images,
            states=states,
            prev_actions=prev_actions,
            meta=meta,
            sem_name=sem_name,
            sem_state_name=sem_state_name,
        )

    @classmethod
    def from_bytes(cls, b: bytes) -> "InferenceBufferV2":
        obj = ForkingPickler.loads(b)
        if isinstance(obj, dict):
            return cls.from_plain_payload(obj)
        raise TypeError(f"Unsupported payload type: {type(obj)}")

    @classmethod
    def from_base64(cls, s: str) -> "InferenceBufferV2":
        return cls.from_bytes(_b64d(s))

    # ---------- POSIX named semaphore helpers ----------
    def _gen_sem_name(self) -> str:
        # POSIX requires names to start with '/'
        return f"/ifbuf_{uuid.uuid4().hex}"

    # ----- Low-level open helpers for each semaphore -----
    def _open_or_create_images_sem(self, *, create: bool, initial_value: int = 1, exclusive: bool = False) -> posix_ipc.Semaphore:
        if create:
            if not self.sem_name:
                self.sem_name = self._gen_sem_name()
            flags = posix_ipc.O_CREAT | (posix_ipc.O_EXCL if exclusive else 0)
            self._sem_images = posix_ipc.Semaphore(self.sem_name, flags=flags, initial_value=initial_value)
            return self._sem_images
        if not self.sem_name:
            raise ValueError("sem_name is not set; cannot open existing images semaphore")
        self._sem_images = posix_ipc.Semaphore(self.sem_name)
        return self._sem_images

    def _open_or_create_state_sem(self, *, create: bool, initial_value: int = 1, exclusive: bool = False) -> posix_ipc.Semaphore:
        if create:
            if not self.sem_state_name:
                self.sem_state_name = self._gen_sem_name()
            flags = posix_ipc.O_CREAT | (posix_ipc.O_EXCL if exclusive else 0)
            self._sem_state = posix_ipc.Semaphore(self.sem_state_name, flags=flags, initial_value=initial_value)
            return self._sem_state
        if not self.sem_state_name:
            raise ValueError("sem_state_name is not set; cannot open existing state semaphore")
        self._sem_state = posix_ipc.Semaphore(self.sem_state_name)
        return self._sem_state

    def attach_semaphore(
        self,
        name: Optional[str] = None,
        *,
        create: bool = True,
        initial_value: int = 1,
        exclusive: bool = False,
    ) -> str:
        """
        Attach images (video) semaphore; returns its name.

        - If `create=True` (default), create the semaphore (O_CREAT) with
          `initial_value` (1 => unlocked mutex). If `name` is not provided,
          a fresh random name is generated.
        - If `create=False`, `name` (or an existing `self.sem_name`) must be set
          and the function will open it.
        - The semaphore handle is cached in-process; only `sem_name` is serialized.
        """
        if name is not None:
            if not name.startswith("/"):
                name = f"/{name}"
            self.sem_name = name
        self._open_or_create_images_sem(create=create, initial_value=initial_value, exclusive=exclusive)
        return str(self.sem_name)

    def attach_state_semaphore(
        self,
        name: Optional[str] = None,
        *,
        create: bool = True,
        initial_value: int = 1,
        exclusive: bool = False,
    ) -> str:
        """Attach state semaphore; returns its name."""
        if name is not None:
            if not name.startswith("/"):
                name = f"/{name}"
            self.sem_state_name = name
        self._open_or_create_state_sem(create=create, initial_value=initial_value, exclusive=exclusive)
        return str(self.sem_state_name)

    def _ensure_images_sem(self) -> posix_ipc.Semaphore:
        if self._sem_images is not None:
            return self._sem_images
        if not self.sem_name:
            return self._open_or_create_images_sem(create=True, initial_value=1, exclusive=False)
        return self._open_or_create_images_sem(create=False)

    def _ensure_state_sem(self) -> posix_ipc.Semaphore:
        if self._sem_state is not None:
            return self._sem_state
        if not self.sem_state_name:
            return self._open_or_create_state_sem(create=True, initial_value=1, exclusive=False)
        return self._open_or_create_state_sem(create=False)

    def try_lock(self) -> bool:
        """Try acquire images semaphore (legacy)."""
        sem = self._ensure_images_sem()
        try:
            sem.acquire(timeout=0)
            return True
        except posix_ipc.BusyError:
            return False

    def lock(self, timeout: Optional[float] = None) -> None:
        """Acquire images semaphore (legacy)."""
        sem = self._ensure_images_sem()
        sem.acquire(timeout=None if timeout is None else float(timeout))

    def unlock(self) -> None:
        """Release images semaphore (legacy)."""
        sem = self._ensure_images_sem()
        sem.release()

    @property
    def is_locked(self) -> bool:
        """Best-effort check on images semaphore (legacy)."""
        sem = self._ensure_images_sem()
        try:
            return sem.value == 0
        except Exception:
            # Some platforms may not expose .value; fall back to non-blocking try
            acquired = self.try_lock()
            if acquired:
                self.unlock()
                return False
            return True

    @contextmanager
    def hold_lock(self, timeout: Optional[float] = None):
        """Context manager for images semaphore (legacy)."""
        self.lock(timeout=timeout)
        try:
            yield self
        finally:
            self.unlock()

    # ----- Dedicated state semaphore helpers -----
    def try_state_lock(self) -> bool:
        sem = self._ensure_state_sem()
        try:
            sem.acquire(timeout=0)
            return True
        except posix_ipc.BusyError:
            return False

    def state_lock(self, timeout: Optional[float] = None) -> None:
        sem = self._ensure_state_sem()
        sem.acquire(timeout=None if timeout is None else float(timeout))

    def state_unlock(self) -> None:
        sem = self._ensure_state_sem()
        sem.release()

    @property
    def is_state_locked(self) -> bool:
        sem = self._ensure_state_sem()
        try:
            return sem.value == 0
        except Exception:
            acquired = self.try_state_lock()
            if acquired:
                self.state_unlock()
                return False
            return True

    @contextmanager
    def hold_state_lock(self, timeout: Optional[float] = None):
        self.state_lock(timeout=timeout)
        try:
            yield self
        finally:
            self.state_unlock()

    def close_semaphore(self) -> None:
        """Close both in-process semaphore handles (no unlink)."""
        if self._sem_images is not None:
            try:
                self._sem_images.close()
            finally:
                self._sem_images = None
        if self._sem_state is not None:
            try:
                self._sem_state.close()
            finally:
                self._sem_state = None

    def unlink_semaphore(self) -> None:
        """Unlink both semaphores if present. Safe to call multiple times."""
        if self.sem_name:
            try:
                posix_ipc.unlink_semaphore(self.sem_name)
            except posix_ipc.ExistentialError:
                pass
        if self.sem_state_name:
            try:
                posix_ipc.unlink_semaphore(self.sem_state_name)
            except posix_ipc.ExistentialError:
                pass

    # Metadata alias for compatibility with request wording
    @property
    def metadata(self) -> MutableMapping[str, Any]:  # type: ignore[override]
        return self.meta


# -------- Module-level stateless helpers (no class required by receiver) --------

def pack_base64(
    *,
    images: Mapping[str, torch.Tensor],
    states: Optional[Mapping[str, torch.Tensor]] = None,
    prev_actions: Optional[Mapping[str, torch.Tensor]] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Pack fields into a self-contained base64 pickle.

    Returns a base64 string that any Python process with `torch` installed can
    decode using only the stdlib `pickle` module. No import of this module is
    necessary to recover a plain `dict` with tensors and metadata.
    """
    buf = InferenceBufferV2(
        images=images or {},
        states=states or {},
        prev_actions=prev_actions or {},
        meta=dict(meta or {}),
    )
    # If a caller wants a semaphore, they can attach afterward; for the
    # stateless helper we keep behavior unchanged.
    return buf.pack_base64()


def unpack_base64(s: str) -> Dict[str, Any]:
    """
    Unpack a base64 string into a plain dict (no class required).

    The returned dict has keys: `header`, `meta`, `images`, `states`, `prev_actions`.
    CUDA tensors inside are views over the original GPU memory via CUDA IPC.
    """
    return ForkingPickler.loads(_b64d(s))
