"""CPU unit tests for the pure helpers in pi.data."""

from types import SimpleNamespace

import numpy as np
import pytest

# pi.data imports lerobot at module load time; skip gracefully if it's missing.
pytest.importorskip("lerobot")

from pi.data import (  # noqa: E402
    SimpleLeRobotLoader,
    _apply_delta_actions,
    _make_bool_mask,
    _normalize_array,
)


# --------------------------- _make_bool_mask ---------------------------

def test_make_bool_mask_mixed_signs():
    assert _make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)


def test_make_bool_mask_zero_dim():
    # dim == 0 contributes nothing; surrounding positive runs concatenate.
    assert _make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_make_bool_mask_empty():
    assert _make_bool_mask() == ()


# --------------------------- _apply_delta_actions ---------------------------

def test_apply_delta_single_frame_state():
    # Joint dims become relative; gripper dim (mask=False) stays absolute.
    state = np.array([1.0, 2.0, 0.5], dtype=np.float64)
    actions = np.array([[3.0, 5.0, 0.7], [4.0, 6.0, 0.9]], dtype=np.float64)
    step = {"state": state, "actions": actions.copy()}

    out = _apply_delta_actions(step, (True, True, False))

    expected = np.array(
        [[2.0, 3.0, 0.7], [3.0, 4.0, 0.9]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(out["actions"], expected)


def test_apply_delta_uses_first_frame_of_history():
    # state shape (history, dim): only frame 0 (most recent) is subtracted.
    state = np.array([[10.0, 20.0], [99.0, 99.0]], dtype=np.float64)
    actions = np.array([[13.0, 25.0]], dtype=np.float64)
    step = {"state": state, "actions": actions.copy()}

    out = _apply_delta_actions(step, (True, True))

    np.testing.assert_allclose(out["actions"], [[3.0, 5.0]])


def test_apply_delta_noop_without_actions():
    step = {"state": np.array([1.0, 2.0])}
    assert _apply_delta_actions(step, (True, True)) is step


# --------------------------- _normalize_array ---------------------------

def _stats(*, mean=None, std=None, q01=None, q99=None):
    return SimpleNamespace(
        mean=None if mean is None else np.asarray(mean, dtype=np.float64),
        std=None if std is None else np.asarray(std, dtype=np.float64),
        q01=None if q01 is None else np.asarray(q01, dtype=np.float64),
        q99=None if q99 is None else np.asarray(q99, dtype=np.float64),
    )


def test_normalize_zscore():
    stats = _stats(mean=[1.0, 2.0], std=[2.0, 4.0])
    x = np.array([[3.0, 10.0]], dtype=np.float64)
    out = _normalize_array(x, stats, use_quantiles=False)
    np.testing.assert_allclose(out, [[1.0, 2.0]], rtol=1e-5)


def test_normalize_truncates_stats_to_input_last_dim():
    # stats are stored at the padded action_dim, but callers may pass narrower vectors.
    stats = _stats(mean=np.arange(8, dtype=np.float64), std=np.ones(8))
    x = np.array([[10.0, 11.0, 12.0]], dtype=np.float64)
    out = _normalize_array(x, stats, use_quantiles=False)
    np.testing.assert_allclose(out, [[10.0, 10.0, 10.0]])


def test_normalize_quantile():
    stats = _stats(mean=[0.0, 0.0], std=[1.0, 1.0], q01=[0.0, -2.0], q99=[10.0, 2.0])
    x = np.array([[5.0, 0.0]], dtype=np.float64)
    out = _normalize_array(x, stats, use_quantiles=True)
    # midpoint of [q01, q99] maps to 0.0
    np.testing.assert_allclose(out, [[0.0, 0.0]], atol=1e-5)


def test_normalize_quantile_raises_without_q01_q99():
    stats = _stats(mean=[0.0], std=[1.0])
    with pytest.raises(ValueError, match="Quantile stats required"):
        _normalize_array(np.array([0.0]), stats, use_quantiles=True)


# --------------------------- SimpleLeRobotLoader._transform (airbot) ---------------------------

class _StubTokenizer:
    """Returns deterministic tokens / mask so we can assert shapes without touching SentencePiece."""

    def __init__(self, max_len: int) -> None:
        self.max_len = max_len

    def tokenize(self, prompt, state=None):
        tokens = np.zeros(self.max_len, dtype=np.int64)
        mask = np.zeros(self.max_len, dtype=bool)
        return tokens, mask


def test_simple_loader_transform_airbot_path():
    """Verify the airbot repack -> normalize -> resize -> tokenize -> pad pipeline composes.

    Bypasses __init__ (which would hit the real LeRobot dataset) and feeds a synthetic
    sample directly through the transform chain.
    """
    loader = SimpleLeRobotLoader.__new__(SimpleLeRobotLoader)
    loader.policy_name = "airbot_smoke"
    loader.apply_delta_transform = False
    loader.use_quantile_norm = False
    loader.action_dim = 32
    loader.state_history_frames = 1
    loader.state_delay_frames = 0
    loader.discrete_state_input = False
    loader.norm_stats = {
        "state": _stats(mean=np.zeros(32), std=np.ones(32)),
        "actions": _stats(mean=np.zeros(32), std=np.ones(32)),
    }
    loader.tokenizer = _StubTokenizer(max_len=48)

    raw_sample = {
        "observation.images.cam_env": np.full((180, 320, 3), 128, dtype=np.uint8),
        "observation.images.cam_left_wrist": np.full((180, 320, 3), 64, dtype=np.uint8),
        "observation.images.cam_right_wrist": np.full((180, 320, 3), 200, dtype=np.uint8),
        "observation.state": np.arange(6, dtype=np.float32),
        "action": np.zeros((10, 6), dtype=np.float32),
        "task": "pick up the cup",
    }

    out = loader._transform(raw_sample)

    assert set(out["image"].keys()) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    for img in out["image"].values():
        assert img.shape == (224, 224, 3)
        assert img.dtype == np.uint8
    assert all(out["image_mask"].values())

    # State/actions are padded to action_dim=32.
    assert out["state"].shape[-1] == 32
    assert out["actions"].shape == (10, 32)

    # Tokenizer output is preserved verbatim.
    assert out["tokenized_prompt"].shape == (48,)
    assert out["tokenized_prompt_mask"].shape == (48,)
