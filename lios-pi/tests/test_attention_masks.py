"""CPU unit tests for make_att_2d_masks (PaliGemma-style prefix-LM mask)."""

import pytest

torch = pytest.importorskip("torch")

from pi.models_pytorch.pi0_pytorch import make_att_2d_masks  # noqa: E402


def test_prefix_lm_pattern():
    # mask_ar = [0, 0, 1, 1]: tokens 0,1 are a bidirectional prefix; tokens 2,3
    # attend causally to everything with cumsum <= their own.
    pad = torch.ones(1, 4, dtype=torch.bool)
    ar = torch.tensor([[0, 0, 1, 1]], dtype=torch.int32)

    mask = make_att_2d_masks(pad, ar)

    expected = torch.tensor([[
        [True,  True,  False, False],
        [True,  True,  False, False],
        [True,  True,  True,  False],
        [True,  True,  True,  True],
    ]])
    assert torch.equal(mask, expected)


def test_padding_zeros_row_and_col():
    # Token 2 is padding: nobody attends to it, and it attends to nothing.
    pad = torch.tensor([[True, True, False]], dtype=torch.bool)
    ar = torch.tensor([[0, 1, 1]], dtype=torch.int32)

    mask = make_att_2d_masks(pad, ar)

    assert not mask[0, :, 2].any()
    assert not mask[0, 2, :].any()


def test_rejects_non_2d_input():
    pad = torch.ones(4, dtype=torch.bool)
    ar = torch.zeros(4, dtype=torch.int32)
    with pytest.raises(ValueError):
        make_att_2d_masks(pad, ar)
