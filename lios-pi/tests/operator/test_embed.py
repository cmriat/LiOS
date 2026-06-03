"""Test embedding utilities for Gemma model."""

import types

import torch


def create_gemma_model():
    import pi.training.instance_config as train_config
    from pi.models_pytorch.pi0_pytorch import PI0Pytorch

    base_config = train_config.get_config("pi05_airbot")
    model_cfg = base_config.model
    device = torch.device("cuda:0")
    with torch.device(device):
        raw_model = PI0Pytorch(model_cfg)
    print("Model ready on", device)
    return raw_model


def new_embed_images(self, images, img_masks):
    """Vectorized version of _embed_images: one call to image_embed_func."""
    assert isinstance(images, list)
    num_images = len(images)
    B = images[0].shape[0]

    pixel_values = torch.cat(images, dim=0)  # [num_images * B, 3, H, W]
    img_emb_flat = self.image_embed_func(pixel_values)  # [num_images * B, L_img, D]
    _, L_img, D = img_emb_flat.shape

    img_emb_grouped = img_emb_flat.view(num_images, B, L_img, D)
    img_emb_list = [img_emb_grouped[i] for i in range(num_images)]

    img_masks_stack = torch.stack(list(img_masks), dim=0)  # [num_images, B]
    pad_masks_stack = img_masks_stack[:, :, None].expand(num_images, B, L_img)
    pad_masks_list = [pad_masks_stack[i] for i in range(num_images)]

    att_masks = [0] * (num_images * L_img)
    return img_emb_list, pad_masks_list, att_masks


def run_unit_test(model):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()

    B, C, H, W = 4, 3, 224, 224
    num_images = 3

    images = [torch.randn(B, C, H, W, dtype=torch.bfloat16, device=device) for _ in range(num_images)]
    img_masks = [torch.randint(0, 2, (B,), dtype=torch.bool, device=device) for _ in range(num_images)]

    with torch.no_grad():
        embs_old, pad_old, att_old = model._embed_images(images, img_masks)

    embed_images_new = types.MethodType(new_embed_images, model)
    with torch.no_grad():
        embs_new, pad_new, att_new = embed_images_new(images, img_masks)

    assert att_old == att_new, "att_masks mismatch"
    assert len(embs_old) == len(embs_new) == len(pad_old) == len(pad_new)

    max_diff = 0.0
    for e_old, e_new, p_old, p_new in zip(embs_old, embs_new, pad_old, pad_new, strict=True):
        assert torch.equal(p_old, p_new), "pad_masks mismatch"
        diff = (e_old.float() - e_new.float()).abs().max().item()
        max_diff = max(max_diff, diff)

    print(f"[INFO] max abs diff in embeddings = {max_diff}")
    if not torch.allclose(
        torch.cat([t.float() for t in embs_old]),
        torch.cat([t.float() for t in embs_new]),
        rtol=1e-3,
        atol=1e-3,
    ):
        raise AssertionError("embeddings mismatch beyond tolerance")

    print("✅ Test Passed: new _embed_images matches original (within tolerance).")


def main():
    print("=== Loading model... ===")
    model = create_gemma_model()

    print("=== Running unit test ===")
    run_unit_test(model)


if __name__ == "__main__":
    main()
