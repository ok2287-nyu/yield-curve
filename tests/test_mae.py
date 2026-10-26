"""ViT encoder + MAE tests:  python -m pytest tests/test_mae.py   or   python tests/test_mae.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from roundabout_wm.models.encoders import build_encoder
from roundabout_wm.models.mae import MAE, patchify
from tests.test_world_model import check_encoder_contract


def test_encoder_contract_vit():
    check_encoder_contract(build_encoder("vit"))


def test_token_i_is_grid_cell_i():
    """Token i must be the 8x8 patch at row i // 8, column i % 8 (the encoder contract)."""
    enc = build_encoder("vit").eval()
    x = torch.zeros(1, 3, 64, 64)
    y0 = enc(x)
    x[:, :, 8:16, 24:32] = 1.0                                                  # patch (row 1, col 3) only
    t0, t1 = enc.tokens(torch.zeros(1, 3, 64, 64)), enc.tokens(x)
    assert t0.shape == (1, 64, 256) and y0.shape == (1, 256, 8, 8)
    # patch embedding is local: before the blocks, only token 1 * 8 + 3 changes
    e0 = enc.patch_embed(torch.zeros(1, 3, 64, 64)).flatten(2).transpose(1, 2)
    e1 = enc.patch_embed(x).flatten(2).transpose(1, 2)
    changed = (e0 - e1).abs().amax((0, 2)).nonzero().flatten().tolist()
    assert changed == [1 * 8 + 3], changed
    assert patchify(x)[0, 11].min() == 1.0 and patchify(x)[0].sum() == 3 * 8 * 8      # same order in patchify


def test_masking_keeps_exactly_16_and_loss_is_masked_only():
    torch.manual_seed(0)
    m = MAE().eval()
    x = (torch.rand(4, 3, 64, 64) > 0.9).float()
    idx, masked = m.random_keep(4, x.device)
    assert idx.shape == (4, 16) and (masked.sum(1) == 48).all()
    assert all(len(set(r.tolist())) == 16 for r in idx)                          # distinct patches per sample
    loss, pred, per_patch, masked2 = m(x, idx)
    assert pred.shape == (4, 64, 192) and torch.equal(masked, masked2)
    assert torch.allclose(loss, (per_patch * masked).sum() / masked.sum())
    # changing the pixels of a VISIBLE patch's target must not matter beyond the encoder input; changing a MASKED
    # patch's pixels must change the loss only through that patch: zero loss weight on visible patches
    vis_only = (per_patch * ~masked).sum()
    assert vis_only > 0 and loss.item() != vis_only.item()                       # visible patches exist but are not in the loss


def test_encoder_sees_visible_tokens_only():
    """The masked patches' pixels must not influence the encoded visible tokens."""
    torch.manual_seed(0)
    enc = build_encoder("vit").eval()
    x = torch.rand(1, 3, 64, 64)
    keep = torch.arange(0, 64, 4)[None]                                          # 16 visible patches
    x2 = x.clone()
    for p in range(64):
        if p not in keep[0].tolist():
            r, c = divmod(p, 8)
            x2[:, :, 8 * r:8 * r + 8, 8 * c:8 * c + 8] = torch.rand(1, 3, 8, 8)
    assert torch.allclose(enc.tokens(x, keep), enc.tokens(x2, keep), atol=1e-6)


def test_backward_and_weights_load_into_vit():
    m = MAE()
    loss, *_ = m((torch.rand(2, 3, 64, 64) > 0.9).float())
    loss.backward()
    assert all(p.grad is not None for p in m.encoder.parameters())
    fresh = build_encoder("vit")
    fresh.load_state_dict(m.encoder.state_dict(), strict=True)                   # what train_wm.py --init_encoder does
    check_encoder_contract(fresh)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
