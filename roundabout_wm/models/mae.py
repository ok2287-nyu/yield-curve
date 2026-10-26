"""MAE pretraining for the vit encoder (He et al., CVPR 2022), on single rendered frames.

    64 patches of 8x8 px; keep a random 25% (16), drop 48. The ViTEncoder runs on the visible tokens only (no mask
    tokens). The decoder (2 blocks, width 128) gets the encoded tokens plus 48 learned mask tokens at their positions
    and predicts 192 pixel values per patch. Loss: MSE on masked patches only, targets normalised per patch
    (x - mean) / (std + 1e-3). After pretraining keep `mae.encoder` (saved as runs/mae/encoder.pt), drop the decoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import ViTEncoder

GREEN = 1


def patchify(x, p: int = 8):
    """(B, C, H, W) -> (B, (H/p)*(W/p), C*p*p); patch order is row-major, the same as the encoder's tokens."""
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (H // p) * (W // p), C * p * p)


class MAE(nn.Module):
    def __init__(self, encoder: ViTEncoder | None = None, keep: int = 16, dec_dim: int = 128, dec_depth: int = 2,
                 dec_heads: int = 4, patch: int = 8):
        super().__init__()
        self.encoder = encoder or ViTEncoder()
        self.keep, self.patch = keep, patch
        n, hidden = self.encoder.grid ** 2, self.encoder.pos.shape[-1]
        self.embed = nn.Linear(hidden, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos = nn.Parameter(torch.zeros(1, n, dec_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.dec_pos, std=0.02)
        block = nn.TransformerEncoderLayer(dec_dim, dec_heads, dec_dim * 2, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.dec_blocks = nn.TransformerEncoder(block, dec_depth, enable_nested_tensor=False)
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.head = nn.Linear(dec_dim, 3 * patch * patch)

    def random_keep(self, B, device):
        """(B, keep) indices of visible patches, (B, n) boolean mask (True = masked)."""
        n = self.encoder.grid ** 2
        order = torch.rand(B, n, device=device).argsort(1)
        idx = order[:, :self.keep]
        masked = torch.ones(B, n, dtype=torch.bool, device=device)
        masked.scatter_(1, idx, False)
        return idx, masked

    def forward(self, x, keep_idx=None):
        """x: (B, 3, 64, 64) in [0, 1]. Returns loss, pred (B, 64, 192), per-patch loss (B, 64), masked (B, 64)."""
        B = x.shape[0]
        if keep_idx is None:
            keep_idx, masked = self.random_keep(B, x.device)
        else:
            masked = torch.ones(B, self.encoder.grid ** 2, dtype=torch.bool, device=x.device)
            masked.scatter_(1, keep_idx, False)
        vis = self.embed(self.encoder.tokens(x, keep_idx))                         # (B, keep, dec_dim)
        full = self.mask_token.expand(B, masked.shape[1], -1).to(vis.dtype).clone()   # (B, 64, dec_dim); bf16 under autocast
        full.scatter_(1, keep_idx[..., None].expand_as(vis), vis)
        pred = self.head(self.dec_norm(self.dec_blocks(full + self.dec_pos)))
        tgt = patchify(x, self.patch)
        tgt = (tgt - tgt.mean(-1, keepdim=True)) / (tgt.std(-1, keepdim=True) + 1e-3)
        per_patch = ((pred - tgt) ** 2).mean(-1)                                   # (B, 64)
        loss = (per_patch * masked).sum() / masked.sum()
        return loss, pred, per_patch, masked

    @torch.no_grad()
    def car_patches(self, x):
        """(B, 64) True where the patch contains any other road user (green channel > 0)."""
        return patchify(x[:, GREEN:GREEN + 1], self.patch).amax(-1) > 0
