"""Per-frame encoders: the ONLY part of the world model that differs between the three runs.

Contract every encoder must meet (checked by tests/test_world_model.py::check_encoder_contract):
    input : (B, 3, 64, 64)   one rendered frame, float in [0, 1] (road, others, ego)
    output: (B, 256, 8, 8)   one 256-d feature per 6 m x 6 m cell; cell (r, c) covers pixels [8r, 8r+8) x [8c, 8c+8)

No frame stacking: the ConvGRU carries time. Decision 17: no dropout in the encoder (in a latent rollout it only
sees real, fully observed warm-up frames); epistemic dropout lives in Pred and Dec.

Runs:
    cnn       this file (Ojaswi)
    vit       to be added by its owner
    vit_mae   the vit encoder, initialised from MAE pretraining (owner's script), loaded with --init_encoder
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """conv 4x4 stride 2: 3->64 (32x32), 64->128 (16x16), 128->256 (8x8), then conv 3x3 256->256.
    GroupNorm(8) + LeakyReLU(0.2) after each strided conv. Trained from scratch."""

    def __init__(self, in_ch: int = 3, hidden: int = 256):
        super().__init__()
        chans = [in_ch, 64, 128, hidden]
        layers = []
        for i in range(3):
            layers += [nn.Conv2d(chans[i], chans[i + 1], 4, stride=2, padding=1),
                       nn.GroupNorm(8, chans[i + 1]), nn.LeakyReLU(0.2)]
        layers += [nn.Conv2d(hidden, hidden, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ViTEncoder(nn.Module):
    """8x8-pixel patches -> 64 tokens -> linear to `hidden` + learned positional embedding -> 6 pre-norm transformer
    blocks (8 heads, MLP ratio 2, no dropout: decision 17) -> LayerNorm -> reshape to (hidden, 8, 8).
    Token i is grid cell i, row-major. `tokens(x, keep)` runs only the kept tokens (used by MAE pretraining)."""

    def __init__(self, in_ch: int = 3, hidden: int = 256, patch: int = 8, img: int = 64,
                 depth: int = 6, heads: int = 8, mlp_ratio: int = 2):
        super().__init__()
        self.grid = img // patch
        self.patch_embed = nn.Conv2d(in_ch, hidden, patch, stride=patch)      # = linear on each flattened patch
        self.pos = nn.Parameter(torch.zeros(1, self.grid ** 2, hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)
        block = nn.TransformerEncoderLayer(hidden, heads, hidden * mlp_ratio, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(block, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)

    def tokens(self, x, keep=None):
        """x: (B, 3, 64, 64). keep: optional (B, K) indices of the tokens to run. Returns (B, K or 64, hidden)."""
        t = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos          # (B, 64, hidden), row-major
        if keep is not None:
            t = t.gather(1, keep[..., None].expand(-1, -1, t.shape[-1]))
        return self.norm(self.blocks(t))

    def forward(self, x):
        t = self.tokens(x)
        return t.transpose(1, 2).reshape(x.shape[0], -1, self.grid, self.grid)


def build_encoder(kind: str, hidden: int = 256) -> nn.Module:
    if kind == "cnn":
        return CNNEncoder(3, hidden)
    if kind == "vit":
        return ViTEncoder(3, hidden)
    raise ValueError(f"unknown encoder {kind!r}")
