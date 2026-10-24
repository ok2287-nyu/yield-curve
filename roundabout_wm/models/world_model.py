"""World model v1: a spatial RSSM-style model with an SVG-style learned prior, a kinematic ego and a decoder that
predicts only the other road users (green channel). Design: context doc section 3 / WORLD_MODEL_BRIEF.

Indexing (one window = n_cond + n_pred frames, 10 + 20 by default; act[i] moves frame i -> i+1; v[i] = speed at i):

    warm-up, i = 0 .. n_cond-1                       (observed frames)
        e_i  = Enc(frame_i)                                       256x8x8
        u_i  = P_e(e_i) + A(act[i-1], v[i-1])                     A-term is 0 for i = 0
        h_i  = Pred(h_{i-1}, u_i),  h_{-1} = 0
        loss : MSE(Dec(h_i), green_i)                             warm-up reconstruction

    rollout, i = n_cond-1 .. n_cond+n_pred-2, predicting frame i+1
        prior      p(z_i | h_i)                                   learned, per cell (8 channels x 8 x 8)
        posterior  q(z_i | h_i, e_{i+1})                          training / diagnostics only
        z_i        ~ q with prob 1-p, ~ p with prob p             z-dropout, per sample and step, whole grid
        u_i        = P_z(z_i) + A(act[i], v[i])
        h_{i+1}    = Pred(h_i, u_i)
        s_{i+1}    = Dec(h_{i+1})                                 green channel only, no residual
        loss : MSE(s_{i+1}, green_{i+1}) + beta * KL(q || p)

The action (with the ego speed) enters ONLY Pred, through A. It never enters the prior or the posterior.
Only h is autoregressive: decoded frames are never re-encoded, and the encoder only ever sees real frames.

Stop-gradient on prior samples (prior_sg, adopted Oct 23): on z-dropout steps that sample the prior,
    z = sg(mu_p) + sg(sigma_p) * eps
The forward value is unchanged, so Pred/Dec still learn to cope with prior samples, but the MSE no longer pushes
sigma_p -> 0 through the reparameterisation; the prior head is trained by the KL alone (as in SVG-LP).
--no-prior_sg turns it off for the ablation. It applies only to z_source="mix" (training); imagination is
unaffected because there the world model is frozen anyway.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import build_encoder

GREEN = 1


@dataclass
class WMConfig:
    encoder: str = "cnn"
    n_cond: int = 10
    hidden: int = 256
    z_ch: int = 8               # latent channels per 6 m cell -> z is z_ch x 8 x 8 = 512 numbers
    dropout: float = 0.1        # Dropout2d in Pred (input u, output h) and Dec; NOT in the encoder or the prior head
    p_zdrop: float = 0.5        # probability of sampling z from the prior on a training step
    beta: float = 1e-6          # KL weight, relative to a per-pixel MEAN squared error
    prior_sg: bool = True       # stop-gradient on prior samples during z-dropout training
    logvar_min: float = -8.0
    logvar_max: float = 6.0


# ---------------------------------------------------------------------------------------------------------------
class StickyDropout2d(nn.Module):
    """Dropout2d (drops whole channels of every cell) whose mask can be held fixed across calls.

    Normal training: a fresh mask on every call. Inside `WorldModel.fixed_dropout_masks()`, the first call samples
    one mask and every later call reuses it, so a whole rollout behaves like one member of an ensemble
    (epistemic estimate for the policy; context doc 3.8).
    """

    def __init__(self, p: float):
        super().__init__()
        self.p, self.sticky, self.mask = p, False, None

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        if not self.sticky:
            return F.dropout2d(x, self.p, True)
        if self.mask is None or self.mask.shape[:2] != x.shape[:2]:
            keep = torch.rand(x.shape[0], x.shape[1], 1, 1, device=x.device) >= self.p
            self.mask = keep.to(x.dtype) / (1 - self.p)
        return x * self.mask.to(x.dtype)


class ConvGRUCell(nn.Module):
    """r = s(W_r*u + U_r*h); g = s(W_g*u + U_g*h); c = tanh(W*u + U*(r.h)); h' = (1-g).h + g.c   (* = 3x3 conv).

    Dropout2d on the input u and on the output h' (the carried state).
    """

    def __init__(self, ch: int, k: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv_u = nn.Conv2d(ch, 3 * ch, k, padding=k // 2)                   # W_r, W_g, W (with biases)
        self.conv_h = nn.Conv2d(ch, 2 * ch, k, padding=k // 2, bias=False)       # U_r, U_g
        self.conv_c = nn.Conv2d(ch, ch, k, padding=k // 2, bias=False)           # U
        self.drop_in, self.drop_out = StickyDropout2d(dropout), StickyDropout2d(dropout)

    def forward(self, h, u):
        u = self.drop_in(u)
        ur, ug, uc = self.conv_u(u).chunk(3, 1)
        hr, hg = self.conv_h(h).chunk(2, 1)
        r, g = torch.sigmoid(ur + hr), torch.sigmoid(ug + hg)
        c = torch.tanh(uc + self.conv_c(r * h))
        return self.drop_out((1 - g) * h + g * c)


class GaussHead(nn.Module):
    """conv3x3 in->hidden, LeakyReLU(0.2), conv3x3 hidden->2*z_ch  ->  (mu, logvar) per cell, each (B, z_ch, 8, 8).

    The last layer runs in fp32 even under bf16 autocast, so the log-variance and the KL are computed in fp32.
    No dropout here: sigma_p is meant to be an aleatoric signal.
    """

    def __init__(self, in_ch: int, hidden: int, z_ch: int, lv_min: float, lv_max: float):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(in_ch, hidden, 3, padding=1), nn.LeakyReLU(0.2))
        self.out = nn.Conv2d(hidden, 2 * z_ch, 3, padding=1)
        self.lv_min, self.lv_max = lv_min, lv_max

    def forward(self, x):
        y = self.body(x)
        with torch.autocast(device_type=y.device.type, enabled=False):
            mu, lv = self.out(y.float()).chunk(2, 1)
        return mu, lv.clamp(self.lv_min, self.lv_max)


def gauss_kl(mu_q, lv_q, mu_p, lv_p):
    """KL(q || p) for diagonal Gaussians, summed over every latent element (z_ch x 8 x 8) -> (B,). fp32."""
    mu_q, lv_q, mu_p, lv_p = (t.float() for t in (mu_q, lv_q, mu_p, lv_p))
    return 0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p) ** 2) / lv_p.exp() - 1).flatten(1).sum(1)


# ---------------------------------------------------------------------------------------------------------------
class WorldModel(nn.Module):
    def __init__(self, cfg: WMConfig):
        super().__init__()
        self.cfg = cfg
        C, Z, p = cfg.hidden, cfg.z_ch, cfg.dropout
        self.encoder = build_encoder(cfg.encoder, C)                               # Enc: the only part that differs
        self.P_e = nn.Conv2d(C, C, 1)                                              # encoder -> Pred input (warm-up)
        self.P_z = nn.Conv2d(Z, C, 1)                                              # z -> Pred input, per cell
        self.A = nn.Sequential(nn.Linear(3, C), nn.LeakyReLU(0.2), nn.Linear(C, C * 64))  # (acc, curv, v) -> 256x8x8
        self.pred = ConvGRUCell(C, 3, p)                                           # Pred
        self.prior = GaussHead(C, C, Z, cfg.logvar_min, cfg.logvar_max)
        self.posterior = GaussHead(2 * C, C, Z, cfg.logvar_min, cfg.logvar_max)
        self.dec = nn.Sequential(                                                  # Dec: 8 -> 16 -> 32 -> 64, green only
            nn.Conv2d(C, C, 3, padding=1), nn.GroupNorm(8, C), nn.LeakyReLU(0.2), StickyDropout2d(p),
            nn.ConvTranspose2d(C, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.LeakyReLU(0.2), StickyDropout2d(p),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 1, 4, 2, 1))

    # ---- pieces -------------------------------------------------------------------------------------------------
    def encode(self, frames):
        """(B, T, 3, 64, 64) -> (B, T, C, 8, 8); one batched encoder call."""
        B, T = frames.shape[:2]
        return self.encoder(frames.flatten(0, 1)).unflatten(0, (B, T))

    def action_map(self, act, spd):
        """act (B, 2) normalised (acc, curv); spd (B, 1) normalised speed -> (B, C, 8, 8)."""
        return self.A(torch.cat([act, spd], -1)).view(act.shape[0], self.cfg.hidden, 8, 8)

    def decode(self, h):
        return torch.sigmoid(self.dec(h))                                          # (B, 1, 64, 64)

    def _sample(self, mu, lv, detach=False):
        if detach:
            mu, lv = mu.detach(), lv.detach()
        return mu + (0.5 * lv).exp() * torch.randn_like(mu)

    # ---- warm-up ------------------------------------------------------------------------------------------------
    def warmup(self, e, actions, speeds):
        """e (B, n, C, 8, 8) encoded observed frames; actions (B, >=n-1, 2); speeds (B, >=n, 1).

        Returns h_{n-1} and the list of h_i for the warm-up reconstruction.
        """
        n = e.shape[1]
        h = torch.zeros_like(e[:, 0])
        hs = []
        for i in range(n):
            u = self.P_e(e[:, i])
            if i > 0:
                u = u + self.action_map(actions[:, i - 1], speeds[:, i - 1])
            h = self.pred(h, u)
            hs.append(h)
        return h, hs

    # ---- full pass ----------------------------------------------------------------------------------------------
    def forward(self, images, actions, speeds, n_pred, z_source="mix", z_seq=None):
        """images  (B, L, 3, 64, 64) in [0, 1]; for z_source "prior"/"given" only the first n_cond are read
        actions (B, L-1, 2) normalised; speeds (B, L, 1) normalised; act[i] moves frame i -> i+1
        z_source "mix"   training: posterior or (prob p_zdrop) prior, per sample and step
                 "post"  always the posterior (diagnostic: how well z can explain the future)
                 "prior" always the prior (open-loop evaluation, imagination)
                 "given" z_seq[k] (B, z_ch, 8, 8) at rollout step k (e.g. one z shared across dropout masks)
        Returns dict:
            green        (B, n_pred, 1, 64, 64)  predicted frames n_cond .. n_cond+n_pred-1
            green_warm   (B, n_cond, 1, 64, 64)  warm-up reconstructions of frames 0 .. n_cond-1
            kl           (B, n_pred) or None     KL(q || p) per step (fp32), when the posterior is computed
            prior_std    (B, n_pred)             mean prior sigma over the latent, per step
            prior_std_map(B, n_pred, 8, 8)       prior sigma averaged over the z channels, per cell
            z            (n_pred, B, z_ch, 8, 8) the latents used (detached), for reuse via z_source="given"
        """
        n, cfg = self.cfg.n_cond, self.cfg
        use_post = z_source in ("mix", "post")
        T_enc = n + n_pred if use_post else n
        e = self.encode(images[:, :T_enc])
        h, hs = self.warmup(e[:, :n], actions, speeds)
        # decoded step by step (not as one B*n batch) so a held dropout mask is per sample: see StickyDropout2d
        green_warm = torch.stack([self.decode(hi) for hi in hs], 1)

        greens, kls, stds, std_maps, zs = [], [], [], [], []
        B = h.shape[0]
        for k in range(n_pred):
            i = n - 1 + k                                                          # current frame; predict i+1
            mu_p, lv_p = self.prior(h)
            if use_post:
                mu_q, lv_q = self.posterior(torch.cat([h, e[:, i + 1]], 1))
                kls.append(gauss_kl(mu_q, lv_q, mu_p, lv_p))
                z_q = self._sample(mu_q, lv_q)
            if z_source == "mix":
                z_p = self._sample(mu_p, lv_p, detach=cfg.prior_sg)
                use_prior = (torch.rand(B, device=h.device) < cfg.p_zdrop)[:, None, None, None]
                z = torch.where(use_prior, z_p, z_q)
            elif z_source == "post":
                z = z_q
            elif z_source == "prior":
                z = self._sample(mu_p, lv_p)
            elif z_source == "given":
                z = z_seq[k]
            else:
                raise ValueError(z_source)
            u = self.P_z(z.to(h.dtype)) + self.action_map(actions[:, i], speeds[:, i])
            h = self.pred(h, u)
            greens.append(self.decode(h))
            sd = (0.5 * lv_p).exp()
            stds.append(sd.flatten(1).mean(1)); std_maps.append(sd.mean(1))
            zs.append(z.detach())
        return {"green": torch.stack(greens, 1), "green_warm": green_warm,
                "kl": torch.stack(kls, 1) if kls else None,
                "prior_std": torch.stack(stds, 1).detach(), "prior_std_map": torch.stack(std_maps, 1).detach(),
                "z": torch.stack(zs, 0)}

    def loss(self, batch, n_pred):
        """L = mean_t MSE(rollout) + mean_t MSE(warm-up) + beta * mean_t KL; MSE averaged over pixels."""
        n = self.cfg.n_cond
        imgs, act, spd = batch["images"], batch["actions"], batch["speed"]
        out = self(imgs, act, spd, n_pred, z_source="mix")
        tgt = imgs[:, n:n + n_pred, GREEN:GREEN + 1]
        mse_t = ((out["green"].float() - tgt) ** 2).mean((0, 2, 3, 4))           # per rollout step
        mse_warm = ((out["green_warm"].float() - imgs[:, :n, GREEN:GREEN + 1]) ** 2).mean()
        kl_t = out["kl"].mean(0)
        loss = mse_t.mean() + mse_warm + self.cfg.beta * kl_t.mean()
        return loss, {"loss": loss.item(), "mse": mse_t.mean().item(), "mse_warm": mse_warm.item(),
                      "kl": kl_t.mean().item(), "prior_std": out["prior_std"].mean().item()}

    # ---- epistemic helper (for the policy later) ---------------------------------------------------------------
    @contextmanager
    def fixed_dropout_masks(self):
        """Within this block every StickyDropout2d samples one mask on first use and keeps it.
        Typical use: model.train() for dropout, a fixed z_seq, M passes, each inside its own block."""
        mods = [m for m in self.modules() if isinstance(m, StickyDropout2d)]
        for m in mods:
            m.sticky, m.mask = True, None
        try:
            yield
        finally:
            for m in mods:
                m.sticky, m.mask = False, None


def param_counts(model: WorldModel) -> dict:
    count = lambda m: sum(p.numel() for p in m.parameters())
    parts = {k: count(getattr(model, k)) for k in ("encoder", "P_e", "P_z", "A", "pred", "prior", "posterior", "dec")}
    parts["total"] = count(model)
    return parts
