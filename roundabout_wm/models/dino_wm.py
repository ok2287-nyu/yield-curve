"""DINO-WM-style world model (Zhou, Pan, LeCun & Pinto, ICML 2025) for our rasters, plus an optional per-step latent.

    WorldModel(WMConfig(arch="dino", ...)) builds this class, so every evaluation script runs unchanged.

What it is
    encoder    FROZEN pretrained DINOv2 (default ViT-S/14). Each 64x64 frame (road, others, ego as RGB) is upsampled
               to 112x112, so the 14-px patches give an 8x8 grid = our 6 m cells; features D = 384 per cell.
    predictor  a ViT over the last K frames (K = hist, default 4 = 0.8 s), no recurrent state. Each frame's tokens
               are its 64 patch features plus the ego action that LEAVES that frame and the ego speed at it
               (acc, curv, v; DINO-WM concatenates the action to every patch). Full attention over K x 64 tokens;
               the last frame's output tokens predict the next frame's features (residual on the last frame's).
    latent     optional (deterministic=False): z (z_ch x 8 x 8) per step, learned prior p(z | last two frames),
               posterior q(z | last two frames, true next frame), added to the predictor's output tokens. As in
               the old model the action never enters the prior or the posterior; KL warm-up / beta / anchor /
               z-dropout / prior stop-gradient work the same way.
    decoder    features -> others channel (64x64), trained on the TRUE features of the warm-up frames (DINO-WM
               trains its decoder apart from the dynamics) and, with pix_weight > 0, also on predicted frames.

Losses (per training window)
    feat   MSE(predicted features, true DINO features) at every predicted step (DINO-WM's dynamics loss)
    warm   MSE(Dec(true features), true others channel) over the warm-up frames (the decoder)
    pix    pix_weight * MSE(Dec(predicted features), true others channel)
    kl     beta * KL(q || p) (+ anchor_prior * KL(p || N(0, I)))
Training is teacher-forced by default (DINO-WM): after each step the buffer gets the TRUE next features, so all
steps learn from real history. teacher_forcing=False rolls out on its own predictions instead.

State h, so imagine() / warm_states() / the planners work unchanged: the buffer of the last K frames, each frame
D feature channels + 3 channels (acc, curv, v, broadcast over the grid), flattened to (B, K * (D + 3), 8, 8).
The last frame's action slots are empty until rollout_step writes the plan's action into them.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

GREEN = 1
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)
DINO_DIMS = {"dinov2_vits14": 384, "dinov2_vits14_reg": 384, "dinov2_vitb14": 768, "dinov2_vitb14_reg": 768,
             "stub": 384}


def feat_dim(name: str) -> int:
    return DINO_DIMS[name]


def hidden_size(name: str, hist: int) -> int:
    """Channels of the state h: K frames x (D features + acc, curv, v)."""
    return hist * (feat_dim(name) + 3)


class StubDino(nn.Module):
    """Tests / offline: a frozen random patch embedding with DINOv2's interface (x_norm_patchtokens)."""

    def __init__(self, dim=384, patch=14):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, stride=patch)
        self.norm = nn.LayerNorm(dim)

    def forward_features(self, x):
        return {"x_norm_patchtokens": self.norm(self.proj(x).flatten(2).transpose(1, 2))}


HF_NAMES = {"dinov2_vits14": "facebook/dinov2-small", "dinov2_vitb14": "facebook/dinov2-base",
            "dinov2_vits14_reg": "facebook/dinov2-with-registers-small",
            "dinov2_vitb14_reg": "facebook/dinov2-with-registers-base"}


class HFDino(nn.Module):
    """A Hugging Face transformers DINOv2 with DINOv2's own interface: forward_features(x)["x_norm_patchtokens"].
    last_hidden_state is already layer-normed (Dinov2Model applies its final LayerNorm), as x_norm_patchtokens is;
    the CLS token (and register tokens) are dropped. Position embeddings are interpolated to the input size."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_prefix = 1 + int(getattr(model.config, "num_register_tokens", 0) or 0)

    def forward_features(self, x):
        out = self.model(pixel_values=x).last_hidden_state
        return {"x_norm_patchtokens": out[:, self.n_prefix:]}


def load_dino(name: str) -> nn.Module:
    """Frozen DINOv2. First choice: Hugging Face transformers (works on Python 3.9; the facebookresearch/dinov2
    torch.hub code now needs Python >= 3.10). Tries the local cache first (compute nodes have no internet)."""
    if name == "stub":
        torch.manual_seed(0)
        return StubDino(DINO_DIMS["stub"])
    errors = []
    if name in HF_NAMES:
        try:
            from transformers import AutoModel
            try:
                m = AutoModel.from_pretrained(HF_NAMES[name], local_files_only=True)
            except Exception:  # noqa: BLE001 - not cached yet: download (login node)
                m = AutoModel.from_pretrained(HF_NAMES[name])
            return HFDino(m)
        except Exception as e:  # noqa: BLE001
            errors.append(f"transformers: {e}")
    try:
        return torch.hub.load("facebookresearch/dinov2", name, trust_repo=True)
    except Exception as e:  # noqa: BLE001
        errors.append(f"torch.hub: {e}")
    raise RuntimeError(
        f"could not load {name}: {' | '.join(errors)}. Compute nodes have no internet: run once on a login node: "
        f"pip install 'transformers<5' && python -c \"from transformers import AutoModel; "
        f"AutoModel.from_pretrained('{HF_NAMES.get(name, name)}')\"")


class FrozenDino(nn.Module):
    """(B, 3, 64, 64) in [0, 1] -> (B, D, 8, 8). Frozen, always in eval mode, no gradients."""

    def __init__(self, name: str, size: int = 112):
        super().__init__()
        self.net, self.size = load_dino(name), size
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()
        self.register_buffer("mean", torch.tensor(IMNET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMNET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):          # the backbone never leaves eval mode
        super().train(mode)
        self.net.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        x = F.interpolate(x.float(), size=(self.size, self.size), mode="bilinear", align_corners=False)
        t = self.net.forward_features((x - self.mean) / self.std)["x_norm_patchtokens"]      # (B, 64, D)
        g = int(math.isqrt(t.shape[1]))
        return t.float().transpose(1, 2).reshape(x.shape[0], -1, g, g)


class DinoWM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from .world_model import GaussHead, StickyDropout2d   # same building blocks as the old model
        self.cfg = cfg
        K, D, W, Z, p = cfg.hist, feat_dim(cfg.dino), cfg.pred_width, cfg.z_ch, cfg.dropout
        if cfg.hidden != hidden_size(cfg.dino, K):      # the state's channel count (imagine() reads cfg.hidden)
            import dataclasses
            cfg = self.cfg = dataclasses.replace(cfg, hidden=hidden_size(cfg.dino, K))
        self.K, self.D = K, D
        self.encoder = FrozenDino(cfg.dino)
        self.embed = nn.Linear(D + 3, W)
        self.pos_space = nn.Parameter(torch.zeros(1, 1, 64, W))
        self.pos_time = nn.Parameter(torch.zeros(1, K, 1, W))
        nn.init.trunc_normal_(self.pos_space, std=0.02)
        nn.init.trunc_normal_(self.pos_time, std=0.02)
        block = nn.TransformerEncoderLayer(W, cfg.pred_heads, 4 * W, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.pred = nn.TransformerEncoder(block, cfg.pred_depth, enable_nested_tensor=False)
        self.drop = StickyDropout2d(p)                                            # epistemic masks (planner)
        self.P_z = nn.Linear(Z, W)
        self.head = nn.Sequential(nn.LayerNorm(W), nn.Linear(W, D))
        nn.init.zeros_(self.head[1].weight); nn.init.zeros_(self.head[1].bias)     # step 0: copy the last frame
        self.prior = GaussHead(2 * D, 256, Z, cfg.logvar_min, cfg.logvar_max)
        self.posterior = GaussHead(3 * D, 256, Z, cfg.logvar_min, cfg.logvar_max)
        C = 256
        self.dec = nn.Sequential(
            nn.Conv2d(D, C, 3, padding=1), nn.GroupNorm(8, C), nn.LeakyReLU(0.2), StickyDropout2d(p),
            nn.ConvTranspose2d(C, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.LeakyReLU(0.2), StickyDropout2d(p),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 1, 4, 2, 1))

    # ---- state buffer ---------------------------------------------------------------------------------------------
    def _split(self, h):
        """(B, K*(D+3), 8, 8) -> features (B, K, D, 8, 8), action slots (B, K, 3, 8, 8)."""
        h = h.view(h.shape[0], self.K, self.D + 3, 8, 8)
        return h[:, :, :self.D], h[:, :, self.D:]

    @staticmethod
    def _join(f, a):
        return torch.cat([f, a], 2).flatten(1, 2)

    @staticmethod
    def _slot(act, spd):
        """act (B, 2), spd (B, 1) -> (B, 3, 8, 8)."""
        return torch.cat([act, spd], -1).float()[:, :, None, None].expand(-1, 3, 8, 8)

    # ---- pieces (same names as the old WorldModel) ------------------------------------------------------------------
    def encode(self, frames):
        """(B, T, 3, 64, 64) -> (B, T, D, 8, 8)."""
        B, T = frames.shape[:2]
        return self.encoder(frames.flatten(0, 1)).unflatten(0, (B, T))

    def zero_z(self, B, device):
        return torch.zeros(B, self.cfg.z_ch, 8, 8, device=device)

    def decode(self, f, prev=None):
        """(B, D, 8, 8) features -> (B, 1, 64, 64) others channel."""
        return torch.sigmoid(self.dec(f))

    def prior_params(self, h):
        f, _ = self._split(h)
        return self.prior(torch.cat([f[:, -1], f[:, -1] - f[:, -2]], 1))

    def posterior_params(self, h, f_next):
        f, _ = self._split(h)
        return self.posterior(torch.cat([f[:, -1], f[:, -1] - f[:, -2], f_next], 1))

    def predict(self, h, z):
        """h with the current action already in the last slot -> predicted next features (B, D, 8, 8)."""
        f, a = self._split(h)
        B = f.shape[0]
        tok = torch.cat([f, a], 2).permute(0, 1, 3, 4, 2).reshape(B, self.K, 64, self.D + 3)
        x = self.embed(tok) + self.pos_space + self.pos_time
        y = self.pred(x.flatten(1, 2))[:, -64:]                                    # the last frame's tokens
        y = y + self.P_z(z.float().permute(0, 2, 3, 1).reshape(B, 64, -1)).to(y.dtype)
        y = self.drop(y.transpose(1, 2).reshape(B, -1, 8, 8)).flatten(2).transpose(1, 2)
        return f[:, -1] + self.head(y).float().transpose(1, 2).reshape(B, self.D, 8, 8)

    def advance(self, h, act, spd, f_next):
        """Write (act, spd) into the last frame's slot, then shift in f_next with empty slots."""
        f, a = self._split(h)
        a = torch.cat([a[:, :-1], self._slot(act, spd)[:, None]], 1)
        f = torch.cat([f[:, 1:], f_next[:, None].to(f.dtype)], 1)
        a = torch.cat([a[:, 1:], torch.zeros_like(a[:, :1])], 1)
        return self._join(f, a)

    def with_action(self, h, act, spd):
        f, a = self._split(h)
        a = torch.cat([a[:, :-1], self._slot(act, spd)[:, None]], 1)
        return self._join(f, a)

    def rollout_step(self, h, z, act, spd, frame=None, prev=None):
        """One imagined step (frame i -> i+1) under the plan's action. Returns (h, green)."""
        f_next = self.predict(self.with_action(h, act, spd), z)
        return self.advance(h, act, spd, f_next), self.decode(f_next)

    def warmup(self, e, actions, speeds):
        """e (B, n, D, 8, 8): the observed frames' features -> (h after frame n-1, []). The last K frames go into the
        buffer, each with the action that left it and its speed; the last frame's action slot stays empty."""
        n, K = e.shape[1], self.K
        if n < K:
            raise ValueError(f"need at least hist={K} observed frames, got {n}")
        idx = range(n - K, n)
        slots = [self._slot(actions[:, j], speeds[:, j]) if j < n - 1
                 else torch.zeros(e.shape[0], 3, 8, 8, device=e.device) for j in idx]
        return self._join(e[:, n - K:].float(), torch.stack(slots, 1)), []

    def _sample(self, mu, lv, detach=False):
        if detach:
            mu, lv = mu.detach(), lv.detach()
        return mu + (0.5 * lv).exp() * torch.randn_like(mu)

    # ---- full pass ----------------------------------------------------------------------------------------------------
    def forward(self, images, actions, speeds, n_pred, z_source="mix", z_seq=None, open_loop=None):
        """Same signature and outputs as the old WorldModel.forward (+ "feat", "feat_true")."""
        cfg, n = self.cfg, self.cfg.n_cond
        if cfg.deterministic:
            z_source = "zero"
        use_post = z_source in ("mix", "post")
        need_true = use_post or self.training
        T_enc = n + n_pred if need_true and images.shape[1] >= n + n_pred else n
        e = self.encode(images[:, :T_enc])
        h, _ = self.warmup(e[:, :n], actions, speeds)
        green_warm = torch.stack([self.decode(e[:, j]) for j in range(n)], 1)
        B, dev = h.shape[0], h.device
        if z_source == "mix" and open_loop is None:
            open_loop = torch.rand(B, device=dev) < cfg.p_openloop
        tf = self.training and cfg.teacher_forcing and T_enc > n
        feats, greens, kls, anchors, stds, std_maps, zs, lat = [], [], [], [], [], [], [], []
        for k in range(n_pred):
            i = n - 1 + k
            if z_source == "zero":
                z = self.zero_z(B, dev)
                stds.append(torch.zeros(B, device=dev)); std_maps.append(torch.zeros(B, 8, 8, device=dev))
            else:
                mu_p, lv_p = self.prior_params(h)
                if use_post:
                    mu_q, lv_q = self.posterior_params(h, e[:, i + 1].float())
                    from .world_model import gauss_kl
                    kls.append(gauss_kl(mu_q, lv_q, mu_p, lv_p))
                    if cfg.anchor_prior > 0:
                        anchors.append(gauss_kl(mu_p, lv_p, torch.zeros_like(mu_p), torch.zeros_like(lv_p)))
                    with torch.no_grad():
                        lat.append(torch.stack([
                            lv_q.float().mean(), lv_p.float().mean(),
                            ((mu_q - mu_p).abs() / (0.5 * lv_p).exp()).float().mean(),
                            mu_q.float().abs().mean(), mu_p.float().abs().mean(),
                            mu_q.float().std(0).mean() if B > 1 else mu_q.new_zeros((), dtype=torch.float32)]))
                    z_q = self._sample(mu_q, lv_q)
                if z_source == "mix":
                    z_p = self._sample(mu_p, lv_p, detach=cfg.prior_sg)
                    use_prior = ((torch.rand(B, device=dev) < cfg.p_zdrop) | open_loop).view(B, 1, 1, 1)
                    z = torch.where(use_prior, z_p, z_q)
                elif z_source == "post":
                    z = z_q
                elif z_source == "prior":
                    z = self._sample(mu_p, lv_p)
                elif z_source == "given":
                    z = z_seq[k]
                else:
                    raise ValueError(z_source)
                sd = (0.5 * lv_p).exp()
                stds.append(sd.flatten(1).mean(1)); std_maps.append(sd.mean(1))
            f_next = self.predict(self.with_action(h, actions[:, i], speeds[:, i]), z)
            feats.append(f_next)
            greens.append(self.decode(f_next))
            zs.append(z.detach())
            h = self.advance(h, actions[:, i], speeds[:, i], e[:, i + 1].float() if tf else f_next)
        return {"green": torch.stack(greens, 1), "green_warm": green_warm,
                "feat": torch.stack(feats, 1), "feat_true": e[:, n:n + n_pred].float() if T_enc > n else None,
                "kl": torch.stack(kls, 1) if kls else None,
                "kl_anchor": torch.stack(anchors, 1) if anchors else None,
                "prior_std": torch.stack(stds, 1).detach(), "prior_std_map": torch.stack(std_maps, 1).detach(),
                "latent_stats": torch.stack(lat).mean(0) if lat else None, "z": torch.stack(zs, 0)}

    def loss(self, batch, n_pred, beta: float | None = None):
        """feat MSE + warm-up decoder MSE + pix_weight * predicted-frame MSE + beta * KL (+ anchor)."""
        cfg, n = self.cfg, self.cfg.n_cond
        imgs, act, spd = batch["images"], batch["actions"], batch["speed"]
        out = self(imgs, act, spd, n_pred, z_source="mix")
        tgt = imgs[:, n:n + n_pred, GREEN:GREEN + 1]
        mse_t = ((out["green"].float() - tgt) ** 2).mean((0, 2, 3, 4))
        feat = ((out["feat"] - out["feat_true"]) ** 2).mean()
        mse_warm = ((out["green_warm"].float() - imgs[:, :n, GREEN:GREEN + 1]) ** 2).mean()
        kl = out["kl"].mean() if out["kl"] is not None else torch.zeros((), device=feat.device)
        anchor = out["kl_anchor"].mean() if out["kl_anchor"] is not None else torch.zeros((), device=feat.device)
        loss = (feat + mse_warm + cfg.pix_weight * mse_t.mean() + (cfg.beta if beta is None else beta) * kl
                + cfg.anchor_prior * anchor)
        ls = out["latent_stats"]
        names = ("post_logvar", "prior_logvar", "mean_gap", "post_mu_abs", "prior_mu_abs", "post_mu_spread")
        lat = {k: (ls[i].item() if ls is not None else 0.0) for i, k in enumerate(names)}
        return loss, {"loss": loss.item(), "mse": mse_t.mean().item(), "mse_warm": mse_warm.item(),
                      "feat": feat.item(), "kl": kl.item(), "kl_anchor": anchor.item(),
                      "prior_std": out["prior_std"].mean().item()} | lat

    # ---- stage-2 style warm start: a deterministic DinoWM into a stochastic one ---------------------------------
    FRESH = ("prior.", "posterior.", "P_z.")

    def load_warm_start(self, state_dict, pz_init_scale: float = 0.0):
        own = self.state_dict()
        keep = {k: v for k, v in state_dict.items() if not k.startswith(self.FRESH) and k in own}
        self.load_state_dict(keep, strict=False)
        with torch.no_grad():
            self.P_z.weight.mul_(pz_init_scale)
            if "P_z.bias" in state_dict:
                self.P_z.bias.copy_(state_dict["P_z.bias"])
        return sorted(keep)

    @contextmanager
    def fixed_dropout_masks(self):
        from .world_model import StickyDropout2d
        mods = [m for m in self.modules() if isinstance(m, StickyDropout2d)]
        for m in mods:
            m.sticky, m.mask = True, None
        try:
            yield
        finally:
            for m in mods:
                m.sticky, m.mask = False, None
