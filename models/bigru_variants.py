"""BiGRU structure ablation (post hoc, exploratory; not preregistered). models/bigru.py is not modified.

All four variants share one encoder: per wavelength 4 inputs [standardised A, wavelength position, input HbI, target HbI],
one bidirectional GRU layer (hidden 8 per direction), shared Linear(16, 1) initialised to zero. 689 parameters each.
Only the way the linear-layer output o(lambda) is assembled into a spectrum differs:

  BiGRU_ResidualSlope       A* = A_in + (h* - h_in) (b + o)      (1)  current structure
  BiGRU_FullSlope           A* = A_in + (h* - h_in) o            (2)  without the Global Linear prior
  BiGRU_AdditiveCorrection  A* = A_in + o                        (3)  also without the proportionality to the HbI change
  BiGRU_DirectOutput        A* = mu + sigma o                    (4)  also without the input pass-through

b = Global Linear slope of the training subjects (models.linear.fit_slope); mu, sigma = wavelength-wise mean and SD of the
training measurements (the same values that standardise the input). In every variant A*(570) = h* in training and prediction,
and the loss uses the 30 wavelengths other than 570 nm (subject-weighted MSE, no slope penalty).
Because the linear layer starts at zero, the epoch-0 prediction is: (1) Global Linear, (2) and (3) the input spectrum,
(4) the training mean spectrum (570 nm = h* in all cases).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from common.data_loader import FREE, HBI_INDEX, WL, Spectra, make_pairs
from models.base import finish, prepare
from models.bigru import _seed_everything
from models.linear import fit_slope

VARIANTS = {"BiGRU_ResidualSlope": "residual_slope", "BiGRU_FullSlope": "full_slope",
            "BiGRU_AdditiveCorrection": "additive_correction", "BiGRU_DirectOutput": "direct_output"}
REMOVED = {"BiGRU_ResidualSlope": "nothing (current structure)", "BiGRU_FullSlope": "Global Linear prior",
           "BiGRU_AdditiveCorrection": "Global Linear prior, proportionality to the HbI change",
           "BiGRU_DirectOutput": "Global Linear prior, proportionality to the HbI change, input pass-through"}
STRUCTURAL_IDENTITY = {"BiGRU_ResidualSlope": True, "BiGRU_FullSlope": True, "BiGRU_AdditiveCorrection": False, "BiGRU_DirectOutput": False}
HIDDEN, CLIP = 8, 1.0


class VariantNet(nn.Module):
    """Same construction order as models.bigru.ResidualBiGRU (buffers, GRU, zero-initialised head), so the same seed gives the same GRU weights."""

    def __init__(self, mean, std, b2, mode: str, hidden: int = HIDDEN):
        super().__init__()
        self.mode = mode
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.register_buffer("b2", torch.tensor(b2, dtype=torch.float32))
        self.register_buffer("pos", torch.linspace(-1, 1, len(WL)))
        self.register_buffer("free", torch.tensor(FREE))
        self.gru = nn.GRU(4, hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.head = nn.Linear(2 * hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, a, target):
        h = a[:, HBI_INDEX]
        target = torch.as_tensor(target, dtype=a.dtype).expand(len(a))
        x = torch.stack(((a - self.mean) / self.std, self.pos.expand_as(a), h[:, None].expand_as(a), target[:, None].expand_as(a)), dim=-1)
        o = self.head(self.gru(x)[0]).squeeze(-1) * self.free          # linear-layer output, 0 at 570 nm
        dh = (target - h)[:, None]
        if self.mode == "residual_slope":
            out = a + dh * (self.b2 + o)
        elif self.mode == "full_slope":
            out = a + dh * o
        elif self.mode == "additive_correction":
            out = a + o
        elif self.mode == "direct_output":
            out = self.mean + self.std * o
        else:
            raise ValueError(self.mode)
        return torch.where(self.free[None, :], out, target[:, None]), o


class BiGRUVariant:
    def __init__(self, name: str, seed: int = 0, lr: float = 3e-3, hidden: int = HIDDEN, clip: float = CLIP):
        self.name, self.mode, self.seed, self.lr, self.hidden, self.clip = name, VARIANTS[name], seed, lr, hidden, clip
        self.net = self.b2 = self.mean = self.std = self.train_subjects = None
        self.history, self.epochs_done = [], 0

    def fit(self, train: Spectra, epochs: int, checkpoints=(), on_checkpoint=None) -> dict:
        """Full batch, Adam, gradient clipping; same order of operations as models.bigru.BiGRUModel.fit.
        Returns {epoch: on_checkpoint(self)} for the requested checkpoints (epoch 0 = before any update)."""
        _seed_everything(self.seed)
        self.b2 = fit_slope(train)
        self.train_subjects = train.subjects
        self.mean, self.std = train.a.mean(0), np.maximum(train.a.std(0), 1e-6)
        self.net = VariantNet(self.mean, self.std, self.b2, self.mode, self.hidden)
        X, Y, W = make_pairs(train)[:3]
        a, y, w = (torch.tensor(v, dtype=torch.float32) for v in (X, Y, W))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        cps, out = set(checkpoints), {}
        self.history, self.epochs_done = [], 0
        if 0 in cps and on_checkpoint is not None:
            out[0] = on_checkpoint(self)
        for ep in range(1, epochs + 1):
            self.net.train(); opt.zero_grad(set_to_none=True)
            pred, _ = self.net(a, y[:, HBI_INDEX])
            loss = (w * ((pred[:, self.net.free] - y[:, self.net.free]) ** 2).mean(1)).sum()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
            opt.step()
            self.history.append((ep, loss.item()))
            self.epochs_done = ep
            if ep in cps and on_checkpoint is not None:
                out[ep] = on_checkpoint(self)
        return out

    @torch.no_grad()
    def raw_output(self, a, target) -> np.ndarray:
        self.net.eval()
        _, o = self.net(torch.tensor(a, dtype=torch.float32), torch.tensor(target, dtype=torch.float32))
        return o.numpy().astype(np.float64)

    def predict(self, a, target) -> np.ndarray:
        """Final assembly in float64 (as in BiGRUModel.predict). 570 nm is set to the target HbI."""
        if self.net is None:
            raise RuntimeError("fit() first")
        a, target, single = prepare(a, target)
        o = self.raw_output(a, target)
        dh = (target - a[:, HBI_INDEX])[:, None]
        if self.mode == "residual_slope":
            out = a + dh * (self.b2 + o)
        elif self.mode == "full_slope":
            out = a + dh * o
        elif self.mode == "additive_correction":
            out = a + o
        else:
            out = self.mean[None, :] + self.std[None, :] * o
        out[:, HBI_INDEX] = target
        return finish(out, single)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())
