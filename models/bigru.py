"""제안 모델: B2(선형 기울기) + BiGRU 잔차 기울기.

Model_Ablation/ablation.py 의 Residual(cell="GRU", hidden=8, bidirectional=True), objective(), predict() 와
같은 구조·손실·학습 설정이다. 달라진 점은 하나뿐이다.
  - 검증 세트가 없으므로 epoch 수를 고르지 않고 EPOCHS로 고정한다 (학습 전에 고정, 테스트 결과로 바꾸지 않음).

구조
  입력 특징 (파장마다 4개): [표준화한 A_λ, 파장 위치(-1~1), 입력 HbI, 목표 HbI]  -> [N, 31, 4]
  1층 양방향 GRU (방향별 hidden 8) -> [N, 31, 16] -> 공유 선형층 -> 파장별 잔차 기울기 r(λ)
  보정: A*(λ) = A(λ) + (h* - h) · (b_B2(λ) + r(λ)),   r(570) = 0,  A*(570) = h*
  선형층을 0으로 초기화하므로 학습 시작 시점의 모델은 B2와 같다.

손실 (같은 피험자 안의 측정 쌍 i -> j, 피험자별 가중치 동일)
  MSE_λ≠570(A*, A_j) + SLOPE_L2 · mean(r²)
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common.config import BIGRU, BIGRU_SEEDS
from common.data_loader import FREE, HBI_INDEX, WL, Spectra, make_pairs
from models.base import finish, prepare
from models.linear import fit_slope

# 학습 설정은 common/config.py (Model_Ablation/ablation.py 의 CFG와 같은 값; epoch 150 고정, 시드 0/1/2)
CFG = dict(BIGRU, seeds=BIGRU_SEEDS)


class ResidualBiGRU(nn.Module):
    def __init__(self, mean, std, b2, hidden=CFG["hidden"]):
        super().__init__()
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
        x = torch.stack(((a - self.mean) / self.std, self.pos.expand_as(a),
                         h[:, None].expand_as(a), target[:, None].expand_as(a)), dim=-1)
        r = self.head(self.gru(x)[0]).squeeze(-1) * self.free          # 570 nm 잔차 = 0
        out = a + (target - h)[:, None] * (self.b2 + r)
        out = torch.where(self.free[None, :], out, target[:, None])
        return out, r


def _seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _objective(model, batch, slope_l2):
    a, y, w = batch
    out, r = model(a, y[:, HBI_INDEX])
    mse = (w * ((out[:, model.free] - y[:, model.free]) ** 2).mean(1)).sum()
    pen = (w * r[:, model.free].square().mean(1)).sum()
    return mse + slope_l2 * pen, mse


class BiGRUModel:
    name = "BiGRU"

    def __init__(self, seed=CFG["seeds"][0], epochs=CFG["epochs"], cfg=CFG):
        self.seed, self.epochs, self.cfg = seed, epochs, cfg
        self.net, self.b2, self.history = None, None, []
        self.train_subjects = None

    def fit(self, train: Spectra, pairs=None) -> "BiGRUModel":
        """pairs: None이면 make_pairs(train) (P1 All Pairs, 기존 방식). Pair 전략 실험에서는 (X, Y, W)를 직접 전달."""
        _seed_everything(self.seed)
        # 표준화 평균/표준편차와 B2 기울기는 학습 피험자의 측정에서만 계산 (Pair 구성과 무관)
        self.b2 = fit_slope(train)
        self.train_subjects = train.subjects
        mean, std = train.a.mean(0), np.maximum(train.a.std(0), 1e-6)
        self.net = ResidualBiGRU(mean, std, self.b2, self.cfg["hidden"])
        X, Y, W = make_pairs(train)[:3] if pairs is None else pairs
        batch = tuple(torch.tensor(v, dtype=torch.float32) for v in (X, Y, W))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg["lr"])
        self.history = []
        for ep in range(1, self.epochs + 1):                 # full batch
            self.net.train(); opt.zero_grad(set_to_none=True)
            loss, mse = _objective(self.net, batch, self.cfg["slope_l2"])
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg["clip"])   # clip 전 전체 norm
            opt.step()
            self.history.append((ep, loss.item(), mse.item(), grad_norm.item()))
        return self

    @torch.no_grad()
    def residual(self, a, target) -> np.ndarray:
        self.net.eval()
        a, target, single = prepare(a, target)
        _, r = self.net(torch.tensor(a, dtype=torch.float32), torch.tensor(target, dtype=torch.float32))
        return finish(r.numpy().astype(np.float64), single)

    def predict(self, a, target) -> np.ndarray:
        """float64로 최종 조립 (원 흡광도 정밀도 보존, Δh = 0이면 입력 그대로)."""
        if self.net is None:
            raise RuntimeError("fit()을 먼저 호출해야 함")
        a, target, single = prepare(a, target)
        out = a + (target - a[:, HBI_INDEX])[:, None] * (self.b2 + self.residual(a, target))
        out[:, HBI_INDEX] = target
        return finish(out, single)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def save(self, path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=self.name, wavelengths_nm=WL.tolist(), seed=self.seed, epochs=self.epochs,
                        cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in self.cfg.items()},
                        b2=self.b2.tolist(), mean=self.net.mean.tolist(), std=self.net.std.tolist(),
                        train_subjects=self.train_subjects, state_dict=self.net.state_dict()), path)

    @classmethod
    def load(cls, path) -> "BiGRUModel":
        d = torch.load(path, map_location="cpu", weights_only=True)
        if d["model"] != cls.name or d["wavelengths_nm"] != WL.tolist():
            raise ValueError("BiGRU 체크포인트가 아니거나 파장이 다름")
        m = cls(seed=d["seed"], epochs=d["epochs"], cfg=d["cfg"])
        m.b2 = np.asarray(d["b2"], dtype=np.float64)
        m.net = ResidualBiGRU(d["mean"], d["std"], m.b2, d["cfg"]["hidden"])
        m.net.load_state_dict(d["state_dict"], strict=True)
        m.net.eval()
        m.train_subjects = d["train_subjects"]
        return m
