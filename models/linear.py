"""비교군: HbI–흡광도 선형관계 기반 보정 (B2).

Model_Ablation/ablation.py 의 fit_pre()/b1_pred() 와 같은 수식이다.

  학습: 학습 피험자마다 A와 HbI에서 그 피험자의 평균을 뺀 뒤(피험자 간 피부색 차이 제거),
        모든 학습 피험자를 모아 파장별로 원점을 지나는 최소제곱 회귀
            b(λ) = Σ (h - h̄_s)(A_λ - Ā_s,λ) / Σ (h - h̄_s)²
        570 nm는 b = 1 로 고정 (보정 후 HbI가 목표값과 정확히 같아지도록)
  보정: A*(λ) = A(λ) + (h* - h) · b(λ),   A*(570) = h*

고를 하이퍼파라미터가 없으므로 검증 세트가 필요 없다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.data_loader import HBI_INDEX, WL, Spectra
from models.base import finish, prepare


def fit_slope(train: Spectra) -> np.ndarray:
    """피험자 내 중심화 후 모은(pooled within-subject) 파장별 기울기 b(λ)."""
    ac, hc = train.a.copy(), train.hbi.copy()
    for s in train.subjects:
        m = train.subject == s
        ac[m] -= ac[m].mean(axis=0)
        hc[m] -= hc[m].mean()
    denom = hc @ hc
    if denom < 1e-12:
        raise ValueError("학습 피험자 안에서 HbI 변화가 없어 기울기를 추정할 수 없음")
    slope = hc @ ac / denom
    slope[HBI_INDEX] = 1.0
    return slope


def apply_slope(a: np.ndarray, target, slope: np.ndarray) -> np.ndarray:
    """A*(λ) = A(λ) + (h* - h) · b(λ). target은 스칼라(예: 0.6) 또는 측정별 배열."""
    a = np.asarray(a, dtype=np.float64)
    target = np.broadcast_to(np.asarray(target, dtype=np.float64), (len(a),))
    out = a + (target - a[:, HBI_INDEX])[:, None] * slope
    out[:, HBI_INDEX] = target           # 부동소수점 오차 없이 목표 HbI를 정확히 지정
    return out


class LinearModel:
    name = "B2"

    def __init__(self):
        self.slope = None
        self.train_subjects = None

    def fit(self, train: Spectra) -> "LinearModel":
        self.slope = fit_slope(train)
        self.train_subjects = train.subjects
        return self

    def predict(self, a: np.ndarray, target) -> np.ndarray:
        if self.slope is None:
            raise RuntimeError("fit()을 먼저 호출해야 함")
        a, target, single = prepare(a, target)
        out = apply_slope(a, target, self.slope)
        out[target == a[:, HBI_INDEX]] = a[target == a[:, HBI_INDEX]]   # Identity (이미 성립, 명시적 보장)
        return finish(out, single)

    def save(self, path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(model=self.name, wavelengths_nm=WL.tolist(), slope=self.slope.tolist(),
                                        train_subjects=self.train_subjects), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "LinearModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d["model"] != cls.name or d["wavelengths_nm"] != WL.tolist():
            raise ValueError("B2 파라미터 파일이 아니거나 파장이 다름")
        m = cls(); m.slope = np.asarray(d["slope"], dtype=np.float64); m.train_subjects = d["train_subjects"]
        return m
