"""PLS 모델 = B2 + PLS 잔차 보정 (BiGRU와 같은 '보정 단계'에 PLS를 넣은 모델).

  A_B2(λ) = A_in(λ) + Δh · b(λ)                         (models/linear.py 의 B2 그대로)
  Δh      = h* - h_in,   h_in = A_in(570)
  x(λ)    = A_in(λ) - A_in(570)                         570 nm를 뺀 30밴드
  u       = Δh · x                                      PLS 입력 (30)
  e       = A_target - A_B2   (570 nm 제외 30밴드)       PLS 학습 표적, Δh로 나누지 않음
  f       = 표준 PLS2 회귀 (X·Y 열별 평균 0, SD 1 표준화 -> 절편 있음, models/pls.py 의 nipals_pls2 재사용)
  최종     A*(λ) = A_B2(λ) + f(u) - f(0)                 Δh = 0 이면 추가 보정 = 0
           A*(570) = h*                                 570 nm에는 PLS 보정을 적용하지 않음

  - B2 기울기, 표준화 평균/SD, PLS 계수는 학습 피험자의 측정 쌍에서만 추정.
  - 성분 수: fit(n_components=None)이면 학습 피험자 안 피험자 단위 LOSO로 1..max_components 비교,
    570 nm 제외 흡광도 RMSE(내부 검증 쌍 전체 pooled)가 최소인 값. 안쪽 fold마다 B2도 다시 추정.
  - 학습 쌍 = 같은 피험자의 모든 순서쌍 i != j (make_pairs, 다른 모델과 같음). 피험자마다 측정 수가 같아 무가중.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.data_loader import FREE, HBI_INDEX, WL, Spectra, make_pairs
from models.base import finish, prepare
from models.linear import apply_slope, fit_slope
from models.pls import nipals_pls2

MAX_COMPONENTS = 5


def features(a, target):
    """u = Δh · (A_in - A_in(570)), 570 nm 제외 30밴드."""
    a = np.atleast_2d(a)
    dh = np.asarray(target, dtype=np.float64) - a[:, HBI_INDEX]
    return dh[:, None] * (a - a[:, [HBI_INDEX]])[:, FREE]


def _pairs_ue(data: Spectra, slope):
    Xa, Ya, _, I, J = make_pairs(data)
    T = data.hbi[J]
    return features(Xa, T), (Ya - apply_slope(Xa, T, slope))[:, FREE]


class _Fit:
    """표준화 + 성분 1..K 계수. correction(u, k) = f_k(u) - f_k(0)."""

    def __init__(self, U, E, K):
        self.um, self.us = U.mean(0), np.maximum(U.std(0), 1e-12)
        self.em, self.es = E.mean(0), np.maximum(E.std(0), 1e-12)
        self.coefs = nipals_pls2((U - self.um) / self.us, (E - self.em) / self.es, K)

    def f(self, U, k):
        return ((U - self.um) / self.us) @ self.coefs[k - 1] * self.es + self.em

    def correction(self, U, k):
        return self.f(U, k) - self.f(np.zeros((1, U.shape[1])), k)


class B2PLSModel:
    name = "PLS"

    def __init__(self, n_components: int | None = None, max_components: int = MAX_COMPONENTS):
        self.n_components, self.max_components = n_components, max_components
        self.slope = self.train_subjects = self.cv = self.rank = None
        self.um = self.us = self.em = self.es = self.coef = None

    def fit(self, train: Spectra) -> "B2PLSModel":
        self.train_subjects = train.subjects
        self.slope = fit_slope(train)
        U, E = _pairs_ue(train, self.slope)
        self.rank = int(np.linalg.matrix_rank((U - U.mean(0)) / np.maximum(U.std(0), 1e-12)))
        K = min(self.max_components, self.rank)
        if self.n_components is None:
            subs = train.subjects
            sse, cnt = np.zeros((len(subs), K)), np.zeros(len(subs))
            for n, s in enumerate(subs):
                tr_in, va = train.select([x for x in subs if x != s]), train.select([s])
                sl = fit_slope(tr_in)                                  # 안쪽 학습 피험자로 B2 다시 추정
                f = _Fit(*_pairs_ue(tr_in, sl), K)
                Uv, Ev = _pairs_ue(va, sl)
                for k in range(1, K + 1):
                    sse[n, k - 1] = np.sum((f.correction(Uv, k) - Ev) ** 2)   # = (A* - A_target)², 570 제외
                cnt[n] = Ev.size
            rmse = np.sqrt(sse.sum(0) / cnt.sum())
            self.n_components = int(np.argmin(rmse) + 1)
            self.cv = dict(method="inner leave-one-subject-out over training subjects (B2 refit per inner fold)",
                           criterion="pooled absorbance RMSE excluding 570 nm",
                           components=list(range(1, K + 1)), rmse=rmse.tolist(),
                           per_subject_rmse={s: np.sqrt(sse[n] / cnt[n]).tolist() for n, s in enumerate(subs)},
                           chosen=self.n_components)
        f = _Fit(U, E, self.n_components)
        self.um, self.us, self.em, self.es, self.coef = f.um, f.us, f.em, f.es, f.coefs[-1]
        return self

    def _correction(self, a, target):
        U = features(a, target)
        z = lambda V: ((V - self.um) / self.us) @ self.coef * self.es + self.em   # noqa: E731
        return z(U) - z(np.zeros((1, U.shape[1])))

    def predict(self, a, target) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("fit()을 먼저 호출해야 함")
        a, target, single = prepare(a, target)
        out = apply_slope(a, target, self.slope)          # B2, 570 nm = h*
        out[:, FREE] += self._correction(a, target)       # 570 nm에는 보정 없음
        return finish(out, single)

    def save(self, path: Path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(model=self.name, wavelengths_nm=WL.tolist(), n_components=self.n_components,
                                        rank_u=self.rank, train_subjects=self.train_subjects, b2_slope=self.slope.tolist(),
                                        u_mean=self.um.tolist(), u_std=self.us.tolist(), e_mean=self.em.tolist(),
                                        e_std=self.es.tolist(), coef=self.coef.tolist(), inner_cv=self.cv),
                                   ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "B2PLSModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        m = cls(n_components=d["n_components"])
        m.train_subjects, m.cv, m.rank = d["train_subjects"], d["inner_cv"], d["rank_u"]
        m.slope = np.array(d["b2_slope"])
        m.um, m.us, m.em, m.es, m.coef = (np.array(d[k]) for k in ("u_mean", "u_std", "e_mean", "e_std", "coef"))
        return m
