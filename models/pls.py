"""PLS 회귀 기반 스펙트럼 변화 예측 (v3).

  입력 X = [입력 흡광도 A_in (31), 입력 HbI h_in, 목표 HbI h*]   (33개 특징)
  출력 Y = A_target - A_in (31)                                 (스펙트럼 변화량)
  복원   A*(λ) = A_in(λ) + Ŷ(λ),  A*(570) = h*  (다른 모델과 같이 570 nm는 목표값으로 고정)

  - PLS2 NIPALS (numpy 구현, sklearn PLSRegression(scale=True)과 같은 정의: X·Y 열별 평균 0, 표준편차 1로 표준화)
  - 성분 수는 학습 피험자 안에서만 고른다 (fit(..., n_components=None) -> 내부 LOSO, 1..max_components).
  - 학습 쌍 = 같은 피험자의 모든 순서쌍 i != j (P1 All Pairs, 다른 모델과 같음).
  - 알려진 차이: 선형 모델에 절편이 있으므로 h* = h_in 이어도 변화량이 정확히 0이라는 보장이 없다
    (B1/B2/BiGRU는 구조적으로 입력을 그대로 반환). identity_error()로 크기를 기록한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.data_loader import FREE, HBI_INDEX, Spectra, make_pairs
from models.base import finish, prepare

MAX_COMPONENTS = 10


def nipals_pls2(X, Y, n_components, tol=1e-12, max_iter=1000):
    """표준화된 X (n,p), Y (n,q) -> 성분 1..K 각각의 회귀계수 B_k (p,q) 리스트 (성분이 누적되므로 한 번에 계산)."""
    X, Y = X.copy(), Y.copy()
    p, q = X.shape[1], Y.shape[1]
    W, P, Q = np.zeros((p, n_components)), np.zeros((p, n_components)), np.zeros((q, n_components))
    coefs = []
    for k in range(n_components):
        u = Y[:, np.argmax((Y ** 2).sum(0))].copy()
        w_old = None
        for _ in range(max_iter):
            w = X.T @ u
            w /= np.linalg.norm(w)
            t = X @ w
            c = Y.T @ t / (t @ t)
            u = Y @ c / (c @ c)
            if w_old is not None and np.linalg.norm(w - w_old) < tol:
                break
            w_old = w
        t = X @ w
        pk = X.T @ t / (t @ t)
        qk = Y.T @ t / (t @ t)
        X -= np.outer(t, pk)
        Y -= np.outer(t, qk)
        W[:, k], P[:, k], Q[:, k] = w, pk, qk
        Wk, Pk, Qk = W[:, :k + 1], P[:, :k + 1], Q[:, :k + 1]
        coefs.append(Wk @ np.linalg.pinv(Pk.T @ Wk) @ Qk.T)
    return coefs


def features(a, h_in, target):
    return np.column_stack([a, h_in, target])


def _pairs_xy(data: Spectra):
    Xa, Ya, _, I, J = make_pairs(data)
    return features(Xa, data.hbi[I], data.hbi[J]), Ya - Xa


class _Fitted:
    """표준화 + 성분 1..K 계수."""

    def __init__(self, X, Y, K):
        self.xm, self.xs = X.mean(0), np.maximum(X.std(0), 1e-12)
        self.ym, self.ys = Y.mean(0), np.maximum(Y.std(0), 1e-12)
        self.coefs = nipals_pls2((X - self.xm) / self.xs, (Y - self.ym) / self.ys, K)

    def predict(self, X, k):
        return ((X - self.xm) / self.xs) @ self.coefs[k - 1] * self.ys + self.ym


class PLSModel:
    name = "PLS"

    def __init__(self, n_components: int | None = None, max_components: int = MAX_COMPONENTS):
        self.n_components, self.max_components = n_components, max_components
        self.train_subjects, self.cv = None, None
        self.xm = self.xs = self.ym = self.ys = self.coef = None

    def fit(self, train: Spectra) -> "PLSModel":
        self.train_subjects = train.subjects
        subs = train.subjects
        K = min(self.max_components, 33)
        if self.n_components is None:
            # 내부 LOSO: 학습 피험자 1명씩 빼고 나머지로 학습, 성분 수별 MSE (570 nm 제외, 피험자 평균)
            mse = np.zeros((len(subs), K))
            for n, s in enumerate(subs):
                inner_tr = train.select([x for x in subs if x != s])
                inner_te = train.select([s])
                f = _Fitted(*_pairs_xy(inner_tr), K)
                Xv, Yv = _pairs_xy(inner_te)
                for k in range(1, K + 1):
                    mse[n, k - 1] = np.mean((f.predict(Xv, k) - Yv)[:, FREE] ** 2)
            curve = mse.mean(0)
            self.n_components = int(np.argmin(curve) + 1)
            self.cv = dict(method="inner LOSO over training subjects", components=list(range(1, K + 1)),
                           mean_mse=curve.tolist(), per_subject_mse={s: mse[n].tolist() for n, s in enumerate(subs)},
                           chosen=self.n_components)
        f = _Fitted(*_pairs_xy(train), self.n_components)
        self.xm, self.xs, self.ym, self.ys = f.xm, f.xs, f.ym, f.ys
        self.coef = f.coefs[-1]
        return self

    def _delta(self, a, target):
        X = features(a, a[:, HBI_INDEX], target)
        return ((X - self.xm) / self.xs) @ self.coef * self.ys + self.ym

    def predict(self, a, target) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("fit()을 먼저 호출해야 함")
        a, target, single = prepare(a, target)
        out = a + self._delta(a, target)
        out[:, HBI_INDEX] = target
        return finish(out, single)

    def identity_error(self, a) -> float:
        """h* = h_in 일 때 출력과 입력의 최대 절대 차이 (570 nm 제외). 0이 아니면 Identity 계약 위반 크기."""
        a = np.atleast_2d(np.asarray(a, dtype=np.float64))
        return float(np.abs(self._delta(a, a[:, HBI_INDEX])[:, FREE]).max())

    def save(self, path: Path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(model="PLS", n_components=self.n_components, train_subjects=self.train_subjects,
                                        x_mean=self.xm.tolist(), x_std=self.xs.tolist(), y_mean=self.ym.tolist(),
                                        y_std=self.ys.tolist(), coef=self.coef.tolist(), inner_cv=self.cv),
                                   ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PLSModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        m = cls(n_components=d["n_components"])
        m.train_subjects, m.cv = d["train_subjects"], d["inner_cv"]
        m.xm, m.xs, m.ym, m.ys, m.coef = (np.array(d[k]) for k in ("x_mean", "x_std", "y_mean", "y_std", "coef"))
        return m
