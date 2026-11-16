"""B2 + Ridge 잔차 보정 모델 (B2 + PLS와 같은 입력·표적·출력 제약, 회귀만 Ridge).

  u = Δh · (A_in - A_in(570))   (570 제외 30밴드),   Δh = h* - A_in(570)
  e = A_target - A_B2           (570 제외 30밴드)
  입력 스케일링: u를 학습 쌍의 열별 평균·SD로 표준화 (B2+PLS와 같은 방식, 학습 피험자로만 추정)
  다중출력 Ridge (30개 출력 동시):  B = (Z'Z + alpha·I)^-1 Z'(E - Ē)   (Z = 표준화 u, 절편 = Ē - 0, 절편은 벌점 없음)
    - 벌점 정의는 sklearn Ridge와 같음: ||E - ZB - c||² (제곱합) + alpha·||B||²
    - Ridge는 출력 열마다 독립이고 Y 스케일에 대해 등변이므로 출력 표준화 여부는 예측에 영향 없음 (여기선 하지 않음)
  최종: A* = A_B2 + f(u) - f(0),  A*(570) = h*  -> Δh = 0 이면 추가 보정 0

  alpha 선택: 학습 피험자 안 피험자 단위 LOSO (안쪽 fold마다 B2 다시 추정), 570 제외 흡광도 RMSE(pooled) 최소.
  격자 ALPHAS. 최솟값이 격자 끝이면 그 방향으로 10배씩 넓혀 (최대 EXTEND_MAX번) 내부 검증에서만 다시 확인.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.data_loader import FREE, WL, Spectra
from models.b2_pls import _pairs_ue, features
from models.base import finish, prepare
from models.linear import apply_slope, fit_slope

ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
EXTEND_MAX = 6


class _Fit:
    def __init__(self, U, E):
        self.um, self.us = U.mean(0), np.maximum(U.std(0), 1e-12)
        self.em = E.mean(0)
        Z = (U - self.um) / self.us
        self.ZtZ, self.ZtE = Z.T @ Z, Z.T @ (E - self.em)
        self._cache = {}

    def coef(self, alpha):
        if alpha not in self._cache:
            self._cache[alpha] = np.linalg.solve(self.ZtZ + alpha * np.eye(len(self.ZtZ)), self.ZtE)
        return self._cache[alpha]

    def correction(self, U, alpha):                      # f(u) - f(0) = (u / s) · B
        return (U / self.us) @ self.coef(alpha)


class B2RidgeModel:
    name = "Ridge"

    def __init__(self, alpha: float | None = None, alphas=ALPHAS):
        self.alpha, self.alphas = alpha, list(alphas)
        self.slope = self.train_subjects = self.cv = None
        self.us = self.coef = None

    def _inner_rmse(self, folds, alphas):
        sse = np.zeros(len(alphas)); cnt = 0
        for f, Uv, Ev in folds:
            for n, a in enumerate(alphas):
                sse[n] += np.sum((f.correction(Uv, a) - Ev) ** 2)
            cnt += Ev.size
        return np.sqrt(sse / cnt)

    def fit(self, train: Spectra) -> "B2RidgeModel":
        self.train_subjects = train.subjects
        self.slope = fit_slope(train)
        if self.alpha is None:
            subs, folds = train.subjects, []
            for s in subs:
                tr_in, va = train.select([x for x in subs if x != s]), train.select([s])
                sl = fit_slope(tr_in)
                folds.append((_Fit(*_pairs_ue(tr_in, sl)), *_pairs_ue(va, sl)))
            grid = sorted(self.alphas)
            rmse = dict(zip(grid, self._inner_rmse(folds, grid)))
            extended = []
            for _ in range(EXTEND_MAX):                      # 최솟값이 끝이면 그 방향으로 확장
                g = sorted(rmse); best = min(rmse, key=rmse.get)
                if best == g[0]:
                    new = g[0] / 10
                elif best == g[-1]:
                    new = g[-1] * 10
                else:
                    break
                rmse[new] = float(self._inner_rmse(folds, [new])[0]); extended.append(new)
            g = sorted(rmse); best = min(rmse, key=rmse.get)
            self.alpha = float(best)
            self.cv = dict(method="inner leave-one-subject-out over training subjects (B2 refit per inner fold)",
                           criterion="pooled absorbance RMSE excluding 570 nm", alphas=g, rmse=[float(rmse[a]) for a in g],
                           base_grid=sorted(self.alphas), extended=extended,
                           at_boundary_after_extension=bool(best in (g[0], g[-1])), chosen=self.alpha)
        f = _Fit(*_pairs_ue(train, self.slope))
        self.us, self.coef = f.us, f.coef(self.alpha)
        return self

    def predict(self, a, target) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("fit()을 먼저 호출해야 함")
        a, target, single = prepare(a, target)
        out = apply_slope(a, target, self.slope)               # B2, 570 nm = h*
        out[:, FREE] += (features(a, target) / self.us) @ self.coef
        return finish(out, single)

    def save(self, path: Path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(model=self.name, wavelengths_nm=WL.tolist(), alpha=self.alpha,
                                        train_subjects=self.train_subjects, b2_slope=self.slope.tolist(),
                                        u_std=self.us.tolist(), coef=self.coef.tolist(), inner_cv=self.cv),
                                   ensure_ascii=False), encoding="utf-8")
