"""평가지표 (세 모델 공통). 정의는 Model_Ablation/ablation.py 의 run() 과 같다.

  흡광도 오차      e_A(λ) = A_pred(λ) - A_true(λ)                         [absorbance]
  반사율           R(λ) = 100 · 10^(-A(λ))                                 [%]  (A = -log10(R/100)의 역변환)
  반사율 오차      e_R(λ) = R_pred(λ) - R_true(λ)                          [%p]
  카로티노이드 Peak  P = A(490) - (A(470) + A(510)) / 2                      [absorbance]  (Hwang et al. 2023 식 (1))
  Peak 오차        e_P = P_pred - P_true   (Bias = 평균 e_P, 양수 = 과대예측)

RMSE 계산 단위 (서로 다를 수 있으므로 구분)
  - group RMSE  : 여러 쌍의 모든 (쌍, 파장) 원소를 모아 sqrt(mean(e²))  -> 피험자별 지표의 기본 정의
  - pair RMSE   : 쌍 하나의 30개 파장에 대해 sqrt(mean(e²))              -> 쌍별 기록용
흡광도·반사율 RMSE는 기본적으로 570 nm를 제외한 30개 파장(FREE)으로 계산한다
(570 nm는 세 모델 모두 목표 HbI로 고정되어 구조적으로 오차가 0).
"""
from __future__ import annotations

import numpy as np

from common.data_loader import FREE, WL

I470, I490, I510 = (int(np.flatnonzero(WL == w)[0]) for w in (470, 490, 510))


def reflectance_pct(a):
    return 100.0 * np.power(10.0, -np.asarray(a, dtype=np.float64))


def peak(a):
    a = np.asarray(a, dtype=np.float64)
    return a[..., I490] - (a[..., I470] + a[..., I510]) / 2


def group_metrics(pred, true) -> dict:
    """여러 쌍을 모은 지표 (group RMSE)."""
    pred, true = np.atleast_2d(pred), np.atleast_2d(true)
    ea = (pred - true)[:, FREE]
    er = (reflectance_pct(pred) - reflectance_pct(true))[:, FREE]
    ep = peak(pred) - peak(true)
    return dict(n_pairs=int(len(pred)),
                a_rmse=float(np.sqrt(np.mean(ea ** 2))), a_mae=float(np.mean(np.abs(ea))),
                r_rmse_pp=float(np.sqrt(np.mean(er ** 2))),
                peak_rmse=float(np.sqrt(np.mean(ep ** 2))), peak_mae=float(np.mean(np.abs(ep))),
                peak_bias=float(np.mean(ep)))


def pair_metrics(pred, true) -> dict:
    """쌍별 지표 (각 쌍의 30개 파장 RMSE, Peak 오차)."""
    ea = (pred - true)[:, FREE]
    er = (reflectance_pct(pred) - reflectance_pct(true))[:, FREE]
    return dict(a_rmse=np.sqrt(np.mean(ea ** 2, axis=1)), r_rmse_pp=np.sqrt(np.mean(er ** 2, axis=1)),
                peak_err=peak(pred) - peak(true))


def wavelength_metrics(pred, true) -> dict:
    """파장별 흡광도 RMSE / MAE / Bias (31개 파장, 570 nm 포함 - 값은 0)."""
    e = np.atleast_2d(pred) - np.atleast_2d(true)
    return dict(rmse=np.sqrt(np.mean(e ** 2, axis=0)).tolist(), mae=np.mean(np.abs(e), axis=0).tolist(),
                bias=np.mean(e, axis=0).tolist())


def consistency(pred) -> dict:
    """같은 피험자의 반복 측정을 같은 목표로 복원한 결과의 개인 내 변동 (ddof=1)."""
    pred = np.atleast_2d(pred)
    return dict(n=int(len(pred)), peak_sd=float(np.std(peak(pred), ddof=1)),
                spectral_sd=float(np.mean(np.std(pred[:, FREE], axis=0, ddof=1))),
                peak_mean=float(np.mean(peak(pred))))
