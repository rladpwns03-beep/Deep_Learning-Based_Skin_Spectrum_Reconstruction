"""세 모델(B1, B2, BiGRU)이 공유하는 입력·출력 처리.

공통 인터페이스
  model.fit(train: Spectra) -> model           학습 데이터로만 파라미터 추정
  model.predict(a, target) -> 복원 흡광도       a: (31,) 또는 (N, 31), target: 스칼라 또는 (N,)
                                                 출력 shape = 입력 shape
  model.save(path) / Model.load(path)           파라미터 저장·불러오기

공통 조건 (predict에서 항상 보장·검사)
  - 출력 570 nm 값 = 목표 HbI
  - 목표 HbI = 입력 HbI 이면 입력 스펙트럼을 그대로 반환 (Identity)
  - 입력·목표·출력에 NaN/Inf가 있으면 오류
"""
from __future__ import annotations

import numpy as np

from common.data_loader import WL

N_WL = len(WL)


def prepare(a, target):
    """(31,) 또는 (N, 31) 입력을 (N, 31) float64로, target을 (N,)으로 맞춘다."""
    a = np.ascontiguousarray(a, dtype=np.float64)   # a[::-1] 같은 음수 stride 배열도 torch로 넘길 수 있게
    single = a.ndim == 1
    if single:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != N_WL:
        raise ValueError(f"입력 shape는 ({N_WL},) 또는 (N, {N_WL})이어야 함: {a.shape}")
    target = np.asarray(target, dtype=np.float64)
    if target.ndim > 1 or (target.ndim == 1 and len(target) not in (1, len(a))):
        raise ValueError(f"target shape가 입력 개수와 맞지 않음: {target.shape} vs N={len(a)}")
    target = np.broadcast_to(target.reshape(-1) if target.ndim else target, (len(a),)).copy()
    if not np.isfinite(a).all():
        raise ValueError("입력 스펙트럼에 NaN/Inf가 있음")
    if not np.isfinite(target).all():
        raise ValueError("목표 HbI에 NaN/Inf가 있음")
    return a, target, single


def finish(out: np.ndarray, single: bool) -> np.ndarray:
    if not np.isfinite(out).all():
        raise FloatingPointError("출력에 NaN/Inf가 있음")
    return out[0] if single else out
