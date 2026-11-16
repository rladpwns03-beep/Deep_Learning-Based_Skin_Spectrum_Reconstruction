"""비교군 B1: HbI-Binned Mean Correction.

  A_pred(λ) = A_input(λ) + MeanSpectrum(target_bin, λ) - MeanSpectrum(input_bin, λ)

설정 (common/config.py)
  - bin 폭 0.05, 경계 0.40, 0.45, ..., 0.75 고정 (모든 분할·실행에서 동일)
  - bin = [왼쪽 경계, 오른쪽 경계). 경계 밖 HbI는 가장 가까운 끝 bin으로 넣고 그 횟수를 기록
  - 각 bin의 평균 스펙트럼은 Train 데이터만으로 계산
  - 빈 bin은 가장 가까운 유효 bin(bin 중심 거리 기준, 같으면 낮은 쪽)의 평균으로 대체
  - bin별 측정 수·고유 피험자 수(bin_table), fallback 사용 횟수(fallback_uses) 기록
  - 출력의 570 nm 값은 목표 HbI로 고정
  - 입력 HbI와 목표 HbI가 같으면 원본 스펙트럼을 그대로 반환

한계 (결과 보고 시 함께 기록)
  1. Bin 경계 불연속성: 보정량이 bin 단위 계단 함수다. 같은 bin 안에서는 HbI가 달라도 보정량이 0이고
     (570 nm만 목표값으로 바뀜), 경계를 조금만 넘어도 보정량이 bin 평균 차이만큼 갑자기 바뀐다.
  2. 피험자별 피부색 차이에 의한 편향: bin 평균은 서로 다른 피험자의 스펙트럼을 그대로 평균한다.
     bin마다 들어간 피험자 구성이 다르면 두 bin 평균의 차이에 HbI 효과뿐 아니라 피부색(멜라닌 등)
     차이가 섞인다. 특히 측정 수·피험자 수가 적은 bin(높은 HbI 구간)에서 크게 작용할 수 있다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import B1_BIN_START, B1_BIN_STOP, B1_BIN_WIDTH
from common.data_loader import HBI_INDEX, WL, Spectra
from models.base import finish, prepare

BIN_EDGES = np.round(np.arange(B1_BIN_START, B1_BIN_STOP + B1_BIN_WIDTH / 2, B1_BIN_WIDTH), 10)
N_BINS = len(BIN_EDGES) - 1
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2


def bin_index(h) -> tuple[np.ndarray, np.ndarray]:
    """HbI -> bin 번호 (0..N_BINS-1). 두 번째 반환값: 경계 밖이라 끝 bin으로 넣었는지 여부."""
    h = np.asarray(h, dtype=np.float64)
    raw = np.floor((h - BIN_EDGES[0]) / B1_BIN_WIDTH + 1e-9).astype(int)
    out = (raw < 0) | (raw >= N_BINS)
    return np.clip(raw, 0, N_BINS - 1), out


class BinnedMeanModel:
    name = "B1"

    def __init__(self):
        self.means = None            # [N_BINS, 31] fallback 적용 후 bin 평균
        self.source = None           # [N_BINS] 실제로 쓰는 평균이 어느 bin 것인지
        self.bin_table = None
        self.train_subjects = None
        self.reset_counters()

    def reset_counters(self):
        self.fallback_uses = 0       # 빈 bin을 대체한 평균을 사용한 횟수 (입력·목표 bin 조회마다 1회)
        self.n_lookups = 0           # 전체 bin 조회 횟수
        self.out_of_range = 0        # 경계 밖 HbI를 끝 bin으로 넣은 횟수

    def fit(self, train: Spectra) -> "BinnedMeanModel":
        b, out = bin_index(train.hbi)
        raw = np.full((N_BINS, len(WL)), np.nan)
        rows = []
        for k in range(N_BINS):
            m = b == k
            if m.any():
                raw[k] = train.a[m].mean(axis=0)
            rows.append(dict(bin=k, lo=BIN_EDGES[k], hi=BIN_EDGES[k + 1], n_measurements=int(m.sum()),
                             n_subjects=len(set(train.subject[m].tolist())),
                             subjects=" ".join(sorted(set(train.subject[m].tolist())))))
        valid = np.flatnonzero(~np.isnan(raw[:, 0]))
        if len(valid) == 0:
            raise ValueError("학습 데이터로 채워진 bin이 없음")
        self.source = np.array([k if k in valid else valid[np.argmin(np.abs(valid - k))] for k in range(N_BINS)])
        self.means = raw[self.source]
        self.bin_table = pd.DataFrame(rows)
        self.bin_table["fallback_from"] = [int(s) if s != k else -1 for k, s in enumerate(self.source)]
        self.bin_table["train_out_of_range"] = int(out.sum())
        self.train_subjects = train.subjects
        self.reset_counters()
        return self

    def predict(self, a: np.ndarray, target) -> np.ndarray:
        if self.means is None:
            raise RuntimeError("fit()을 먼저 호출해야 함")
        a, target, single = prepare(a, target)
        h = a[:, HBI_INDEX]
        bi, oi = bin_index(h)
        bt, ot = bin_index(target)
        for b, o in ((bi, oi), (bt, ot)):
            self.fallback_uses += int((self.source[b] != b).sum())
            self.out_of_range += int(o.sum())
            self.n_lookups += len(b)
        out = a + (self.means[bt] - self.means[bi])  # 차이를 먼저 계산: 같은 bin이면 보정량이 정확히 0
        out[:, HBI_INDEX] = target                  # 570 nm = 목표 HbI
        same = target == h                          # 입력 HbI = 목표 HbI -> 원본 그대로
        out[same] = a[same]
        return finish(out, single)

    def counters(self) -> dict:
        return dict(fallback_uses=self.fallback_uses, n_lookups=self.n_lookups, out_of_range=self.out_of_range)

    def save(self, path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(
            model=self.name, wavelengths_nm=WL.tolist(), bin_edges=BIN_EDGES.tolist(),
            means=self.means.tolist(), source=self.source.tolist(),
            bin_table=self.bin_table.to_dict(orient="records"), train_subjects=self.train_subjects), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "BinnedMeanModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d["model"] != cls.name or d["wavelengths_nm"] != WL.tolist() or not np.allclose(d["bin_edges"], BIN_EDGES):
            raise ValueError("B1 파라미터 파일이 아니거나 파장/bin 경계가 다름")
        m = cls()
        m.means = np.asarray(d["means"], dtype=np.float64)
        m.source = np.asarray(d["source"], dtype=int)
        m.bin_table = pd.DataFrame(d["bin_table"])
        m.train_subjects = d["train_subjects"]
        return m
