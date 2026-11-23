"""재학습·튜닝 실험(2026-11)의 새 스크립트들이 함께 쓰는 정의. 기존 스크립트는 이 파일을 쓰지 않는다.

  - 분할: data/splits_final/<분할 세트>/ 의 split_plan.csv, test_pairs.csv 를 읽는다.
  - 파장 구간 정의 (고정): Soret, Band 450-460, Carotenoid, Q-band (570 제외), Red
  - 쌍체 비교: 차이 = 앞 모델 - 뒤 모델, 개선 = 차이 < 0 (오차·표준편차 지표는 낮을수록 좋음).
    p 값은 다중 비교 보정이 없는 기술적 참고값.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))
from common.data_loader import FREE, HBI_INDEX, WL, Spectra, load_spectra      # noqa: E402
from loso import sign_test, wilcoxon_exact                                    # noqa: E402

RES = ROOT / "results"
SPLITS = ROOT / "data" / "splits_final"
DEV_CSV = ROOT / "data" / "processed" / "spectra.csv"
UNION_CSV = ROOT / "data" / "processed" / "spectra_union16.csv"

BANDS = {
    "Soret": [400, 410, 420, 430, 440],
    "Band 450-460": [450, 460],
    "Carotenoid": [470, 480, 490, 500, 510],
    "Q-band": [520, 530, 540, 550, 560, 580, 590],
    "Red": list(range(600, 701, 10)),
}
BAND_INDEX = {b: np.array([int(np.flatnonzero(WL == w)[0]) for w in ws]) for b, ws in BANDS.items()}
assert sum(len(v) for v in BANDS.values()) == 30 and 570 not in sum(BANDS.values(), [])
assert sorted(sum(BANDS.values(), [])) == WL[FREE].tolist()
SLOPE_BANDS = ["Soret", "Band 450-460", "Q-band", "Red"]          # 잔차 기울기 집계 구간 (Carotenoid, 570 제외)

# 보고 명칭 <-> 기존 코드 이름
REPORT_NAME = {"B1": "HbI-binned mean", "B2": "Global Linear", "BiGRU": "BiGRU (fixed setting)",
               "pls_v2_k10": "PLS_v2", "pls_v1_k10": "PLS_v1", "ridge": "Ridge",
               "pls_v2": "PLS_v2 (max 5 components)", "pls_v1": "PLS_v1 (max 5 components)"}
MAIN_MODELS = ["HbI-binned mean", "Global Linear", "PLS_v2", "BiGRU"]
SUPP_MODELS = ["PLS_v1", "Ridge", "BiGRU (fixed setting)", "BiGRU (Peak-constrained selection)",
               "PLS_v2 (max 5 components)", "PLS_v1 (max 5 components)"]
MAIN_COMPARISONS = [("PLS_v2", "Global Linear"), ("BiGRU", "Global Linear"), ("PLS_v2", "HbI-binned mean"),
                    ("BiGRU", "HbI-binned mean"), ("BiGRU", "PLS_v2"), ("BiGRU", "BiGRU (fixed setting)")]
SUPP_COMPARISONS = [(m, "Global Linear") for m in SUPP_MODELS] + [("BiGRU (Peak-constrained selection)", "BiGRU")]
BAND_COMPARISONS = [("PLS_v2", "Global Linear"), ("BiGRU", "Global Linear"), ("BiGRU", "PLS_v2")]
P_NOTE = "p 값은 다중 비교 보정이 없는 기술적 참고값 (정확 Wilcoxon signed-rank, 부호검정, 양측)."


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jdump(p, obj, indent=1):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=indent, default=_default), encoding="utf-8")


def _default(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_splits(split_set: str) -> list[dict]:
    plan = pd.read_csv(SPLITS / split_set / "split_plan.csv")
    return [dict(split_id=r.split_id, train=r.train.split(), test=r.test.split()) for r in plan.itertuples()]


def row_index(data: Spectra) -> dict:
    return {(s, int(t)): i for i, (s, t) in enumerate(zip(data.subject, data.trial))}


def test_pairs(split_set: str, split_id: str, data: Spectra):
    """General reconstruction 테스트 쌍 -> (ii, jj, subject 배열). X = a[ii], 정답 = a[jj], 목표 HbI = hbi[jj]."""
    d = pd.read_csv(SPLITS / split_set / split_id / "test_pairs.csv")
    row = row_index(data)
    ii = np.array([row[(s, t)] for s, t in zip(d.subject, d.input_trial)])
    jj = np.array([row[(s, t)] for s, t in zip(d.subject, d.target_trial)])
    assert np.allclose(data.hbi[ii], d.input_hbi, atol=1e-12) and np.allclose(data.hbi[jj], d.target_hbi, atol=1e-12)
    return ii, jj, d.subject.to_numpy()


def paired_diff(a: dict, b: dict, lower_is_better_abs: bool = False) -> dict | None:
    """피험자 쌍체 차이 (a - b). a, b = {피험자: 값}. 값이 없는(None/NaN) 피험자는 n에서 빠진다.
    개선 = 차이 < 0. lower_is_better_abs=True 이면 |a| - |b| 로 개선을 판정한다 (부호 있는 bias용)."""
    subs = sorted(s for s in set(a) & set(b) if a[s] is not None and b[s] is not None
                  and np.isfinite(a[s]) and np.isfinite(b[s]))
    if not subs:
        return None
    x, y = np.array([a[s] for s in subs], float), np.array([b[s] for s in subs], float)
    d = x - y
    judge = np.abs(x) - np.abs(y) if lower_is_better_abs else d
    n = len(subs)
    return dict(n=n, subjects=subs, mean_a=float(x.mean()), mean_b=float(y.mean()), diff_mean=float(d.mean()),
                diff_sd=float(d.std(ddof=1)) if n > 1 else None,
                diff_se=float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
                n_improved=int((judge < 0).sum()), improved_definition="|a| < |b|" if lower_is_better_abs else "a < b",
                wilcoxon_p=wilcoxon_exact(judge)["p_two_sided"] if n >= 2 else None,
                sign_p=sign_test(judge)["p_two_sided"] if n >= 2 else None,
                per_subject_diff={s: float(v) for s, v in zip(subs, d)})


def mean_sd(values) -> dict:
    v = np.array([x for x in values if x is not None and np.isfinite(x)], float)
    return dict(n=int(len(v)), mean=float(v.mean()) if len(v) else None, sd=float(v.std(ddof=1)) if len(v) > 1 else None)


__all__ = ["BANDS", "BAND_INDEX", "SLOPE_BANDS", "FREE", "HBI_INDEX", "WL", "RES", "ROOT", "SPLITS", "DEV_CSV", "UNION_CSV",
           "jload", "jdump", "load_splits", "row_index", "test_pairs", "paired_diff", "mean_sd", "load_spectra"]
