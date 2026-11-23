"""재학습·튜닝 실험(2026-11)의 공용 평가 함수: 모델 불러오기/학습, 한 분할의 피험자별 지표, 집계와 쌍체 비교.

final_comparison.py, band_analysis.py, sensitivity_exclude_S006.py, external_test.py, run_union16.py 가 함께 쓴다.
지표는 common/metrics.py 의 함수만 쓴다. BiGRU의 피험자 값 = seed별 지표의 평균.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))
from common import config as C                                                # noqa: E402
from common.data_loader import HBI_INDEX, Spectra                             # noqa: E402
from common.metrics import consistency, group_metrics                         # noqa: E402
from evaluate import DH_BINS                                                  # noqa: E402
from final_utils import BAND_INDEX, BANDS, RES, jload, paired_diff            # noqa: E402
from models.b2_pls import B2PLSModel                                          # noqa: E402
from models.b2_ridge import B2RidgeModel                                      # noqa: E402
from models.bigru import BiGRUModel                                           # noqa: E402
from models.hb_index import BinnedMeanModel                                   # noqa: E402
from models.linear import LinearModel                                         # noqa: E402
from models.pls import PLSModel                                               # noqa: E402

SEEDS = tuple(C.BIGRU_SEEDS)
SCHEME_DIR = {"Development10_Holdout73": "holdout", "Development10_LeaveOneSubjectOut": "loso"}
TOLS = {"near_0.02": C.NEAR_TOL_MAIN, "near_0.03": C.NEAR_TOL_SUPP}
ABS_JUDGED = {"peak_bias", "bias"}                 # 부호 있는 지표: |값| 감소를 개선으로 판정
BIGRU_LR, BIGRU_CLIP = 3e-3, 1.0
GENERAL_KEYS = ("a_rmse", "r_rmse_pp", "peak_rmse", "peak_bias")


# ---------------------------------------------------------------------------
# 모델 불러오기 / 학습 (보고 명칭 -> seed별 모델 목록)
# ---------------------------------------------------------------------------
def load_ridge(path) -> B2RidgeModel:
    """B2RidgeModel 에는 load 가 없으므로 저장된 JSON에서 예측에 필요한 값만 복원한다."""
    d = jload(path)
    m = B2RidgeModel(alpha=d["alpha"])
    m.train_subjects, m.cv = d["train_subjects"], d["inner_cv"]
    m.slope, m.us, m.coef = np.array(d["b2_slope"]), np.array(d["u_std"]), np.array(d["coef"])
    return m


def load_dev_models(split_set: str, split_id: str, with_peak_constrained: bool = True) -> dict:
    """Development cohort 분할의 저장 모델 전부. {보고 명칭: [모델 (seed별)]}"""
    sc = SCHEME_DIR[split_set]
    base, tuned = RES / sc / "models", RES / "bigru_tuned" / split_set
    out = {
        "HbI-binned mean": [BinnedMeanModel.load(base / "B1" / f"{split_id}.json")],
        "Global Linear": [LinearModel.load(base / "B2" / f"{split_id}.json")],
        "PLS_v2": [B2PLSModel.load(RES / "pls_v2_k10" / sc / "models" / f"{split_id}.json")],
        "BiGRU": [BiGRUModel.load(tuned / "models" / f"{split_id}_seed_{k}.pt") for k in SEEDS],
        "PLS_v1": [PLSModel.load(RES / "pls_v1_k10" / sc / "models" / f"{split_id}.json")],
        "Ridge": [load_ridge(RES / "ridge" / sc / "models" / f"{split_id}.json")],
        "BiGRU (fixed setting)": [BiGRUModel.load(base / "BiGRU" / f"{split_id}_seed_{k}.pt") for k in SEEDS],
    }
    if with_peak_constrained:
        out["BiGRU (Peak-constrained selection)"] = [
            BiGRUModel.load(tuned / "peak_constrained" / "models" / f"{split_id}_seed_{k}.pt") for k in SEEDS]
    out["PLS_v2 (max 5 components)"] = [B2PLSModel.load(RES / "pls_v2" / sc / "models" / f"{split_id}.json")]
    out["PLS_v1 (max 5 components)"] = [PLSModel.load(RES / "pls_v1" / sc / "models" / f"{split_id}.json")]
    return out


def bigru_cfg(hp: dict) -> dict:
    return dict(hidden=int(hp["hidden"]), lr=BIGRU_LR, slope_l2=float(hp["slope_l2"]), clip=BIGRU_CLIP)


def fit_models(tr: Spectra, hp: dict, names) -> dict:
    """고정된 하이퍼파라미터 hp로 학습 (다시 고르지 않음).
    hp 키: pls_v2_components, pls_v1_components, ridge_alpha, bigru, bigru_peak_constrained (각각 hidden, slope_l2, epoch)."""
    out = {}
    for n in names:
        if n == "HbI-binned mean":
            out[n] = [BinnedMeanModel().fit(tr)]
        elif n == "Global Linear":
            out[n] = [LinearModel().fit(tr)]
        elif n == "PLS_v2":
            out[n] = [B2PLSModel(n_components=int(hp["pls_v2_components"])).fit(tr)]
        elif n == "PLS_v1":
            out[n] = [PLSModel(n_components=int(hp["pls_v1_components"])).fit(tr)]
        elif n == "Ridge":
            out[n] = [B2RidgeModel(alpha=float(hp["ridge_alpha"])).fit(tr)]
        elif n == "BiGRU (fixed setting)":
            out[n] = [BiGRUModel(seed=k).fit(tr) for k in SEEDS]
        elif n in ("BiGRU", "BiGRU (Peak-constrained selection)"):
            h = hp["bigru" if n == "BiGRU" else "bigru_peak_constrained"]
            out[n] = [BiGRUModel(seed=k, epochs=int(h["epoch"]), cfg=bigru_cfg(h)).fit(tr) for k in SEEDS]
        else:
            raise KeyError(n)
    return out


# ---------------------------------------------------------------------------
# 한 분할의 피험자별 지표 (모든 실험·구간)
# ---------------------------------------------------------------------------
def _seed_mean(dicts: list) -> dict:
    return {k: float(np.mean([d[k] for d in dicts])) for k in dicts[0]}


def _general(P, Y) -> dict:
    g = group_metrics(P, Y)
    return {k: g[k] for k in GENERAL_KEYS}


def band_errors(P, Y):
    """구간별 absorbance RMSE와 bias (예측 - 정답), 그리고 5개 구간 제곱합으로 다시 계산한 전체 RMSE."""
    out, sse = {}, 0.0
    for b, idx in BAND_INDEX.items():
        e = (P - Y)[:, idx]
        out[b] = dict(a_rmse=float(np.sqrt(np.mean(e ** 2))), bias=float(np.mean(e)))
        sse += float(np.sum(e ** 2))
    return out, float(np.sqrt(sse / (len(P) * 30)))


def split_records(models: dict, te: Spectra, X, Y, T, h_in, subj) -> list:
    """models = {보고 명칭: [모델 (seed별)]}. 반환: dict(subject, model, section, metric, value) 목록.
    section: general / near_0.02 / near_0.03 / dh|<구간>|<방향> / band|<구간> / repeatability(_in_range|_other)"""
    recs = []

    def add(s, model, section, d):
        recs.extend(dict(subject=s, model=model, section=section, metric=k, value=v) for k, v in d.items())

    dh = T - h_in
    for name, ms in models.items():
        Ps = [m.predict(X, T) for m in ms]
        for P in Ps:
            assert np.array_equal(P[:, HBI_INDEX], T) and np.isfinite(P).all()
        for s in sorted(set(subj.tolist())):
            sm = subj == s
            add(s, name, "general", _seed_mean([_general(P[sm], Y[sm]) for P in Ps]))
            bands = []
            for P in Ps:
                b, total = band_errors(P[sm], Y[sm])
                assert abs(total - group_metrics(P[sm], Y[sm])["a_rmse"]) < 1e-12, "구간 제곱합으로 다시 계산한 전체 RMSE가 다름"
                bands.append(b)
            for b in BANDS:
                add(s, name, f"band|{b}", _seed_mean([x[b] for x in bands]))
            for sec, tol in TOLS.items():
                sl = sm & (np.abs(T - C.TARGET_HBI) <= tol)
                if sl.any():
                    g = _seed_mean([_general(P[sl], Y[sl]) for P in Ps])
                    add(s, name, sec, dict(a_rmse=g["a_rmse"], peak_rmse=g["peak_rmse"], n_pairs=float(sl.sum())))
            for lo, hi in zip(DH_BINS[:-1], DH_BINS[1:]):
                inb = sm & (np.abs(dh) >= lo) & (np.abs(dh) < hi)
                for direction, dm in (("increase", dh > 0), ("decrease", dh < 0), ("all", np.ones_like(inb))):
                    mm = inb & dm
                    if mm.any():
                        g = _seed_mean([_general(P[mm], Y[mm]) for P in Ps])
                        g["n_pairs"] = float(mm.sum())
                        add(s, name, f"dh|{lo:.3f}-{hi:.3f}|{direction}", g)
    for s in te.subjects:                                   # Fixed-target repeatability (정답 없음)
        A0 = te.a[te.subject == s]
        h = A0[:, HBI_INDEX]
        grp = "repeatability_in_range" if h.min() <= C.TARGET_HBI <= h.max() else "repeatability_other"
        fixed = {"Raw": [A0], **{n: [m.predict(A0, C.TARGET_HBI) for m in ms] for n, ms in models.items()}}
        for name, Fs in fixed.items():
            d = _seed_mean([{k: consistency(F)[k] for k in ("peak_sd", "spectral_sd", "peak_mean")} for F in Fs])
            add(s, name, "repeatability", d)
            add(s, name, grp, d)
    return recs


# ---------------------------------------------------------------------------
# 집계와 쌍체 비교
# ---------------------------------------------------------------------------
def summary_table(df: pd.DataFrame, by_split: bool) -> pd.DataFrame:
    """by_split=False: 피험자 값의 평균 ± 표준편차. by_split=True: 피험자 -> 분할 안 평균 -> 분할 평균 ± 표준편차."""
    unit = "split_id" if by_split else "subject"
    g = df.groupby(["section", "metric", "model", unit], sort=False).value.mean()
    out = g.groupby(["section", "metric", "model"], sort=False).agg(["mean", "std", "count"]).reset_index()
    out = out.rename(columns={"std": "sd", "count": "n"})
    out["unit"] = "split" if by_split else "subject"
    return out


def subject_values(df: pd.DataFrame) -> pd.Series:
    """(section, metric, model, subject) -> 값. 한 피험자가 여러 분할에 나오면 분할 평균."""
    return df.groupby(["section", "metric", "model", "subject"], sort=False).value.mean()


def comparison_table(df: pd.DataFrame, comparisons, sections=None, metrics=None) -> pd.DataFrame:
    """피험자 쌍체 차이 표. 비교 이름 'A minus B', 차이 = A - B, 개선 = 차이 < 0 (ABS_JUDGED 지표는 |A| < |B|)."""
    sv = subject_values(df)
    rows = []
    for sec, met in sv.index.droplevel([2, 3]).unique():
        if met in ("n_pairs", "peak_mean") or (metrics is not None and met not in metrics):
            continue
        if sections is not None and not any(sec == s or sec.startswith(s) for s in sections):
            continue
        cur = sv.loc[(sec, met)]
        have = set(cur.index.get_level_values(0))
        for a, b in comparisons:
            if a not in have or b not in have:
                continue
            p = paired_diff(cur.loc[a].to_dict(), cur.loc[b].to_dict(), met in ABS_JUDGED)
            if p is None:
                continue
            p.pop("per_subject_diff")
            subs = p.pop("subjects")
            rows.append(dict(section=sec, metric=met, comparison=f"{a} minus {b}", **p, subjects=" ".join(subs)))
    return pd.DataFrame(rows)
