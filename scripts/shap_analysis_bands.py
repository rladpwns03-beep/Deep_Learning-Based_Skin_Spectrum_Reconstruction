"""구간 단위 SHAP (입력 그룹 7개, 정확한 Shapley 값). 기존 scripts/shap_analysis.py 는 수정하지 않고 계산 방식만 그대로 쓴다.

실행 (code 폴더, conda 환경 skin_spectra):
    python -X utf8 scripts/shap_analysis_bands.py    # results/shap_bands/Development10_LeaveOneSubjectOut/ (이미 있으면 중단)

설계 (결과를 보기 전에 고정)
  - 특징 z = [A_in 31파장, h*], 배경 = 그 fold 학습 쌍 100개 (seed 0), 조합 2^7 = 128개 전부 계산,
    설명 대상 = 예측 변화량 30파장 (570 제외)과 Peak 변화량. (shap_analysis.py 와 같음)
  - 입력 그룹 7개: h* (target), h_in (570 nm), Soret 400-440, Band 450-460, Carotenoid 470-510,
    Q-band 520-560·580·590 (570 제외), Red 600-700. 특징 32개를 빠짐없이 한 번씩 덮는다.
  - 모델: HbI-binned mean, Global Linear, PLS_v2, BiGRU (튜닝, seed 0/1/2 각각 계산 후 평균), PLS_v1, Ridge.
  - 출력 파장도 5개 구간으로 묶은 "입력 그룹 x 출력 구간" 표를 낸다.
  - 검사: 기존 6개 그룹을 넣으면 shap_analysis.shapley 와 같은 값 (fold 1개, 모델 1개). 효율성 base + Σφ = f(x).

해석 주의
  PLS_v2와 Ridge는 입력이 Δh x (A_in - A_in(570)) 형태라 h_in 그룹과 스펙트럼 그룹의 기여가 반대 부호로 상쇄될 수 있다.
  그룹별 크기만으로 중요도를 단정하지 않고, h_in 기여와 스펙트럼 그룹(5개 합) 기여의 상관을 함께 보고한다.
  그룹을 서로 다른 측정에서 섞으면 실제로 없는 스펙트럼이 생길 수 있다 (interventional SHAP의 한계).
"""
from __future__ import annotations

import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))
import shap_analysis as old                                                    # noqa: E402  (기존 구현, 수정하지 않음)
from common.data_loader import FREE, HBI_INDEX, WL, make_pairs                # noqa: E402
from final_utils import BAND_INDEX, BANDS, DEV_CSV, RES, jdump, jload, load_splits, load_spectra, test_pairs   # noqa: E402
from final_eval import load_dev_models                                         # noqa: E402
from models.b2_ridge import B2RidgeModel                                       # noqa: E402

SPLIT_SET = "Development10_LeaveOneSubjectOut"
OUT = RES / "shap_bands" / SPLIT_SET
N_BG, BG_SEED, H_STAR = old.N_BG, old.BG_SEED, old.H_STAR
GROUPS = {"h* (target)": [H_STAR], "h_in (570 nm)": [HBI_INDEX], **{b: [int(i) for i in idx] for b, idx in BAND_INDEX.items()}}
GNAMES = list(GROUPS)
SPECTRAL_GROUPS = list(BANDS)
MODELS = ["HbI-binned mean", "Global Linear", "PLS_v2", "BiGRU", "PLS_v1", "Ridge"]
FREE_WL = WL[FREE]
OUT_BAND_COLS = {b: [int(np.flatnonzero(FREE_WL == w)[0]) for w in ws] for b, ws in BANDS.items()}   # 출력 30열 안의 위치


def check_groups(groups: dict):
    flat = sorted(sum(groups.values(), []))
    if flat != list(range(32)):
        raise ValueError("입력 그룹이 특징 32개를 빠짐없이 한 번씩 덮지 않음")


check_groups(GROUPS)
assert len(GROUPS) == 7


def shapley(model, X, BG, groups: dict):
    """정확한 그룹 Shapley 값. X (N,32), BG (B,32) -> phi (N, 그룹 수, 31), base (31,), fx (N, 31)."""
    check_groups(groups)
    names = list(groups)
    ng, N, B = len(names), len(X), len(BG)
    v = {}
    for S in itertools.product([0, 1], repeat=ng):
        mask = np.zeros(32, bool)
        for g, on in zip(names, S):
            if on:
                mask[groups[g]] = True
        Z = np.where(mask[None, None, :], X[:, None, :], BG[None, :, :]).reshape(N * B, 32)
        v[S] = old.outputs(model, Z).reshape(N, B, -1).mean(1)
    phi = np.zeros((N, ng, v[(0,) * ng].shape[1]))
    for i in range(ng):
        for S in v:
            if S[i]:
                continue
            k = sum(S)
            w = math.factorial(k) * math.factorial(ng - k - 1) / math.factorial(ng)
            Si = tuple(1 if j == i else s for j, s in enumerate(S))
            phi[:, i] += w * (v[Si] - v[S])
    return phi, v[(0,) * ng].mean(0), v[(1,) * ng]


def _corr(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None


def main():
    if OUT.exists() and any(OUT.iterdir()):
        sys.exit(f"[중단] {OUT} 에 이미 결과가 있음. 덮어쓰지 않음.")
    t0 = time.time()
    torch.set_num_threads(1)
    data = load_spectra(DEV_CSV)
    recs, checks, phis = [], [], {}
    for n_fold, sp in enumerate(load_splits(SPLIT_SET)):
        fid, s = sp["split_id"], sp["test"][0]
        tr = data.select(sp["train"])
        Xa, Ya, *_ = make_pairs(tr)
        pick = np.random.default_rng(BG_SEED).choice(len(Xa), N_BG, replace=False)
        BG = np.column_stack([Xa[pick], Ya[pick, HBI_INDEX]])
        ii, jj, _ = test_pairs(SPLIT_SET, fid, data)
        X = np.column_stack([data.a[ii], data.hbi[jj]])
        loaded = load_dev_models(SPLIT_SET, fid, with_peak_constrained=False)
        ridge = B2RidgeModel().fit(tr)                       # 결정적 재학습, 저장 계수와 같은지 확인
        saved = jload(RES / f"ridge/loso/models/{fid}.json")
        checks.append(dict(fold=fid, ridge_refit_matches_saved=bool(np.allclose(ridge.coef, saved["coef"], rtol=1e-10, atol=1e-14))))
        models = {m: loaded[m] for m in MODELS}
        models["Ridge"] = [ridge]
        if n_fold == 0:                                      # 기존 6개 그룹 -> 기존 shapley 와 같은 값
            ref = old.shapley(models["PLS_v2"][0], X, BG)
            new = shapley(models["PLS_v2"][0], X, BG, old.GROUPS)
            same = all(np.array_equal(a, b) for a, b in zip(ref, new))
            checks.append(dict(fold=fid, model="PLS_v2", old_groups_match_shap_analysis=bool(same),
                               max_abs_diff=float(max(np.abs(a - b).max() for a, b in zip(ref, new)))))
            if not same and checks[-1]["max_abs_diff"] > 1e-12:
                raise RuntimeError(f"기존 6개 그룹에서 shap_analysis.shapley 와 다름: {checks[-1]}")
        for name, ms in models.items():
            for k, m in enumerate(ms):
                label = name if len(ms) == 1 else f"{name}|seed{k}"
                phi, base, fx = shapley(m, X, BG, GROUPS)
                checks.append(dict(fold=fid, model=label, efficiency_max_abs_err=float(np.abs(base[None] + phi.sum(1) - fx).max())))
                phis.setdefault(label, []).append(phi)
                for n in range(len(X)):
                    for gi, g in enumerate(GNAMES):
                        recs.append(dict(fold=fid, subject=s, pair=n, model=label, group=g,
                                         mean_abs_spectral=float(np.abs(phi[n, gi, :30]).mean()), peak=float(phi[n, gi, 30]),
                                         **{f"wl{int(w)}": float(phi[n, gi, c]) for c, w in enumerate(FREE_WL)}))
        print(f"[{fid}] 완료 ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(recs)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "shap_values_per_pair.csv", index=False, encoding="utf-8-sig", float_format="%.6g")
    df["model_base"] = df.model.str.replace(r"\|seed\d", "", regex=True)
    df["abs_peak"] = df.peak.abs()
    wl_cols = [c for c in df.columns if c.startswith("wl")]
    for b, cols in OUT_BAND_COLS.items():
        df[f"out|{b}"] = df[[wl_cols[c] for c in cols]].abs().mean(axis=1)
    absw = df[wl_cols].abs().add_prefix("abs_")
    df = pd.concat([df, absw], axis=1)

    def seed_then_model(cols):                               # seed별 평균을 먼저 내고 그 뒤에 seed 평균
        per = df.groupby(["model_base", "model", "group"], sort=False)[cols].mean()
        return per.groupby(level=["model_base", "group"], sort=False).mean()

    summ = seed_then_model(["mean_abs_spectral", "abs_peak"])
    by_wl = seed_then_model(list(absw.columns)).rename(columns=lambda c: c.replace("abs_", ""))
    by_band = seed_then_model([f"out|{b}" for b in BANDS])
    summ["share_pct_spectral"] = 100 * summ.mean_abs_spectral / summ.groupby(level=0).mean_abs_spectral.transform("sum")
    summ["share_pct_peak"] = 100 * summ.abs_peak / summ.groupby(level=0).abs_peak.transform("sum")
    summ.to_csv(OUT / "shap_summary.csv", encoding="utf-8-sig", float_format="%.6g")
    by_wl.to_csv(OUT / "shap_input_group_by_output_wavelength.csv", encoding="utf-8-sig", float_format="%.6g")
    by_band.to_csv(OUT / "shap_input_group_by_output_band.csv", encoding="utf-8-sig", float_format="%.6g")

    # h_in 기여와 스펙트럼 그룹(5개 합) 기여의 상관 (쌍 x 출력 파장 전체, Peak 따로). BiGRU는 seed별 상관의 평균.
    gi_h = GNAMES.index("h_in (570 nm)")
    gi_s = [GNAMES.index(g) for g in SPECTRAL_GROUPS]
    corr = {}
    for label, parts in phis.items():
        phi = np.concatenate(parts)                           # (900, 7, 31)
        c = dict(spectral=_corr(phi[:, gi_h, :30], phi[:, gi_s, :30].sum(1)), peak=_corr(phi[:, gi_h, 30], phi[:, gi_s, 30].sum(1)))
        corr.setdefault(label.split("|")[0], []).append(c)
    corr = {m: {k: (None if any(c[k] is None for c in cs) else float(np.mean([c[k] for c in cs]))) for k in ("spectral", "peak")}
            for m, cs in corr.items()}
    pd.DataFrame(corr).T.rename_axis("model").to_csv(OUT / "shap_hin_vs_spectral_correlation.csv", encoding="utf-8-sig", float_format="%.6g")
    eff = max(c.get("efficiency_max_abs_err", 0.0) for c in checks)
    jdump(OUT / "shap_summary.json", dict(
        design=__doc__, split_set=SPLIT_SET, groups={g: [int(WL[i]) if i < 31 else "h*" for i in v] for g, v in GROUPS.items()},
        n_groups=len(GROUPS), n_coalitions=2 ** len(GROUPS), n_background=N_BG, background_seed=BG_SEED,
        n_pairs=int(df.groupby("fold").pair.nunique().sum()), models=MODELS, output_bands=BANDS,
        checks=dict(groups_cover_32_features_once=True, max_efficiency_error=eff,
                    old_groups_match_shap_analysis=[c for c in checks if "old_groups_match_shap_analysis" in c],
                    ridge_refit_matches_saved=all(c["ridge_refit_matches_saved"] for c in checks if "ridge_refit_matches_saved" in c)),
        hin_vs_spectral_groups_correlation=corr,
        interpretation_caution=("PLS_v2와 Ridge는 입력이 Δh x (A_in - A_in(570)) 형태라 h_in 그룹과 스펙트럼 그룹의 기여가 반대 부호로 "
                                "상쇄될 수 있다. 그룹별 크기만으로 중요도를 단정하지 않는다."),
        summary={m: summ.loc[m].to_dict(orient="index") for m in summ.index.get_level_values(0).unique()},
        elapsed_seconds=time.time() - t0))
    print(f"저장: {OUT} ({time.time() - t0:.0f}s) | 효율성 최대 오차 {eff:.2e} | h_in-스펙트럼 상관 {corr}")


if __name__ == "__main__":
    main()
