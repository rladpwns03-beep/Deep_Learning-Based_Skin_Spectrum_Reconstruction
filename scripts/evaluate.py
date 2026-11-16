"""Phase 3 — General HbI Reconstruction & Target HbI = 0.6 Reconstruction 평가.

실행 (code 폴더, conda 환경 skin_spectra):
    python scripts/evaluate.py                    # results/holdout/{metrics,predictions,figures,logs} 에 저장 (기존 파일이 있으면 중단)
    python scripts/evaluate.py --run-id NAME      # results/holdout/runs/NAME/ 에 별도 저장
    python scripts/evaluate.py --version NAME     # results/NAME/models 를 평가해 results/NAME/ 에 저장

원칙
  - Phase 2에서 저장한 모델(results/holdout/models)을 불러와 그대로 사용 (재학습·하이퍼파라미터 변경 없음)
  - 기존 분할·평가 Pair(data/splits/repNN) 재사용. 목표 HbI는 CSV 반올림값이 아니라 원 데이터의 정확한 HbI
  - 세 모델(B1, B2, BiGRU seed 0/1/2)에 같은 입력·목표·정답 적용
  - 피험자별 지표를 먼저 계산 -> 피험자 평균 -> split 평균. 같은 피험자의 Pair를 독립 표본으로 보지 않음
  - BiGRU 요약값 = seed별 지표의 평균 (예측 평균이 아님)
  - Fixed-target(0.6) 결과는 반복성이며 정답 기반 정확도가 아님
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]   # 프로젝트 루트 (scripts/ 의 상위)
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "Model_Ablation"))

import ablation as legacy                                         # noqa: E402  Peak 정의 비교용 (수정하지 않음)
from common import config as C                                    # noqa: E402
from common.data_loader import FREE, HBI_INDEX, WL, load_spectra  # noqa: E402
from common.metrics import (consistency, group_metrics, pair_metrics, peak,  # noqa: E402
                            reflectance_pct, wavelength_metrics)
from models.bigru import BiGRUModel                               # noqa: E402
from models.hb_index import BinnedMeanModel                       # noqa: E402
from models.linear import LinearModel                             # noqa: E402

VERSION = "holdout"
BASE = ROOT / "results" / VERSION
MODELS = ["B1", "B2", "BiGRU"]
INSTANCES = [("B1", None), ("B2", None)] + [("BiGRU", s) for s in C.BIGRU_SEEDS]
DH_BINS = [0.0, 0.025, 0.05, 0.10, 0.30]          # |ΔHbI| 구간 (평가 전에 고정)
TOLS = {"main": C.NEAR_TOL_MAIN, "supplementary": C.NEAR_TOL_SUPP}
METRICS = ["a_rmse", "r_rmse_pp", "peak_rmse", "peak_mae", "peak_bias"]
UNITS = dict(a_rmse="absorbance", a_mae="absorbance", r_rmse_pp="%p (reflectance percentage points)",
             peak_rmse="absorbance", peak_mae="absorbance", peak_bias="absorbance (pred - true)",
             peak_sd="absorbance", spectral_sd="absorbance", target_hbi_error="absorbance (HbI)")
COLORS = {"Raw": "#9a9893", "B2": "#2a78d6", "B1": "#eb6834", "BiGRU": "#1baf7a"}
OUT_FILES = ["metrics/general_reconstruction.json", "metrics/target06_reconstruction.json",
             "metrics/model_comparison.json", "predictions/predictions.json", "logs/evaluation_verification.json",
             "figures/model_rmse_comparison.png", "figures/spectral_reconstruction.png",
             "figures/wavelength_error.png", "figures/rmse_by_delta_hbi.png", "figures/target06_accuracy.png",
             "figures/target06_reconstruction.png", "figures/target06_consistency.png"]


# ---------------------------------------------------------------------------
def jdump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")


def jload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha_tree(paths) -> dict:
    return {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def phase2_files():
    return [p for p in (BASE / "models").rglob("*") if p.is_file()] + \
        [BASE / "logs" / f for f in ("training_history.json", "training_summary.json", "training_verification.json")] + \
        [BASE / "config" / "training_config.json"]


def inst_name(model, seed):
    return model if seed is None else f"{model}_seed{seed}"


def nan_to_none(x):
    if isinstance(x, dict):
        return {k: nan_to_none(v) for k, v in x.items()}
    if isinstance(x, list):
        return [nan_to_none(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, metric: str) -> dict:
    """df: (split_id, model, seed, subject, metric) 피험자 단위 행. 변동성을 seed / subject / split으로 구분."""
    out = {}
    for model in MODELS:
        d = df[df.model == model]
        if d.empty or d[metric].isna().all():
            out[model] = None
            continue
        seedlvl = d.groupby(["split_id", "seed"])[metric].mean()               # split·seed별 피험자 평균
        split_mean = seedlvl.groupby("split_id").mean()                        # seed 평균
        subj = d.groupby(["split_id", "subject"])[metric].mean()               # 피험자별 (seed 평균)
        uniq = subj.groupby("subject").mean()                                  # 고유 피험자별 (여러 split 평균)
        rec = dict(mean=float(split_mean.mean()), n_splits=int(split_mean.size),
                   split_sd=float(split_mean.std(ddof=1)) if split_mean.size > 1 else None,
                   subject_sd_within_split=float(subj.groupby("split_id").std(ddof=1).mean())
                   if subj.groupby("split_id").size().max() > 1 else None,
                   n_subject_evaluations=int(subj.size), n_unique_subjects=int(uniq.size),
                   unique_subject_mean=float(uniq.mean()),
                   unique_subject_sd=float(uniq.std(ddof=1)) if uniq.size > 1 else None,
                   per_unique_subject={k: float(v) for k, v in uniq.items()},
                   per_split={k: float(v) for k, v in split_mean.items()})
        if model == "BiGRU":
            per_seed = seedlvl.groupby("seed").mean()
            rec["seed_sd_within_split"] = float(seedlvl.groupby("split_id").std(ddof=1).mean())
            rec["per_seed_mean"] = {f"seed_{int(k)}": float(v) for k, v in per_seed.items()}
            rec["seed_sd_of_overall_means"] = float(per_seed.std(ddof=1))
        out[model] = rec
    return out


def improvement(summ: dict, base: str) -> dict:
    if summ.get(base) is None or summ.get("BiGRU") is None:
        return None
    b, g = summ[base], summ["BiGRU"]
    subs = sorted(set(b["per_unique_subject"]) & set(g["per_unique_subject"]))
    spl = sorted(set(b["per_split"]) & set(g["per_split"]))
    per_subj = {s: 100 * (b["per_unique_subject"][s] - g["per_unique_subject"][s]) / b["per_unique_subject"][s] for s in subs}
    return dict(improvement_pct=100 * (b["mean"] - g["mean"]) / b["mean"],
                baseline_mean=b["mean"], bigru_mean=g["mean"],
                per_unique_subject_pct=per_subj,
                n_subjects_bigru_better=int(sum(v > 0 for v in per_subj.values())), n_subjects=len(subs),
                n_splits_bigru_better=int(sum(g["per_split"][k] < b["per_split"][k] for k in spl)), n_splits=len(spl))


# ---------------------------------------------------------------------------
def main():
    global VERSION, BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--version", default=VERSION, help="평가할 결과 버전 폴더 (results/<version>)")
    args = ap.parse_args()
    VERSION, BASE = args.version, ROOT / "results" / args.version
    if not (BASE / "config" / "training_config.json").exists():
        sys.exit(f"[중단] {BASE} 에 학습 결과(training_config.json)가 없음. train_models.py --version 먼저 실행")
    out = BASE / "runs" / args.run_id if args.run_id else BASE
    existing = [f for f in OUT_FILES if (out / f).exists()]
    if existing:
        sys.exit(f"[중단] 이미 존재하는 평가 결과: {existing}. 덮어쓰지 않음. 새 실행은 --run-id 지정.")
    run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    t0 = time.time()
    torch.set_num_threads(1)

    hash_before = sha_tree(phase2_files())
    tcfg = jload(BASE / "config" / "training_config.json")
    data = load_spectra()
    row = {(s, t): i for i, (s, t) in enumerate(zip(data.subject, data.trial))}
    checks = dict(models_from_phase2=[], pairs_match_files=[], pair_keys_complete=[], target_570_exact=True,
                  same_inputs_for_all_models=True)

    gen_rows, wl_rows, dh_rows, near_rows, cons_rows, ref_rows, b1_counters = [], [], [], [], [], [], []
    pred_general, pred_fixed = [], []

    for sp in tcfg["splits"]:
        sid, rep = sp["split_id"], sp["rep"]
        sdir = ROOT / "data" / "splits" / f"rep{rep:02d}"
        subs_file = pd.read_csv(sdir / "subjects.csv")
        test = sorted(subs_file[subs_file.role == "test"].subject)
        if test != sorted(sp["test"]):
            raise RuntimeError(f"{sid}: training_config와 split 파일의 test 피험자가 다름")

        # ---- 모델 불러오기 (Phase 2) ---------------------------------------
        mdir = BASE / "models"
        models = {("B2", None): LinearModel.load(mdir / "B2" / f"{sid}.json"),
                  ("B1", None): BinnedMeanModel.load(mdir / "B1" / f"{sid}.json")}
        for s in C.BIGRU_SEEDS:
            models[("BiGRU", s)] = BiGRUModel.load(mdir / "BiGRU" / f"{sid}_seed_{s}.pt")
        for k, m in models.items():
            checks["models_from_phase2"].append(dict(split_id=sid, model=inst_name(*k),
                                                     train_subjects_match=sorted(m.train_subjects) == sorted(sp["train"]),
                                                     no_test_subject_in_training=not set(m.train_subjects) & set(test)))

        # ---- Experiment 1: 평가 Pair -----------------------------------------
        pairs = pd.read_csv(sdir / "exp1_test_pairs.csv")
        ii = np.array([row[(s, t)] for s, t in zip(pairs.subject, pairs.input_trial)])
        jj = np.array([row[(s, t)] for s, t in zip(pairs.subject, pairs.target_trial)])
        X, Y, T = data.a[ii], data.a[jj], data.hbi[jj]                     # 정확한 HbI 사용
        expected = {(s, a, b) for s in test for a in data.trial[data.subject == s]
                    for b in data.trial[data.subject == s] if a != b}
        got = set(zip(pairs.subject, pairs.input_trial, pairs.target_trial))
        checks["pair_keys_complete"].append(dict(split_id=sid, n_pairs=len(pairs), complete=got == expected))
        checks["pairs_match_files"].append(dict(
            split_id=sid, csv_hbi_matches_data=bool(np.allclose(pairs.target_hbi, T, atol=5e-7) and
                                                    np.allclose(pairs.input_hbi, data.hbi[ii], atol=5e-7))))
        dh = T - data.hbi[ii]
        key = hashlib.sha256(np.concatenate([X, Y, T[:, None]], 1).tobytes()).hexdigest()

        preds = {}
        for k, m in models.items():
            if isinstance(m, BinnedMeanModel):
                m.reset_counters()
            P = m.predict(X, T)
            if not np.array_equal(P[:, HBI_INDEX], T):
                checks["target_570_exact"] = False
            preds[k] = P
            if isinstance(m, BinnedMeanModel):
                b1_counters.append(dict(split_id=sid, experiment="general", **m.counters()))
            name = dict(model=k[0], seed=-1 if k[1] is None else k[1], split_id=sid)
            pm = pair_metrics(P, Y)
            for s in test:
                msk = pairs.subject.to_numpy() == s
                g = group_metrics(P[msk], Y[msk])
                gen_rows.append(dict(**name, subject=s, **g, mean_pair_a_rmse=float(pm["a_rmse"][msk].mean())))
                wl_rows.append(dict(**name, subject=s, **wavelength_metrics(P[msk], Y[msk])))
                for lo, hi in zip(DH_BINS[:-1], DH_BINS[1:]):
                    for direction, dmask in (("increase", dh > 0), ("decrease", dh < 0)):
                        mm = msk & dmask & (np.abs(dh) >= lo) & (np.abs(dh) < hi)
                        if mm.any():
                            dh_rows.append(dict(**name, subject=s, dh_bin=f"{lo:.3f}-{hi:.3f}", direction=direction,
                                                **group_metrics(P[mm], Y[mm])))
        if len({hashlib.sha256(np.ascontiguousarray(v[:, HBI_INDEX]).tobytes()).hexdigest() for v in preds.values()}) != 1:
            checks["same_inputs_for_all_models"] = False

        for n, (s, a, b) in enumerate(zip(pairs.subject, pairs.input_trial, pairs.target_trial)):
            pred_general.append(dict(split_id=sid, subject=s, input_trial=int(a), target_trial=int(b),
                                     input_hbi=float(data.hbi[ii[n]]), target_hbi=float(T[n]),
                                     actual=Y[n].tolist(),
                                     predictions={inst_name(*k): v[n].tolist() for k, v in preds.items()}))

        # ---- Experiment 2.1: Near-target accuracy ------------------------------
        for tname, tol in TOLS.items():
            sel = np.abs(T - C.TARGET_HBI) <= tol
            fpairs = pd.read_csv(sdir / f"exp2_accuracy_pairs_tol{tol:.2f}.csv")
            checks["pairs_match_files"].append(dict(split_id=sid, tol=tol,
                                                    near_pairs_match_file=int(sel.sum()) == len(fpairs)))
            for k, P in preds.items():
                for s in test:
                    msk = sel & (pairs.subject.to_numpy() == s)
                    if not msk.any():
                        continue
                    gt_trials = sorted(set(pairs.target_trial[msk]))
                    near_rows.append(dict(model=k[0], seed=-1 if k[1] is None else k[1], split_id=sid, tolerance=tname,
                                          tol=tol, subject=s, n_gt_measurements=len(gt_trials), gt_trials=gt_trials,
                                          gt_hbi=sorted({float(x) for x in T[msk]}), **group_metrics(P[msk], Y[msk])))

        # ---- Experiment 2.2: Fixed-target (0.6) consistency ----------------------
        rep_in = pd.read_csv(sdir / "exp2_repeatability_inputs.csv")
        rr = np.array([row[(s, t)] for s, t in zip(rep_in.subject, rep_in.trial)])
        A0 = data.a[rr]
        fixed = {("Raw", None): A0}
        for k, m in models.items():
            if isinstance(m, BinnedMeanModel):
                m.reset_counters()
            fixed[k] = m.predict(A0, C.TARGET_HBI)
            if isinstance(m, BinnedMeanModel):
                b1_counters.append(dict(split_id=sid, experiment="fixed_target_0.6", **m.counters()))
        for k, P in fixed.items():
            for s in test:
                msk = rep_in.subject.to_numpy() == s
                hs = data.hbi[rr][msk]
                cons_rows.append(dict(model=k[0], seed=-1 if k[1] is None else k[1], split_id=sid, subject=s,
                                      **consistency(P[msk]),
                                      target_hbi_error_max=None if k[0] == "Raw" else
                                      float(np.abs(P[msk, HBI_INDEX] - C.TARGET_HBI).max()),
                                      n_reflectance_over_100=int((P[msk] < 0).sum()),
                                      n_nonfinite=int((~np.isfinite(P[msk])).sum()),
                                      target_in_subject_range=bool(hs.min() <= C.TARGET_HBI <= hs.max())))
        for n, (s, t) in enumerate(zip(rep_in.subject, rep_in.trial)):
            pred_fixed.append(dict(split_id=sid, subject=s, input_trial=int(t), input_hbi=float(data.hbi[rr[n]]),
                                   target_hbi=C.TARGET_HBI, actual=None,
                                   actual_note="정확히 HbI 0.6인 실측 스펙트럼 없음",
                                   predictions={inst_name(*k): v[n].tolist() for k, v in fixed.items() if k[0] != "Raw"}))

        # ---- Experiment 2.3: 0.6 복원 vs 0.6 근처 실측 (참고) ---------------------
        for s in test:
            ms = rep_in.subject.to_numpy() == s
            hs, trials = data.hbi[rr][ms], rep_in.trial.to_numpy()[ms]
            near = np.flatnonzero(np.abs(hs - C.TARGET_HBI) <= C.NEAR_TOL_SUPP)
            if len(near) == 0:
                ref_rows.append(dict(split_id=sid, subject=s, available=False,
                                     reason=f"|HbI-0.6| <= {C.NEAR_TOL_SUPP} 인 실측 없음 (범위 {hs.min():.3f}-{hs.max():.3f})"))
                continue
            for kk in near:
                others = np.array([i for i in range(len(hs)) if i != kk])
                for k, P in fixed.items():
                    if k[0] == "Raw":
                        continue
                    Ps = P[ms][others]
                    ref = np.repeat(A0[ms][kk][None], len(others), 0)
                    ref_rows.append(dict(split_id=sid, subject=s, available=True, model=k[0],
                                         seed=-1 if k[1] is None else k[1], reference_trial=int(trials[kk]),
                                         reference_hbi=float(hs[kk]), hbi_minus_06=float(hs[kk] - C.TARGET_HBI),
                                         within_main_tol=bool(abs(hs[kk] - C.TARGET_HBI) <= C.NEAR_TOL_MAIN),
                                         **group_metrics(Ps, ref),
                                         peak_difference_mean=float(np.mean(peak(Ps) - peak(ref)))))
        print(f"[{sid}] test {test}: Exp1 {len(pairs)} pairs, near(±0.02) {int((np.abs(T-0.6)<=0.02).sum())} pairs",
              flush=True)

    # ======================= 집계 및 JSON =======================================
    G, WLd, DH, NR, CS = map(pd.DataFrame, (gen_rows, wl_rows, dh_rows, near_rows, cons_rows))
    meta = dict(experiment_version=VERSION, run_id=run_id, phase="Phase 3 - Evaluation",
                config=dict(models=MODELS, bigru_seeds=list(C.BIGRU_SEEDS), training_pairs="P1_all_pairs",
                            target_hbi=C.TARGET_HBI, near_tolerances=TOLS, dh_bins=DH_BINS,
                            split_ids=[s["split_id"] for s in tcfg["splits"]],
                            split_source=f"data/splits/repNN + results/{VERSION}/config/training_config.json",
                            model_source=f"results/{VERSION}/models (Phase 2)", bigru_epochs=tcfg["bigru"]["epochs"]),
                units=UNITS,
                aggregation="pair -> subject(group RMSE over all pairs x 30 wavelengths) -> subject mean per split "
                            "-> mean over splits. BiGRU = mean of per-seed metrics. seed=-1 means not applicable.")

    general = dict(**meta, model="B1, B2, BiGRU", experiment="Experiment 1 - General HbI Reconstruction",
                   n_splits=int(G.split_id.nunique()), n_pairs_per_split=int(G[G.model == "B2"].groupby("split_id").n_pairs.sum().iloc[0]),
                   summary={m: summarize(G, m) for m in METRICS + ["mean_pair_a_rmse"]},
                   subject_wise=G.to_dict(orient="records"),
                   wavelength={"wavelengths_nm": WL.tolist(), "excluded_from_rmse_nm": [570],
                               "per_model": {m: {k: np.mean(np.vstack(WLd[WLd.model == m].groupby("split_id")[k].apply(
                                   lambda s: np.mean(np.vstack(s.tolist()), 0)).tolist()), 0).tolist()
                                   for k in ("rmse", "mae", "bias")} for m in MODELS},
                               "note": "피험자별 파장 지표 -> split 내 평균(BiGRU는 seed 포함) -> split 평균"},
                   rmse_by_delta_hbi=[dict(dh_bin=b, direction=d, model=m, **(lambda x: dict(
                       a_rmse=float(x.groupby("split_id").a_rmse.mean().mean()),
                       peak_rmse=float(x.groupby("split_id").peak_rmse.mean().mean()),
                       n_pairs=int(x[x.seed.isin([-1, 0])].n_pairs.sum()),
                       n_subject_evaluations=int(x.groupby(["split_id", "subject"]).ngroups)))(g))
                       for (b, d, m), g in DH.groupby(["dh_bin", "direction", "model"])],
                   b1_fallback_counters=[c for c in b1_counters if c["experiment"] == "general"],
                   status="completed")
    jdump(out / "metrics" / "general_reconstruction.json", nan_to_none(general))

    near_summary = {}
    for tname in TOLS:
        d = NR[NR.tolerance == tname]
        if d.empty:
            near_summary[tname] = dict(available=False, reason="해당 허용오차의 정답 측정이 test에 없음")
            continue
        b2 = d[d.model == "B2"]
        near_summary[tname] = dict(
            available=True, tol=TOLS[tname],
            n_splits_with_gt=int(b2.split_id.nunique()), n_subject_evaluations=int(len(b2)),
            unique_subjects=sorted(b2.subject.unique()), n_unique_subjects=int(b2.subject.nunique()),
            n_gt_measurements_total=int(b2.n_gt_measurements.sum()), n_pairs_total=int(b2.n_pairs.sum()),
            unique_gt_measurements=sorted({f"{r.subject}-t{t}" for r in b2.itertuples() for t in r.gt_trials}),
            metrics={m: summarize(d, m) for m in METRICS},
            note="정확히 HbI 0.6이 아니라 0.6 근처 실측을 정답으로 하고, 목표 HbI는 정답 측정의 실제 HbI")
    cons_summary = {m: {} for m in ("peak_sd", "spectral_sd")}
    for met in cons_summary:
        for model in ["Raw"] + MODELS:
            d = CS[CS.model == model]
            subj = d.groupby(["split_id", "subject"])[met].mean()
            uniq = subj.groupby("subject").mean()
            sm = subj.groupby("split_id").mean()
            cons_summary[met][model] = dict(mean=float(sm.mean()), split_sd=float(sm.std(ddof=1)),
                                            unique_subject_mean=float(uniq.mean()), unique_subject_sd=float(uniq.std(ddof=1)),
                                            per_unique_subject={k: float(v) for k, v in uniq.items()},
                                            n_subject_evaluations=int(subj.size))
            if model == "BiGRU":
                cons_summary[met][model]["seed_sd_within_split"] = float(
                    d.groupby(["split_id", "seed"])[met].mean().groupby("split_id").std(ddof=1).mean())
    for met in cons_summary:   # 0.6이 측정 범위 안/밖 피험자 구분
        for flag, lab in ((True, "target_in_range"), (False, "target_out_of_range")):
            d = CS[CS.target_in_subject_range == flag]
            cons_summary[met][lab] = {m: float(d[d.model == m].groupby(["split_id", "subject"])[met].mean().mean())
                                      for m in ["Raw"] + MODELS} | {"n_subject_evaluations": int(len(d[d.model == "Raw"]))}
    target06 = dict(
        **meta, model="Raw, B1, B2, BiGRU", experiment="Experiment 2 - Target HbI = 0.6 Reconstruction",
        near_target_accuracy=dict(summary=near_summary, subject_wise=NR.to_dict(orient="records"),
                                  ground_truth="실측 (0.6 근처, 목표 HbI = 정답 측정의 실제 HbI)"),
        fixed_target_consistency=dict(
            summary=cons_summary, subject_wise=CS.to_dict(orient="records"),
            target_hbi_error_max=float(CS[CS.model != "Raw"].target_hbi_error_max.max()),
            n_reflectance_over_100_total=int(CS[CS.model != "Raw"].n_reflectance_over_100.sum()),
            n_nonfinite_total=int(CS.n_nonfinite.sum()),
            ground_truth="없음 - 반복성(개인 내 변동) 지표이며 정확도가 아님",
            b1_fallback_counters=[c for c in b1_counters if c["experiment"] != "general"]),
        near06_reference_comparison=dict(
            rows=ref_rows,
            note="정확히 0.6으로 복원한 스펙트럼 vs 0.6 근처 실측. RMSE에는 목표 HbI 불일치(hbi_minus_06) 오차가 포함되므로 "
                 "정답 기반 정확도가 아닌 참고값"),
        status="completed")
    jdump(out / "metrics" / "target06_reconstruction.json", nan_to_none(target06))

    gsum = general["summary"]
    comparison = dict(
        **meta, model="B1, B2, BiGRU", experiment="Model comparison",
        general={m: {mod: (None if gsum[m][mod] is None else
                           {k: gsum[m][mod][k] for k in ("mean", "split_sd", "subject_sd_within_split",
                                                          "unique_subject_mean", "unique_subject_sd")}
                           | ({"seed_sd_within_split": gsum[m][mod]["seed_sd_within_split"]} if mod == "BiGRU" else {}))
                     for mod in MODELS} for m in METRICS},
        improvement_general={m: {"vs_B2": improvement(gsum[m], "B2"), "vs_B1": improvement(gsum[m], "B1"),
                                 "B2_vs_B1_pct": 100 * (gsum[m]["B1"]["mean"] - gsum[m]["B2"]["mean"]) / gsum[m]["B1"]["mean"]}
                             for m in ("a_rmse", "r_rmse_pp", "peak_rmse", "peak_mae")},
        improvement_near_target={t: ({m: {"vs_B2": improvement(near_summary[t]["metrics"][m], "B2"),
                                          "vs_B1": improvement(near_summary[t]["metrics"][m], "B1")}
                                      for m in ("a_rmse", "r_rmse_pp", "peak_rmse")} if near_summary[t]["available"] else None)
                                 for t in TOLS},
        consistency_peak_sd={m: cons_summary["peak_sd"][m]["mean"] for m in ["Raw"] + MODELS},
        consistency_spectral_sd={m: cons_summary["spectral_sd"][m]["mean"] for m in ["Raw"] + MODELS},
        variability_note="split_sd: split 평균값의 split 간 SD / subject_sd_within_split: split 안 test 피험자 간 SD의 평균 / "
                         "seed_sd_within_split: split 안 BiGRU seed 3개 간 SD의 평균 / unique_subject_*: 피험자별(여러 split 평균) 10명 기준",
        improvement_formula="(RMSE_baseline - RMSE_BiGRU) / RMSE_baseline x 100 (양수 = BiGRU 개선)",
        status="completed")
    jdump(out / "metrics" / "model_comparison.json", nan_to_none(comparison))

    jdump(out / "predictions" / "predictions.json", nan_to_none(dict(
        **meta, model="B1, B2, BiGRU (seed별)", wavelengths_nm=WL.tolist(), unit="absorbance",
        general_reconstruction=pred_general, fixed_target_0p6=pred_fixed, status="completed")))

    # ======================= 그림 (JSON에서 읽어 그림) ===========================
    fig_sources = make_figures(out, data)

    # ======================= 검증 ==============================================
    hash_after = sha_tree(phase2_files())
    proc = pd.read_csv(ROOT / "data" / "processed" / "spectra.csv")
    R_proc = proc[[f"{w}nm" for w in WL]].to_numpy(float)
    # 지표 계산 교차검증: 기존 ablation.py 방식(직접 식)으로 split_00 B2, BiGRU seed0 재계산
    xs = [r for r in pred_general if r["split_id"] == "split_00"]
    xc = []
    for inst in ("B2", "BiGRU_seed0"):
        for s in sorted({r["subject"] for r in xs}):
            P = np.array([r["predictions"][inst] for r in xs if r["subject"] == s])
            Yt = np.array([r["actual"] for r in xs if r["subject"] == s])
            ae, re = P - Yt, 100 * (10 ** -P - 10 ** -Yt)
            pe = legacy.peak(P) - legacy.peak(Yt)
            ref = G[(G.split_id == "split_00") & (G.subject == s) & (G.model == inst.split("_")[0]) &
                    (G.seed == (0 if "seed" in inst else -1))].iloc[0]
            xc.append(bool(np.isclose(np.sqrt(np.mean(ae[:, FREE] ** 2)), ref.a_rmse, rtol=1e-12) and
                           np.isclose(np.sqrt(np.mean(re[:, FREE] ** 2)), ref.r_rmse_pp, rtol=1e-12) and
                           np.isclose(np.sqrt(np.mean(pe ** 2)), ref.peak_rmse, rtol=1e-12) and
                           np.isclose(np.mean(np.abs(pe)), ref.peak_mae, rtol=1e-12) and
                           np.isclose(np.mean(pe), ref.peak_bias, rtol=1e-12, atol=1e-15)))
    verification = dict(
        **meta, model="B1, B2, BiGRU",
        checks={
            "1_phase2_models_used_unchanged": dict(
                passed=hash_before == hash_after and all(c["train_subjects_match"] and c["no_test_subject_in_training"]
                                                         for c in checks["models_from_phase2"]),
                n_model_files=len([k for k in hash_before if k.startswith("models")]), files_unchanged=hash_before == hash_after),
            "2_splits_and_pairs_reused": dict(
                passed=all(c["complete"] for c in checks["pair_keys_complete"]) and
                all(all(v for k, v in c.items() if k not in ("split_id", "tol")) for c in checks["pairs_match_files"]),
                pairs_per_split=[c["n_pairs"] for c in checks["pair_keys_complete"]]),
            "3_same_test_data_for_all_models": dict(passed=checks["same_inputs_for_all_models"]),
            "4_target_570_exact": dict(passed=checks["target_570_exact"] and
                                       target06["fixed_target_consistency"]["target_hbi_error_max"] == 0.0),
            "5_metric_formulas_cross_checked": dict(passed=all(xc), n_cases=len(xc),
                                                     method="ablation.py와 같은 직접 식으로 split_00 재계산"),
            "6_reflectance_formula": dict(passed=bool(np.allclose(reflectance_pct(data.a), R_proc, rtol=0, atol=1e-9)),
                                          formula="R(%) = 100 * 10^(-A); processed 반사율 복원 확인"),
            "7_peak_matches_legacy": dict(passed=bool(np.array_equal(peak(data.a), legacy.peak(data.a))),
                                          formula="A490 - (A470 + A510)/2"),
            "8_accuracy_vs_consistency_separated": dict(passed=True,
                                                        note="near_target_accuracy(정답 있음)와 fixed_target_consistency(정답 없음), "
                                                             "near06_reference_comparison(참고)를 별도 키로 저장"),
            "9_figures_from_json": dict(passed=True, sources=fig_sources),
            "10_no_imputed_values": dict(passed=True, n_reference_unavailable=int(sum(not r["available"] for r in ref_rows)),
                                         note="정답이 없는 경우 actual=null 또는 available=false + reason"),
        },
        b1_fallback_total=int(sum(c["fallback_uses"] for c in b1_counters)),
        abnormal_reflectance_over_100=target06["fixed_target_consistency"]["n_reflectance_over_100_total"],
        elapsed_seconds=round(time.time() - t0, 1), status="completed")
    verification["all_checks_passed"] = all(v["passed"] for v in verification["checks"].values())
    jdump(out / "logs" / "evaluation_verification.json", nan_to_none(verification))
    print(f"\n완료: {out}  ({verification['elapsed_seconds']}s)  all_checks_passed={verification['all_checks_passed']}")


# ======================= Figures ===================================================
def _style():
    plt.rcParams.update({
        "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 8,
        "mathtext.fontset": "custom", "mathtext.rm": "Malgun Gothic", "mathtext.it": "Malgun Gothic",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "axes.edgecolor": "#7a7a7a", "xtick.color": "#7a7a7a", "ytick.color": "#7a7a7a",
        "xtick.labelcolor": "#444444", "ytick.labelcolor": "#444444", "axes.titlesize": 8.5,
        "legend.frameon": False, "legend.fontsize": 7.5})


def _title(fig, text):
    fig.suptitle(text, x=0.01, ha="left", fontsize=10, fontweight="bold")


def plot_target06_reconstruction(out: Path, data, pr: dict, t06: dict) -> str:
    """0.6 근처 실측(|HbI-0.6|<=0.02)이 있는 test 피험자만, 그 피험자가 test였던 모든 split을 패널로 그린다.

    - 정답 = 0.6 근처 실측, 목표 HbI = 정답의 실제 HbI (near_target_accuracy와 같은 쌍).
    - 입력 = 같은 피험자의 정답이 아닌 측정 (자기 자신 -> 자기 자신 쌍은 없음).
    - BiGRU seed = 그 split에서 최종 학습 loss가 가장 작은 seed (test 결과는 선택에 쓰지 않음).
    """
    tsum = jload(BASE / "logs" / "training_summary.json")
    best_seed = {}
    for r in tsum["bigru"]["runs"]:
        if r["split_id"] not in best_seed or r["final_loss"] < best_seed[r["split_id"]][1]:
            best_seed[r["split_id"]] = (r["seed"], r["final_loss"])
    tol = C.NEAR_TOL_MAIN
    gen = [r for r in pr["general_reconstruction"] if abs(r["target_hbi"] - C.TARGET_HBI) <= tol]
    sw = [r for r in t06["near_target_accuracy"]["subject_wise"] if r["tolerance"] == "main"]
    panels = sorted({(r["subject"], r["split_id"]) for r in gen})
    subs = sorted({s for s, _ in panels})
    ncol = max(sum(s == sub for s, _ in panels) for sub in subs)
    fig, axes = plt.subplots(len(subs), ncol, figsize=(3.0 * ncol, 2.7 * len(subs)), sharex=True, sharey=True, squeeze=False)
    for i, sub in enumerate(subs):
        sps = [sp for s, sp in panels if s == sub]
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(sps):
                ax.axis("off"); continue
            sp = sps[j]
            rs = [r for r in gen if r["subject"] == sub and r["split_id"] == sp]
            seed = best_seed[sp][0]
            for mod, key in (("B1", "B1"), ("B2", "B2"), ("BiGRU", f"BiGRU_seed{seed}")):
                P = np.array([r["predictions"][key] for r in rs])
                rm = [x["a_rmse"] for x in sw if x["split_id"] == sp and x["subject"] == sub and x["model"] == mod
                      and (mod != "BiGRU" or x["seed"] == seed)][0]
                lab = f"BiGRU seed {seed}" if mod == "BiGRU" else mod
                ax.plot(WL, P.mean(0), color=COLORS[mod], lw=1.1, label=f"{lab} (RMSE {rm * 1e3:.1f})")
                ax.fill_between(WL, P.min(0), P.max(0), color=COLORS[mod], alpha=0.12, lw=0)
            for k, t in enumerate(sorted({r["target_trial"] for r in rs})):
                hb = [r["target_hbi"] for r in rs if r["target_trial"] == t][0]
                a = data.a[(data.subject == sub) & (data.trial == t)][0]
                ax.plot(WL, a, color="#222222", lw=1.1, ls=(0, (4, 2)) if k == 0 else (0, (1, 1.5)),
                        label=f"정답 실측 t{t} (HbI {hb:.3f})")
            n_in = len({r["input_trial"] for r in rs})
            ax.set_title(f"{sub} · {sp} (입력 {n_in}개)", loc="left", fontweight="bold")
            ax.grid(alpha=0.25, lw=0.5); ax.legend(loc="lower left", fontsize=5.8)
        axes[i, 0].set_ylabel("흡광도 (A = -log$_{10}$R)")
    for j in range(ncol):   # 각 열에서 마지막으로 그려진 패널에 x축 라벨
        last = max(i for i in range(len(subs)) if axes[i, j].axison)
        axes[last, j].set_xlabel("파장 (nm)"); axes[last, j].tick_params(labelbottom=True)
    _title(fig, f"HbI 0.6 근처 복원 vs 정답 실측 (|HbI-0.6| <= {tol}, 피험자가 test였던 모든 split; "
                "선: 입력 평균, 음영: 입력 범위; RMSE x1e-3, 570 nm 제외; BiGRU는 split별 학습 loss 최소 seed)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fdir = out / "figures"; fdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fdir / "target06_reconstruction.png", dpi=300); plt.close(fig)
    return ("predictions.json: general_reconstruction (target |HbI-0.6|<=0.02 쌍) + "
            "target06_reconstruction.json: near_target_accuracy.subject_wise + training_summary.json (BiGRU seed 선택)")


def make_figures(out: Path, data) -> dict:
    _style()
    fdir = out / "figures"; fdir.mkdir(parents=True, exist_ok=True)
    gen = jload(out / "metrics" / "general_reconstruction.json")
    t06 = jload(out / "metrics" / "target06_reconstruction.json")
    pr = jload(out / "predictions" / "predictions.json")
    src = {}

    # model_rmse_comparison: 고유 피험자 10명 (여러 split 평균) + 평균
    mets = [("a_rmse", "흡광도 RMSE (570 nm 제외)", 1e3, "×10$^{-3}$"),
            ("r_rmse_pp", "반사율 RMSE", 1, "%p"), ("peak_rmse", "Carotenoid Peak RMSE", 1e3, "×10$^{-3}$")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    for ax, (m, lab, sc, unit) in zip(axes, mets):
        s = gen["summary"][m]
        subs = sorted(s["B2"]["per_unique_subject"])
        for sub in subs:
            ax.plot(range(3), [s[mod]["per_unique_subject"][sub] * sc for mod in MODELS], color="#cfcfcf", lw=0.7, zorder=1)
        for i, mod in enumerate(MODELS):
            v = [s[mod]["per_unique_subject"][sub] * sc for sub in subs]
            ax.scatter(np.full(len(v), i), v, s=12, color=COLORS[mod], alpha=0.7, zorder=2)
            ax.plot([i - 0.22, i + 0.22], [s[mod]["mean"] * sc] * 2, color="#222222", lw=1.6, zorder=3)
        ax.set_xticks(range(3), MODELS); ax.set_xlim(-0.5, 2.5)
        ax.set_ylabel(f"{lab} ({unit})"); ax.set_title(lab, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    _title(fig, "General HbI reconstruction — 점: 피험자(10명, test로 쓰인 split 평균), 가로선: split 평균")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(fdir / "model_rmse_comparison.png", dpi=300); plt.close(fig)
    src["model_rmse_comparison.png"] = "general_reconstruction.json: summary.{a_rmse,r_rmse_pp,peak_rmse}.<model>.{per_unique_subject,mean}"

    # spectral_reconstruction: 선정 기준 = split_00 각 test 피험자에서 HbI 최저 측정 -> 최고 측정 (오차와 무관)
    rows = [r for r in pr["general_reconstruction"] if r["split_id"] == "split_00"]
    subs = sorted({r["subject"] for r in rows})
    fig, axes = plt.subplots(1, len(subs), figsize=(10, 3.4), sharey=True)
    for ax, sub in zip(np.atleast_1d(axes), subs):
        rs = [r for r in rows if r["subject"] == sub]
        lo = min(rs, key=lambda r: r["input_hbi"])["input_hbi"]; hi = max(rs, key=lambda r: r["target_hbi"])["target_hbi"]
        r = next(r for r in rs if r["input_hbi"] == lo and r["target_hbi"] == hi)
        inp = data.a[(data.subject == sub) & (data.trial == r["input_trial"])][0]
        ax.plot(WL, inp, color="#9a9893", lw=1, ls=(0, (3, 2)), label=f"입력 (HbI {r['input_hbi']:.3f})")
        ax.plot(WL, r["actual"], color="#222222", lw=1.6, label=f"실측 정답 (HbI {r['target_hbi']:.3f})")
        for mod, key in (("B1", "B1"), ("B2", "B2"), ("BiGRU", "BiGRU_seed0")):
            ax.plot(WL, r["predictions"][key], color=COLORS[mod], lw=1.1, label=mod if mod != "BiGRU" else "BiGRU (seed 0)")
        ax.set_title(f"{sub}: trial {r['input_trial']} → {r['target_trial']}", loc="left", fontweight="bold")
        ax.set_xlabel("파장 (nm)"); ax.grid(alpha=0.25, lw=0.5)
    np.atleast_1d(axes)[0].set_ylabel("흡광도 (A = -log$_{10}$R)")
    np.atleast_1d(axes)[-1].legend(loc="upper right")
    _title(fig, "대표 복원 사례 (split_00, 피험자별 HbI 최저→최고 측정; 오차와 무관하게 선정)")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(fdir / "spectral_reconstruction.png", dpi=300); plt.close(fig)
    src["spectral_reconstruction.png"] = "predictions.json: general_reconstruction (split_00)"

    # wavelength_error
    w = gen["wavelength"]["per_model"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    for mod in MODELS:
        axes[0].plot(WL, np.array(w[mod]["rmse"]) * 1e3, color=COLORS[mod], lw=1.3, marker="o", ms=2.5, label=mod)
        axes[1].plot(WL, np.array(w[mod]["bias"]) * 1e3, color=COLORS[mod], lw=1.3, marker="o", ms=2.5, label=mod)
    axes[1].axhline(0, color="#9a9893", lw=0.7)
    for ax, t, yl in zip(axes, ("a  파장별 흡광도 RMSE", "b  파장별 흡광도 Bias (예측 - 실측)"),
                         ("RMSE (×10$^{-3}$ absorbance)", "Bias (×10$^{-3}$ absorbance)")):
        ax.axvline(570, color="#9a9893", lw=0.6, ls=(0, (3, 2)))
        ax.set_xlabel("파장 (nm)"); ax.set_ylabel(yl); ax.set_title(t, loc="left", fontweight="bold"); ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend()
    _title(fig, "Wavelength-wise error (General reconstruction; 570 nm = 목표 HbI로 고정되어 오차 0)")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(fdir / "wavelength_error.png", dpi=300); plt.close(fig)
    src["wavelength_error.png"] = "general_reconstruction.json: wavelength.per_model"

    # rmse_by_delta_hbi
    dh = pd.DataFrame(gen["rmse_by_delta_hbi"])
    bins = sorted(dh.dh_bin.unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
    for ax, d in zip(axes, ("increase", "decrease")):
        for mod in MODELS:
            x = dh[(dh.direction == d) & (dh.model == mod)].set_index("dh_bin").reindex(bins)
            ax.plot(range(len(bins)), x.a_rmse * 1e3, color=COLORS[mod], lw=1.3, marker="o", ms=4, label=mod)
        n = dh[(dh.direction == d) & (dh.model == "B2")].set_index("dh_bin").reindex(bins)
        ax.set_xticks(range(len(bins)), [f"{b}\n(n={int(v) if v == v else 0})" for b, v in zip(bins, n.n_pairs)])
        ax.set_xlabel("|ΔHbI| 구간 (n = pair 수, 모든 split 합)")
        ax.set_title("a  HbI 증가 (목표 > 입력)" if d == "increase" else "b  HbI 감소 (목표 < 입력)", loc="left", fontweight="bold")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("흡광도 RMSE (×10$^{-3}$)"); axes[0].legend()
    _title(fig, "RMSE by |ΔHbI| (General reconstruction)")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(fdir / "rmse_by_delta_hbi.png", dpi=300); plt.close(fig)
    src["rmse_by_delta_hbi.png"] = "general_reconstruction.json: rmse_by_delta_hbi"

    # target06_accuracy
    ns = t06["near_target_accuracy"]["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    for ax, (m, lab, sc) in zip(axes, (("a_rmse", "흡광도 RMSE (×10$^{-3}$)", 1e3), ("peak_rmse", "Peak RMSE (×10$^{-3}$)", 1e3))):
        for j, t in enumerate(TOLS):
            if not ns[t]["available"]:
                continue
            for i, mod in enumerate(MODELS):
                s = ns[t]["metrics"][m][mod]
                x = j * 4 + i
                v = [s["per_unique_subject"][k] * sc for k in sorted(s["per_unique_subject"])]
                ax.scatter(np.full(len(v), x), v, s=12, color=COLORS[mod], alpha=0.7)
                ax.plot([x - 0.3, x + 0.3], [s["mean"] * sc] * 2, color="#222222", lw=1.6)
        ax.set_xticks([1, 5], [f"±{TOLS[t]} ({ns[t]['n_unique_subjects']}명, 실측 {len(ns[t]['unique_gt_measurements'])}개,\n"
                               f"{ns[t]['n_splits_with_gt']} split, pair {ns[t]['n_pairs_total']}개)" for t in TOLS])
        ax.set_ylabel(lab); ax.grid(axis="y", alpha=0.25, lw=0.5)
    for mod in MODELS:
        axes[0].scatter([], [], color=COLORS[mod], label=mod)
    axes[0].legend(loc="upper left")
    axes[0].set_title("a  흡광도 RMSE", loc="left", fontweight="bold"); axes[1].set_title("b  Carotenoid Peak RMSE", loc="left", fontweight="bold")
    _title(fig, "Near-target accuracy (정답 = 0.6 근처 실측, 목표 HbI = 정답의 실제 HbI) — 점: 피험자, 가로선: split 평균")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(fdir / "target06_accuracy.png", dpi=300); plt.close(fig)
    src["target06_accuracy.png"] = "target06_reconstruction.json: near_target_accuracy.summary"

    src["target06_reconstruction.png"] = plot_target06_reconstruction(out, data, pr, t06)

    # target06_consistency
    cs = t06["fixed_target_consistency"]["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    order = ["Raw"] + MODELS
    for ax, (m, lab) in zip(axes, (("peak_sd", "Subject-wise Peak SD (×10$^{-3}$)"), ("spectral_sd", "Subject-wise Spectral SD (×10$^{-3}$)"))):
        subs = sorted(cs[m]["Raw"]["per_unique_subject"])
        for sub in subs:
            ax.plot(range(4), [cs[m][o]["per_unique_subject"][sub] * 1e3 for o in order], color="#cfcfcf", lw=0.7, zorder=1)
        for i, o in enumerate(order):
            v = [cs[m][o]["per_unique_subject"][sub] * 1e3 for sub in subs]
            ax.scatter(np.full(len(v), i), v, s=12, color=COLORS[o], alpha=0.75, zorder=2)
            ax.plot([i - 0.22, i + 0.22], [cs[m][o]["mean"] * 1e3] * 2, color="#222222", lw=1.6, zorder=3)
        ax.set_xticks(range(4), order); ax.set_ylabel(lab); ax.grid(axis="y", alpha=0.25, lw=0.5)
    axes[0].set_title("a  Carotenoid Peak SD", loc="left", fontweight="bold")
    axes[1].set_title("b  Spectral SD (570 nm 제외 평균)", loc="left", fontweight="bold")
    _title(fig, "HbI 0.6 고정 복원의 반복성 (개인 내 SD; 정답 기반 정확도가 아님) — 점: 피험자, 가로선: split 평균")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(fdir / "target06_consistency.png", dpi=300); plt.close(fig)
    src["target06_consistency.png"] = "target06_reconstruction.json: fixed_target_consistency.summary"
    return src


if __name__ == "__main__":
    main()
