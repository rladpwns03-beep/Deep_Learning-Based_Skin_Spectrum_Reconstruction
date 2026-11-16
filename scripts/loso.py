"""추가 검증 — Leave-One-Subject-Out (LOSO) Cross-Validation (B1, B2, BiGRU).

실행 (code 폴더, conda 환경 skin_spectra):
    python scripts/loso.py                          # results/holdout/loso/ 에 저장 (기존 결과가 있으면 중단)
    python scripts/loso.py --out-version loso       # results/loso/ 에 저장 (Hold-out 비교 기준은 계속 results/holdout 읽기 전용)
    python scripts/loso.py --run-id NAME            # <출력 폴더>/runs/NAME/ 에 별도 저장

원칙
  - Fold = 피험자 1명 Test / 나머지 9명 Train (10 fold). 기존 7:3 split 파일·모델·결과는 읽기만 한다.
  - 모델·학습 설정은 Phase 2와 같다 (B2 pooled within-subject slope, B1 bin 0.05 + nearest-bin fallback,
    BiGRU hidden 8 / 150 epoch / seed 0,1,2 / P1 All Pairs). 7:3 Test 결과로 바꾼 설정 없음.
  - 평가 지표·계산 함수는 Phase 3(evaluate.py)와 같다 (common/metrics.py).
  - 평가 단위 = 피험자. BiGRU 피험자 값 = seed별 지표의 평균. 요약 = 피험자 10명의 평균 ± SD (ddof=1).
  - 통계 검정(정확 Wilcoxon signed-rank, 부호 검정)은 n=10 기술적 참고값이며 다중비교 보정을 하지 않았다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import sys
import time
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]   # 프로젝트 루트 (scripts/ 의 상위)
sys.path.insert(0, str(ROOT))

from evaluate import COLORS, DH_BINS, METRICS, TOLS, UNITS, _style, _title, legacy, nan_to_none  # noqa: E402
from common import config as C                                              # noqa: E402
from common.data_loader import FREE, HBI_INDEX, WL, load_spectra           # noqa: E402
from common.metrics import (consistency, group_metrics, pair_metrics, peak,  # noqa: E402
                            wavelength_metrics)
from models.bigru import CFG as BIGRU_CFG, BiGRUModel                      # noqa: E402
from models.hb_index import BIN_EDGES, BinnedMeanModel                     # noqa: E402
from models.linear import LinearModel                                       # noqa: E402
from train_models import DIVERGENCE_FACTOR, contracts, roundtrip           # noqa: E402

VERSION = "holdout"                      # Hold-out 비교 기준 (읽기 전용)
BASE = ROOT / "results" / VERSION
OUT_VERSION = VERSION               # 결과 버전 태그 (--out-version)
MODELS = ["B1", "B2", "BiGRU"]
PAIRS_CMP = [("B1", "B2"), ("B1", "BiGRU"), ("B2", "BiGRU")]


def jdump(path: Path, obj, indent=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nan_to_none(obj), ensure_ascii=False, indent=indent, allow_nan=False), encoding="utf-8")


def jload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha_files(paths) -> dict:
    return {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


# ---------------------------------------------------------------------------
# 기술 통계 (n = 10 피험자, 정확 분포 열거)
# ---------------------------------------------------------------------------
def wilcoxon_exact(d) -> dict:
    d = np.asarray([x for x in d if x != 0], float)
    n = len(d)
    if n == 0:
        return dict(n_nonzero=0, w_plus=None, p_two_sided=None)
    a = np.abs(d)
    order = a.argsort()
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    for v in np.unique(a):                                    # 동률 평균 순위
        ranks[a == v] = ranks[a == v].mean()
    w = float(ranks[d > 0].sum())
    mu = ranks.sum() / 2
    dist = np.array([sum(r for r, s in zip(ranks, signs) if s) for signs in itertools.product([0, 1], repeat=n)])
    p = float(np.mean(np.abs(dist - mu) >= abs(w - mu) - 1e-12))
    return dict(n_nonzero=n, w_plus=w, p_two_sided=p)


def sign_test(d) -> dict:
    d = [x for x in d if x != 0]
    n, k = len(d), sum(x > 0 for x in d)
    if n == 0:
        return dict(n_nonzero=0, n_positive=0, p_two_sided=None)
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return dict(n_nonzero=n, n_positive=int(k), p_two_sided=float(min(1.0, 2 * tail)))


def subj_summary(df: pd.DataFrame, metric: str, model: str) -> dict | None:
    d = df[df.model == model]
    if d.empty or d[metric].isna().all():
        return None
    per_seed = d.groupby(["subject", "seed"])[metric].mean()
    subj = per_seed.groupby("subject").mean()                 # BiGRU: seed 평균 = 피험자 대표값
    rec = dict(mean=float(subj.mean()), sd=float(subj.std(ddof=1)) if subj.size > 1 else None,
               n_subjects=int(subj.size), per_subject={k: float(v) for k, v in subj.items()},
               min=float(subj.min()), max=float(subj.max()), sd_definition="피험자 값의 SD (ddof=1)")
    if d.seed.nunique() > 1:
        ps = per_seed.unstack("seed")
        rec["per_seed_mean"] = {f"seed_{int(c)}": float(ps[c].mean()) for c in ps.columns}
        rec["seed_sd_of_means"] = float(ps.mean().std(ddof=1))
        rec["seed_sd_within_subject_mean"] = float(ps.std(axis=1, ddof=1).mean())
        rec["per_subject_per_seed"] = {s: {f"seed_{int(c)}": float(ps.loc[s, c]) for c in ps.columns} for s in ps.index}
    return rec


def paired(s1: dict, s2: dict, name1: str, name2: str) -> dict | None:
    if s1 is None or s2 is None:
        return None
    subs = sorted(set(s1["per_subject"]) & set(s2["per_subject"]))
    a = np.array([s1["per_subject"][s] for s in subs]); b = np.array([s2["per_subject"][s] for s in subs])
    d = a - b                                                  # 양수 = name2 오차가 더 작음
    return dict(comparison=f"{name1} vs {name2}", n_subjects=len(subs),
                mean_1=float(a.mean()), mean_2=float(b.mean()),
                diff_mean=float(d.mean()), diff_sd=float(d.std(ddof=1)) if len(d) > 1 else None,
                improvement_pct_of_2_vs_1=float(100 * (a.mean() - b.mean()) / a.mean()),
                per_subject_improvement_pct={s: float(100 * x / y) for s, x, y in zip(subs, d, a)},
                n_subjects_2_better=int((d > 0).sum()),
                wilcoxon_exact=wilcoxon_exact(d) if len(d) >= 2 else None, sign_test=sign_test(d) if len(d) >= 2 else None,
                note="diff = metric_1 - metric_2 (양수 = 2가 개선). p값은 n이 작은 기술적 참고값, 다중비교 보정 없음")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out-version", default=None, help="지정하면 results/<out-version>/ 에 저장 (기본: results/holdout/loso)")
    args = ap.parse_args()
    global OUT_VERSION
    OUT_VERSION = args.out_version or VERSION
    root_out = ROOT / "results" / args.out_version if args.out_version else BASE / "loso"
    out = root_out / "runs" / args.run_id if args.run_id else root_out
    if out.exists() and any(p.is_file() for p in out.rglob("*") if "runs" not in p.relative_to(out).parts):
        sys.exit(f"[중단] {out} 에 이미 결과가 있음. 덮어쓰지 않음. 새 실행은 --run-id 지정.")
    run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    t0 = time.time()
    torch.set_num_threads(1)
    prev = [p for p in BASE.rglob("*") if p.is_file() and "loso" not in p.relative_to(BASE).parts]
    hash_before = sha_files(prev)

    data = load_spectra()
    subjects = data.subjects
    # ======================= Fold 구성 =========================================
    folds = []
    for s in subjects:
        train = [x for x in subjects if x != s]
        ix = np.flatnonzero(data.subject == s)
        tp = [(int(data.trial[i]), int(data.trial[j])) for i in ix for j in ix if i != j]
        folds.append(dict(fold_id=f"fold_{s}", test=[s], train=train, n_train_measurements=int(np.isin(data.subject, train).sum()),
                          n_test_measurements=int(len(ix)), test_pairs=[dict(input_trial=a, target_trial=b) for a, b in tp]))
    fold_checks = dict(
        n_folds=len(folds) == 10,
        one_test_subject_each=all(len(f["test"]) == 1 for f in folds),
        each_subject_tested_once=sorted(s for f in folds for s in f["test"]) == subjects,
        no_train_test_overlap=all(not set(f["train"]) & set(f["test"]) for f in folds),
        train_has_9_subjects=all(len(f["train"]) == 9 for f in folds),
        pairs_90_per_fold=all(len(f["test_pairs"]) == 90 for f in folds))
    jdump(out / "config" / "loso_folds.json", dict(
        experiment_version=OUT_VERSION, holdout_reference_version=VERSION, run_id=run_id, scheme="Leave-One-Subject-Out", n_folds=len(folds), folds=folds,
        checks=fold_checks, note="기존 7:3 split 파일(data/splits)과 별도. 모든 모델이 같은 fold·같은 Test Pair 사용"))
    config = dict(
        experiment_version=OUT_VERSION, holdout_reference_version=VERSION, run_id=run_id, phase="Additional validation - LOSO", models=MODELS,
        folds="config/loso_folds.json", training_pairs="P1_all_pairs (같은 피험자 모든 i != j)",
        b2=dict(method="pooled within-subject linear slope, b(570)=1, fold Train 9명만"),
        b1=dict(bin_width=C.B1_BIN_WIDTH, bin_edges=BIN_EDGES.tolist(), empty_bin_policy="nearest valid bin (tie -> lower)"),
        bigru={k: (list(v) if isinstance(v, tuple) else v) for k, v in BIGRU_CFG.items()}, seeds=list(C.BIGRU_SEEDS),
        bigru_note="hidden 8 bidirectional GRU, Adam, full batch, zero-init head; fold마다 새로 초기화 (가중치 재사용 없음)",
        target_hbi=C.TARGET_HBI, near_tolerances=TOLS, dh_bins=DH_BINS, device="cpu", torch_threads=1,
        hyperparameters_changed_from_holdout=False, units=UNITS,
        aggregation="pair -> subject(group 지표, 90 pair x 30 파장) -> 피험자 10명 평균 ± SD. BiGRU 피험자 값 = seed별 지표의 평균")
    jdump(out / "config" / "loso_config.json", config)

    row = {(s, t): i for i, (s, t) in enumerate(zip(data.subject, data.trial))}
    hist, b1_info, b2_info, train_checks, models_all = [], [], [], [], {}
    G, WLr, DH, NR, NRnull, CS, REF, PH, b1c, chk570, same_inp, pred_g, pred_f = ([] for _ in range(13))
    test_hash = {}

    for f in folds:
        fid, s = f["fold_id"], f["test"][0]
        tr, te = data.select(f["train"]), data.select(f["test"])
        assert s not in set(tr.subject.tolist()) and set(te.subject.tolist()) == {s}
        # ---------------- 학습 ----------------
        b2 = LinearModel().fit(tr)
        b1 = BinnedMeanModel().fit(tr)
        tab = b1.bin_table
        b1.reset_counters(); c2 = contracts(b1, tr); b1.reset_counters()
        tc = dict(fold_id=fid,
                  B2=dict(train_subjects_ok=b2.train_subjects == f["train"], slope_570_is_1=bool(b2.slope[HBI_INDEX] == 1.0),
                          slope_finite=bool(np.isfinite(b2.slope).all()),
                          save_load_identical=roundtrip(b2, out / "models" / "B2" / f"{fid}.json", tr), contracts=contracts(b2, tr)),
                  B1=dict(train_subjects_ok=b1.train_subjects == f["train"],
                          bin_counts_sum_to_train=int(tab.n_measurements.sum()) == len(tr),
                          save_load_identical=roundtrip(b1, out / "models" / "B1" / f"{fid}.json", tr), contracts=c2),
                  BiGRU={})
        b1.reset_counters()
        b2_info.append(dict(fold_id=fid, slope=b2.slope.tolist()))
        b1_info.append(dict(fold_id=fid, bin_table=tab.to_dict(orient="records"),
                            n_empty_bins=int((tab.n_measurements == 0).sum()),
                            empty_bins=tab["bin"][tab.n_measurements == 0].astype(int).tolist()))
        models = {("B2", -1): b2, ("B1", -1): b1}
        for seed in C.BIGRU_SEEDS:
            t = time.time()
            m = BiGRUModel(seed=seed).fit(tr)               # fold마다 새 객체·새 초기화
            secs = time.time() - t
            H = np.array(m.history)
            finite = bool(np.isfinite(H).all())
            tc["BiGRU"][f"seed_{seed}"] = dict(
                epochs_recorded=len(H) == C.BIGRU["epochs"], finite=finite,
                not_diverged=finite and bool(H[:, 1].max() <= DIVERGENCE_FACTOR * H[0, 1]),
                train_subjects_ok=m.train_subjects == f["train"],
                b2_backbone_equals_fold_B2=bool(np.array_equal(m.b2, b2.slope)),
                standardization_from_train=bool(np.allclose(m.net.mean.numpy(), tr.a.mean(0).astype(np.float32))),
                save_load_identical=roundtrip(m, out / "models" / "BiGRU" / f"{fid}_seed_{seed}.pt", tr),
                contracts=contracts(m, tr))
            hist.append(dict(fold_id=fid, test_subject=s, seed=seed, n_train_pairs=810, epochs=len(H),
                             train_seconds=round(secs, 2), initial_loss=float(H[0, 1]), final_loss=float(H[-1, 1]),
                             loss=H[:, 1].tolist(), mse=H[:, 2].tolist(), grad_norm=H[:, 3].tolist()))
            models[("BiGRU", seed)] = m
            models_all[(fid, seed)] = m
            print(f"[{fid}] BiGRU seed {seed}: loss {H[0, 1]:.3e} -> {H[-1, 1]:.3e} ({secs:.1f}s)", flush=True)
        train_checks.append(tc)

        # ---------------- Experiment A ----------------
        ii = np.array([row[(s, p["input_trial"])] for p in f["test_pairs"]])
        jj = np.array([row[(s, p["target_trial"])] for p in f["test_pairs"]])
        X, Y, T = data.a[ii], data.a[jj], data.hbi[jj]
        dh = T - data.hbi[ii]
        test_hash[fid] = hashlib.sha256(np.concatenate([X, Y, T[:, None]], 1).tobytes()).hexdigest()
        preds, h570 = {}, set()
        for (mn, seed), m in models.items():
            if mn == "B1":
                m.reset_counters()
            P = m.predict(X, T)
            if mn == "B1":
                b1c.append(dict(fold_id=fid, experiment="general", **m.counters()))
            preds[(mn, seed)] = P
            chk570.append(bool(np.array_equal(P[:, HBI_INDEX], T)))
            h570.add(hashlib.sha256(np.ascontiguousarray(P[:, HBI_INDEX]).tobytes()).hexdigest())
            name = dict(model=mn, seed=seed, fold_id=fid, subject=s)
            PH.append(dict(**name, experiment="general", n_reflectance_over_100=int((P < 0).sum()), n_nonfinite=int((~np.isfinite(P)).sum())))
            G.append(dict(**name, **group_metrics(P, Y), mean_pair_a_rmse=float(pair_metrics(P, Y)["a_rmse"].mean())))
            WLr.append(dict(**name, **wavelength_metrics(P, Y)))
            for lo, hi in zip(DH_BINS[:-1], DH_BINS[1:]):
                inb = (np.abs(dh) >= lo) & (np.abs(dh) < hi)
                for direction, dm in (("increase", dh > 0), ("decrease", dh < 0), ("both", np.ones_like(inb))):
                    mm = inb & dm
                    if mm.any():
                        DH.append(dict(**name, dh_bin=f"{lo:.3f}-{hi:.3f}", direction=direction, **group_metrics(P[mm], Y[mm])))
            # B-1 near-target accuracy
            for tname, tol in TOLS.items():
                sel = np.abs(T - C.TARGET_HBI) <= tol
                if sel.any():
                    gt = sorted({int(f["test_pairs"][k]["target_trial"]) for k in np.flatnonzero(sel)})
                    NR.append(dict(**name, tolerance=tname, tol=tol, n_gt_measurements=len(gt), gt_trials=gt,
                                   gt_hbi=sorted({float(x) for x in T[sel]}), **group_metrics(P[sel], Y[sel])))
        same_inp.append(len(h570) == 1)
        for tname, tol in TOLS.items():
            if not (np.abs(te.hbi - C.TARGET_HBI) <= tol).any():
                NRnull.append(dict(fold_id=fid, subject=s, tolerance=tname, tol=tol, available=False, metrics=None,
                                   reason=f"|HbI-0.6| <= {tol} 인 실측 없음 (HbI 범위 {te.hbi.min():.3f}-{te.hbi.max():.3f})"))
        for n, p in enumerate(f["test_pairs"]):
            pred_g.append(dict(fold_id=fid, subject=s, input_trial=p["input_trial"], target_trial=p["target_trial"],
                               input_hbi=float(data.hbi[ii[n]]), target_hbi=float(T[n]), actual=Y[n].tolist(),
                               predictions={(mn if sd < 0 else f"{mn}_seed{sd}"): v[n].tolist() for (mn, sd), v in preds.items()}))

        # ---------------- Experiment B-2 / B-3 ----------------
        order = np.argsort(te.trial)
        A0, trials, hs = te.a[order], te.trial[order], te.hbi[order]
        fixed = {("Raw", -1): A0}
        for (mn, seed), m in models.items():
            if mn == "B1":
                m.reset_counters()
            fixed[(mn, seed)] = m.predict(A0, C.TARGET_HBI)
            if mn == "B1":
                b1c.append(dict(fold_id=fid, experiment="fixed_target_0.6", **m.counters()))
        for (mn, seed), F in fixed.items():
            CS.append(dict(model=mn, seed=seed, fold_id=fid, subject=s, **consistency(F),
                           target_hbi_error_max=None if mn == "Raw" else float(np.abs(F[:, HBI_INDEX] - C.TARGET_HBI).max()),
                           n_reflectance_over_100=int((F < 0).sum()), n_nonfinite=int((~np.isfinite(F)).sum()),
                           target_in_subject_range=bool(hs.min() <= C.TARGET_HBI <= hs.max()),
                           mean_spectrum=F.mean(0).tolist()))
            if mn != "Raw":
                PH.append(dict(model=mn, seed=seed, fold_id=fid, subject=s, experiment="fixed_target_0.6",
                               n_reflectance_over_100=int((F < 0).sum()), n_nonfinite=int((~np.isfinite(F)).sum())))
        near = np.flatnonzero(np.abs(hs - C.TARGET_HBI) <= C.NEAR_TOL_SUPP)
        if len(near) == 0:
            REF.append(dict(fold_id=fid, subject=s, available=False,
                            reason=f"|HbI-0.6| <= {C.NEAR_TOL_SUPP} 인 실측 없음 (범위 {hs.min():.3f}-{hs.max():.3f})"))
        for kk in near:
            others = np.array([i for i in range(len(hs)) if i != kk])
            for (mn, seed), F in fixed.items():
                if mn == "Raw":
                    continue
                ref = np.repeat(A0[kk][None], len(others), 0)
                REF.append(dict(fold_id=fid, subject=s, available=True, model=mn, seed=seed, reference_trial=int(trials[kk]),
                                reference_hbi=float(hs[kk]), hbi_minus_06=float(hs[kk] - C.TARGET_HBI),
                                within_main_tol=bool(abs(hs[kk] - C.TARGET_HBI) <= C.NEAR_TOL_MAIN),
                                **group_metrics(F[others], ref), peak_difference_mean=float(np.mean(peak(F[others]) - peak(ref)))))
        for n in range(len(A0)):
            pred_f.append(dict(fold_id=fid, subject=s, input_trial=int(trials[n]), input_hbi=float(hs[n]), target_hbi=C.TARGET_HBI,
                               actual=None, actual_note="정확히 HbI 0.6인 실측 스펙트럼 없음",
                               predictions={(mn if sd < 0 else f"{mn}_seed{sd}"): v[n].tolist() for (mn, sd), v in fixed.items() if mn != "Raw"}))

    jdump(out / "logs" / "loso_training_history.json", dict(
        experiment_version=OUT_VERSION, holdout_reference_version=VERSION, run_id=run_id, model="BiGRU (B2 slope, B1 bin table 포함)", bigru_config=config["bigru"],
        runs=hist, b2=b2_info, b1=b1_info,
        bigru_final_loss=dict(mean=float(np.mean([h["final_loss"] for h in hist])), sd=float(np.std([h["final_loss"] for h in hist], ddof=1))),
        note="Training loss는 학습 정상 작동의 참고 정보이며 Test 성능을 보장하지 않음"), indent=None)

    G, WLd, DH, NR, CS, PH = map(pd.DataFrame, (G, WLr, DH, NR, CS, PH))
    meta = dict(experiment_version=OUT_VERSION, holdout_reference_version=VERSION, run_id=run_id, phase="Additional validation - LOSO", units=UNITS,
                aggregation=config["aggregation"], n_folds=len(folds))

    # ======================= Experiment A 집계 =================================
    gsum = {m: {mod: subj_summary(G, m, mod) for mod in MODELS} for m in METRICS + ["mean_pair_a_rmse"]}
    wl = {}
    for mod in MODELS:
        d = WLd[WLd.model == mod]
        per_subj = {q: np.mean(np.vstack([np.mean(np.vstack(g[q].tolist()), 0) for _, g in d.groupby("subject")]), 0).tolist()
                    for q in ("rmse", "mae", "bias")}
        wl[mod] = per_subj
    dh_rows = []
    for (b, dr, mod), x in DH.groupby(["dh_bin", "direction", "model"]):
        sv = x.groupby(["subject", "seed"])[["a_rmse", "peak_rmse", "r_rmse_pp"]].mean().groupby("subject").mean()
        dh_rows.append(dict(dh_bin=b, direction=dr, model=mod, a_rmse=float(sv.a_rmse.mean()), peak_rmse=float(sv.peak_rmse.mean()),
                            r_rmse_pp=float(sv.r_rmse_pp.mean()),
                            a_rmse_sd=float(sv.a_rmse.std(ddof=1)) if len(sv) > 1 else None,
                            per_subject_a_rmse={k: float(v) for k, v in sv.a_rmse.items()},
                            n_subjects=int(len(sv)), n_pairs=int(x[x.seed.isin([-1, 0])].n_pairs.sum())))
    ranks = {}
    for m in ("a_rmse", "peak_rmse"):
        tab = pd.DataFrame({mod: gsum[m][mod]["per_subject"] for mod in MODELS})
        rk = tab.rank(axis=1, method="min")
        ranks[m] = dict(per_subject={s: {mod: int(rk.loc[s, mod]) for mod in MODELS} for s in rk.index},
                        n_rank1={mod: int((rk[mod] == 1).sum()) for mod in MODELS},
                        mean_rank={mod: float(rk[mod].mean()) for mod in MODELS})
    general = dict(**meta, model="B1, B2, BiGRU", experiment="Experiment A - General HbI Reconstruction (LOSO)",
                   n_pairs_per_subject=90, summary=gsum, subject_wise=G.to_dict(orient="records"),
                   wavelength={"wavelengths_nm": WL.tolist(), "excluded_from_rmse_nm": [570], "per_model": wl,
                               "note": "피험자별 파장 지표 (BiGRU seed 평균) -> 피험자 10명 평균"},
                   rmse_by_delta_hbi=dh_rows, model_rank_per_subject=ranks,
                   b1_fallback_counters=[c for c in b1c if c["experiment"] == "general"],
                   peak_bias_note="B1·B2는 피험자 안의 i->j와 j->i 오차가 정확히 상쇄되어 피험자별 Peak Bias가 구조적으로 0. "
                                  "Bias는 BiGRU에서만 해석 가능", status="completed")
    jdump(out / "metrics" / "loso_general_reconstruction.json", general)

    # ======================= Experiment B 집계 =================================
    near_sum = {}
    for tname, tol in TOLS.items():
        d = NR[NR.tolerance == tname]
        b2 = d[d.model == "B2"]
        near_sum[tname] = dict(tol=tol, available=not d.empty, n_subjects_evaluable=int(len(b2)),
                               evaluable_subjects=sorted(b2.subject.tolist()),
                               n_gt_measurements_total=int(b2.n_gt_measurements.sum()), n_pairs_total=int(b2.n_pairs.sum()),
                               gt_measurements=sorted(f"{r.subject}-t{t}" for r in b2.itertuples() for t in r.gt_trials),
                               per_subject_counts={r.subject: dict(n_gt_measurements=int(r.n_gt_measurements), n_pairs=int(r.n_pairs))
                                                   for r in b2.itertuples()},
                               unavailable=[x for x in NRnull if x["tolerance"] == tname],
                               metrics={m: {mod: subj_summary(d, m, mod) for mod in MODELS} for m in METRICS})
    cons = {met: {mod: subj_summary(CS, met, mod) for mod in ["Raw"] + MODELS} for met in ("peak_sd", "spectral_sd")}
    for met in cons:
        for flag, lab in ((True, "target_in_range"), (False, "target_out_of_range")):
            d = CS[CS.target_in_subject_range == flag]
            cons[met][lab] = {mod: (float(d[d.model == mod].groupby("subject")[met].mean().mean()) if len(d) else None)
                              for mod in ["Raw"] + MODELS} | {"n_subjects": int(len(d[d.model == "Raw"]))}
    # 복원 후 피험자 고유 특성 유지 (참고): 피험자 평균 스펙트럼의 피험자 간 편차 패턴과 Peak 순위
    raw_mean = np.vstack([np.array(r) for r in CS[CS.model == "Raw"].sort_values("subject").mean_spectrum])
    retention = {}
    for mod in MODELS:
        d = CS[CS.model == mod]
        rest = np.vstack([np.mean(np.vstack(g.mean_spectrum.tolist()), 0) for _, g in d.sort_values("subject").groupby("subject")])
        dr, dm = raw_mean - raw_mean.mean(0), rest - rest.mean(0)
        pr, pm = peak(raw_mean), peak(rest)
        rr, rm = pd.Series(pr).rank(), pd.Series(pm).rank()
        retention[mod] = dict(
            between_subject_deviation_corr=float(np.corrcoef(dr[:, FREE].ravel(), dm[:, FREE].ravel())[0, 1]),
            peak_rank_spearman=float(np.corrcoef(rr, rm)[0, 1]),
            mean_abs_peak_shift=float(np.mean(np.abs(pm - pr))),
            per_subject_peak_raw_mean={s: float(v) for s, v in zip(subjects, pr)},
            per_subject_peak_restored_mean={s: float(v) for s, v in zip(subjects, pm)})
    retention["definition"] = ("피험자별 10개 측정의 평균 스펙트럼 기준. between_subject_deviation_corr = (피험자 평균 - 10명 평균) 패턴의 "
                               "Raw vs 복원 상관 (570 nm 제외), peak_rank_spearman = 피험자 평균 Peak 순위 상관. "
                               "HbI 차이는 의도적으로 제거되므로 1이 아니어도 오류가 아님 - 참고 지표")
    target06 = dict(
        **meta, model="Raw, B1, B2, BiGRU", experiment="Experiment B - Target HbI = 0.6 Reconstruction (LOSO)",
        near_target_accuracy=dict(summary=near_sum, subject_wise=NR.to_dict(orient="records"),
                                  ground_truth="0.6 근처 실측, 목표 HbI = 정답 측정의 실제 HbI"),
        fixed_target_consistency=dict(
            summary=cons, subject_wise=CS.drop(columns="mean_spectrum").to_dict(orient="records"),
            target_hbi_error_max=float(CS[CS.model != "Raw"].target_hbi_error_max.max()),
            n_reflectance_over_100_total=int(CS[CS.model != "Raw"].n_reflectance_over_100.sum()),
            n_nonfinite_total=int(CS.n_nonfinite.sum()), subject_characteristic_retention=retention,
            ground_truth="없음 - 반복성(개인 내 변동) 지표이며 정확도가 아님",
            b1_fallback_counters=[c for c in b1c if c["experiment"] != "general"]),
        near06_reference_comparison=dict(
            rows=REF, n_subjects_unavailable=int(sum(not r["available"] for r in REF)),
            summary={mod: (lambda d: dict(a_rmse_mean=float(d.groupby(["subject", "seed"]).a_rmse.mean().groupby("subject").mean().mean()),
                                          n_subjects=int(d.subject.nunique())) if len(d) else None)(
                pd.DataFrame([r for r in REF if r["available"] and r["model"] == mod])) for mod in MODELS},
            note="정확히 0.6으로 복원한 스펙트럼 vs 0.6 근처 실측 (|HbI-0.6| <= 0.03). 목표 HbI 불일치(hbi_minus_06)에 의한 오차가 "
                 "포함되므로 정확한 복원 오차가 아닌 참고값"),
        status="completed")
    jdump(out / "metrics" / "loso_target06_reconstruction.json", target06)

    # ======================= 비교·통계 =========================================
    ho_g = jload(BASE / "metrics" / "general_reconstruction.json")["summary"]
    ho_t = jload(BASE / "metrics" / "target06_reconstruction.json")
    comps = {m: {f"{a}_vs_{b}": paired(gsum[m][a], gsum[m][b], a, b) for a, b in PAIRS_CMP}
             for m in ("a_rmse", "r_rmse_pp", "peak_rmse", "peak_mae")}
    comps["abs_peak_bias"] = "B1·B2의 Peak Bias는 구조적으로 0이므로 Bias 비교는 하지 않음 (BiGRU Bias만 보고)"
    near_comps = {t: {m: {f"{a}_vs_{b}": paired(near_sum[t]["metrics"][m][a], near_sum[t]["metrics"][m][b], a, b) for a, b in PAIRS_CMP}
                      for m in ("a_rmse", "peak_rmse")} for t in TOLS}
    cons_comps = {met: {f"{a}_vs_{b}": paired(cons[met][a], cons[met][b], a, b) for a, b in [("Raw", "B2"), ("Raw", "B1"), ("Raw", "BiGRU")] + PAIRS_CMP}
                  for met in cons}

    def ho_vs(ho_sum, lo_sum):
        if ho_sum is None or lo_sum is None:
            return None
        hs = ho_sum["per_unique_subject"]; ls = lo_sum["per_subject"]
        subs = sorted(set(hs) & set(ls))
        d = np.array([ls[s] - hs[s] for s in subs])
        return dict(holdout_subject_mean=float(np.mean([hs[s] for s in subs])), holdout_split_mean=ho_sum.get("mean"),
                    loso_subject_mean=float(np.mean([ls[s] for s in subs])), diff_loso_minus_holdout=float(d.mean()),
                    diff_pct=float(100 * d.mean() / np.mean([hs[s] for s in subs])),
                    n_subjects_loso_lower=int((d < 0).sum()), n_subjects=len(subs),
                    per_subject={s: dict(holdout=hs[s], loso=ls[s]) for s in subs})
    ho_cmp = dict(
        general={m: {mod: ho_vs(ho_g[m][mod], gsum[m][mod]) for mod in MODELS} for m in ("a_rmse", "r_rmse_pp", "peak_rmse", "peak_mae")},
        near_main={m: {mod: ho_vs(ho_t["near_target_accuracy"]["summary"]["main"]["metrics"][m][mod], near_sum["main"]["metrics"][m][mod])
                       for mod in MODELS} for m in ("a_rmse", "peak_rmse")},
        consistency={met: {mod: ho_vs(ho_t["fixed_target_consistency"]["summary"][met][mod], cons[met][mod]) for mod in ["Raw"] + MODELS}
                     for met in cons},
        bigru_improvement_pct={sch: {f"vs_{b}": {m: (100 * (src[m][b] - src[m]["BiGRU"]) / src[m][b]) for m in ("a_rmse", "peak_rmse")}
                                     for b in ("B1", "B2")}
                               for sch, src in (("holdout_subject_mean", {m: {mod: float(np.mean(list(ho_g[m][mod]["per_unique_subject"].values())))
                                                                              for mod in MODELS} for m in ("a_rmse", "peak_rmse")}),
                                                ("loso", {m: {mod: gsum[m][mod]["mean"] for mod in MODELS} for m in ("a_rmse", "peak_rmse")}))},
        caution="7:3 Hold-out은 Train 7명, LOSO는 Train 9명. 차이를 분할 방식만의 효과로 단정할 수 없음 (Train 피험자 수·구성, "
                "평가 반복 횟수(Hold-out은 피험자당 2-4회 평균)가 함께 다름). Hold-out 값은 results/holdout/metrics (Phase 3) 읽기 전용",
        holdout_reference="per_unique_subject = 피험자가 Test로 쓰인 split들의 평균")
    comparison = dict(
        **meta, model="B1, B2, BiGRU", experiment="LOSO model comparison",
        general={m: {mod: {k: gsum[m][mod].get(k) for k in ("mean", "sd", "min", "max", "per_seed_mean", "seed_sd_of_means",
                                                            "seed_sd_within_subject_mean")} for mod in MODELS} for m in METRICS},
        subject_wise={m: {mod: gsum[m][mod]["per_subject"] for mod in MODELS} for m in ("a_rmse", "r_rmse_pp", "peak_rmse", "peak_bias")},
        paired_comparisons_general=comps, paired_comparisons_near_target=near_comps, paired_comparisons_consistency=cons_comps,
        model_rank_per_subject=ranks, holdout_vs_loso=ho_cmp,
        statistics_note="피험자 10명(근접 정답은 3명/8명)의 대응 비교. 정확 Wilcoxon signed-rank와 부호 검정 p값은 기술적 참고이며 "
                        "다중비교 보정 없음 -> 유의성 주장 근거로 쓰지 않음", status="completed")
    jdump(out / "metrics" / "loso_model_comparison.json", comparison)
    jdump(out / "predictions" / "loso_predictions.json", dict(
        **meta, model="B1, B2, BiGRU (seed별)", wavelengths_nm=WL.tolist(), unit="absorbance",
        general_reconstruction=pred_g, fixed_target_0p6=pred_f, status="completed"), indent=None)

    # ======================= 그림 =============================================
    fig_ok, fig_src = make_figures(out)

    # ======================= 검증 =============================================
    hash_after = sha_files(prev)
    # 지표 계산 일치 (a) 기존 7:3 모델(split_00 B2)을 이 스크립트의 평가 경로로 S002에 적용 -> Phase 3 저장값과 비교
    ph3_rows = jload(BASE / "metrics" / "general_reconstruction.json")["subject_wise"]
    ref_row = next(r for r in ph3_rows if r["split_id"] == "split_00" and r["model"] == "B2" and r["subject"] == "S002")
    f2 = next(f for f in folds if f["test"] == ["S002"])
    ii = np.array([row[("S002", p["input_trial"])] for p in f2["test_pairs"]]); jj = np.array([row[("S002", p["target_trial"])] for p in f2["test_pairs"]])
    gm = group_metrics(LinearModel.load(BASE / "models" / "B2" / "split_00.json").predict(data.a[ii], data.hbi[jj]), data.a[jj])
    same_calc = all(np.isclose(gm[k], ref_row[k], rtol=1e-12, atol=1e-15) for k in METRICS)
    ph3_pairs = pd.read_csv(ROOT / "data" / "splits" / "rep00" / "exp1_test_pairs.csv")
    same_pairs = set(zip(ph3_pairs[ph3_pairs.subject == "S002"].input_trial, ph3_pairs[ph3_pairs.subject == "S002"].target_trial)) == \
        {(p["input_trial"], p["target_trial"]) for p in f2["test_pairs"]}
    # (b) 직접 식(ablation.py 방식)으로 재계산
    xc = []
    for r in G[(G.fold_id == "fold_S001") & G.seed.isin([-1, 0])].itertuples():
        key = r.model if r.seed < 0 else f"{r.model}_seed{r.seed}"
        P = np.array([x["predictions"][key] for x in pred_g if x["fold_id"] == "fold_S001"])
        Yt = np.array([x["actual"] for x in pred_g if x["fold_id"] == "fold_S001"])
        ae, re, pe = P - Yt, 100 * (10 ** -P - 10 ** -Yt), legacy.peak(P) - legacy.peak(Yt)
        xc.append(bool(np.isclose(np.sqrt(np.mean(ae[:, FREE] ** 2)), r.a_rmse, rtol=1e-12) and
                       np.isclose(np.sqrt(np.mean(re[:, FREE] ** 2)), r.r_rmse_pp, rtol=1e-12) and
                       np.isclose(np.sqrt(np.mean(pe ** 2)), r.peak_rmse, rtol=1e-12)))
    # fold 독립성: 같은 seed의 fold 간 최종 가중치가 모두 다르고, 각 모델의 Train 피험자 = 그 fold Train
    indep = []
    for seed in C.BIGRU_SEEDS:
        fids = [f["fold_id"] for f in folds]
        for a, b in itertools.combinations(fids, 2):
            sa, sb = models_all[(a, seed)].net.state_dict(), models_all[(b, seed)].net.state_dict()
            indep.append(not all(torch.equal(sa[k], sb[k]) for k in sa))
    flat = lambda d: all(flat(v) for v in d.values()) if isinstance(d, dict) else (bool(d) if isinstance(d, (bool, np.bool_)) else True)
    contr_ok = all(flat({k: v for k, v in tc[m].items() if k != "contracts"}) and
                   all(v for k, v in tc[m]["contracts"].items() if isinstance(v, bool))
                   for tc in train_checks for m in ("B1", "B2")) and \
        all(flat({k: v for k, v in x.items() if k != "contracts"}) and all(v for k, v in x["contracts"].items() if isinstance(v, bool))
            for tc in train_checks for x in tc["BiGRU"].values())
    vr = dict(
        **{k: meta[k] for k in ("experiment_version", "run_id", "phase")},
        checks={
            "1_each_subject_tested_once": dict(passed=fold_checks["each_subject_tested_once"] and fold_checks["one_test_subject_each"]
                                               and fold_checks["n_folds"], fold_checks=fold_checks),
            "2_no_train_test_leakage": dict(passed=fold_checks["no_train_test_overlap"] and
                                            all(tc[m]["train_subjects_ok"] for tc in train_checks for m in ("B1", "B2")) and
                                            all(x["train_subjects_ok"] and x["standardization_from_train"] for tc in train_checks for x in tc["BiGRU"].values()),
                                            note="모든 모델의 train_subjects = fold Train 9명, Test 피험자 측정은 학습·표준화·bin 평균에 미사용"),
            "3_folds_trained_independently": dict(passed=all(indep), n_pairs_compared=len(indep),
                                                  note="fold마다 새 BiGRUModel 객체 + 새 초기화(seed 고정). 같은 seed의 fold 간 최종 가중치가 모두 다름"),
            "4_B1_B2_train_only": dict(passed=all(tc["B2"]["train_subjects_ok"] and tc["B1"]["train_subjects_ok"] and tc["B1"]["bin_counts_sum_to_train"]
                                                  for tc in train_checks)),
            "5_bigru_30_runs_ok": dict(passed=len(hist) == 30 and all(x["epochs_recorded"] and x["finite"] and x["not_diverged"]
                                                                       and x["b2_backbone_equals_fold_B2"]
                                                                       for tc in train_checks for x in tc["BiGRU"].values()),
                                       n_runs=len(hist)),
            "6_same_test_pairs_for_all_models": dict(passed=all(same_inp) and fold_checks["pairs_90_per_fold"], test_input_hash=test_hash),
            "7_metrics_match_holdout_calculation": dict(passed=bool(same_calc and same_pairs and all(xc)), same_function_value_on_phase3_model=bool(same_calc),
                                                        test_pairs_match_holdout_files=bool(same_pairs), direct_formula_crosscheck=all(xc),
                                                        method="Phase 3 split_00 B2 모델을 LOSO 평가 경로로 S002에 적용 -> 저장값과 일치 확인 + 직접 식 재계산"),
            "8_target_hbi_constraint": dict(passed=all(chk570) and target06["fixed_target_consistency"]["target_hbi_error_max"] == 0.0,
                                            reflectance_over_100=int(PH.n_reflectance_over_100.sum()), nonfinite=int(PH.n_nonfinite.sum())),
            "9_json_png_consistent": dict(passed=fig_ok, sources=fig_src),
            "10_existing_v1_preserved": dict(passed=hash_before == hash_after, n_files=len(hash_before)),
            "extra_save_load_and_output_contracts": dict(passed=contr_ok),
        },
        missing=dict(near_target_unavailable=NRnull, reference_unavailable=[r for r in REF if not r["available"]]),
        b1_fallback_total=int(sum(c["fallback_uses"] for c in b1c)),
        elapsed_seconds=round(time.time() - t0, 1), status="completed")
    vr["all_checks_passed"] = all(v["passed"] for v in vr["checks"].values())
    jdump(out / "logs" / "loso_verification.json", vr)
    print(f"\n완료: {out} ({vr['elapsed_seconds']}s) all_checks_passed={vr['all_checks_passed']}")
    for k, v in vr["checks"].items():
        print(f"  {k}: {v['passed']}")


# ======================= Figures ===================================================
def make_figures(out: Path):
    _style()
    fdir = out / "figures"; fdir.mkdir(parents=True, exist_ok=True)
    gen = jload(out / "metrics" / "loso_general_reconstruction.json")
    t06 = jload(out / "metrics" / "loso_target06_reconstruction.json")
    cmp_ = jload(out / "metrics" / "loso_model_comparison.json")
    src, plotted = {}, []
    mets = [("a_rmse", "흡광도 RMSE (570 nm 제외)", 1e3, "×10$^{-3}$"), ("r_rmse_pp", "반사율 RMSE", 1, "%p"),
            ("peak_rmse", "Carotenoid Peak RMSE", 1e3, "×10$^{-3}$")]

    def strip(ax, summ, keys, sc, label_mean=True):
        subs = sorted(summ[keys[0]]["per_subject"])
        for s in subs:
            ax.plot(range(len(keys)), [summ[k]["per_subject"][s] * sc for k in keys], color="#cfcfcf", lw=0.7, zorder=1)
        for i, k in enumerate(keys):
            v = [summ[k]["per_subject"][s] * sc for s in subs]
            ax.scatter(np.full(len(v), i), v, s=13, color=COLORS[k], alpha=0.75, zorder=2)
            mu, sd = summ[k]["mean"] * sc, (summ[k]["sd"] or 0) * sc
            ax.plot([i - 0.22, i + 0.22], [mu] * 2, color="#222222", lw=1.6, zorder=3)
            ax.errorbar(i + 0.3, mu, yerr=sd, color="#222222", lw=0.9, capsize=2.5, zorder=3)
            if label_mean:
                ax.text(i + 0.36, mu, f"{mu:.3g}", va="center", fontsize=6.5)
            plotted.append((summ[k]["mean"], float(np.mean(list(summ[k]["per_subject"].values())))))
        ax.set_xticks(range(len(keys)), keys); ax.set_xlim(-0.5, len(keys) - 0.3); ax.grid(axis="y", alpha=0.25, lw=0.5)

    # loso_model_comparison
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    for ax, (m, lab, sc, unit) in zip(axes, mets):
        strip(ax, gen["summary"][m], MODELS, sc)
        ax.set_ylabel(f"{lab} ({unit})"); ax.set_title(lab, loc="left", fontweight="bold")
    _title(fig, "LOSO General reconstruction — 점: 피험자 10명, 가로선: 평균, 막대: ±SD (피험자 기준)")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(fdir / "loso_model_comparison.png", dpi=300); plt.close(fig)
    src["loso_model_comparison.png"] = "loso_general_reconstruction.json: summary.<metric>.<model>.{per_subject,mean,sd}"

    # loso_subject_wise_rmse
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True)
    subs = sorted(gen["summary"]["a_rmse"]["B2"]["per_subject"])
    for ax, (m, lab, sc, unit) in zip(axes, [mets[0], mets[2]]):
        for i, mod in enumerate(MODELS):
            s = gen["summary"][m][mod]
            x = np.arange(len(subs)) + (i - 1) * 0.22
            y = [s["per_subject"][k] * sc for k in subs]
            ax.scatter(x, y, s=22, color=COLORS[mod], zorder=3, label=mod if mod != "BiGRU" else "BiGRU (seed 평균, 선: seed 범위)")
            if mod == "BiGRU":
                for xx, k in zip(x, subs):
                    v = [vv * sc for vv in s["per_subject_per_seed"][k].values()]
                    ax.plot([xx, xx], [min(v), max(v)], color=COLORS[mod], lw=1.2, zorder=2)
            plotted.append((np.array(y) / sc, np.array([s["per_subject"][k] for k in subs])))
        ax.set_ylabel(f"{lab} ({unit})"); ax.set_title(lab, loc="left", fontweight="bold"); ax.grid(axis="y", alpha=0.25, lw=0.5)
    axes[1].set_xticks(range(len(subs)), subs); axes[1].set_xlabel("Test 피험자 (fold)")
    axes[0].legend(loc="upper left", ncol=3)
    _title(fig, "LOSO 피험자별 RMSE (각 피험자 = 해당 fold의 Test)")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(fdir / "loso_subject_wise_rmse.png", dpi=300); plt.close(fig)
    src["loso_subject_wise_rmse.png"] = "loso_general_reconstruction.json: summary.{a_rmse,peak_rmse}.<model>.per_subject(+per_subject_per_seed)"

    # loso_wavelength_error
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
    _title(fig, "LOSO wavelength-wise error (피험자 10명 평균; 570 nm = 목표 HbI로 고정되어 오차 0)")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(fdir / "loso_wavelength_error.png", dpi=300); plt.close(fig)
    src["loso_wavelength_error.png"] = "loso_general_reconstruction.json: wavelength.per_model"

    # loso_target06_accuracy
    ns = t06["near_target_accuracy"]["summary"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    for ax, (t, m, lab) in zip(axes, [("main", "a_rmse", "흡광도 RMSE"), ("main", "peak_rmse", "Peak RMSE"),
                                      ("supplementary", "a_rmse", "흡광도 RMSE"), ("supplementary", "peak_rmse", "Peak RMSE")]):
        strip(ax, ns[t]["metrics"][m], MODELS, 1e3)
        ax.set_ylabel("×10$^{-3}$ absorbance")
        ax.set_title(f"±{ns[t]['tol']:.2f}: {lab}\n(피험자 {ns[t]['n_subjects_evaluable']}명, 실측 {ns[t]['n_gt_measurements_total']}개)",
                     loc="left", fontweight="bold", fontsize=7.5)
    _title(fig, "LOSO 0.6 근처 실측 기반 정확도 (목표 HbI = 정답 측정의 실제 HbI; 점: 피험자)")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(fdir / "loso_target06_accuracy.png", dpi=300); plt.close(fig)
    src["loso_target06_accuracy.png"] = "loso_target06_reconstruction.json: near_target_accuracy.summary.{main,supplementary}.metrics"

    # loso_target06_consistency
    cs = t06["fixed_target_consistency"]["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for ax, (met, lab) in zip(axes, (("peak_sd", "a  개인 내 Carotenoid Peak SD"), ("spectral_sd", "b  개인 내 Spectral SD (파장 평균)"))):
        strip(ax, cs[met], ["Raw"] + MODELS, 1e3)
        ax.set_ylabel("×10$^{-3}$ absorbance"); ax.set_title(lab, loc="left", fontweight="bold")
    _title(fig, "LOSO: 모든 측정을 정확히 HbI 0.6으로 복원했을 때의 반복성 (정답 없음 - 정확도가 아님)")
    fig.tight_layout(rect=(0, 0, 1, 0.91)); fig.savefig(fdir / "loso_target06_consistency.png", dpi=300); plt.close(fig)
    src["loso_target06_consistency.png"] = "loso_target06_reconstruction.json: fixed_target_consistency.summary"

    # loso_vs_holdout
    hv = cmp_["holdout_vs_loso"]
    panels = [("general", "a_rmse", "General 흡광도 RMSE"), ("general", "peak_rmse", "General Peak RMSE"),
              ("near_main", "a_rmse", "0.6 근처 ±0.02 흡광도 RMSE"), ("consistency", "peak_sd", "0.6 고정 복원 Peak SD")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, (sec, m, lab) in zip(axes, panels):
        for i, mod in enumerate(MODELS):
            r = hv[sec][m][mod]
            ps = r["per_subject"]
            for s, v in ps.items():
                ax.plot([i - 0.15, i + 0.15], [v["holdout"] * 1e3, v["loso"] * 1e3], color="#cfcfcf", lw=0.6, zorder=1)
            ax.scatter(np.full(len(ps), i - 0.15), [v["holdout"] * 1e3 for v in ps.values()], s=11, facecolor="white",
                       edgecolor=COLORS[mod], lw=0.9, zorder=2)
            ax.scatter(np.full(len(ps), i + 0.15), [v["loso"] * 1e3 for v in ps.values()], s=11, color=COLORS[mod], zorder=2)
            ax.plot([i - 0.28, i - 0.02], [r["holdout_subject_mean"] * 1e3] * 2, color="#222222", lw=1.4, ls=(0, (2, 1)))
            ax.plot([i + 0.02, i + 0.28], [r["loso_subject_mean"] * 1e3] * 2, color="#222222", lw=1.6)
            plotted.append((r["loso_subject_mean"], float(np.mean([v["loso"] for v in ps.values()]))))
        ax.set_xticks(range(3), MODELS); ax.set_ylabel("×10$^{-3}$ absorbance")
        ax.set_title(f"{lab}\n(n={len(ps)}명)", loc="left", fontweight="bold", fontsize=7.5); ax.grid(axis="y", alpha=0.25, lw=0.5)
    axes[0].scatter([], [], s=11, facecolor="white", edgecolor="#555555", label="7:3 Hold-out (피험자별 split 평균)")
    axes[0].scatter([], [], s=11, color="#555555", label="LOSO")
    axes[0].legend(loc="upper left", fontsize=6.5)
    _title(fig, "7:3 Hold-out(Train 7명) vs LOSO(Train 9명) — 같은 피험자끼리 선으로 연결, 가로선: 피험자 평균 (점선 Hold-out, 실선 LOSO)")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(fdir / "loso_vs_holdout.png", dpi=300); plt.close(fig)
    src["loso_vs_holdout.png"] = "loso_model_comparison.json: holdout_vs_loso"
    ok = all(np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-12) for a, b in plotted)
    return ok, src


if __name__ == "__main__":
    main()
