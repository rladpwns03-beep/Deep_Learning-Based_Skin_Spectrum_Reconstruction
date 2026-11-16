"""Phase 4 — BiGRU 학습 Pair 선택 전략 실험 (P1 All / P2 Adjacent / P3 Small-Delta).

실행 (code 폴더, conda 환경 skin_spectra):
    python scripts/pair_strategy.py                 # results/holdout/pair_strategy/ 에 저장 (기존 결과가 있으면 중단)
    python scripts/pair_strategy.py --run-id NAME   # results/holdout/pair_strategy/runs/NAME/ 에 별도 저장
    python scripts/pair_strategy.py --version NAME --epochs 300   # results/NAME/pair_strategy/ (train_models·evaluate --version NAME 먼저 실행)

바꾸는 것은 학습 Pair 구성뿐이다. BiGRU 구조·손실·optimizer·lr·clip·입력 특징·seed·분할·평가 Pair는 Phase 2/3과 같다.
  P1  같은 피험자의 모든 순서쌍 i != j (피험자당 90개, 기존 방식 = make_pairs)
  P2  피험자별 HbI 오름차순(동률은 trial 번호 순) 정렬 후 인접 측정끼리 양방향 (피험자당 9 x 2 = 18개)
  P3  |h_j - h_i| <= δ, i != j.  δ = 그 split의 Train P1 Pair |ΔHbI| 중앙값 (Train 데이터만 사용, 평가 전에 고정)
손실의 피험자 가중치 규칙(피험자마다 총합 1/피험자 수, 피험자 안에서는 Pair 균등)은 모든 전략에 똑같이 적용한다.
표준화 평균/표준편차와 B2 기울기는 Pair가 아니라 Train 측정에서 계산하므로 전략과 무관하게 같다.

Comparison A (fixed_epoch): 모든 전략 같은 epoch (기본 150, --epochs로 새 버전에서만 변경).
Comparison B (fixed_steps): 총 optimizer update 수 = 같은 split·seed의 P1(fixed_epoch) 실제 update 수.
  기존 학습은 full batch(1 epoch = 1 update, 매 update에 모든 Pair 사용)이므로 필요한 epoch = update 수 / 1 이고
  반복 샘플링이 필요 없다. 실제 update 수는 optimizer step hook으로 직접 센다.

평가는 evaluate.py(Phase 3)와 같은 계산 방식: 피험자별 group 지표 -> split 안 평균 -> split 평균, BiGRU는 seed별 지표의 평균.
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
import torch.optim.optimizer as topt

ROOT = Path(__file__).resolve().parents[1]   # 프로젝트 루트 (scripts/ 의 상위)
sys.path.insert(0, str(ROOT))

from common import config as C                                              # noqa: E402
from common.data_loader import HBI_INDEX, WL, load_spectra, make_pairs     # noqa: E402
from common.metrics import consistency, group_metrics, pair_metrics, wavelength_metrics  # noqa: E402
from evaluate import DH_BINS, TOLS, UNITS, _style, _title, nan_to_none     # noqa: E402  Phase 3와 같은 구간·허용오차
from models.bigru import CFG as BIGRU_CFG, BiGRUModel                      # noqa: E402
from models.linear import LinearModel                                       # noqa: E402
from train_models import load_fixed_splits, roundtrip                      # noqa: E402

VERSION = "holdout"
BASE = ROOT / "results" / VERSION
EPOCHS = BIGRU_CFG["epochs"]          # 기본 150
STRATS = ["P1", "P2", "P3"]
STRAT_DESC = {
    "P1": "All Pairs: 같은 피험자의 모든 순서쌍 i != j (피험자당 90개, 기존 학습과 동일)",
    "P2": "Adjacent HbI Pairs: 피험자별 HbI 오름차순(동률은 trial 번호 순) 정렬 후 인접 측정끼리 양방향 (피험자당 18개)",
    "P3": "Small-Delta HbI Pairs: |h_target - h_input| <= δ, i != j (자기 자신 제외)",
}
P3_RULE = ("δ(split) = 해당 split Train 피험자의 모든 순서쌍(P1) |ΔHbI|의 중앙값. Train 데이터만 사용하며 "
           "Test 평가 전에 계산·고정 (Test 결과로 바꾸지 않음). 경계값은 포함(<=).")
COMPARISONS = {
    "fixed_epoch": "Comparison A - 모든 전략 같은 epoch (EPOCHS)",
    "fixed_steps": "Comparison B - 총 optimizer update 수를 같은 split·seed의 P1(fixed_epoch) 실제 update 수에 맞춤",
}
METRICS = ["a_rmse", "r_rmse_pp", "peak_rmse", "peak_mae", "peak_bias"]
KEYS = STRATS + ["B2"]                                   # B2 = 전략과 무관한 참고 기준 (Phase 2 모델)
COLORS = {"P1": "#009E73", "P2": "#CC79A7", "P3": "#E69F00", "B2": "#0072B2", "Raw": "#9a9893"}
MARKERS = {"P1": "o", "P2": "s", "P3": "^", "B2": "D"}
HIST_EDGES = np.round(np.arange(0, 0.2501, 0.01), 4)

STEP_COUNTER = [0]
topt.register_optimizer_step_post_hook(lambda opt, a, k: STEP_COUNTER.__setitem__(0, STEP_COUNTER[0] + 1))


def jdump(path: Path, obj, indent=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nan_to_none(obj), ensure_ascii=False, indent=indent, allow_nan=False), encoding="utf-8")


def jload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha_files(paths, base) -> dict:
    return {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


# ---------------------------------------------------------------------------
# Pair 생성
# ---------------------------------------------------------------------------
def p3_delta(train) -> float:
    _, _, _, I, J = make_pairs(train)
    return float(np.median(np.abs(train.hbi[J] - train.hbi[I])))


def build_pairs(train, strategy, delta=None) -> dict:
    """make_pairs(P1)의 순서를 유지한 채 조건에 맞는 Pair만 남기고, 피험자 가중치 규칙을 다시 적용한다."""
    X, Y, W, I, J = make_pairs(train)
    subj = train.subject[I]
    if strategy == "P1":
        mask = np.ones(len(I), bool)
    elif strategy == "P2":
        rank = np.empty(len(train), int)
        for s in train.subjects:
            ix = np.flatnonzero(train.subject == s)
            order = ix[np.lexsort((train.trial[ix], train.hbi[ix]))]      # HbI 오름차순, 동률은 trial 순
            rank[order] = np.arange(len(order))
        mask = np.abs(rank[I] - rank[J]) == 1
    elif strategy == "P3":
        mask = np.abs(train.hbi[J] - train.hbi[I]) <= delta
    else:
        raise ValueError(strategy)
    subs, cnt = np.unique(subj[mask], return_counts=True)
    c = dict(zip(subs.tolist(), cnt.tolist()))
    w_rule = np.array([1.0 / (len(subs) * c[s]) for s in subj[mask]])
    w = W if strategy == "P1" else w_rule          # P1은 make_pairs 가중치 그대로 (검증에서 w_rule과 같은지 확인)
    return dict(X=X[mask], Y=Y[mask], W=w, W_rule=w_rule, I=I[mask], J=J[mask], subject=subj[mask],
                dh=train.hbi[J[mask]] - train.hbi[I[mask]], per_subject=c)


def check_pairs(train, strategy, p, delta) -> dict:
    """Pair 생성 조건 검사 (Train 데이터만)."""
    ok_same_subject = bool(np.all(train.subject[p["I"]] == train.subject[p["J"]]))
    ok_no_self = bool(np.all(p["I"] != p["J"]))
    keys = set(zip(p["I"].tolist(), p["J"].tolist()))
    expected = set()
    for s in train.subjects:
        ix = np.flatnonzero(train.subject == s)
        if strategy == "P1":
            expected |= {(i, j) for i in ix for j in ix if i != j}
        elif strategy == "P2":
            o = ix[np.lexsort((train.trial[ix], train.hbi[ix]))]
            expected |= {(int(o[k]), int(o[k + 1])) for k in range(len(o) - 1)} | {(int(o[k + 1]), int(o[k])) for k in range(len(o) - 1)}
        else:
            expected |= {(i, j) for i in ix for j in ix if i != j and abs(train.hbi[j] - train.hbi[i]) <= delta}
    w_per_subj = {s: float(p["W"][p["subject"] == s].sum()) for s in p["per_subject"]}
    return dict(same_subject_only=ok_same_subject, no_self_pairs=ok_no_self, no_duplicate=len(keys) == len(p["I"]),
                exact_condition_set=keys == expected,
                expected_count_per_subject=(all(v == 90 for v in p["per_subject"].values()) if strategy == "P1" else
                                            all(v == 18 for v in p["per_subject"].values()) if strategy == "P2" else True),
                weights_sum_to_1=bool(np.isclose(p["W"].sum(), 1.0, rtol=0, atol=1e-12)),
                equal_subject_weight=bool(np.allclose(list(w_per_subj.values()), 1 / len(w_per_subj), rtol=0, atol=1e-12)),
                weights_follow_rule=bool(np.allclose(p["W"], p["W_rule"], rtol=1e-15, atol=0)),
                all_train_subjects_have_pairs=sorted(p["per_subject"]) == train.subjects)


# ---------------------------------------------------------------------------
# 집계 (evaluate.py summarize 와 같은 방식)
# ---------------------------------------------------------------------------
def summarize(d: pd.DataFrame, metric: str):
    if d.empty or d[metric].isna().all():
        return None
    seedlvl = d.groupby(["split_id", "seed"])[metric].mean()
    split_mean = seedlvl.groupby("split_id").mean()
    subj = d.groupby(["split_id", "subject"])[metric].mean()
    uniq = subj.groupby("subject").mean()
    rec = dict(mean=float(split_mean.mean()), sd=float(split_mean.std(ddof=1)) if split_mean.size > 1 else None,
               sd_definition="split 평균값 10개의 SD (ddof=1)", n_splits=int(split_mean.size),
               subject_sd_within_split=float(subj.groupby("split_id").std(ddof=1).mean())
               if subj.groupby("split_id").size().max() > 1 else None,
               n_subject_evaluations=int(subj.size), n_unique_subjects=int(uniq.size),
               unique_subject_mean=float(uniq.mean()), unique_subject_sd=float(uniq.std(ddof=1)) if uniq.size > 1 else None,
               per_unique_subject={k: float(v) for k, v in uniq.items()},
               per_split={k: float(v) for k, v in split_mean.items()})
    if d.seed.nunique() > 1:
        per_seed = seedlvl.groupby("seed").mean()
        rec["seed_sd_within_split"] = float(seedlvl.groupby("split_id").std(ddof=1).mean())
        rec["per_seed_mean"] = {f"seed_{int(k)}": float(v) for k, v in per_seed.items()}
        rec["seed_sd_of_overall_means"] = float(per_seed.std(ddof=1))
    return rec


def improvement(base: dict, other: dict):
    """(base - other) / base x 100, 양수 = other가 더 낮음(개선)."""
    if base is None or other is None:
        return None
    subs = sorted(set(base["per_unique_subject"]) & set(other["per_unique_subject"]))
    spl = sorted(set(base["per_split"]) & set(other["per_split"]))
    ps = {s: 100 * (base["per_unique_subject"][s] - other["per_unique_subject"][s]) / base["per_unique_subject"][s] for s in subs}
    return dict(improvement_pct=100 * (base["mean"] - other["mean"]) / base["mean"], base_mean=base["mean"],
                other_mean=other["mean"], per_unique_subject_pct=ps,
                n_subjects_better=int(sum(v > 0 for v in ps.values())), n_subjects=len(subs),
                n_splits_better=int(sum(other["per_split"][k] < base["per_split"][k] for k in spl)), n_splits=len(spl))


def bin_summary(D: pd.DataFrame, col: str) -> list:
    out = []
    for (b, d, k), x in D.groupby([col, "direction", "strategy"]):
        rec = {col: b, "direction": d, "strategy": k}
        for m in ("a_rmse", "r_rmse_pp", "peak_rmse"):
            sm = x.groupby("split_id")[m].mean()
            rec[m] = float(sm.mean())
            rec[f"{m}_sd"] = float(sm.std(ddof=1)) if sm.size > 1 else None
            rec[f"{m}_per_split"] = {kk: float(v) for kk, v in sm.items()}
        rec.update(n_pairs=int(x[x.seed.isin([-1, 0])].n_pairs.sum()),
                   n_subject_evaluations=int(x.groupby(["split_id", "subject"]).ngroups), n_splits=int(x.split_id.nunique()))
        out.append(rec)
    return out


def bin_improvement(rows: list, col: str) -> list:
    """같은 구간에서 P1 대비 개선율 (구간마다 split 평균 기준)."""
    res = []
    idx = {(r[col], r["direction"], r["strategy"]): r for r in rows}
    for (b, d, k), r in idx.items():
        base = idx.get((b, d, "P1"))
        if k == "P1" or base is None:
            continue
        rec = {col: b, "direction": d, "strategy": k}
        for m in ("a_rmse", "peak_rmse"):
            spl = sorted(set(base[f"{m}_per_split"]) & set(r[f"{m}_per_split"]))
            rec[f"{m}_improvement_pct_vs_P1"] = 100 * (base[m] - r[m]) / base[m]
            rec[f"{m}_n_splits_better"] = int(sum(r[f"{m}_per_split"][s] < base[f"{m}_per_split"][s] for s in spl))
            rec["n_splits"] = len(spl)
        res.append(rec)
    return res


# ---------------------------------------------------------------------------
def main():
    global VERSION, BASE, EPOCHS
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--version", default=VERSION, help="결과 버전 폴더 (results/<version>)")
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="BiGRU epoch 수 (기본 150)")
    args = ap.parse_args()
    VERSION, BASE, EPOCHS = args.version, ROOT / "results" / args.version, args.epochs
    COMPARISONS["fixed_epoch"] = f"Comparison A - 모든 전략 {EPOCHS} epoch"
    if VERSION == "holdout" and EPOCHS != BIGRU_CFG["epochs"]:
        sys.exit("[중단] holdout은 150 epoch 고정. 다른 epoch는 --version NAME 처럼 새 버전으로 저장하세요.")
    need = [BASE / "config" / "training_config.json", BASE / "metrics" / "general_reconstruction.json"]
    if [x for x in need if not x.exists()]:
        sys.exit(f"[중단] {BASE} 에 학습·평가 결과가 없음. train_models.py / evaluate.py --version {VERSION} 먼저 실행")
    tcfg_epochs = json.loads(need[0].read_text(encoding="utf-8"))["bigru"]["epochs"]
    if tcfg_epochs != EPOCHS:
        sys.exit(f"[중단] {VERSION}의 학습 epoch({tcfg_epochs})와 --epochs({EPOCHS})가 다름")
    root_out = BASE / "pair_strategy"
    out = root_out / "runs" / args.run_id if args.run_id else root_out
    if out.exists() and any(p.is_file() for p in out.rglob("*") if "runs" not in p.relative_to(out).parts):
        sys.exit(f"[중단] {out} 에 이미 결과가 있음. 덮어쓰지 않음. 새 실행은 --run-id 지정.")
    run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    t0 = time.time()
    torch.set_num_threads(1)

    prev_files = [p for p in BASE.rglob("*") if p.is_file() and "pair_strategy" not in p.relative_to(BASE).parts]
    hash_before = sha_files(prev_files, BASE)
    data = load_spectra()
    splits = load_fixed_splits(data)
    row = {(s, t): i for i, (s, t) in enumerate(zip(data.subject, data.trial))}

    # ======================= 1. Pair 생성 (Train만) ============================
    pairs, deltas, pair_checks, pair_counts, hist = {}, {}, [], [], {k: np.zeros(len(HIST_EDGES) - 1) for k in STRATS}
    for sp in splits:
        sid = sp["split_id"]
        tr = data.select(sp["train"])
        deltas[sid] = p3_delta(tr)
        for st in STRATS:
            p = build_pairs(tr, st, deltas[sid])
            pairs[(sid, st)] = p
            pair_checks.append(dict(split_id=sid, strategy=st, **check_pairs(tr, st, p, deltas[sid]),
                                    pair_subjects_in_train_only=set(p["per_subject"]) <= set(sp["train"]),
                                    no_test_subject=not set(p["per_subject"]) & set(sp["test"])))
            ad = np.abs(p["dh"])
            pair_counts.append(dict(split_id=sid, strategy=st, n_pairs=int(len(ad)), per_subject=p["per_subject"],
                                    min_per_subject=int(min(p["per_subject"].values())),
                                    max_per_subject=int(max(p["per_subject"].values())),
                                    abs_dhbi_min=float(ad.min()), abs_dhbi_median=float(np.median(ad)),
                                    abs_dhbi_max=float(ad.max()), n_zero_dhbi_pairs=int((ad == 0).sum())))
            hist[st] += np.histogram(ad, HIST_EDGES)[0]
    print("δ per split:", {k: round(v, 4) for k, v in deltas.items()}, flush=True)

    config = dict(
        experiment_version=VERSION, run_id=run_id, phase="Phase 4 - BiGRU Pair Selection Strategy", model="BiGRU",
        strategies=STRAT_DESC, p3_delta_rule=P3_RULE, p3_delta_per_split=deltas,
        p3_delta_summary=dict(min=min(deltas.values()), max=max(deltas.values()), mean=float(np.mean(list(deltas.values())))),
        p2_tie_rule="같은 피험자 안 HbI 동률은 trial 번호 오름차순으로 정렬 (결정적)",
        loss_weighting="w = 1 / (Pair가 있는 Train 피험자 수 x 그 피험자의 Pair 수) - 기존 make_pairs 규칙과 같음 (P1에서 동일)",
        comparisons=COMPARISONS, bigru={k: (list(v) if isinstance(v, tuple) else v) for k, v in dict(BIGRU_CFG, epochs=EPOCHS).items()},
        seeds=list(C.BIGRU_SEEDS), batching="full batch (1 epoch = optimizer update 1회, 매 update에 전체 Pair 사용) - 기존과 동일",
        fixed_steps_rule="target_steps = 같은 split·seed의 P1 fixed_epoch 실제 update 수; epochs = ceil(target_steps / steps_per_epoch)",
        resampling="불필요 - full batch에서 steps_per_epoch = 1 이므로 모든 전략이 target_steps epoch로 정확히 맞음. "
                   "각 Pair는 update마다 1회씩 사용 (Pair별 사용 횟수 = update 수)",
        device="cpu", torch_threads=1, splits=splits, split_source="data/splits/split_plan.csv",
        evaluation_pairs="data/splits/repNN/exp1_test_pairs.csv, exp2_* (Phase 3와 같은 파일)",
        dh_bins=DH_BINS, near_tolerances=TOLS, pair_counts=pair_counts,
        abs_dhbi_histogram=dict(edges=HIST_EDGES.tolist(), counts_sum_over_splits={k: v.astype(int).tolist() for k, v in hist.items()},
                                note="Train Pair |ΔHbI| 분포, 10개 split 합계"),
        b2_reference="results/holdout/models/B2 (Phase 2, Pair 전략과 무관) - 비교 참고용으로만 평가")
    jdump(out / "config" / "pair_strategy_config.json", config)

    # ======================= 2. 학습 ===========================================
    hist_rows, models, rt_checks, arch_checks = [], {}, [], []
    ref_state = None
    for comp in COMPARISONS:
        for sp in splits:
            sid = sp["split_id"]
            tr = data.select(sp["train"])
            for st in STRATS:
                p = pairs[(sid, st)]
                for seed in C.BIGRU_SEEDS:
                    if comp == "fixed_epoch":
                        epochs, target_steps = EPOCHS, None
                    else:
                        target_steps = next(h["update_steps"] for h in hist_rows if h["comparison"] == "fixed_epoch"
                                            and h["strategy"] == "P1" and h["split_id"] == sid and h["seed"] == seed)
                        epochs = int(np.ceil(target_steps / 1))       # steps_per_epoch = 1 (full batch)
                    STEP_COUNTER[0] = 0
                    t = time.time()
                    m = BiGRUModel(seed=seed, epochs=epochs, cfg=dict(BIGRU_CFG, epochs=epochs)).fit(tr, pairs=(p["X"], p["Y"], p["W"]))
                    secs = time.time() - t
                    steps = STEP_COUNTER[0]
                    path = out / "models" / st / comp / f"{sid}_seed_{seed}.pt"
                    rt_checks.append(dict(comparison=comp, strategy=st, split_id=sid, seed=seed,
                                          save_load_identical=roundtrip(m, path, tr)))
                    sd = {k: tuple(v.shape) for k, v in m.net.state_dict().items()}
                    ref_state = ref_state or sd
                    arch_checks.append(dict(comparison=comp, strategy=st, split_id=sid, seed=seed, n_params=m.n_params,
                                            same_state_dict_shapes=sd == ref_state, same_cfg={k: v for k, v in m.cfg.items() if k != "epochs"} == {k: v for k, v in BIGRU_CFG.items() if k != "epochs"},
                                            train_subjects_match=m.train_subjects == sorted(sp["train"])))
                    models[(comp, st, sid, seed)] = m
                    H = np.array(m.history)
                    hist_rows.append(dict(
                        comparison=comp, strategy=st, split_id=sid, seed=seed, n_pairs=int(len(p["I"])),
                        n_pairs_per_subject=p["per_subject"], epochs=epochs, steps_per_epoch=1, update_steps=steps,
                        target_update_steps=target_steps, epoch_records=int(len(H)),
                        pair_uses_per_pair=steps, total_pair_terms=int(steps * len(p["I"])),
                        resampling="none (full batch)", initial_loss=float(H[0, 1]), final_loss=float(H[-1, 1]),
                        final_mse=float(H[-1, 2]), max_loss=float(H[:, 1].max()), finite=bool(np.isfinite(H[:, 1:]).all()),
                        train_seconds=round(secs, 2),
                        history=dict(loss=H[:, 1].tolist(), mse=H[:, 2].tolist(), grad_norm=H[:, 3].tolist())))
                print(f"[{comp}] {sid} {st}: pairs {len(p['I'])}, steps {steps}, "
                      f"final loss {np.mean([h['final_loss'] for h in hist_rows[-3:]]):.3e}", flush=True)
    jdump(out / "logs" / "pair_training_history.json", dict(
        experiment_version=VERSION, run_id=run_id, model="BiGRU", loss_unit=
        "absorbance^2 (subject-weighted MSE over 30 wavelengths + 0.001 x residual-slope L2). 전략마다 Pair 집합이 달라 "
        "전략 간 training loss 값은 직접 비교할 수 없음", runs=hist_rows), indent=None)

    # 재현·동일성 검사
    p1_match_v1, ab_identical = [], []
    for sp in splits:
        sid = sp["split_id"]
        for seed in C.BIGRU_SEEDS:
            old = torch.load(BASE / "models" / "BiGRU" / f"{sid}_seed_{seed}.pt", map_location="cpu", weights_only=True)
            new = models[("fixed_epoch", "P1", sid, seed)]
            p1_match_v1.append(dict(split_id=sid, seed=seed, identical=all(
                torch.equal(old["state_dict"][k], v) for k, v in new.net.state_dict().items()) and
                np.array_equal(np.asarray(old["b2"]), new.b2)))
            for st in STRATS:
                a, b = models[("fixed_epoch", st, sid, seed)], models[("fixed_steps", st, sid, seed)]
                ab_identical.append(dict(split_id=sid, seed=seed, strategy=st, identical=all(
                    torch.equal(v, b.net.state_dict()[k]) for k, v in a.net.state_dict().items())))

    # ======================= 3. 평가 (Phase 3와 같은 Test Pair) ===================
    G, WLr, DH, DC, NR, CS, PH, chk570, same_inp = [], [], [], [], [], [], [], [], []
    test_hash = {}
    for sp in splits:
        sid, rep = sp["split_id"], sp["rep"]
        sdir = ROOT / "data" / "splits" / f"rep{rep:02d}"
        test = sorted(sp["test"])
        insts = {("reference", "B2", -1): LinearModel.load(BASE / "models" / "B2" / f"{sid}.json")}
        for comp in COMPARISONS:
            for st in STRATS:
                for seed in C.BIGRU_SEEDS:
                    insts[(comp, st, seed)] = models[(comp, st, sid, seed)]
        pairs_df = pd.read_csv(sdir / "exp1_test_pairs.csv")
        ii = np.array([row[(s, t)] for s, t in zip(pairs_df.subject, pairs_df.input_trial)])
        jj = np.array([row[(s, t)] for s, t in zip(pairs_df.subject, pairs_df.target_trial)])
        X, Y, T = data.a[ii], data.a[jj], data.hbi[jj]
        dh = T - data.hbi[ii]
        subj = pairs_df.subject.to_numpy()
        test_hash[sid] = hashlib.sha256(np.concatenate([X, Y, T[:, None]], 1).tobytes()).hexdigest()
        delta = deltas[sid]
        rep_in = pd.read_csv(sdir / "exp2_repeatability_inputs.csv")
        rr = np.array([row[(s, t)] for s, t in zip(rep_in.subject, rep_in.trial)])
        A0 = data.a[rr]
        for s in test:
            ms = rep_in.subject.to_numpy() == s
            CS.append(dict(comparison="reference", strategy="Raw", seed=-1, split_id=sid, subject=s, **consistency(A0[ms])))
        inp570 = set()
        for (comp, st, seed), m in insts.items():
            P = m.predict(X, T)
            inp570.add(hashlib.sha256(np.ascontiguousarray(P[:, HBI_INDEX]).tobytes()).hexdigest())
            chk570.append(bool(np.array_equal(P[:, HBI_INDEX], T)))
            name = dict(comparison=comp, strategy=st, seed=seed, split_id=sid)
            PH.append(dict(**name, experiment="general", n_reflectance_over_100=int((P < 0).sum()),
                           n_nonfinite=int((~np.isfinite(P)).sum())))
            pm = pair_metrics(P, Y)
            for s in test:
                msk = subj == s
                G.append(dict(**name, subject=s, **group_metrics(P[msk], Y[msk]), mean_pair_a_rmse=float(pm["a_rmse"][msk].mean())))
                WLr.append(dict(**name, subject=s, **wavelength_metrics(P[msk], Y[msk])))
                for lo, hi in zip(DH_BINS[:-1], DH_BINS[1:]):
                    inb = (np.abs(dh) >= lo) & (np.abs(dh) < hi)
                    for direction, dm in (("increase", dh > 0), ("decrease", dh < 0), ("both", np.ones_like(inb))):
                        mm = msk & dm & inb
                        if mm.any():
                            DH.append(dict(**name, subject=s, dh_bin=f"{lo:.3f}-{hi:.3f}", direction=direction,
                                           **group_metrics(P[mm], Y[mm])))
                for cls, cm in (("le_delta", np.abs(dh) <= delta), ("gt_delta", np.abs(dh) > delta)):
                    mm = msk & cm
                    if mm.any():
                        DC.append(dict(**name, subject=s, dh_class=cls, direction="both", delta=delta, **group_metrics(P[mm], Y[mm])))
            for tname, tol in TOLS.items():
                sel = np.abs(T - C.TARGET_HBI) <= tol
                for s in test:
                    mm = sel & (subj == s)
                    if mm.any():
                        gt = sorted(set(pairs_df.target_trial[mm]))
                        NR.append(dict(**name, tolerance=tname, tol=tol, subject=s, n_gt_measurements=len(gt), gt_trials=gt,
                                       **group_metrics(P[mm], Y[mm])))
            F = m.predict(A0, C.TARGET_HBI)
            PH.append(dict(**name, experiment="fixed_target_0.6", n_reflectance_over_100=int((F < 0).sum()),
                           n_nonfinite=int((~np.isfinite(F)).sum())))
            for s in test:
                ms = rep_in.subject.to_numpy() == s
                CS.append(dict(**name, subject=s, **consistency(F[ms]),
                               target_hbi_error_max=float(np.abs(F[ms, HBI_INDEX] - C.TARGET_HBI).max())))
        same_inp.append(len(inp570) == 1)
        print(f"[eval] {sid} done", flush=True)

    G, WLd, DH, DC, NR, CS, PH = map(pd.DataFrame, (G, WLr, DH, DC, NR, CS, PH))

    def per_comp(df, comp):
        return df[df.comparison.isin([comp, "reference"])]

    meta = dict(experiment_version=VERSION, run_id=run_id, phase="Phase 4 - BiGRU Pair Selection Strategy",
                units=UNITS, aggregation="pair -> subject(group RMSE) -> subject mean per split -> mean over splits; "
                "BiGRU 전략 = seed별 지표의 평균. sd = split 평균값의 SD. seed=-1 = 해당 없음(B2, Raw).",
                strategies=STRAT_DESC, reference="B2 = Phase 2 모델 (Pair 전략과 무관한 참고 기준)")
    results = {}
    for comp in COMPARISONS:
        g, dhc, dcc, nr, cs, ph = (per_comp(x, comp) for x in (G, DH, DC, NR, CS, PH))
        wl = per_comp(WLd, comp)
        general = {m: {k: summarize(g[g.strategy == k], m) for k in KEYS} for m in METRICS + ["mean_pair_a_rmse"]}
        dh_rows, dc_rows = bin_summary(dhc, "dh_bin"), bin_summary(dcc, "dh_class")
        near = {}
        for tname in TOLS:
            d = nr[nr.tolerance == tname]
            if d.empty:
                near[tname] = dict(available=False, reason="해당 허용오차의 정답 측정이 test에 없음")
                continue
            b2 = d[d.strategy == "B2"]
            near[tname] = dict(available=True, tol=TOLS[tname], n_subject_evaluations=int(len(b2)),
                               unique_subjects=sorted(b2.subject.unique()), n_pairs_total=int(b2.n_pairs.sum()),
                               metrics={m: {k: summarize(d[d.strategy == k], m) for k in KEYS} for m in METRICS})
        cons = {met: {k: summarize(cs[cs.strategy == k], met) for k in ["Raw"] + KEYS} for met in ("peak_sd", "spectral_sd")}
        imp = dict(
            general={m: {k: improvement(general[m]["P1"], general[m][k]) for k in ("P2", "P3", "B2")}
                     for m in ("a_rmse", "r_rmse_pp", "peak_rmse", "peak_mae")},
            near_target={t: ({m: {k: improvement(near[t]["metrics"][m]["P1"], near[t]["metrics"][m][k]) for k in ("P2", "P3", "B2")}
                              for m in ("a_rmse", "peak_rmse")} if near[t]["available"] else None) for t in TOLS},
            consistency={met: {k: improvement(cons[met]["P1"], cons[met][k]) for k in ("P2", "P3", "B2")} for met in cons},
            by_dh_bin=bin_improvement(dh_rows, "dh_bin"), by_delta_class=bin_improvement(dc_rows, "dh_class"),
            formula="(metric_P1 - metric_X) / metric_P1 x 100 (양수 = X가 P1보다 오차·변동이 작음)")
        tr_runs = [h for h in hist_rows if h["comparison"] == comp]
        training = {st: dict(n_pairs_per_split={h["split_id"]: h["n_pairs"] for h in tr_runs if h["strategy"] == st and h["seed"] == 0},
                             epochs=sorted({h["epochs"] for h in tr_runs if h["strategy"] == st}),
                             update_steps=sorted({h["update_steps"] for h in tr_runs if h["strategy"] == st}),
                             final_loss_mean=float(np.mean([h["final_loss"] for h in tr_runs if h["strategy"] == st])),
                             final_loss_sd=float(np.std([h["final_loss"] for h in tr_runs if h["strategy"] == st], ddof=1)))
                    for st in STRATS}
        results[comp] = dict(
            **meta, comparison=comp, condition=COMPARISONS[comp], training=training,
            general=dict(summary=general, subject_wise=g.to_dict(orient="records"),
                         wavelength={"wavelengths_nm": WL.tolist(), "excluded_from_rmse_nm": [570], "per_strategy": {
                             k: {q: np.mean(np.vstack(wl[wl.strategy == k].groupby("split_id")[q].apply(
                                 lambda s: np.mean(np.vstack(s.tolist()), 0)).tolist()), 0).tolist() for q in ("rmse", "mae", "bias")}
                             for k in KEYS}},
                         rmse_by_delta_hbi=dh_rows, rmse_by_delta_class=dc_rows,
                         delta_class_definition="le_delta: |ΔHbI| <= δ(split) (P3 학습 범위 안), gt_delta: > δ (P3 학습 범위 밖)"),
            target06=dict(near_target_accuracy=dict(summary=near, subject_wise=nr.to_dict(orient="records"),
                                                    ground_truth="0.6 근처 실측, 목표 HbI = 정답 측정의 실제 HbI"),
                          fixed_target_consistency=dict(summary=cons, subject_wise=cs.to_dict(orient="records"),
                                                        ground_truth="없음 - 반복성 지표이며 정확도가 아님"),
                          target_hbi_error_max=float(cs[cs.strategy != "Raw"].target_hbi_error_max.max())),
            physical_validity=dict(n_reflectance_over_100=int(ph.n_reflectance_over_100.sum()),
                                   n_nonfinite=int(ph.n_nonfinite.sum()), rows=ph.to_dict(orient="records"),
                                   definition="반사율 > 100% <=> 흡광도 < 0; 반사율 <= 0%는 수식상 불가능"),
            improvement_vs_P1=imp, status="completed")
        jdump(out / "metrics" / f"{comp}_metrics.json", results[comp])

    def table(res):
        gs = res["general"]["summary"]
        return {m: {k: (None if gs[m][k] is None else {q: gs[m][k].get(q) for q in
                       ("mean", "sd", "subject_sd_within_split", "seed_sd_within_split", "seed_sd_of_overall_means", "per_seed_mean")})
                    for k in KEYS} for m in METRICS}
    comparison = dict(
        **meta, experiment="Pair strategy comparison",
        pair_counts={st: dict(per_split={c["split_id"]: c["n_pairs"] for c in pair_counts if c["strategy"] == st},
                              per_subject_min=min(c["min_per_subject"] for c in pair_counts if c["strategy"] == st),
                              per_subject_max=max(c["max_per_subject"] for c in pair_counts if c["strategy"] == st))
                     for st in STRATS},
        p3_delta_per_split=deltas,
        conditions={comp: dict(condition=COMPARISONS[comp], training=results[comp]["training"],
                               general=table(results[comp]),
                               near_target={t: ({m: {k: (lambda r: None if r is None else dict(mean=r["mean"], sd=r["sd"]))(
                                   results[comp]["target06"]["near_target_accuracy"]["summary"][t]["metrics"][m][k]) for k in KEYS}
                                   for m in ("a_rmse", "peak_rmse")} if results[comp]["target06"]["near_target_accuracy"]["summary"][t]["available"] else None)
                                   for t in TOLS},
                               consistency={met: {k: dict(mean=v["mean"], sd=v["sd"]) for k, v in d.items()}
                                            for met, d in results[comp]["target06"]["fixed_target_consistency"]["summary"].items()},
                               improvement_vs_P1={sec: ({m: {k: (None if v is None else dict(
                                   improvement_pct=v["improvement_pct"], n_subjects_better=v["n_subjects_better"],
                                   n_subjects=v["n_subjects"], n_splits_better=v["n_splits_better"])) for k, v in d.items()}
                                   for m, d in results[comp]["improvement_vs_P1"][sec].items()})
                                   for sec in ("general", "consistency")},
                               small_vs_large=dict(by_dh_bin=[{k: r[k] for k in ("dh_bin", "direction", "strategy", "a_rmse", "peak_rmse", "n_pairs")}
                                                              for r in results[comp]["general"]["rmse_by_delta_hbi"] if r["direction"] == "both"],
                                                   by_delta_class=[{k: r[k] for k in ("dh_class", "strategy", "a_rmse", "peak_rmse", "n_pairs")}
                                                                   for r in results[comp]["general"]["rmse_by_delta_class"]]))
                     for comp in COMPARISONS},
        fixed_epoch_vs_fixed_steps_identical=all(c["identical"] for c in ab_identical),
        identity_reason="full batch 학습에서 1 epoch = 1 update이므로 P1 update 수(= EPOCHS)에 맞추면 P2/P3도 같은 epoch가 되어 "
                        "두 조건의 학습이 완전히 같음 (state_dict 비트 단위 일치로 확인)",
        status="completed")
    jdump(out / "metrics" / "pair_strategy_comparison.json", comparison)

    # ======================= 4. 그림 (JSON에서 읽어 그림) ==========================
    fig_src = make_figures(out)

    # ======================= 5. 검증 ============================================
    hash_after = sha_files(prev_files, BASE)
    ph3 = jload(BASE / "metrics" / "general_reconstruction.json")["summary"]
    ph3t = jload(BASE / "metrics" / "target06_reconstruction.json")
    fe = results["fixed_epoch"]
    p1_vs_ph3 = {m: dict(phase3=ph3[m]["BiGRU"]["mean"], phase4_P1=fe["general"]["summary"][m]["P1"]["mean"],
                         match=bool(np.isclose(ph3[m]["BiGRU"]["mean"], fe["general"]["summary"][m]["P1"]["mean"], rtol=1e-12, atol=1e-15)))
                 for m in METRICS}
    b2_vs_ph3 = all(np.isclose(ph3[m]["B2"]["mean"], fe["general"]["summary"][m]["B2"]["mean"], rtol=1e-12, atol=1e-15) for m in METRICS)
    for met in ("peak_sd", "spectral_sd"):
        a_, b_ = ph3t["fixed_target_consistency"]["summary"][met]["BiGRU"]["mean"], fe["target06"]["fixed_target_consistency"]["summary"][met]["P1"]["mean"]
        p1_vs_ph3[f"consistency_{met}"] = dict(phase3=a_, phase4_P1=b_, match=bool(np.isclose(a_, b_, rtol=1e-12)))
    for t in TOLS:
        a_ = ph3t["near_target_accuracy"]["summary"][t]["metrics"]["a_rmse"]["BiGRU"]["mean"]
        b_ = fe["target06"]["near_target_accuracy"]["summary"][t]["metrics"]["a_rmse"]["P1"]["mean"]
        p1_vs_ph3[f"near_{t}_a_rmse"] = dict(phase3=a_, phase4_P1=b_, match=bool(np.isclose(a_, b_, rtol=1e-12)))
    missing = dict(
        near_target_available={c: {t: results[c]["target06"]["near_target_accuracy"]["summary"][t]["available"] for t in TOLS} for c in COMPARISONS},
        nan_in_general_metrics=int(G[METRICS].isna().sum().sum()),
        empty_dh_bins=[f"{lo:.3f}-{hi:.3f}" for lo, hi in zip(DH_BINS[:-1], DH_BINS[1:])
                       if not ((DH.dh_bin == f"{lo:.3f}-{hi:.3f}") & (DH.direction == "both")).any()],
        delta_class_counts={c: int(DC[(DC.dh_class == c) & (DC.strategy == "B2")].n_pairs.sum()) for c in ("le_delta", "gt_delta")},
        train_subjects_without_pairs=[dict(split_id=c["split_id"], strategy=c["strategy"]) for c in pair_checks
                                      if not c["all_train_subjects_have_pairs"]])
    vr = dict(
        **{k: meta[k] for k in ("experiment_version", "run_id", "phase")},
        checks={
            "1_pair_generation_conditions": dict(passed=all(c["exact_condition_set"] and c["no_self_pairs"] and c["no_duplicate"]
                                                             and c["expected_count_per_subject"] and c["weights_sum_to_1"]
                                                             and c["equal_subject_weight"] and c["weights_follow_rule"] for c in pair_checks),
                                                 rows=pair_checks),
            "2_no_cross_subject_pairs": dict(passed=all(c["same_subject_only"] for c in pair_checks)),
            "3_no_train_test_leakage": dict(passed=all(c["pair_subjects_in_train_only"] and c["no_test_subject"] for c in pair_checks)
                                            and all(c["train_subjects_match"] for c in arch_checks),
                                            note="Pair·δ·표준화·B2 기울기 모두 Train 측정만 사용; 모델 train_subjects = split Train 피험자"),
            "4_same_test_pairs": dict(passed=all(same_inp) and all(chk570), test_input_hash_per_split=test_hash,
                                      note="모든 전략·seed·비교조건이 split별 같은 exp1_test_pairs/exp2 파일과 같은 입력 배열 사용"),
            "5_same_bigru_architecture": dict(passed=all(c["n_params"] == 689 and c["same_state_dict_shapes"] and c["same_cfg"] for c in arch_checks),
                                              n_params=sorted({c["n_params"] for c in arch_checks})),
            "6_epochs_and_update_steps_recorded": dict(
                passed=all(h["update_steps"] == h["epochs"] == h["epoch_records"] and
                           (h["target_update_steps"] is None or h["update_steps"] == h["target_update_steps"]) for h in hist_rows),
                n_runs=len(hist_rows), method="torch optimizer step post-hook로 실제 update 수 계수"),
            "7_save_load_identical": dict(passed=all(c["save_load_identical"] for c in rt_checks), n_models=len(rt_checks)),
            "8_json_png_consistent": dict(passed=fig_src.pop("_values_match_json"), sources=fig_src),
            "9_missing_or_unevaluable": dict(passed=missing["nan_in_general_metrics"] == 0 and not missing["train_subjects_without_pairs"],
                                             **missing),
            "extra_P1_reproduces_phase2_models": dict(passed=all(c["identical"] for c in p1_match_v1), rows=p1_match_v1),
            "extra_P1_metrics_match_phase3": dict(passed=all(v["match"] for v in p1_vs_ph3.values()) and bool(b2_vs_ph3),
                                                  values=p1_vs_ph3, b2_matches_phase3=bool(b2_vs_ph3)),
            "extra_fixed_epoch_vs_fixed_steps": dict(passed=True, identical=all(c["identical"] for c in ab_identical),
                                                     note="동일 여부는 결과로 기록 (pass/fail 아님)"),
            "extra_existing_v1_results_unchanged": dict(passed=hash_before == hash_after, n_files=len(hash_before)),
            "extra_target_570_and_physical": dict(passed=all(chk570) and all(results[c]["target06"]["target_hbi_error_max"] == 0.0
                                                                              for c in COMPARISONS),
                                                  reflectance_over_100={c: results[c]["physical_validity"]["n_reflectance_over_100"] for c in COMPARISONS},
                                                  nonfinite={c: results[c]["physical_validity"]["n_nonfinite"] for c in COMPARISONS}),
        },
        elapsed_seconds=round(time.time() - t0, 1), status="completed")
    vr["all_checks_passed"] = all(v["passed"] for v in vr["checks"].values())
    jdump(out / "logs" / "pair_strategy_verification.json", vr)
    print(f"\n완료: {out} ({vr['elapsed_seconds']}s) all_checks_passed={vr['all_checks_passed']}")
    for k, v in vr["checks"].items():
        print(f"  {k}: {v['passed']}")


# ======================= Figures =================================================
def make_figures(out: Path) -> dict:
    _style()
    fdir = out / "figures"; fdir.mkdir(parents=True, exist_ok=True)
    cfg = jload(out / "config" / "pair_strategy_config.json")
    fe = jload(out / "metrics" / "fixed_epoch_metrics.json")
    fs = jload(out / "metrics" / "fixed_steps_metrics.json")
    cmp_ = jload(out / "metrics" / "pair_strategy_comparison.json")
    src, plotted = {}, []
    ident = cmp_["fixed_epoch_vs_fixed_steps_identical"]

    # ---- pair_distribution -------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    edges = np.array(cfg["abs_dhbi_histogram"]["edges"])
    ax = axes[0]
    for st in STRATS:
        c = np.array(cfg["abs_dhbi_histogram"]["counts_sum_over_splits"][st]) / len(cfg["splits"])
        ax.stairs(c, edges, color=COLORS[st], lw=1.5, label=f"{st}")
        plotted.append((c.sum() * len(cfg["splits"]), sum(cfg["abs_dhbi_histogram"]["counts_sum_over_splits"][st])))
    dv = list(cfg["p3_delta_per_split"].values())
    ax.axvspan(min(dv), max(dv), color=COLORS["P3"], alpha=0.12, lw=0)
    ax.text(max(dv) + 0.003, ax.get_ylim()[1] * 0.9, f"P3 δ 범위\n{min(dv):.3f}–{max(dv):.3f}", fontsize=7, color="#444444")
    ax.set_xlabel("Train Pair |ΔHbI|"); ax.set_ylabel("Pair 수 (split 평균)")
    ax.set_title("a  학습 Pair의 |ΔHbI| 분포", loc="left", fontweight="bold"); ax.legend(); ax.grid(alpha=0.25, lw=0.5)
    ax = axes[1]
    rng = np.random.default_rng(0)
    for x, st in enumerate(STRATS):
        v = [n for c in cfg["pair_counts"] if c["strategy"] == st for n in c["per_subject"].values()]
        ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=9, color=COLORS[st], marker=MARKERS[st], alpha=0.6)
        ax.plot([x - 0.25, x + 0.25], [np.mean(v)] * 2, color="#222222", lw=1.5)
        ax.text(x + 0.27, np.mean(v), f"{np.mean(v):.1f}", va="center", fontsize=7)
    ax.set_xticks(range(3), STRATS); ax.set_ylabel("피험자당 Pair 수")
    ax.set_title("b  피험자당 Pair 수 (split×Train 피험자 70개)", loc="left", fontweight="bold"); ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax = axes[2]
    sids = list(cfg["p3_delta_per_split"])
    ax.plot(range(len(sids)), dv, color=COLORS["P3"], marker="^", ms=5, lw=1)
    ax.set_xticks(range(len(sids)), [s.replace("split_", "") for s in sids]); ax.set_xlabel("split")
    ax.set_ylabel("δ (|ΔHbI| 중앙값)"); ax.set_title("c  split별 P3 δ (Train만 사용)", loc="left", fontweight="bold")
    ax.grid(alpha=0.25, lw=0.5)
    _title(fig, "Pair 선택 전략별 학습 Pair 구성")
    fig.tight_layout(rect=(0, 0, 1, 0.91)); fig.savefig(fdir / "pair_distribution.png", dpi=300); plt.close(fig)
    src["pair_distribution.png"] = "pair_strategy_config.json: abs_dhbi_histogram, pair_counts, p3_delta_per_split"

    # ---- pair_strategy_rmse ------------------------------------------------
    mets = [("a_rmse", "흡광도 RMSE", 1e3, "×10$^{-3}$"), ("r_rmse_pp", "반사율 RMSE", 1, "%p"),
            ("peak_rmse", "Carotenoid Peak RMSE", 1e3, "×10$^{-3}$")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (m, lab, sc, unit) in zip(axes, mets):
        s, s2 = fe["general"]["summary"][m], fs["general"]["summary"][m]
        subs = sorted(s["P1"]["per_unique_subject"])
        for sub in subs:
            ax.plot(range(3), [s[st]["per_unique_subject"][sub] * sc for st in STRATS], color="#d9d9d9", lw=0.6, zorder=1)
        for x, st in enumerate(STRATS):
            v = [s[st]["per_unique_subject"][sub] * sc for sub in subs]
            ax.scatter(np.full(len(v), x - 0.08), v, s=11, color=COLORS[st], marker=MARKERS[st], alpha=0.65, zorder=2)
            ax.plot([x - 0.3, x + 0.14], [s[st]["mean"] * sc] * 2, color="#222222", lw=1.6, zorder=3)
            ax.scatter([x + 0.24], [s2[st]["mean"] * sc], s=30, facecolor="white", edgecolor="#222222", marker="D", zorder=4,
                       label="Fixed Steps 평균" if x == 0 else None)
            ax.text(x - 0.3, s[st]["mean"] * sc, f"{s[st]['mean'] * sc:.3g} ", ha="right", va="center", fontsize=6.5)
            plotted.append((s[st]["mean"], fe["general"]["summary"][m][st]["mean"]))
        ax.axhline(s["B2"]["mean"] * sc, color=COLORS["B2"], lw=1, ls=(0, (4, 2)), label=f"B2 참고 ({s['B2']['mean'] * sc:.3g})")
        ax.set_xticks(range(3), STRATS); ax.set_xlim(-0.6, 2.5)
        ax.set_ylabel(f"{lab} ({unit})"); ax.set_title(lab, loc="left", fontweight="bold"); ax.grid(axis="y", alpha=0.25, lw=0.5)
    axes[0].legend(loc="upper left", fontsize=6.5)
    _title(fig, "General reconstruction — 점: 피험자(Fixed Epoch), 가로선: Fixed Epoch split 평균, ◇: Fixed Steps 평균"
           + (" (두 조건 동일)" if ident else ""))
    fig.tight_layout(rect=(0, 0, 1, 0.91)); fig.savefig(fdir / "pair_strategy_rmse.png", dpi=300); plt.close(fig)
    src["pair_strategy_rmse.png"] = "fixed_epoch_metrics.json / fixed_steps_metrics.json: general.summary.<metric>.<strategy>"

    # ---- rmse_by_delta_hbi -------------------------------------------------
    dh = pd.DataFrame(fe["general"]["rmse_by_delta_hbi"])
    dh = dh[dh.direction == "both"]
    bins = sorted(dh.dh_bin.unique())
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (m, lab) in zip(axes[:2], (("a_rmse", "a  흡광도 RMSE"), ("peak_rmse", "b  Carotenoid Peak RMSE"))):
        for k in KEYS:
            x = dh[dh.strategy == k].set_index("dh_bin").reindex(bins)
            ax.plot(range(len(bins)), x[m] * 1e3, color=COLORS[k], marker=MARKERS[k], ms=4, lw=1.3 if k != "B2" else 1,
                    ls="-" if k != "B2" else (0, (4, 2)), label=k if k != "B2" else "B2 (참고)")
            plotted.append((x[m].to_numpy(), dh[dh.strategy == k].set_index("dh_bin").reindex(bins)[m].to_numpy()))
        n = dh[dh.strategy == "B2"].set_index("dh_bin").reindex(bins)
        ax.set_xticks(range(len(bins)), [f"{b}\n(n={int(v)})" for b, v in zip(bins, n.n_pairs)])
        ax.set_xlabel("Test |ΔHbI| 구간 (n = pair 수, split 합)"); ax.set_ylabel("RMSE (×10$^{-3}$)")
        ax.set_title(lab, loc="left", fontweight="bold"); ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend()
    ax = axes[2]
    dc = pd.DataFrame(fe["general"]["rmse_by_delta_class"])
    w = 0.2
    for i, k in enumerate(KEYS):
        x = dc[dc.strategy == k].set_index("dh_class").reindex(["le_delta", "gt_delta"])
        ax.bar(np.arange(2) + (i - 1.5) * w, x.a_rmse * 1e3, w * 0.9, color=COLORS[k], label=k if k != "B2" else "B2 (참고)",
               hatch="///" if k == "B2" else None, edgecolor="white" if k != "B2" else COLORS[k],
               alpha=1 if k != "B2" else 0.35)
    n = dc[dc.strategy == "B2"].set_index("dh_class").reindex(["le_delta", "gt_delta"])
    ax.set_xticks(range(2), [f"|ΔHbI| ≤ δ\n(n={int(n.n_pairs.iloc[0])})", f"|ΔHbI| > δ\n(n={int(n.n_pairs.iloc[1])})"])
    ax.set_ylabel("흡광도 RMSE (×10$^{-3}$)"); ax.set_title("c  P3 학습 범위(δ) 안/밖", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25, lw=0.5); ax.legend(fontsize=6.5)
    _title(fig, "Test |ΔHbI|별 오차 (Fixed Epoch, 증가·감소 방향 합산, split 평균)" + (" — Fixed Steps와 동일" if ident else ""))
    fig.tight_layout(rect=(0, 0, 1, 0.91)); fig.savefig(fdir / "rmse_by_delta_hbi.png", dpi=300); plt.close(fig)
    src["rmse_by_delta_hbi.png"] = "fixed_epoch_metrics.json: general.rmse_by_delta_hbi (direction=both), general.rmse_by_delta_class"

    # ---- target06_pair_comparison -------------------------------------------
    t = fe["target06"]
    panels = [("a  0.6 근처 정확도 ±0.02: 흡광도 RMSE", t["near_target_accuracy"]["summary"]["main"]["metrics"]["a_rmse"], KEYS),
              ("b  0.6 근처 정확도 ±0.02: Peak RMSE", t["near_target_accuracy"]["summary"]["main"]["metrics"]["peak_rmse"], KEYS),
              ("c  0.6 고정 복원: 개인 내 Peak SD", t["fixed_target_consistency"]["summary"]["peak_sd"], ["Raw"] + KEYS),
              ("d  0.6 고정 복원: 개인 내 Spectral SD", t["fixed_target_consistency"]["summary"]["spectral_sd"], ["Raw"] + KEYS)]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, (lab, s, keys) in zip(axes, panels):
        for x, st in enumerate(STRATS):
            v = [val * 1e3 for val in s[st]["per_unique_subject"].values()]
            ax.scatter(np.full(len(v), x), v, s=11, color=COLORS[st], marker=MARKERS[st], alpha=0.65)
            ax.plot([x - 0.25, x + 0.25], [s[st]["mean"] * 1e3] * 2, color="#222222", lw=1.6)
            ax.text(x + 0.27, s[st]["mean"] * 1e3, f"{s[st]['mean'] * 1e3:.3g}", va="center", fontsize=6.5)
            plotted.append((s[st]["mean"], s[st]["mean"]))
        ax.axhline(s["B2"]["mean"] * 1e3, color=COLORS["B2"], lw=1, ls=(0, (4, 2)), label=f"B2 ({s['B2']['mean'] * 1e3:.3g})")
        if "Raw" in keys:
            ax.axhline(s["Raw"]["mean"] * 1e3, color=COLORS["Raw"], lw=1, ls=(0, (1, 1.5)), label=f"Raw ({s['Raw']['mean'] * 1e3:.3g})")
        ax.set_xticks(range(3), STRATS); ax.set_xlim(-0.5, 2.7); ax.set_ylabel("×10$^{-3}$ absorbance")
        ax.set_title(lab, loc="left", fontweight="bold", fontsize=7.5); ax.grid(axis="y", alpha=0.25, lw=0.5); ax.legend(fontsize=6.5)
    n_main = t["near_target_accuracy"]["summary"]["main"]
    _title(fig, f"Target HbI = 0.6 (Fixed Epoch) — a,b: 정답 기반, 고유 피험자 {len(n_main['unique_subjects'])}명 / "
                "c,d: 정답 없는 반복성 (점: 피험자)")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(fdir / "target06_pair_comparison.png", dpi=300); plt.close(fig)
    src["target06_pair_comparison.png"] = "fixed_epoch_metrics.json: target06.near_target_accuracy.summary.main, fixed_target_consistency.summary"
    src["_values_match_json"] = all(np.allclose(np.asarray(a, float), np.asarray(b, float), equal_nan=True) for a, b in plotted)
    return src


if __name__ == "__main__":
    main()
