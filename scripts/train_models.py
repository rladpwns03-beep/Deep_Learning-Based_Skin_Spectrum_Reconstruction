"""Phase 2 — 모델 학습과 학습 검증 (Test 데이터 평가 없음).

실행 (code 폴더, conda 환경 skin_spectra):
    python scripts/train_models.py                 # results/holdout/ 에 저장 (이미 결과가 있으면 중단)
    python scripts/train_models.py --run-id NAME   # results/holdout/runs/NAME/ 에 별도 실행으로 저장
    python scripts/train_models.py --version NAME --epochs 300   # results/NAME/ 에 BiGRU epoch만 바꿔 새 버전으로 저장

하는 일
  - data/splits/split_plan.csv 의 기존 7:3 분할 10개를 그대로 사용 (분할·평가 Pair 재생성 없음)
  - 각 분할의 Train 데이터만으로 B2 기울기, B1 bin 평균, BiGRU(seed 0/1/2, 150 epoch, P1 All Pairs) 학습
  - 모델 저장·재로드, 출력 조건, 재현성, NaN/Inf·발산 검사 (모두 Train 데이터만 사용)
  - 학습 이력·요약·검증 JSON과 학습 손실 PNG 저장

Training loss 감소는 학습이 정상 작동한다는 참고 정보일 뿐, 일반화(Test) 성능을 보장하지 않는다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
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

import ablation as legacy                                     # noqa: E402  기존 구현 (일치 확인용, 수정하지 않음)
from common import config as C                                # noqa: E402
from common.data_loader import HBI_INDEX, WL, load_spectra, repeated_splits  # noqa: E402
from models.bigru import CFG as BIGRU_CFG, BiGRUModel         # noqa: E402
from models.hb_index import BIN_CENTERS, BIN_EDGES, BinnedMeanModel  # noqa: E402
from models.linear import LinearModel                         # noqa: E402

VERSION = "holdout"
EPOCHS = C.BIGRU["epochs"]       # 기본 150 (common/config.py). --epochs로 새 버전 실험 시에만 변경
PAIR_STRATEGY = "P1_all_pairs"
DIVERGENCE_FACTOR = 10.0       # 학습 중 최대 손실 > 초기 손실 x 10 이면 발산으로 판정
BATCH_TOL = 1e-6               # 단일 입력 vs 배치 입력 예측 허용 차이 (흡광도, float32 연산 오차)
LOSS_UNIT = "absorbance^2 (subject-weighted MSE over 570 nm 제외 30 파장 + 0.001 x residual-slope L2)"


# ---------------------------------------------------------------------------
def jdump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_fixed_splits(data):
    """기존 분할 파일을 읽고, 코드(data_loader)의 분할과 같은지 확인한다."""
    plan = pd.read_csv(ROOT / "data" / "splits" / "split_plan.csv")
    code = {rep: (tr.subjects, te.subjects) for rep, tr, te in repeated_splits(data)}
    splits = []
    for r in plan.itertuples():
        test, train = r.test.split(), r.train.split()
        if (sorted(train), sorted(test)) != code[r.rep]:
            raise RuntimeError(f"split {r.rep}: 파일과 코드의 분할이 다름")
        if set(train) & set(test):
            raise RuntimeError(f"split {r.rep}: train/test 피험자 중복")
        splits.append(dict(split_id=f"split_{r.rep:02d}", rep=int(r.rep), train=train, test=test))
    return splits


def contracts(model, part) -> dict:
    """출력 조건 검사 (Train 데이터만 사용)."""
    a, h = part.a, part.hbi
    perm = h[np.random.default_rng(0).permutation(len(h))]           # 다른 Train 측정의 HbI를 목표로
    single = model.predict(a[0], C.TARGET_HBI)
    batch = model.predict(a, C.TARGET_HBI)
    per_row = model.predict(a, perm)
    ident = model.predict(a, h)
    one_by_one = np.vstack([model.predict(x, C.TARGET_HBI) for x in a])
    diff = float(np.abs(one_by_one - batch).max())
    return {
        "output_shape_single_31": single.shape == (len(WL),),
        "batch_prediction_shape": batch.shape == a.shape,
        # BiGRU는 float32 연산이라 배치 크기에 따라 마지막 자릿수가 다를 수 있음 -> 허용오차로 판정
        "single_vs_batch_max_abs_diff": diff,
        f"single_matches_batch_within_{BATCH_TOL:g}": diff <= BATCH_TOL,
        "target_constraint_0.6": bool(np.array_equal(batch[:, HBI_INDEX], np.full(len(a), C.TARGET_HBI))),
        "target_constraint_per_row": bool(np.array_equal(per_row[:, HBI_INDEX], perm)),
        "identity_condition": bool(np.array_equal(ident, a)),
        "finite_outputs": bool(np.isfinite(batch).all() and np.isfinite(per_row).all()),
    }


def roundtrip(model, path, part) -> bool:
    model.save(path)
    loaded = type(model).load(path)
    return bool(np.array_equal(loaded.predict(part.a, C.TARGET_HBI), model.predict(part.a, C.TARGET_HBI)) and
                np.array_equal(loaded.predict(part.a[::-1], part.hbi), model.predict(part.a[::-1], part.hbi)))


def all_true(d) -> bool:
    if isinstance(d, dict):
        return all(all_true(v) for v in d.values())
    if isinstance(d, list):
        return all(all_true(v) for v in d)
    return bool(d) if isinstance(d, (bool, np.bool_)) else True


# ---------------------------------------------------------------------------
def main():
    global VERSION, EPOCHS
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None, help="기존 결과가 있을 때 별도 실행 폴더 이름")
    ap.add_argument("--version", default=VERSION, help="결과 버전 폴더 (results/<version>)")
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="BiGRU epoch 수 (기본 150)")
    args = ap.parse_args()
    VERSION, EPOCHS = args.version, args.epochs
    if VERSION == "holdout" and EPOCHS != C.BIGRU["epochs"]:
        sys.exit("[중단] holdout은 150 epoch 고정. 다른 epoch는 --version NAME 처럼 새 버전으로 저장하세요.")

    base = ROOT / "results" / VERSION
    if args.run_id:
        out = base / "runs" / args.run_id
    else:
        out = base
    has_history = (out / "logs" / "training_history.json").exists()
    has_models = (out / "models").exists() and any((out / "models").glob("*/*"))
    if has_history or has_models:
        sys.exit(f"[중단] {out} 에 이미 학습 결과가 있음. 덮어쓰지 않음. 새 실행은 --run-id 를 지정하세요.")
    run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    torch.set_num_threads(1)                    # 기존 코드와 같은 CPU 1 thread (재현성)
    data = load_spectra()
    splits = load_fixed_splits(data)
    t_start = time.time()

    config = dict(
        experiment_version=VERSION, run_id=run_id, phase="Phase 2 - Model Training", status="running",
        models=["B1", "B2", "BiGRU"], training_pairs=PAIR_STRATEGY, device="cpu", torch_threads=1,
        target_hbi=C.TARGET_HBI, n_splits=len(splits), split_source="data/splits/split_plan.csv",
        splits=splits,
        b2=dict(method="pooled within-subject linear slope, b(570)=1"),
        b1=dict(bin_width=C.B1_BIN_WIDTH, bin_edges=BIN_EDGES.tolist(), empty_bin_policy="nearest valid bin (tie -> lower)"),
        bigru={k: (list(v) if isinstance(v, tuple) else v) for k, v in dict(BIGRU_CFG, epochs=EPOCHS).items()},
        bigru_note="hidden 8 bidirectional GRU, Adam, full batch, zero-init head, loss = weighted MSE + slope L2",
        loss_unit=LOSS_UNIT, divergence_rule=f"max loss > {DIVERGENCE_FACTOR} x epoch-1 loss or non-finite",
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                         pandas=pd.__version__, platform=platform.platform()),
        test_data_used=False,
    )
    jdump(out / "config" / "training_config.json", config)

    history, b1_info, b2_info, verif_splits = [], [], [], []
    spec_legacy = dict(cell="GRU", hidden=8, bi=True, name="BiGRU8")

    for sp in splits:
        sid = sp["split_id"]
        tr = data.select(sp["train"])
        v = dict(split_id=sid, train_subjects=sp["train"], test_subjects=sp["test"])
        print(f"[{sid}] train {sp['train']}", flush=True)

        # ---- B2 ----------------------------------------------------------
        b2 = LinearModel().fit(tr)
        pre = legacy.fit_pre(tr.a, tr.subject)
        v["B2"] = dict(
            slope_finite=bool(np.isfinite(b2.slope).all()), slope_570_is_1=bool(b2.slope[HBI_INDEX] == 1.0),
            matches_legacy_slope=bool(np.array_equal(b2.slope, pre["b1"])),
            matches_legacy_prediction=bool(np.array_equal(b2.predict(tr.a, 0.6), legacy.b1_pred(tr.a, 0.6, pre))),
            fit_on_train_only=b2.train_subjects == sorted(sp["train"]) and not set(b2.train_subjects) & set(sp["test"]),
            save_reload_identical=roundtrip(b2, out / "models" / "B2" / f"{sid}.json", tr),
            contracts=contracts(b2, tr))
        b2_info.append(dict(split_id=sid, slope=b2.slope.tolist()))

        # ---- B1 ----------------------------------------------------------
        b1 = BinnedMeanModel().fit(tr)
        tab = b1.bin_table
        valid = tab.index[tab.n_measurements > 0].to_numpy()
        empty = tab.index[tab.n_measurements == 0].to_numpy()
        fb_ok = all(b1.source[k] == valid[np.argmin(np.abs(BIN_CENTERS[valid] - BIN_CENTERS[k]) + 1e-12 * valid)]
                    and np.array_equal(b1.means[k], b1.means[b1.source[k]]) for k in empty)
        b1.reset_counters()
        c2 = contracts(b1, tr)
        counters = b1.counters()
        v["B1"] = dict(
            bin_means_finite=bool(np.isfinite(b1.means).all()),
            bin_counts_sum_to_train=int(tab.n_measurements.sum()) == len(tr),
            empty_bins=[int(k) for k in empty], fallback_policy_correct=bool(fb_ok),
            fit_on_train_only=b1.train_subjects == sorted(sp["train"]) and not set(b1.train_subjects) & set(sp["test"]),
            save_reload_identical=roundtrip(b1, out / "models" / "B1" / f"{sid}.json", tr),
            contracts=c2)
        b1_info.append(dict(split_id=sid, bin_table=tab.to_dict(orient="records"),
                            n_empty_bins=len(empty), fallback_counters_during_train_checks=counters))

        # ---- BiGRU -------------------------------------------------------
        v["BiGRU"] = {}
        for seed in C.BIGRU_SEEDS:
            t0 = time.time()
            m = BiGRUModel(seed=seed, epochs=EPOCHS, cfg=dict(BIGRU_CFG, epochs=EPOCHS)).fit(tr)
            secs = time.time() - t0
            hist = np.array(m.history)                       # (epoch, loss, mse, grad_norm)
            finite = bool(np.isfinite(hist).all())
            diverged = (not finite) or bool(hist[:, 1].max() > DIVERGENCE_FACTOR * hist[0, 1])
            path = out / "models" / "BiGRU" / f"{sid}_seed_{seed}.pt"
            m2 = BiGRUModel(seed=seed, epochs=EPOCHS, cfg=dict(BIGRU_CFG, epochs=EPOCHS)).fit(tr)               # 같은 설정·시드로 재실행
            same_w = all(torch.equal(a, b) for a, b in zip(m.net.state_dict().values(), m2.net.state_dict().values()))
            chk = dict(
                epochs_recorded=len(hist) == EPOCHS, loss_finite=finite,
                grad_norm_finite=bool(np.isfinite(hist[:, 3]).all()), diverged=diverged,
                b2_from_train_only=bool(np.array_equal(m.b2, b2.slope)),
                fit_on_train_only=m.train_subjects == sorted(sp["train"]) and not set(m.train_subjects) & set(sp["test"]),
                save_reload_identical=roundtrip(m, path, tr),
                rerun_identical_weights=bool(same_w),
                rerun_identical_predictions=bool(np.array_equal(m.predict(tr.a, 0.6), m2.predict(tr.a, 0.6))),
                rerun_identical_history=bool(np.array_equal(hist, np.array(m2.history))),
                contracts=contracts(m, tr))
            if sp["rep"] == 0 and seed == C.BIGRU_SEEDS[0]:   # 기존 ablation.py와 150 epoch 결과 비교 (1회)
                om, opre, _ = legacy.fit(tr.a, tr.subject, spec_legacy, seed, epochs=EPOCHS)
                chk[f"matches_legacy_{EPOCHS}_epochs"] = bool(np.array_equal(m.predict(tr.a, 0.6),
                                                                       legacy.predict(om, opre, tr.a, 0.6)[0]))
            v["BiGRU"][f"seed_{seed}"] = chk
            history.append(dict(split_id=sid, seed=seed, model="BiGRU", train_seconds=round(secs, 3),
                                epoch=hist[:, 0].astype(int).tolist(), loss=hist[:, 1].tolist(),
                                mse=hist[:, 2].tolist(), grad_norm=hist[:, 3].tolist()))
            print(f"   BiGRU seed {seed}: loss {hist[0, 1]:.4e} -> {hist[-1, 1]:.4e}  ({secs:.1f}s)", flush=True)
        verif_splits.append(v)

    # ---- JSON 저장 ---------------------------------------------------------
    common_meta = dict(experiment_version=VERSION, run_id=run_id, training_pairs=PAIR_STRATEGY,
                       test_data_used=False, loss_unit=LOSS_UNIT,
                       note="Training loss는 학습 정상 작동의 참고 정보이며 일반화(Test) 성능을 보장하지 않음. "
                            "epoch e의 loss는 e번째 업데이트 직전 파라미터로 계산한 학습 손실.")
    jdump(out / "logs" / "training_history.json", dict(**common_meta, model="BiGRU",
          bigru_config=config["bigru"], runs=history, status="completed"))

    hist_df = pd.DataFrame([dict(split_id=h["split_id"], seed=h["seed"], initial_loss=h["loss"][0],
                                 final_loss=h["loss"][-1], final_mse=h["mse"][-1], min_loss=min(h["loss"]),
                                 max_loss=max(h["loss"]), train_seconds=h["train_seconds"]) for h in history])
    hist_df["final_over_initial"] = hist_df.final_loss / hist_df.initial_loss
    per_split = hist_df.groupby("split_id").final_loss
    seed_means = hist_df.groupby("seed").final_loss.mean()
    summary = dict(
        **common_meta, model="BiGRU (B1, B2 parameter summaries included)",
        bigru=dict(
            n_training_runs=len(history), epochs=EPOCHS, seeds=list(C.BIGRU_SEEDS),
            runs=hist_df.to_dict(orient="records"),
            final_loss_overall=dict(mean=hist_df.final_loss.mean(), sd=hist_df.final_loss.std(ddof=1), n=len(hist_df)),
            seed_variability=dict(description="split 안에서 seed 3개 final loss의 SD, split 10개 평균",
                                  mean_within_split_sd=per_split.std(ddof=1).mean(), n_splits=per_split.ngroups),
            split_variability=dict(description="seed 평균 final loss를 split 10개에 대해 SD",
                                   sd_of_split_means=per_split.mean().std(ddof=1), n_splits=per_split.ngroups),
            final_loss_by_seed_mean_over_splits={f"seed_{k}": v for k, v in seed_means.items()},
            total_train_seconds=float(hist_df.train_seconds.sum())),
        b2=dict(parameters="파장별 기울기 31개 (models/B2/split_XX.json)", splits=b2_info),
        b1=dict(bin_edges=BIN_EDGES.tolist(), splits=b1_info),
        status="completed")
    jdump(out / "logs" / "training_summary.json", summary)

    # diverged 는 False 여야 정상이므로 별도 판정
    no_div = all(not s["BiGRU"][k]["diverged"] for s in verif_splits for k in s["BiGRU"])
    flat_ok = all(all_true({k: x for k, x in s["BiGRU"][kk].items() if k != "diverged"})
                  for s in verif_splits for kk in s["BiGRU"]) and \
        all(all_true(s["B2"]) and all_true(s["B1"]) for s in verif_splits)
    verification = dict(
        **common_meta, model="B1, B2, BiGRU",
        checks_use_train_data_only=True,
        splits_reused_from_file=True, n_splits_trained=len(splits),
        all_checks_passed=bool(flat_ok and no_div), no_divergence=bool(no_div),
        per_split=verif_splits, status="completed")
    jdump(out / "logs" / "training_verification.json", verification)

    # ---- 그림 (JSON에서 다시 읽어 그림 -> 그림 값 = JSON 값) ------------------
    make_figures(out)

    config["status"] = "completed"
    config["elapsed_seconds"] = round(time.time() - t_start, 1)
    jdump(out / "config" / "training_config.json", config)
    print(f"\n완료: {out}  ({config['elapsed_seconds']}s)  all_checks_passed={verification['all_checks_passed']}")


# ---------------------------------------------------------------------------
SEED_COLORS = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}


def _style():
    plt.rcParams.update({
        "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 8,
        "mathtext.fontset": "custom", "mathtext.rm": "Malgun Gothic", "mathtext.it": "Malgun Gothic",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "axes.edgecolor": "#7a7a7a", "xtick.color": "#7a7a7a", "ytick.color": "#7a7a7a",
        "xtick.labelcolor": "#444444", "ytick.labelcolor": "#444444", "axes.titlesize": 8.5,
    })


def make_figures(out: Path):
    _style()
    hist = json.loads((out / "logs" / "training_history.json").read_text(encoding="utf-8"))["runs"]
    df = pd.DataFrame(hist)
    splits = sorted(df.split_id.unique())

    # training_loss.png: split별 패널, seed별 곡선
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.6), sharex=True, sharey=True)
    for ax, sid in zip(axes.flat, splits):
        for r in df[df.split_id == sid].itertuples():
            ax.plot(r.epoch, np.array(r.loss) * 1e4, color=SEED_COLORS[r.seed], lw=1.0, label=f"seed {r.seed}")
        ax.set_title(sid.replace("_", " "), loc="left")
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("Training loss (×10$^{-4}$)")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.995, 1.0))
    fig.suptitle(f"BiGRU training loss (P1 All Pairs, {EPOCHS} epochs) — Train 데이터 기준", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figures" / "training_loss.png", dpi=300)
    plt.close(fig)

    # training_stability.png: split·seed별 final loss, final/initial 비
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.arange(len(splits))
    for seed, off in zip(sorted(df.seed.unique()), (-0.18, 0.0, 0.18)):
        d = df[df.seed == seed].set_index("split_id").loc[splits]
        fin = np.array([v[-1] for v in d.loss]) * 1e4
        ratio = np.array([v[-1] / v[0] for v in d.loss])
        axes[0].scatter(x + off, fin, s=18, color=SEED_COLORS[seed], label=f"seed {seed}", zorder=3)
        axes[1].scatter(x + off, ratio, s=18, color=SEED_COLORS[seed], label=f"seed {seed}", zorder=3)
    axes[0].set_ylabel("Final training loss (×10$^{-4}$)")
    axes[1].set_ylabel("Final / initial training loss")
    axes[1].axhline(1.0, color="#9a9893", lw=0.7, ls=(0, (3, 2)))
    for ax, t in zip(axes, (f"a  Final training loss (epoch {EPOCHS})", "b  Loss reduction (1 미만 = 감소)")):
        ax.set_xticks(x, [s.split("_")[1] for s in splits])
        ax.set_xlabel("Split")
        ax.set_title(t, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    fig.suptitle("BiGRU training stability across splits and seeds — Train 데이터 기준", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "figures" / "training_stability.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
