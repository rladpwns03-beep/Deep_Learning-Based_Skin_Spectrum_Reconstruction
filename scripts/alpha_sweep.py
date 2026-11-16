"""v1 추가 실험 — BiGRU 정규화 강도 alpha(slope_l2)에 따른 전체 오차와 Peak 안정도.

실행 (code 폴더, conda 환경 skin_spectra):
    python scripts/alpha_sweep.py          # results/holdout/alpha_sweep/ 에 저장 (기존 결과가 있으면 중단)

설계
  - alpha = loss의 slope_l2:  loss = MSE(570 nm 제외) + alpha · mean(r²),  r = B2 기울기에 더하는 BiGRU 보정.
    alpha = 0 -> 보정 제한 없음, alpha -> ∞ -> r -> 0 -> B2와 같아짐 (B2를 기준선으로 함께 표시).
  - 분할: v1과 같은 7:3 split 10개, seed 0/1/2, 나머지 설정(hidden 8, lr 3e-3, 150 epoch, clip 1.0)은 v1과 같음.
  - 지표 (common/metrics.py, v1과 같은 함수, 평가 단위 = 피험자 -> split 평균 -> split 10개 평균):
      전체 오차      : 일반 복원 흡광도 RMSE (test 피험자 모든 i != j 쌍, 570 nm 제외)
      Peak 정확도    : 일반 복원 490 nm Peak RMSE
      Peak 안정도    : 같은 피험자 측정 10개를 모두 HbI 0.6으로 복원했을 때 Peak의 SD (consistency.peak_sd)
      0.6 근처 정답  : |HbI-0.6| <= 0.02 정답 쌍의 흡광도/Peak RMSE (참고)
  - 주의: test 결과로 alpha를 고르면 test 누출. 이 실험은 민감도 분석이며 v1 설정(alpha=1e-3)은 바꾸지 않는다.
  - 검증: alpha=1e-3 결과가 기존 v1 지표(results/holdout/metrics)와 같은지 확인.
"""
from __future__ import annotations

import datetime as dt
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
from common import config as C                                         # noqa: E402
from common.data_loader import HBI_INDEX, load_spectra, repeated_splits  # noqa: E402
from common.metrics import consistency, group_metrics                  # noqa: E402
from models.bigru import CFG as BIGRU_CFG, BiGRUModel                  # noqa: E402
from models.linear import LinearModel                                  # noqa: E402

ALPHAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
BASE = ROOT / "results" / "holdout"
OUT = BASE / "alpha_sweep"
COL = {"BiGRU": "#1baf7a", "B2": "#2a78d6"}


def jdump(p, obj, indent=1):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def evaluate(model, te):
    """test 피험자별 지표 dict 리스트."""
    out = []
    for s in te.subjects:
        x = te.select([s]); n = len(x)
        I, J = np.array([(i, j) for i in range(n) for j in range(n) if i != j]).T
        P = model.predict(x.a[I], x.hbi[J]); Y = x.a[J]; T = x.hbi[J]
        g = group_metrics(P, Y)
        sel = np.abs(T - C.TARGET_HBI) <= C.NEAR_TOL_MAIN
        nr = group_metrics(P[sel], Y[sel]) if sel.any() else None
        cs = consistency(model.predict(x.a[np.argsort(x.trial)], C.TARGET_HBI))
        out.append(dict(subject=s, a_rmse=g["a_rmse"], peak_rmse=g["peak_rmse"], peak_bias=g["peak_bias"],
                        peak_sd=cs["peak_sd"], spectral_sd=cs["spectral_sd"],
                        near_a_rmse=None if nr is None else nr["a_rmse"], near_peak_rmse=None if nr is None else nr["peak_rmse"]))
    return out


def main():
    if OUT.exists() and any(OUT.rglob("*.json")):
        sys.exit(f"[중단] {OUT} 에 이미 결과가 있음. 덮어쓰지 않음.")
    torch.set_num_threads(1)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S"); t0 = time.time()
    data = load_spectra()
    jdump(OUT / "config" / "alpha_sweep_config.json", dict(
        experiment_version="v1", experiment="alpha (slope_l2) sensitivity", run_id=run_id, alphas=ALPHAS,
        base_config={k: (list(v) if isinstance(v, tuple) else v) for k, v in BIGRU_CFG.items()},
        v1_alpha=BIGRU_CFG["slope_l2"], splits="v1 7:3 x10 (data/splits)", seeds=list(C.BIGRU_SEEDS),
        note="민감도 분석. test 결과로 alpha를 선택하지 않음 (v1 설정 유지)."))
    rows, hist = [], []
    for rep, tr, te in repeated_splits(data):
        sid = f"split_{rep:02d}"
        b2 = LinearModel().fit(tr)
        for r in evaluate(b2, te):
            rows.append(dict(model="B2", alpha=None, seed=-1, split_id=sid, **r))
        for alpha in ALPHAS:
            for seed in C.BIGRU_SEEDS:
                cfg = dict(BIGRU_CFG, slope_l2=alpha)
                m = BiGRUModel(seed=seed, cfg=cfg).fit(tr)
                H = np.array(m.history)
                hist.append(dict(split_id=sid, alpha=alpha, seed=seed, final_loss=float(H[-1, 1]), final_mse=float(H[-1, 2]),
                                 finite=bool(np.isfinite(H).all())))
                for r in evaluate(m, te):
                    rows.append(dict(model="BiGRU", alpha=alpha, seed=seed, split_id=sid, **r))
            print(f"[{sid}] alpha {alpha:g} 완료 ({time.time() - t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)

    # ---- 집계: 피험자 -> (seed 평균) -> split 평균 -> split 10개 평균 / 범위 ----
    mets = ["a_rmse", "peak_rmse", "peak_sd", "spectral_sd", "near_a_rmse", "near_peak_rmse"]
    def agg(d):
        per_split = d.groupby(["split_id", "subject"])[mets].mean().groupby("split_id").mean()
        per_subject = d.groupby(["split_id", "subject"])[mets].mean().groupby("subject").mean()
        return {m: dict(mean=float(per_split[m].mean()), split_min=float(per_split[m].min()), split_max=float(per_split[m].max()),
                        split_sd=float(per_split[m].std(ddof=1)), per_split=per_split[m].dropna().to_dict(),
                        per_subject=per_subject[m].dropna().to_dict()) for m in mets}
    summary = {"B2": agg(D[D.model == "B2"])}
    for a in ALPHAS:
        d = D[(D.model == "BiGRU") & (D.alpha == a)]
        summary[f"BiGRU_alpha_{a:g}"] = agg(d)
        summary[f"BiGRU_alpha_{a:g}"]["seed_sd_a_rmse"] = float(
            d.groupby(["seed", "split_id", "subject"]).a_rmse.mean().groupby(["seed", "split_id"]).mean().groupby("split_id").std(ddof=1).mean())

    # ---- 검증: alpha = v1 값이면 기존 v1 지표와 같아야 함 ----
    v1 = jload(BASE / "metrics" / "general_reconstruction.json")["subject_wise"]
    v1d = pd.DataFrame(v1); v1d = v1d[v1d.model == "BiGRU"]
    mine = D[(D.model == "BiGRU") & (D.alpha == BIGRU_CFG["slope_l2"])]
    mg = mine.merge(v1d[["split_id", "subject", "seed", "a_rmse", "peak_rmse"]], on=["split_id", "subject", "seed"], suffixes=("", "_v1"))
    reproduce = dict(n_compared=int(len(mg)),
                     max_abs_diff_a_rmse=float((mg.a_rmse - mg.a_rmse_v1).abs().max()),
                     max_abs_diff_peak_rmse=float((mg.peak_rmse - mg.peak_rmse_v1).abs().max()))
    reproduce["matches_v1"] = bool(reproduce["n_compared"] == 90 and reproduce["max_abs_diff_a_rmse"] < 1e-9
                                   and reproduce["max_abs_diff_peak_rmse"] < 1e-9)

    jdump(OUT / "metrics" / "alpha_sweep_summary.json", dict(
        experiment_version="v1", run_id=run_id, alphas=ALPHAS, unit="absorbance", summary=summary,
        reproduce_v1_check=reproduce, all_training_finite=bool(all(h["finite"] for h in hist)),
        aggregation="pair -> 피험자(group 지표) -> seed 평균 -> split 평균(test 피험자 3명) -> split 10개 평균; min/max = split 범위",
        near_note="near_* 는 0.6 근처 정답(±0.02)이 test에 있는 피험자만 (split마다 1명)", status="completed"))
    jdump(OUT / "metrics" / "alpha_sweep_rows.json", D.where(D.notna(), None).to_dict(orient="records"), indent=None)
    jdump(OUT / "logs" / "alpha_sweep_training.json", hist)
    make_figures(summary)
    print(f"완료 ({time.time() - t0:.0f}s) | v1 재현: {reproduce}")


def make_figures(summary):
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    fd = OUT / "figures"; fd.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(ALPHAS)); xl = [f"{a:g}" for a in ALPHAS]

    def series(m):
        b = [summary[f"BiGRU_alpha_{a:g}"][m] for a in ALPHAS]
        return (np.array([v["mean"] for v in b]) * 1e3, np.array([v["split_min"] for v in b]) * 1e3,
                np.array([v["split_max"] for v in b]) * 1e3, summary["B2"][m])

    def panel(ax, m, title, ylabel):
        mu, lo, hi, b2 = series(m)
        ax.fill_between(x, lo, hi, color=COL["BiGRU"], alpha=0.15, lw=0, label="BiGRU split 범위 (최소-최대)")
        ax.plot(x, mu, color=COL["BiGRU"], marker="o", ms=4, lw=1.4, label="BiGRU (split 10개 평균)")
        ax.axhline(b2["mean"] * 1e3, color=COL["B2"], ls="--", lw=1.1, label=f"B2 ({b2['mean'] * 1e3:.2f})")
        ax.axhspan(b2["split_min"] * 1e3, b2["split_max"] * 1e3, color=COL["B2"], alpha=0.07, lw=0)
        v1i = ALPHAS.index(BIGRU_CFG["slope_l2"])
        ax.axvline(v1i, color="#888888", lw=0.8, ls=":"); ax.text(v1i, ax.get_ylim()[1], " v1 설정", va="top", fontsize=7, color="#666666")
        ax.set_xticks(x); ax.set_xticklabels(xl); ax.set_xlabel("alpha (slope_l2, 보정 r 크기 제한)")
        ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontweight="bold"); ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=6.5, loc="best")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    panel(axes[0], "a_rmse", "a  일반 복원 흡광도 RMSE (570 nm 제외)", "RMSE x1e-3")
    panel(axes[1], "near_a_rmse", "b  0.6 근처 정답 흡광도 RMSE (참고, split당 1명)", "RMSE x1e-3")
    fig.suptitle("정규화 강도(alpha)에 따른 전체 오차 — v1 7:3 split 10개 x seed 3 (민감도 분석, test로 alpha 선택하지 않음)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(fd / "alpha_total_error.png", dpi=300); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    panel(axes[0], "peak_rmse", "a  Peak 정확도: 490 nm Peak RMSE", "Peak RMSE x1e-3")
    panel(axes[1], "peak_sd", "b  Peak 안정도: 같은 피험자 10개 입력 -> HbI 0.6 복원 시 Peak SD", "Peak SD x1e-3 (낮을수록 안정)")
    fig.suptitle("정규화 강도(alpha)에 따른 Peak 정확도와 안정도 — v1 7:3 split 10개 x seed 3",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(fd / "alpha_peak_stability.png", dpi=300); plt.close(fig)


def subject_figure(summary=None):
    """피험자 10명 각각의 선 (피험자별 평균 = 그 피험자가 test였던 split·seed 평균, 동일 비중).
    오른쪽 끝 'B2' 위치에 같은 피험자의 B2 값 (alpha -> ∞ 극한)."""
    if summary is None:
        summary = jload(OUT / "metrics" / "alpha_sweep_summary.json")["summary"]
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    subs = sorted(summary["B2"]["a_rmse"]["per_subject"])
    cmap = plt.get_cmap("tab10")
    x = np.arange(len(ALPHAS)); xb = len(ALPHAS) + 0.6
    panels = [("a_rmse", "a  전체 오차: 일반 복원 흡광도 RMSE", "RMSE x1e-3"),
              ("peak_rmse", "b  Peak 정확도: 490 nm Peak RMSE", "Peak RMSE x1e-3"),
              ("peak_sd", "c  Peak 안정도: HbI 0.6 복원 시 Peak SD", "Peak SD x1e-3 (낮을수록 안정)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (m, title, yl) in zip(axes, panels):
        allv = []
        for n, s in enumerate(subs):
            v = np.array([summary[f"BiGRU_alpha_{a:g}"][m]["per_subject"][s] for a in ALPHAS]) * 1e3
            b2 = summary["B2"][m]["per_subject"][s] * 1e3
            ax.plot(x, v, color=cmap(n), lw=1.0, marker="o", ms=3, label=s)
            ax.plot([x[-1], xb], [v[-1], b2], color=cmap(n), lw=0.6, ls=":")
            ax.scatter([xb], [b2], color=cmap(n), marker="s", s=16, zorder=3)
            allv.append(v)
        mean = np.mean(allv, 0); b2m = np.mean([summary["B2"][m]["per_subject"][s] for s in subs]) * 1e3
        ax.plot(x, mean, color="#111111", lw=2.2, label="10명 평균 (동일 비중)", zorder=4)
        ax.scatter([xb], [b2m], color="#111111", marker="s", s=30, zorder=5)
        v1i = ALPHAS.index(BIGRU_CFG["slope_l2"])
        ax.axvline(v1i, color="#888888", lw=0.8, ls=":")
        ax.set_xticks(list(x) + [xb]); ax.set_xticklabels([f"{a:g}" for a in ALPHAS] + ["B2"])
        ax.set_xlabel("alpha (slope_l2)   |   B2 = 보정 없음 (alpha → ∞ 극한)")
        ax.set_ylabel(yl); ax.set_title(title, loc="left", fontweight="bold"); ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(fontsize=6.5, ncol=2, loc="upper left")
    fig.suptitle("피험자별 alpha 민감도 (선 1개 = 피험자 1명, 그 피험자가 test였던 split·seed 평균; 점선 세로줄 = v1 설정 1e-3)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fd = OUT / "figures"; fd.mkdir(parents=True, exist_ok=True)
    fig.savefig(fd / "alpha_per_subject.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--subject-figure":
        subject_figure()
    else:
        main()
        subject_figure()
