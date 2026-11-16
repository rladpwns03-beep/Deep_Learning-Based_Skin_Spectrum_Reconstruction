"""490 nm 카로티노이드 Peak 확대 그림 (평가 결과 재사용, 새로 학습/평가하지 않음).

    python scripts/plot_peak.py [--version holdout]

입력: results/<v>/predictions/predictions.json, results/<v>/logs/training_summary.json
출력: results/<v>/figures/peak490_zoom.png       0.6 근처 정답 쌍, 450-530 nm 확대 (개별 예측 전부)
      results/<v>/figures/peak490_error_all.png  쌍별 Peak 오차 전부 (평균 아님)

쌍 정의는 evaluate.py near_target_accuracy와 같다:
  정답 = |HbI-0.6| <= 0.02 test 측정, 목표 HbI = 정답의 실제 HbI, 입력 = 같은 피험자의 나머지 측정.
BiGRU seed = split별 최종 학습 loss가 최소인 seed (test 결과로 고르지 않음).
Peak P = A(490) - (A(470) + A(510)) / 2  (common/metrics.py)
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]   # 프로젝트 루트 (scripts/ 의 상위)
sys.path.insert(0, str(ROOT))
from common import config as C                     # noqa: E402
from common.data_loader import WL, load_spectra    # noqa: E402
from common.metrics import peak                    # noqa: E402

COLORS = {"B1": "#eb6834", "B2": "#2a78d6", "BiGRU": "#1baf7a"}
MODELS = ["B1", "B2", "BiGRU"]
ZOOM = (WL >= 450) & (WL <= 530)


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def style():
    plt.rcParams.update({
        "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "axes.edgecolor": "#7a7a7a", "xtick.color": "#7a7a7a", "ytick.color": "#7a7a7a",
        "xtick.labelcolor": "#444444", "ytick.labelcolor": "#444444", "axes.titlesize": 8.5,
        "legend.frameon": False, "legend.fontsize": 6.5})


def load_panels(base):
    pr = jload(base / "predictions" / "predictions.json")
    best = {}
    for r in jload(base / "logs" / "training_summary.json")["bigru"]["runs"]:
        if r["split_id"] not in best or r["final_loss"] < best[r["split_id"]][1]:
            best[r["split_id"]] = (r["seed"], r["final_loss"])
    rows = [r for r in pr["general_reconstruction"] if abs(r["target_hbi"] - C.TARGET_HBI) <= C.NEAR_TOL_MAIN]
    panels = {}
    for r in rows:
        panels.setdefault((r["subject"], r["split_id"]), []).append(r)
    return panels, {k: v[0] for k, v in best.items()}


def grid(panels):
    subs = sorted({s for s, _ in panels})
    ncol = max(sum(s == sub for s, _ in panels) for sub in subs)
    return subs, ncol


def fig_zoom(panels, seed, data, out):
    subs, ncol = grid(panels)
    fig, axes = plt.subplots(len(subs), ncol, figsize=(3.0 * ncol, 2.8 * len(subs)), squeeze=False)
    w = WL[ZOOM]
    for i, sub in enumerate(subs):
        sps = sorted(sp for s, sp in panels if s == sub)
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(sps):
                ax.axis("off"); continue
            sp = sps[j]; rs = panels[(sub, sp)]
            keys = {"B1": "B1", "B2": "B2", "BiGRU": f"BiGRU_seed{seed[sp]}"}
            seen = {}
            for r in rs:   # 같은 입력은 목표 HbI가 같으면 예측도 같으므로 입력별로 한 번만
                seen.setdefault((r["input_trial"], round(r["target_hbi"], 6)), r)
            for m in MODELS:
                for k, r in enumerate(seen.values()):
                    a = np.asarray(r["predictions"][keys[m]])
                    ax.plot(w, a[ZOOM], color=COLORS[m], lw=0.6, alpha=0.45,
                            label=(f"BiGRU seed {seed[sp]}" if m == "BiGRU" else m) if k == 0 else None)
            for k, t in enumerate(sorted({r["target_trial"] for r in rs})):
                a = data.a[(data.subject == sub) & (data.trial == t)][0]
                ax.plot(w, a[ZOOM], color="#111111", lw=1.6, ls="-" if k == 0 else "--", marker="o", ms=2.5,
                        label=f"정답 t{t} (P {peak(a) * 1e3:.1f})", zorder=5)
                b = (a[WL == 470][0] + a[WL == 510][0]) / 2   # 기준선 (470-510 직선)의 490 nm 값
                ax.plot([470, 510], [a[WL == 470][0], a[WL == 510][0]], color="#111111", lw=0.7, ls=":", zorder=4)
                ax.vlines(490, b, a[WL == 490][0], color="#111111", lw=1.0, zorder=4)
            for x in (470, 490, 510):
                ax.axvline(x, color="#bbbbbb", lw=0.5, zorder=0)
            ax.set_title(f"{sub} · {sp}", loc="left", fontweight="bold")
            ax.set_xticks([450, 470, 490, 510, 530]); ax.grid(alpha=0.2, lw=0.5, axis="y")
            ax.legend(loc="upper right", fontsize=6)
        axes[i, 0].set_ylabel("흡광도 (A = -log$_{10}$R)")
    for j in range(ncol):
        last = max(i for i in range(len(subs)) if axes[i, j].axison)
        axes[last, j].set_xlabel("파장 (nm)")
    fig.suptitle("490 nm 카로티노이드 Peak 확대 (450-530 nm; 가는 선: 입력별 예측 전부, 굵은 선: 정답 실측; "
                 "점선: 470-510 기준선, 세로선: 정답 P = A490 - 기준선; P 단위 x1e-3; 패널마다 y축 다름)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "peak490_zoom.png", dpi=300); plt.close(fig)


def fig_errors(panels, seed, out):
    labels = sorted(panels, key=lambda k: (k[0], k[1]))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    rng = np.random.default_rng(0)
    off = {"B1": -0.25, "B2": 0.0, "BiGRU": 0.25}
    for x, key in enumerate(labels):
        sp = key[1]; rs = panels[key]
        Ptrue = np.array([peak(np.asarray(r["actual"])) for r in rs])
        for m in MODELS:
            k = f"BiGRU_seed{seed[sp]}" if m == "BiGRU" else m
            e = (np.array([peak(np.asarray(r["predictions"][k])) for r in rs]) - Ptrue) * 1e3
            ax.scatter(x + off[m] + rng.uniform(-0.07, 0.07, len(e)), e, s=9, color=COLORS[m], alpha=0.75, lw=0,
                       label=m if x == 0 else None)
    ax.axhline(0, color="#444444", lw=0.8)
    for x in range(len(labels) - 1):
        if labels[x][0] != labels[x + 1][0]:
            ax.axvline(x + 0.5, color="#999999", lw=0.8, ls="--")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{s}\n{sp}\n(BiGRU s{seed[sp]})" for s, sp in labels], fontsize=7)
    ax.set_ylabel("Peak 오차 x1e-3 (예측 - 정답, 양수 = 과대예측)", fontsize=7.5)
    ax.grid(alpha=0.25, lw=0.5, axis="y"); ax.legend(loc="upper left", ncol=3)
    n = sum(len(v) for v in panels.values())
    fig.suptitle(f"쌍별 490 nm Peak 오차 전부 (평균 아님; 점 1개 = 입력-정답 쌍 1개, 모델별 {n}쌍)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "peak490_error_all.png", dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="holdout")
    base = ROOT / "results" / ap.parse_args().version
    style()
    panels, seed = load_panels(base)
    data = load_spectra()
    out = base / "figures"
    fig_zoom(panels, seed, data, out)
    fig_errors(panels, seed, out)
    print("저장:", out / "peak490_zoom.png", out / "peak490_error_all.png")


if __name__ == "__main__":
    main()
