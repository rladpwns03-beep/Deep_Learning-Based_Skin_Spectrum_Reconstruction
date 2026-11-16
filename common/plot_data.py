"""data/processed/spectra.csv -> 논문용 데이터 개요 그림 (data/figures/)

실행:  python common/plot_data.py   (code 폴더에서, prepare_data.py 실행 후)

출력 (.png 600 dpi)
  fig1_spectra_absorbance   피험자별 흡광도 스펙트럼 10회 (색 = HbI)        2단 폭 7.2 in
  fig2_hbi_distribution     피험자별 측정 HbI (측정값만 점으로)             1단 폭 3.5 in
  fig3_variation            파장별 흡광도 SD: 개인 내 vs 개인 간             1단 폭 3.5 in

그림 안에는 제목/설명문을 넣지 않는다 (논문 캡션에서 설명).
측정값만 그리며, 측정 범위 밖 값은 만들지 않는다.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "spectra.csv"
OUT = ROOT / "data" / "figures"

WL = np.arange(400, 701, 10)
HBI_WL = 570
BANDS = [(400, 440, "Soret"), (470, 510, "카로티노이드")]

FONT = "Malgun Gothic"
W, HALF_W = 7.2, 3.5                      # inch, 2단 / 1단 폭
CMAP = plt.cm.viridis
INK, INK2, AXIS, BAND = "#1f1f1f", "#5f5f5f", "#7a7a7a", "#efefef"
# Okabe–Ito 파랑/주홍: 색각이상 시뮬레이션 OKLab ΔE ≥ 21 (정상 31)
C_WITHIN, C_BETWEEN = "#0072B2", "#D55E00"
C_DOT = "#0072B2"

plt.rcParams.update({
    "font.family": FONT, "axes.unicode_minus": False,
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": FONT, "mathtext.bf": FONT,
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.labelpad": 2,
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)


def load():
    d = pd.read_csv(SRC)
    R = d[[f"{w}nm" for w in WL]].to_numpy(float)
    A = -np.log10(R / 100)
    d["HbI"] = A[:, list(WL).index(HBI_WL)]
    return d, R, A


def guides(ax):
    for x0, x1, _ in BANDS:
        ax.axvspan(x0, x1, color=BAND, lw=0, zorder=0)
    ax.axvline(HBI_WL, color=AXIS, lw=0.5, ls=(0, (2.5, 2)), zorder=1)


def wl_axis(ax, ticks):
    ax.set_xlim(398, 702)
    ax.set_xticks(ticks)


def spectra_grid(d, Y, ylabel, name, ylim, yticks):
    subs = sorted(d.subject.unique())
    norm = Normalize(d.HbI.min(), d.HbI.max())     # 전 패널 공통 색 척도
    fig = plt.figure(figsize=(W, 3.1))
    gs = GridSpec(2, 6, figure=fig, width_ratios=[1] * 5 + [0.06],
                  left=0.075, right=0.93, bottom=0.12, top=0.94, wspace=0.12, hspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(5)]
    for i, (ax, sid) in enumerate(zip(axes, subs)):
        m = (d.subject == sid).to_numpy()
        h = d.HbI[m].to_numpy()
        guides(ax)
        for k in np.argsort(h):                    # 낮은 HbI부터 그려 높은 HbI가 위에 보이도록
            ax.plot(WL, Y[m][k], color=CMAP(norm(h[k])), lw=0.7, solid_joinstyle="round")
        ax.set_title(sid, loc="left", pad=2, color=INK)
        wl_axis(ax, [450, 550, 650])
        ax.set_ylim(*ylim); ax.set_yticks(yticks)
        if i % 5:
            ax.tick_params(labelleft=False)
        if i < 5:
            ax.tick_params(labelbottom=False)
    fig.text(0.5, 0.025, "파장 (nm)", ha="center", va="center")
    fig.text(0.022, 0.53, ylabel, ha="center", va="center", rotation=90)
    cax = fig.add_subplot(gs[:, 5])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor(AXIS)
    cb.ax.tick_params(width=0.4, length=1.8)
    cb.set_label("HbI (A$_{570}$)", labelpad=4)
    save(fig, name)


def hbi_distribution(d):
    """피험자별 'o-': 측정 최솟값~최댓값을 잇는 선 위에 측정 HbI를 점으로 (단색)."""
    subs = sorted(d.subject.unique())
    fig, ax = plt.subplots(figsize=(HALF_W, 2.4))
    for i, sid in enumerate(subs):
        h = np.sort(d.HbI[d.subject == sid].to_numpy())
        ax.plot(np.full(len(h), i), h, "o-", color=C_DOT, lw=0.9, ms=3.2, mew=0)
    ax.set_ylim(0.425, 0.73)
    ax.set_xlim(-0.6, len(subs) - 0.4)
    ax.set_xticks(range(len(subs)), subs, fontsize=6)
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    ax.set_xlabel("피험자")
    ax.set_ylabel("HbI (A$_{570}$)")
    save(fig, "fig2_hbi_distribution")


def variation(d, A):
    """파장별 흡광도 SD: 개인 내(같은 피험자 10회 SD의 평균) vs 개인 간(피험자 평균 스펙트럼의 SD)."""
    subs = sorted(d.subject.unique())
    within = np.mean([A[(d.subject == s).to_numpy()].std(0, ddof=1) for s in subs], axis=0) * 1e3
    between = np.vstack([A[(d.subject == s).to_numpy()].mean(0) for s in subs]).std(0, ddof=1) * 1e3
    fig, ax = plt.subplots(figsize=(HALF_W, 2.4))
    guides(ax)
    for x0, x1, name in BANDS:
        ax.text((x0 + x1) / 2, 51.5, name, ha="center", va="top", fontsize=6.3, color=INK2)
    ax.plot(WL, within, color=C_WITHIN, lw=1.2, solid_joinstyle="round", solid_capstyle="round")
    ax.plot(WL, between, color=C_BETWEEN, lw=1.2, solid_joinstyle="round", solid_capstyle="round")
    ax.text(700, within[-1] - 1.8, "개인 내", color=C_WITHIN, ha="right", va="top", fontsize=7)
    ax.text(700, between[-1] + 1.8, "개인 간", color=C_BETWEEN, ha="right", va="bottom", fontsize=7)
    wl_axis(ax, [400, 500, 600, 700])
    ax.set_xticks(np.arange(400, 701, 50), minor=True)
    ax.tick_params(which="minor", length=1.3, width=0.4)
    ax.set_ylim(0, 52); ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel("파장 (nm)")
    ax.set_ylabel("흡광도 SD (×10$^{-3}$)")
    save(fig, "fig3_variation")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*"):            # 이전 버전 그림 정리 (이 스크립트가 만든 파일만 있는 폴더)
        old.unlink()
    d, R, A = load()
    spectra_grid(d, A, "흡광도 (A = -log$_{10}$R)", "fig1_spectra_absorbance", (0.26, 0.87), [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    hbi_distribution(d)
    variation(d, A)
    print("saved:", *sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
