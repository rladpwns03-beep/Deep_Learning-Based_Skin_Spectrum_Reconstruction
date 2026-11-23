"""Figure style shared by scripts/make_figures.py: two output profiles, fixed colours and markers, JPEG saving.

Both profiles call the same drawing functions; only the Style object differs.
All text inside figures is English. Error bars everywhere: mean and standard error across subjects (SD / sqrt(n), ddof = 1).
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
from matplotlib import font_manager                                # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
from matplotlib.text import Text                                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))
from final_utils import BANDS                                      # noqa: E402

PROFILES = {
    "presentation": dict(width=12.0, font=16, tick=14, lw=2.5, ms=9.0, dpi=200, title=True),
    "report": dict(width=6.3, font=8, tick=7, lw=1.2, ms=4.5, dpi=600, title=False),
}
JPEG_QUALITY = 95
FONT = "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"

# colour, marker, filled, line style (fixed for every figure)
MODEL_STYLE = {
    "Measured": dict(color="#000000", marker="", filled=True, ls="-"),
    "Raw": dict(color="#8C8C8C", marker="s", filled=True, ls="-"),
    "HbI-binned mean": dict(color="#E69F00", marker="o", filled=True, ls="-"),
    "Global Linear": dict(color="#0072B2", marker="o", filled=True, ls="-"),
    "PLS_v2": dict(color="#009E73", marker="o", filled=True, ls="-"),
    "BiGRU": dict(color="#D55E00", marker="o", filled=True, ls="-"),
    "PLS_v1": dict(color="#56B4E9", marker="D", filled=False, ls="-"),
    "Ridge": dict(color="#CC79A7", marker="D", filled=False, ls="-"),
    "BiGRU (fixed setting)": dict(color="#D55E00", marker="o", filled=False, ls="--"),
    "BiGRU (Peak-constrained selection)": dict(color="#D55E00", marker="s", filled=False, ls=":"),
    "Global Linear (mean of subject slopes)": dict(color="#000000", marker="*", filled=False, ls="--"),
    "Personal reference": dict(color="#000000", marker="", filled=True, ls="-"),
}
BAND_SHADE = {"Soret": "#EFE6F5", "Band 450-460": "#E3EEF8", "Carotenoid": "#FCEFD9", "Q-band": "#E3F3E6", "Red": "#FBE4E2"}
BAND_RANGE = {"Soret": "400–440", "Band 450-460": "450–460", "Carotenoid": "470–510", "Q-band": "520–590", "Red": "600–700"}
BAND_EDGES = {b: (w[0] - 5, w[-1] + 5) for b, w in BANDS.items()}
COHORT_LABEL = {"Development": "Development cohort (leave-one-subject-out, n = 10)", "External": "External cohort (n = 6)"}
COHORT_SHORT = {"Development": "Development cohort\n(leave-one-subject-out,\nn = 10)", "External": "External cohort\n(n = 6)"}
X3 = "×10$^{-3}$"                       # multiplication sign and superscript via mathtext (no middle dot)
MINUS = "−"


def mean_se(v):
    v = np.asarray(v, float)
    n = len(v)
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")), n


class Style:
    def __init__(self, name: str):
        p = PROFILES[name]
        self.name, self.width, self.font, self.tick = name, p["width"], p["font"], p["tick"]
        self.lw, self.ms, self.dpi, self.show_title = p["lw"], p["ms"], p["dpi"], p["title"]
        self.pres = name == "presentation"

    @contextmanager
    def rc(self):
        with plt.rc_context({
            "font.family": FONT, "font.size": self.font, "axes.titlesize": self.font, "axes.labelsize": self.font,
            "xtick.labelsize": self.tick, "ytick.labelsize": self.tick, "legend.fontsize": self.tick,
            "axes.unicode_minus": True, "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": FONT, "mathtext.bf": FONT,
            "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": self.lw * 0.45,
            "xtick.major.width": self.lw * 0.45, "ytick.major.width": self.lw * 0.45,
            "xtick.major.size": self.ms * 0.45, "ytick.major.size": self.ms * 0.45,
            "lines.linewidth": self.lw, "lines.markersize": self.ms, "figure.facecolor": "white", "axes.facecolor": "white",
            "savefig.facecolor": "white", "hatch.linewidth": self.lw * 0.35, "legend.frameon": False,
        }):
            yield

    def figure(self, h_presentation: float, h_report: float):
        h = h_presentation if self.pres else h_report
        assert (5.0 <= h <= 6.5) if self.pres else h <= 8.5, "figure height outside the profile range"
        return plt.figure(figsize=(self.width, h), layout="constrained")

    def title(self, fig, text: str):
        if self.show_title:
            fig.suptitle(text, fontsize=self.font + 1, fontweight="bold", x=0.01, ha="left")

    def panel(self, ax, letter: str, text: str = ""):
        ax.set_title(f"({letter}) {text}".strip() if letter else text, loc="left", fontsize=self.font, pad=self.font * 0.5)

    def cohort_title(self, ck: str) -> str:
        return COHORT_LABEL[ck].replace(" (", "\n(") if self.pres else COHORT_LABEL[ck]

    def band_title_pad(self) -> float:
        return self.small() * 2.6

    def small(self) -> float:
        return self.tick * 0.85

    # ---- drawing helpers --------------------------------------------------
    def marker_kw(self, model: str, size: float = 1.0, open_triangle: bool = False) -> dict:
        s = MODEL_STYLE[model]
        if open_triangle:                                           # External cohort: open triangle in the model colour
            return dict(marker="^", mfc="white", mec=s["color"], mew=self.lw * 0.55, ms=self.ms * size, ls="none")
        return dict(marker=s["marker"] or "o", mfc=s["color"] if s["filled"] else "white", mec=s["color"],
                    mew=self.lw * 0.55, ms=self.ms * size * (1.35 if s["marker"] == "*" else 1.0), ls="none")

    def subject_points(self, ax, x, values, model, external=False, width=0.28, alpha=0.8):
        """Subject points with a deterministic horizontal spread (no random jitter)."""
        v = np.asarray(values, float)
        off = np.linspace(-width, width, len(v)) if len(v) > 1 else np.zeros(1)
        ax.plot(x + off, v, alpha=alpha, zorder=3, **self.marker_kw(model, 0.62, external))
        return x + off

    def mean_bar(self, ax, x, values, color="#000000", horizontal=False):
        m, se, n = mean_se(values)
        kw = dict(color=color, elinewidth=self.lw * 0.8, capsize=self.ms * 0.45, capthick=self.lw * 0.8, zorder=6, ls="none")
        if horizontal:
            ax.errorbar(m, x, xerr=se, marker="|", ms=self.ms * 2.0, mew=self.lw, **kw)
        else:
            ax.errorbar(x, m, yerr=se, marker="_", ms=self.ms * 2.0, mew=self.lw, **kw)
        return m, se, n

    def zero(self, ax, horizontal=True):
        (ax.axhline if horizontal else ax.axvline)(0, color="#555555", lw=self.lw * 0.5, ls=(0, (4, 3)), zorder=1)

    def shade_bands(self, ax, names=False, bands=None, y=1.0):
        for b in (bands or BANDS):
            lo, hi = BAND_EDGES[b]
            ax.axvspan(lo, hi, color=BAND_SHADE[b], lw=0, zorder=0)
            if names:
                narrow = hi - lo <= 45
                # the narrow band name sits one text row above the others so that neighbouring names do not collide
                ax.text((lo + hi) / 2, y, "Band 450–460\n" if narrow else b, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=self.small(), color="#333333", linespacing=1.05)

    def model_handle(self, model: str, line=False, label=None) -> Line2D:
        s = MODEL_STYLE[model]
        kw = self.marker_kw(model, 0.8)
        if line:
            kw.update(ls=s["ls"], lw=self.lw, color=s["color"])
            if not s["marker"]:
                kw.update(marker="")
        return Line2D([], [], label=label or model, **kw)

    def cohort_handles(self) -> list:
        g = "#555555"
        return [Line2D([], [], marker="o", mfc=g, mec=g, ms=self.ms * 0.7, ls="none", label=COHORT_LABEL["Development"]),
                Line2D([], [], marker="^", mfc="white", mec=g, mew=self.lw * 0.55, ms=self.ms * 0.7, ls="none", label=COHORT_LABEL["External"])]

    def mean_handle(self) -> Line2D:
        return Line2D([], [], marker="_", color="#000000", ms=self.ms * 1.6, mew=self.lw, ls="none", label="Mean and standard error across subjects")

    def ylab(self, ax, *parts):
        """Axis label from parts (the last part is the unit). Presentation: one part per line. Report: text line + unit line."""
        if self.pres or len(parts) == 1:
            ax.set_ylabel("\n".join(parts))
        else:
            ax.set_ylabel(" ".join(parts[:-1]) + "\n" + parts[-1])

    def legend_below(self, fig, handles, ncol=None):
        """Legend below the panels; the number of columns is reduced until the legend fits the figure width."""
        for nc in range(min(ncol or len(handles), len(handles)), 0, -1):
            leg = fig.legend(handles=handles, loc="outside lower center", ncol=nc, handletextpad=0.4, columnspacing=1.2,
                             fontsize=self.tick * (0.9 if self.pres else 1.0))
            fig.canvas.draw()
            if nc == 1 or leg.get_window_extent().width <= fig.bbox.width * 0.97:
                return leg
            leg.remove()

    def save(self, fig, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format="jpeg", dpi=self.dpi, facecolor="white", pil_kwargs=dict(quality=JPEG_QUALITY))


def figure_strings(fig) -> list:
    """Every non-empty text string drawn in the figure (titles, labels, ticks, legends, annotations)."""
    fig.canvas.draw()
    out = []
    for t in fig.findobj(Text):
        s = t.get_text().strip()
        if s and t.get_visible():
            out.append(s)
    return sorted(set(out))
