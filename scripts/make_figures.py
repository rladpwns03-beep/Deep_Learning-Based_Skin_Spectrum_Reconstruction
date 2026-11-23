"""Draw the result figures (.jpg) from files that are already stored under results/. No training or evaluation is rerun.

Run (project root, conda env skin_spectra):
    python -X utf8 scripts/make_figures.py --profile both
    python -X utf8 scripts/make_figures.py --profile presentation --only Figure03

Output (the only place this script writes, apart from docs/Figure_Captions.md):
    results/figures_final/presentation/<FigureName>.jpg     12.0 in wide, 200 dpi, with a one-line title
    results/figures_final/report/<FigureName>.jpg           6.3 in wide, 600 dpi, no title
    results/figures_final/plotted_data/<FigureName>.csv     the numbers that were drawn (panel, series, x, y, error, n, kind, k)
    results/figures_final/figure_manifest.csv, figure_strings.json

Conventions: point = one subject, centre marker = mean across subjects, error bar = standard error (SD / sqrt(n), ddof = 1).
Absorbance-type values are shown in units of 10^-3 absorbance. Difference = first model minus second model.
BiGRU subject values are the mean of the metric over seeds 0, 1, 2. No p values and no evaluative wording in figures.
"""
from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))
import figure_style as FS                                                       # noqa: E402
from figure_style import COHORT_LABEL, COHORT_SHORT, MINUS, MODEL_STYLE, X3, Style, mean_se, plt   # noqa: E402
from matplotlib.lines import Line2D                                             # noqa: E402
from matplotlib.patches import Patch                                            # noqa: E402
from common.data_loader import FREE, HBI_INDEX, WL, load_spectra                # noqa: E402
from final_utils import BANDS, RES, SLOPE_BANDS, UNION_CSV, row_index           # noqa: E402

OUT = RES / "figures_final"
K = 1e3                                                                          # absorbance -> 10^-3 absorbance
DEV = RES / "final_comparison" / "Development10_LeaveOneSubjectOut"
HOLD = RES / "final_comparison" / "Development10_Holdout73"
EXT = RES / "external_test"
U16 = RES / "union16"
BAND_DEV = RES / "band_analysis" / "Development10_LeaveOneSubjectOut"
SHAP = RES / "shap_bands" / "Development10_LeaveOneSubjectOut"
POST = RES / "posthoc_exploratory" / "subject_mean_slope_baseline"
TUNE = RES / "bigru_tuned" / "Development10_AllSubjects" / "selection.json"
COHORTS = {"Development": DEV, "External": EXT}
MAIN4 = ["HbI-binned mean", "Global Linear", "PLS_v2", "BiGRU"]
PAIRS = [("PLS_v2", "Global Linear"), ("BiGRU", "Global Linear"), ("BiGRU", "PLS_v2")]
VARIANT = "Global Linear (mean of subject slopes)"
UNIT = f"({X3} absorbance)"
A_RMSE = ("Absorbance RMSE,", "570 nm excluded", UNIT)
P_RMSE = ("Peak RMSE", UNIT)
BAND_TICK = {b: f"{b.replace('Band 450-460', 'Band')}\n{FS.BAND_RANGE[b]} nm" for b in BANDS}
DH_BINS = ["0.000-0.025", "0.025-0.050", "0.050-0.100", "0.100-0.300"]


class Stop(RuntimeError):
    """A value recomputed for a figure disagrees with the stored result."""


@functools.lru_cache(maxsize=None)
def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def subj(folder: Path, section: str, metric: str, model: str, scale=K) -> pd.Series:
    d = read(folder / "records.csv")
    d = d[(d.section == section) & (d.metric == metric) & (d.model == model)]
    return d.groupby("subject").value.mean().sort_index() * scale


def comp(path: Path, section: str, metric: str, first: str, second: str):
    d = read(path)
    r = d[(d.section == section) & (d.metric == metric) & (d.comparison == f"{first} minus {second}")]
    if len(r) != 1:
        raise Stop(f"{path}: expected one row for {section} / {metric} / {first} minus {second}, found {len(r)}")
    return r.iloc[0]


def row(panel, series, x, y, error=None, n=None, kind="value", k=None):
    return dict(panel=panel, series=series, x=x, y=y, error=error, n=n, kind=kind, k=k)


def check(name, a, b, rel=1e-6):
    if not np.allclose(a, b, rtol=rel, atol=1e-12):
        raise Stop(f"{name}: recomputed {a} differs from stored {b}")


def cohort_of(s: str) -> str:
    return "External" if s >= "S100" else "Development"


@functools.lru_cache(maxsize=None)
def spectra():
    return load_spectra(UNION_CSV)


# ---------------------------------------------------------------------------
# shared panel: subjects and mean ± SE for several models in both cohorts
# ---------------------------------------------------------------------------
def cohort_model_panel(st, ax, panel, section, metric, models, rows, folders=None, series_values=None):
    nm, centers = len(models), []
    for ci, (ck, folder) in enumerate((folders or COHORTS).items()):
        for mi, m in enumerate(models):
            v = series_values(ck, m) if series_values else subj(folder, section, metric, m)
            x = ci * (nm + 1.3) + mi
            xs = st.subject_points(ax, x, v.values, m, external=(ck == "External"))
            mean, se, n = st.mean_bar(ax, x, v.values)
            rows += [row(panel, m, ck, float(val), None, None, f"subject:{s}") for s, val in v.items()]
            rows.append(row(panel, m, ck, mean, se, n, "mean"))
        centers.append(ci * (nm + 1.3) + (nm - 1) / 2)
    ax.set_xticks(centers, [COHORT_SHORT[c] for c in (folders or COHORTS)])
    ax.set_xlim(-0.8, centers[-1] + (nm - 1) / 2 + 0.8)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", length=0)


def model_legend(st, fig, models, extra=(), ncol=None):
    h = [Line2D([], [], marker="s", ls="none", ms=st.ms * 0.8, mfc=MODEL_STYLE[m]["color"], mec=MODEL_STYLE[m]["color"], label=m) for m in models]
    st.legend_below(fig, h + st.cohort_handles() + [st.mean_handle()] + list(extra), ncol=ncol)


# ---------------------------------------------------------------------------
# main figures
# ---------------------------------------------------------------------------
def figure01(st):
    d, rows = spectra(), []
    fig = st.figure(5.6, 5.6)
    axa, axb = fig.subplots(1, 2, width_ratios=[1, 1.25]) if st.pres else fig.subplots(2, 1)
    subs = d.subjects
    axa.axhspan(0.58, 0.62, color="#DDDDDD", lw=0, zorder=0)
    axa.axhline(0.6, color="#555555", lw=st.lw * 0.5, ls=(0, (4, 3)), zorder=1)
    for i, s in enumerate(subs):
        h = d.hbi[d.subject == s]
        ext = cohort_of(s) == "External"
        axa.plot(np.full(len(h), i), h, marker="^" if ext else "o", mfc="white" if ext else "#444444", mec="#444444",
                 mew=st.lw * 0.5, ms=st.ms * 0.6, ls="none", alpha=0.85)
        rows += [row("a", cohort_of(s), s, float(x), None, None, "measurement") for x in h]
    axa.set_xticks(range(len(subs)), subs, rotation=90)
    axa.set_ylabel("Measured HbI = absorbance at 570 nm")
    axa.set_xlabel("Subject")
    axa.text(0.99, 0.6, "target HbI 0.6 ± 0.02", transform=axa.get_yaxis_transform(), ha="right", va="bottom", fontsize=st.small(), color="#333333")
    st.panel(axa, "a", "Measured HbI by subject")
    st.shade_bands(axb, names=True)
    for ck, color, ls in (("Development", "#1A1A1A", "-"), ("External", "#7A7A7A", "--")):
        cs = [s for s in subs if cohort_of(s) == ck]
        means = np.array([d.a[d.subject == s].mean(0) for s in cs])
        for m in means:
            axb.plot(WL, m, color=color, ls=ls, lw=st.lw * 0.35, alpha=0.55, zorder=2)
        axb.plot(WL, means.mean(0), color=color, ls=ls, lw=st.lw * 1.2, zorder=4, label=f"{ck} cohort mean (n = {len(cs)} subjects)")
        rows += [row("b", f"{ck} cohort mean", int(w), float(v), None, len(cs), "mean") for w, v in zip(WL, means.mean(0))]
        rows += [row("b", s, int(w), float(v), None, None, "subject mean") for s, m in zip(cs, means) for w, v in zip(WL, m)]
    for w in (490, 570):
        axb.axvline(w, color="#555555", lw=st.lw * 0.5, ls=":", zorder=1)
    axb.set_xlim(395, 705)
    axb.set_xlabel("Wavelength (nm)")
    axb.set_ylabel("Absorbance")
    axb.legend(loc="upper right", handlelength=2.6, fontsize=st.small())
    st.panel(axb, "b", "Mean absorbance spectrum")
    axb.set_title(axb.get_title(loc="left"), loc="left", pad=st.band_title_pad())
    st.legend_below(fig, st.cohort_handles() + [Line2D([], [], color="#1A1A1A", lw=st.lw * 0.35, label="Thin lines: subject means; dotted lines: 490 nm and 570 nm")])
    st.title(fig, "Measured HbI and absorbance spectra of the 16 subjects (160 measurements)")
    assert len(subs) == 16 and len(d) == 160
    return fig, rows


def figure02(st):
    rows = []
    fig = st.figure(6.2, 6.4)
    axes = fig.subplots(1, 2) if st.pres else fig.subplots(2, 1)
    for ax, p, metric, lab in zip(axes, "ab", ("a_rmse", "peak_rmse"), (A_RMSE, P_RMSE)):
        cohort_model_panel(st, ax, p, "general", metric, MAIN4, rows)
        st.ylab(ax, *lab)
        st.panel(ax, p, "Absorbance RMSE" if metric == "a_rmse" else "Peak RMSE")
    model_legend(st, fig, MAIN4)
    st.title(fig, "General reconstruction error by model and cohort")
    return fig, rows


def figure03(st):
    rows = []
    fig = st.figure(6.5, 5.6)
    axes = fig.subplots(2, 3, sharex=True)
    for r, (metric, lab) in enumerate((("a_rmse", "absorbance RMSE"), ("peak_rmse", "Peak RMSE"))):
        for c, (a, b) in enumerate(PAIRS):
            ax, panel = axes[r, c], f"{'ab'[r]}{c + 1}"
            st.zero(ax)
            tops = []
            for ci, (ck, folder) in enumerate(COHORTS.items()):
                d = subj(folder, "general", metric, a) - subj(folder, "general", metric, b)
                xs = st.subject_points(ax, ci, d.values, a, external=(ck == "External"), width=0.22)
                m, se, n = st.mean_bar(ax, ci, d.values)
                k = int((d.values < 0).sum())
                ref = comp(folder / "comparisons_main.csv", "general", metric, a, b)
                check(f"Figure03 {ck} {metric} {a}-{b} mean", m, ref.diff_mean * K)
                check(f"Figure03 {ck} {metric} {a}-{b} se", se, ref.diff_se * K)
                if k != int(ref.n_improved) or n != int(ref.n):
                    raise Stop(f"Figure03 {ck} {metric} {a}-{b}: k/n {k}/{n} differs from stored {ref.n_improved}/{ref.n}")
                top2 = [list(d.index).index(s) for s in d.abs().sort_values(ascending=False).index[:2]]
                span = np.ptp(np.concatenate([d.values, [0.0]])) or 1.0
                close = abs(d.values[top2[0]] - d.values[top2[1]]) < 0.12 * span
                for i in top2:                                      # label on the outer side; if the two labels are close, split left and right
                    left = (xs[i] == min(xs[top2[0]], xs[top2[1]])) if close else (xs[i] < ci)
                    ax.annotate(d.index[i], (xs[i], d.values[i]), xytext=(st.ms * (-0.7 if left else 0.7), 0), textcoords="offset points",
                                fontsize=st.small() * 0.9, va="center", ha="right" if left else "left", color="#333333")
                tops.append((ci, k, n))
                rows += [row(panel, f"{a} minus {b}", ck, float(v), None, None, f"subject:{s}") for s, v in d.items()]
                rows.append(row(panel, f"{a} minus {b}", ck, m, se, n, "mean", k))
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.16 * (hi - lo))
            for ci, k, n in tops:
                ax.text(ci, 0.985, f"{k}/{n} lower", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=st.small())
            ax.set_xlim(-0.6, 1.75)
            ax.set_xticks([0, 1], ["Development\ncohort", "External\ncohort"])
            ax.tick_params(axis="x", length=0)
            if r == 0:
                ax.set_title(f"{a} {MINUS} {b}", fontsize=st.font)
            if c == 0:
                st.ylab(ax, "Difference in", lab, UNIT)
    st.legend_below(fig, st.cohort_handles() + [st.mean_handle()])
    st.title(fig, "Paired differences between models, by subject")
    return fig, rows


def figure04(st):
    rows, pts = [], {}
    others = ["PLS_v2", "BiGRU", "PLS_v1", "Ridge", "BiGRU (fixed setting)", "BiGRU (Peak-constrained selection)"]
    for ck, folder in COHORTS.items():
        for m in others:
            f = folder / ("comparisons_main.csv" if m in ("PLS_v2", "BiGRU") else "comparisons_supplementary.csv")
            a, p = comp(f, "general", "a_rmse", m, "Global Linear"), comp(f, "general", "peak_rmse", m, "Global Linear")
            pts[(ck, m)] = (a.diff_mean * K, a.diff_se * K, p.diff_mean * K, p.diff_se * K, int(a.n))
    fig = st.figure(5.9, 6.6)
    axes = fig.subplots(1, 2, sharex=True, sharey=True) if st.pres else fig.subplots(2, 1, sharex=True, sharey=True)
    for ax, p, ck in zip(axes, "ab", COHORTS):
        st.zero(ax); st.zero(ax, horizontal=False)
        ax.plot(0, 0, **st.marker_kw("Global Linear", 1.0), zorder=5)
        rows.append(row(p, "Global Linear", 0.0, 0.0, None, None, "origin"))
        for m in others:
            x, xe, y, ye, n = pts[(ck, m)]
            ax.errorbar(x, y, xerr=xe, yerr=ye, color=MODEL_STYLE[m]["color"], elinewidth=st.lw * 0.6, capsize=st.ms * 0.3, ls="none", zorder=3)
            ax.plot(x, y, **st.marker_kw(m, 1.0), zorder=4 if MODEL_STYLE[m]["filled"] else 6)
            rows += [row(p, m, x, y, ye, n, "mean; error = SE of the Peak RMSE difference"),
                     row(p, m, x, y, xe, n, "mean; error = SE of the absorbance RMSE difference")]
        st.panel(ax, p, st.cohort_title(ck))
        if ax is axes[0] or not st.pres:
            st.ylab(ax, "Peak RMSE difference", "from Global Linear", UNIT)
        if st.pres or ax is axes[-1]:
            ax.set_xlabel(f"Absorbance RMSE difference\nfrom Global Linear {UNIT}")
    st.legend_below(fig, [st.model_handle(m) for m in ["Global Linear"] + others], ncol=4 if st.pres else 2)
    st.title(fig, "Absorbance RMSE difference versus Peak RMSE difference, relative to Global Linear")
    return fig, rows


def figure05(st):
    rows = []
    src = {"Development": BAND_DEV / "band_error_comparisons.csv", "External": EXT / "comparisons_band.csv"}
    fig = st.figure(6.0, 6.2)
    axes = fig.subplots(1, 2, sharey=True) if st.pres else fig.subplots(2, 1, sharey=True)
    for ax, p, m in zip(axes, "ab", ("PLS_v2", "BiGRU")):
        st.zero(ax)
        color = MODEL_STYLE[m]["color"]
        for bi, b in enumerate(BANDS):
            for ci, ck in enumerate(src):
                r = comp(src[ck], f"band|{b}", "a_rmse", m, "Global Linear")
                d, se, x = r.diff_mean * K, r.diff_se * K, bi + (ci - 0.5) * 0.4
                ax.bar(x, d, 0.36, color=color if ck == "Development" else "white", edgecolor=color, hatch=None if ck == "Development" else "////",
                       lw=st.lw * 0.5, zorder=2)
                ax.errorbar(x, d, yerr=se, color="#000000", elinewidth=st.lw * 0.6, capsize=st.ms * 0.3, ls="none", zorder=3)
                up = d >= 0
                ax.annotate(f"{int(r.n_improved)}/{int(r.n)}", (x, d + (se if up else -se)), xytext=(0, st.ms * 0.35 * (1 if up else -1)),
                            textcoords="offset points", ha="center", va="bottom" if up else "top", fontsize=st.small() * 0.85)
                rows.append(row(p, f"{m} minus Global Linear", f"{b}|{ck}", d, se, int(r.n), "mean", int(r.n_improved)))
        ax.set_xticks(range(len(BANDS)), [BAND_TICK[b] for b in BANDS])
        ax.tick_params(axis="x", length=0, labelsize=st.small())
        ax.margins(y=0.14)
        st.panel(ax, p, f"{m} {MINUS} Global Linear")
        if ax is axes[0] or not st.pres:
            st.ylab(ax, "Difference in band", "absorbance RMSE", UNIT)
    g = "#666666"
    st.legend_below(fig, [Patch(fc=g, ec=g, label=COHORT_LABEL["Development"]), Patch(fc="white", ec=g, hatch="////", label=COHORT_LABEL["External"]),
                          Line2D([], [], color="#000000", marker="|", ms=st.ms, ls="none", label="Standard error across subjects; k/n = subjects with difference < 0")],
                    ncol=3 if st.pres else 1)
    st.title(fig, "Band-wise absorbance RMSE difference from Global Linear (Q-band excludes 570 nm)")
    return fig, rows


def slope_table(ck):
    return read((BAND_DEV if ck == "Development" else EXT) / "residual_slope_per_subject.csv")


def figure06(st):
    rows = []
    fig = st.figure(6.5, 4.6)
    axes = fig.subplots(2, 4)
    for r, m in enumerate(("PLS_v2", "BiGRU")):
        for c, b in enumerate(SLOPE_BANDS):
            ax, panel, handles = axes[r, c], f"{'ab'[r]}{c + 1}", []
            st.zero(ax); st.zero(ax, horizontal=False)
            allv = []
            for ck in COHORTS:
                t = slope_table(ck)
                x = t[t.model == "Personal reference"].set_index("subject")[f"mean|{b}"].sort_index()
                y = t[t.model == m].set_index("subject")[f"mean|{b}"].sort_index()
                rr = float(np.corrcoef(x.values, y.values)[0, 1])
                s = read((BAND_DEV if ck == "Development" else EXT) / "residual_slope_summary.csv")
                ref = s[(s.item == b) & (s.model == m)].iloc[0]
                check(f"Figure06 {ck} {m} {b} Pearson r", rr, ref.pearson_with_personal_reference)
                ax.plot(x.values, y.values, alpha=0.85, zorder=3, **st.marker_kw(m, 0.7, ck == "External"))
                handles.append(Line2D([], [], label=f"r = {rr:.2f}, n = {len(x)}".replace("-", MINUS), **st.marker_kw(m, 0.6, ck == "External")))
                allv += [x.values, y.values]
                rows += [row(panel, f"{m} | {b}", float(xv), float(yv), None, None, f"subject:{sid}|{ck}") for sid, xv, yv in zip(x.index, x.values, y.values)]
                rows.append(row(panel, f"{m} | {b}", ck, rr, None, len(x), "pearson_r"))
                rows.append(row(panel, f"{m} | {b}", ck, float(y.mean()), None, len(x), "model mean residual slope"))
                rows.append(row(panel, f"Personal reference | {b}", ck, float(x.mean()), None, len(x), "personal reference mean residual slope"))
            lim = np.abs(np.concatenate(allv)).max() * 1.12
            ax.plot([-lim, lim], [-lim, lim], color="#555555", lw=st.lw * 0.5, ls=(0, (2, 2)), zorder=1)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim * 1.55)
            ax.locator_params(nbins=4)
            ax.legend(handles=handles, loc="upper left", fontsize=st.small() * 0.9, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2,
                      frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, borderpad=0.15).set_zorder(7)
            if r == 0:
                ax.set_title(f"{b.replace('Band 450-460', 'Band 450' + chr(0x2013) + '460')}", fontsize=st.font)
            if c == 0:
                ax.set_ylabel(f"{m}\nresidual slope")
    axes[1, 1].set_xlabel("Personal reference residual slope (band mean, absorbance per unit HbI)", x=1.1)
    st.legend_below(fig, st.cohort_handles() + [Line2D([], [], color="#555555", lw=st.lw * 0.5, ls=(0, (2, 2)), label="y = x")])
    st.title(fig, "Model residual slope versus personal reference residual slope, by wavelength band")
    return fig, rows


@functools.lru_cache(maxsize=None)
def external_predictions():
    """Stored ExternalTest6 predictions -> arrays. BiGRU keeps the three seeds separately."""
    d = spectra()
    idx = row_index(d)
    recs = json.loads((EXT / "predictions.json").read_text(encoding="utf-8"))["general_reconstruction"]
    ii = np.array([idx[(r["subject"], r["input_trial"])] for r in recs])
    jj = np.array([idx[(r["subject"], r["target_trial"])] for r in recs])
    assert np.allclose(d.hbi[jj], [r["target_hbi"] for r in recs], atol=1e-12)
    P = {k: np.array([r["predictions"][k] for r in recs]) for k in recs[0]["predictions"]}
    return dict(subject=np.array([r["subject"] for r in recs]), ii=ii, jj=jj, X=d.a[ii], Y=d.a[jj], P=P,
                input_trial=np.array([r["input_trial"] for r in recs]), target_trial=np.array([r["target_trial"] for r in recs]))


def seeds_of(P, model):
    return [P[model]] if model in P else [P[f"{model}|seed{k}"] for k in (0, 1, 2)]


def figure07(st):
    e, d, rows = external_predictions(), spectra(), []
    fig = st.figure(6.5, 7.4)
    gs = fig.add_gridspec(4, 3, height_ratios=[2.2, 1, 2.2, 1])
    shown = ["Global Linear", "PLS_v2", "BiGRU"]
    for n, s in enumerate(sorted(set(e["subject"]))):
        m = np.flatnonzero(e["subject"] == s)
        rmse = np.sqrt(np.mean((e["P"]["Global Linear"][m] - e["Y"][m])[:, FREE] ** 2, axis=1))
        order = sorted(range(len(m)), key=lambda q: (rmse[q], e["input_trial"][m[q]], e["target_trial"][m[q]]))
        q = m[order[44]]                                         # 45th of the 90 pairs
        h = d.hbi[d.subject == s]
        in_range = h.min() <= 0.6 <= h.max()
        top, bot = fig.add_subplot(gs[2 * (n // 3), n % 3]), fig.add_subplot(gs[2 * (n // 3) + 1, n % 3])
        top.plot(WL, e["X"][q], color="#8C8C8C", ls="--", lw=st.lw * 0.7, zorder=2)
        top.plot(WL, e["Y"][q], color="#000000", lw=st.lw * 0.9, zorder=3)
        st.shade_bands(bot); st.zero(bot)
        rows.append(row(s, "selected pair", int(e["input_trial"][q]), int(e["target_trial"][q]), float(rmse[order[44]] * K), 90,
                        "x = input trial, y = target trial, error = Global Linear absorbance RMSE of the pair (10^-3)"))
        rows += [row(s, "Measured input", int(w), float(v), None, None, "absorbance") for w, v in zip(WL, e["X"][q])]
        rows += [row(s, "Measured target", int(w), float(v), None, None, "absorbance") for w, v in zip(WL, e["Y"][q])]
        for model in shown:
            p = np.mean([x[q] for x in seeds_of(e["P"], model)], axis=0)        # BiGRU: mean of the three seed predictions
            top.plot(WL, p, color=MODEL_STYLE[model]["color"], lw=st.lw * 0.6, zorder=4)
            bot.plot(WL, (p - e["Y"][q]) * K, color=MODEL_STYLE[model]["color"], lw=st.lw * 0.6, zorder=4)
            rows += [row(s, model, int(w), float(v), None, None, "absorbance") for w, v in zip(WL, p)]
        top.set_title(f"{s}: HbI {d.hbi[e['ii'][q]]:.2f} to {d.hbi[e['jj'][q]]:.2f}", fontsize=st.tick, loc="left")
        if in_range:
            top.text(0.98, 0.95, "In-range subject", transform=top.transAxes, ha="right", va="top", fontsize=st.small())
        top.set_xlim(395, 705); bot.set_xlim(395, 705)
        top.tick_params(labelbottom=False)
        top.locator_params(axis="y", nbins=4); bot.locator_params(axis="y", nbins=3)
        if n % 3 == 0:
            top.set_ylabel("Absorbance")
            bot.set_ylabel(f"Error\n({X3})")
        if n // 3 == 1:
            bot.set_xlabel("Wavelength (nm)")
        else:
            bot.tick_params(labelbottom=False)
    st.legend_below(fig, [Line2D([], [], color="#8C8C8C", ls="--", lw=st.lw * 0.7, label="Measured input"),
                          Line2D([], [], color="#000000", lw=st.lw * 0.9, label="Measured target")] +
                    [Line2D([], [], color=MODEL_STYLE[m]["color"], lw=st.lw * 0.6, label=m) for m in shown], ncol=5 if st.pres else 3)
    st.title(fig, "Reconstruction examples in the External cohort (one pair per subject, fixed selection rule)")
    return fig, rows


def wavelength_rmse(model):
    """Per subject: RMSE at each wavelength over that subject's pairs (BiGRU: per seed, then seed mean). Also the recombined a_rmse."""
    e = external_predictions()
    subs = sorted(set(e["subject"]))
    per, total = [], []
    for s in subs:
        m = e["subject"] == s
        r = np.array([np.sqrt(np.mean((P[m] - e["Y"][m]) ** 2, axis=0)) for P in seeds_of(e["P"], model)])     # (seeds, 31)
        per.append(r.mean(0))
        total.append(np.mean(np.sqrt(np.mean(r[:, FREE] ** 2, axis=1))))
    return subs, np.array(per) * K, np.array(total) * K


def figure08(st):
    rows = []
    fig = st.figure(5.6, 6.0)
    axa, axb = fig.subplots(1, 2) if st.pres else fig.subplots(2, 1, sharex=True)
    wl = WL.astype(float)
    data = {}
    for m in MAIN4:
        subs, per, total = wavelength_rmse(m)
        check(f"Figure08 {m} recombined absorbance RMSE", total, subj(EXT, "general", "a_rmse", m).reindex(subs).values, rel=1e-9)
        data[m] = per
    for ax, p in ((axa, "a"), (axb, "b")):
        st.shade_bands(ax, names=True)
        ax.axvline(570, color="#555555", lw=st.lw * 0.5, ls=":", zorder=1)
        ax.set_xlim(395, 705)
    for m in MAIN4:
        mean, se = data[m].mean(0), data[m].std(0, ddof=1) / np.sqrt(len(data[m]))
        y, lo, hi = mean.copy(), mean - se, mean + se
        for arr in (y, lo, hi):
            arr[HBI_INDEX] = np.nan
        axa.fill_between(wl, lo, hi, color=MODEL_STYLE[m]["color"], alpha=0.18, lw=0, zorder=2)
        axa.plot(wl, y, color=MODEL_STYLE[m]["color"], lw=st.lw * 0.8, zorder=3, label=m)
        rows += [row("a", m, int(w), float(a), float(b), len(data[m]), "mean") for w, a, b in zip(WL, mean, se) if w != 570]
    st.zero(axb)
    for m in ("PLS_v2", "BiGRU"):
        d = data[m] - data["Global Linear"]
        mean, se = d.mean(0), d.std(0, ddof=1) / np.sqrt(len(d))
        y, lo, hi = mean.copy(), mean - se, mean + se
        for arr in (y, lo, hi):
            arr[HBI_INDEX] = np.nan
        axb.fill_between(wl, lo, hi, color=MODEL_STYLE[m]["color"], alpha=0.18, lw=0, zorder=2)
        axb.plot(wl, y, color=MODEL_STYLE[m]["color"], lw=st.lw * 0.8, zorder=3, label=f"{m} {MINUS} Global Linear")
        rows += [row("b", f"{m} minus Global Linear", int(w), float(a), float(b), len(d), "mean") for w, a, b in zip(WL, mean, se) if w != 570]
    axa.set_ylim(bottom=0)
    st.ylab(axa, "RMSE at each", "wavelength", UNIT)
    st.ylab(axb, "Difference from", "Global Linear", UNIT)
    axb.set_xlabel("Wavelength (nm)")
    if st.pres:
        axa.set_xlabel("Wavelength (nm)")
    st.panel(axa, "a", "RMSE at each wavelength"); axa.set_title(axa.get_title(loc="left"), loc="left", pad=st.band_title_pad())
    st.panel(axb, "b", "Difference from Global Linear"); axb.set_title(axb.get_title(loc="left"), loc="left", pad=st.band_title_pad())
    axb.legend(loc="lower left")
    st.legend_below(fig, [Line2D([], [], color=MODEL_STYLE[m]["color"], lw=st.lw * 0.8, label=m) for m in MAIN4] +
                    [Patch(fc="#999999", alpha=0.35, label="Band: mean ± standard error across 6 subjects"),
                     Line2D([], [], color="#555555", lw=st.lw * 0.5, ls=":", label="570 nm (fixed to the target HbI, error 0)")], ncol=3 if st.pres else 2)
    st.title(fig, "Wavelength-wise reconstruction error in the External cohort (n = 6)")
    return fig, rows


def figure09(st):
    rows, models = [], ["Raw", "HbI-binned mean", "Global Linear", "PLS_v2", "BiGRU", "PLS_v1"]
    fig = st.figure(6.2, 6.6)
    axes = fig.subplots(1, 2) if st.pres else fig.subplots(2, 1)
    for ax, p, metric, lab in zip(axes, "ab", ("peak_sd", "spectral_sd"), ("Peak SD at target HbI 0.6", "Spectral SD at target HbI 0.6")):
        cohort_model_panel(st, ax, p, "repeatability", metric, models, rows)
        st.ylab(ax, lab.split(" at ")[0], "at " + lab.split(" at ")[1], UNIT)
        st.panel(ax, p, lab)
    model_legend(st, fig, models, ncol=4 if st.pres else 2)
    st.title(fig, "Fixed-target repeatability: within-subject SD after reconstruction to HbI 0.6")
    return fig, rows


# ---------------------------------------------------------------------------
# supplementary figures
# ---------------------------------------------------------------------------
def supplementary01(st):
    rows = []
    fig = st.figure(5.6, 6.0)
    axes = fig.subplots(1, 2, sharey=True) if st.pres else fig.subplots(2, 1, sharex=True, sharey=True)
    cols = [f"wl{int(w)}" for w in WL]
    for ax, p, ck in zip(axes, "ab", COHORTS):
        t = slope_table(ck)
        st.shade_bands(ax, names=True); st.zero(ax)
        for m in ("Personal reference", "PLS_v2", "BiGRU"):
            v = t[t.model == m].sort_values("subject")[cols].to_numpy()
            mean, se = v.mean(0), v.std(0, ddof=1) / np.sqrt(len(v))
            y, lo, hi = mean.copy(), mean - se, mean + se
            for arr in (y, lo, hi):
                arr[HBI_INDEX] = np.nan
            ax.fill_between(WL, lo, hi, color=MODEL_STYLE[m]["color"], alpha=0.15, lw=0, zorder=2)
            ax.plot(WL, y, color=MODEL_STYLE[m]["color"], lw=st.lw * 0.8, zorder=3)
            rows += [row(p, m, int(w), float(a), float(b), len(v), "mean") for w, a, b in zip(WL, mean, se) if w != 570]
        ax.set_xlim(395, 705)
        st.panel(ax, p, st.cohort_title(ck)); ax.set_title(ax.get_title(loc="left"), loc="left", pad=st.band_title_pad())
        if ax is axes[0] or not st.pres:
            st.ylab(ax, "Residual slope", "(absorbance per unit HbI)")
        if ax is axes[-1] or st.pres:
            ax.set_xlabel("Wavelength (nm)")
    st.legend_below(fig, [Line2D([], [], color=MODEL_STYLE[m]["color"], lw=st.lw * 0.8, label=m) for m in ("Personal reference", "PLS_v2", "BiGRU")] +
                    [Patch(fc="#999999", alpha=0.35, label="Band: mean ± standard error across subjects (570 nm excluded)")], ncol=4 if st.pres else 2)
    st.title(fig, "Residual slope at each wavelength: models and personal reference")
    return fig, rows


def supplementary02(st):
    rows = []
    fig = st.figure(6.5, 5.8)
    axes = fig.subplots(2, 2, sharex=True, sharey="row")
    for r, (metric, lab) in enumerate((("a_rmse", "absorbance RMSE"), ("peak_rmse", "Peak RMSE"))):
        for c, (ck, folder) in enumerate(COHORTS.items()):
            ax, panel = axes[r, c], f"{'ab'[r]}{c + 1}"
            st.zero(ax)
            ns = []
            for mi, m in enumerate(("PLS_v2", "BiGRU")):
                xs, ys, es = [], [], []
                for bi, b in enumerate(DH_BINS):
                    q = comp(folder / "comparisons_delta_hbi.csv", f"dh|{b}|all", metric, m, "Global Linear")
                    xs.append(bi + (mi - 0.5) * 0.16); ys.append(q.diff_mean * K); es.append(q.diff_se * K)
                    rows.append(row(panel, f"{m} minus Global Linear", f"{b}|{ck}", q.diff_mean * K, q.diff_se * K, int(q.n), "mean", int(q.n_improved)))
                    if mi == 0:
                        ns.append(int(q.n))
                ax.errorbar(xs, ys, yerr=es, color=MODEL_STYLE[m]["color"], elinewidth=st.lw * 0.6, capsize=st.ms * 0.3, lw=st.lw * 0.6,
                            zorder=3, **{k: v for k, v in st.marker_kw(m, 0.8).items() if k != "ls"})
            for bi, n in enumerate(ns):
                ax.text(bi, 0.02, f"n = {n}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=st.small() * 0.9)
            ax.set_xticks(range(4), ["0–0.025", "0.025–0.05", "0.05–0.10", "0.10–0.30"])
            ax.margins(y=0.22)
            if r == 0:
                ax.set_title(COHORT_LABEL[ck], fontsize=st.tick)
            if r == 1:
                ax.set_xlabel("|ΔHbI| between input and target")
            if c == 0:
                st.ylab(ax, "Difference in", lab, "from Global Linear", UNIT)
            ax.tick_params(axis="x", labelsize=st.small())
    st.legend_below(fig, [st.model_handle(m, label=f"{m} {MINUS} Global Linear") for m in ("PLS_v2", "BiGRU")] +
                    [Line2D([], [], color="#000000", marker="|", ms=st.ms, ls="none", label="Mean and standard error across the n subjects with pairs in the bin")],
                    ncol=3 if st.pres else 2)
    st.title(fig, "Difference from Global Linear by size of the HbI change")
    return fig, rows


def supplementary03(st):
    rows, t = [], read(EXT / "peak_rs_reference.csv")
    fig = st.figure(5.6, 3.9)
    ax = fig.subplots()
    st.zero(ax)
    subs = sorted(t.subject.unique())
    for i, s in enumerate(subs):
        g = t[t.subject == s].set_index("model")
        se, ext = float(g.reference_se_peak_rs.iloc[0]), bool(g.extrapolated.iloc[0])
        ax.fill_between([i - 0.42, i + 0.42], -se, se, color="#CCCCCC", lw=0, zorder=1)
        rows.append(row("a", "reference standard error", s, 0.0, se, None, "extrapolated" if ext else "in-range"))
        for mi, m in enumerate(MAIN4):
            y = float(g.loc[m, "model_minus_reference_peak_rs"])
            kw = st.marker_kw(m, 0.85)
            if ext:
                kw["mfc"] = "white"
            ax.plot(i + (mi - 1.5) * 0.19, y, zorder=3, **kw)
            rows.append(row("a", m, s, y, None, None, "extrapolated" if ext else "in-range"))
        rows.append(row("a", "regression slope (AU per HbI)", s, float(g.regression_slope_peak_rs_per_hbi.iloc[0]), None, 10, "within-subject regression"))
    ax.set_xticks(range(len(subs)), [f"{s}\n{t[t.subject == s].regression_slope_peak_rs_per_hbi.iloc[0]:.0f}".replace("-", MINUS) for s in subs])
    ax.set_xlabel("Subject (second line: within-subject regression slope of Peak_RS on HbI, AU per HbI)")
    st.ylab(ax, f"Model {MINUS} regression", "reference at HbI 0.6,", "Peak_RS (AU)")
    g = "#555555"
    st.legend_below(fig, [st.model_handle(m) for m in MAIN4] +
                    [Line2D([], [], marker="o", mfc=g, mec=g, ls="none", ms=st.ms * 0.7, label="Filled: in-range subject"),
                     Line2D([], [], marker="o", mfc="white", mec=g, ls="none", ms=st.ms * 0.7, label="Open: reference extrapolated to HbI 0.6"),
                     Patch(fc="#CCCCCC", label="± standard error of the regression reference")], ncol=4 if st.pres else 2)
    st.title(fig, "Peak_RS at HbI 0.6: model mean minus within-subject regression reference (External cohort)")
    return fig, rows


def supplementary04(st):
    rows, models = [], ["Global Linear", "PLS_v2", "BiGRU"]
    u = U16 / "Union16_LeaveOneSubjectOut"
    cases = [("a", "External cohort (n = 6)", EXT, "10", "External cohort 6명"),
             ("b", "Development cohort (n = 10)", U16 / "Development10_LeaveOneSubjectOut_FixedHyperparameters", "9", "Development cohort 10명")]
    eff = read(U16 / "training_size_effect.csv")
    eff = eff[(eff.role == "주 비교 (같은 고정값)") & (eff.metric == "a_rmse")]
    fig = st.figure(5.8, 6.2)
    axes = fig.subplots(1, 2, sharey=True) if st.pres else fig.subplots(2, 1, sharey=True)
    for ax, (p, lab, small_folder, n_small, cohort_key) in zip(axes, cases):
        ticks, labels = [], []
        for gi, m in enumerate(models):
            lo = subj(small_folder, "general", "a_rmse", m)
            hi = subj(u, "general", "a_rmse", m).reindex(lo.index)
            ref = eff[(eff.cohort == cohort_key) & (eff.model == m)].iloc[0]
            check(f"SupplementaryFigure04 {lab} {m} mean (15)", hi.mean(), ref.mean_a * K)
            check(f"SupplementaryFigure04 {lab} {m} mean ({n_small})", lo.mean(), ref.mean_b * K)
            d = hi - lo
            check(f"SupplementaryFigure04 {lab} {m} difference", d.mean(), ref.diff_mean * K)
            x0, x1, color = gi * 2.6, gi * 2.6 + 1, MODEL_STYLE[m]["color"]
            for a, b in zip(lo.values, hi.values):
                ax.plot([x0, x1], [a, b], color=color, lw=st.lw * 0.3, alpha=0.5, zorder=2)
            ax.plot([x0, x1], [lo.mean(), hi.mean()], color=color, lw=st.lw * 1.1, marker="o", ms=st.ms * 0.8, zorder=4)
            ticks += [x0, x1]; labels += [n_small, "15"]
            rows += [row(p, m, n_small, float(v), None, None, f"subject:{s}") for s, v in lo.items()]
            rows += [row(p, m, "15", float(v), None, None, f"subject:{s}") for s, v in hi.items()]
            rows += [row(p, m, n_small, float(lo.mean()), mean_se(lo.values)[1], len(lo), "mean"), row(p, m, "15", float(hi.mean()), mean_se(hi.values)[1], len(hi), "mean"),
                     row(p, m, f"15 minus {n_small}", float(d.mean()), mean_se(d.values)[1], len(d), "mean difference", int((d.values < 0).sum()))]
        ax.set_xticks(ticks, labels)
        ax.set_xlim(-0.5, ticks[-1] + 0.5)
        ax.set_xlabel("Number of training subjects")
        st.panel(ax, p, lab)
        if ax is axes[0] or not st.pres:
            st.ylab(ax, *A_RMSE)
    st.legend_below(fig, [Line2D([], [], color=MODEL_STYLE[m]["color"], lw=st.lw * 1.1, marker="o", ms=st.ms * 0.8, label=m) for m in models] +
                    [Line2D([], [], color="#777777", lw=st.lw * 0.3, label="Thin lines: individual subjects; thick lines: mean across subjects")],
                    ncol=4 if st.pres else 2)
    st.title(fig, "Absorbance RMSE of the same test subjects with fewer and with 15 training subjects")
    return fig, rows


def supplementary05(st):
    rows = []
    s = json.loads(TUNE.read_text(encoding="utf-8"))["splits"][0]
    c = pd.DataFrame(s["candidates"])
    if len(c) != 84:
        raise Stop(f"SupplementaryFigure05: {len(c)} candidates, expected 84")
    ch, gl = s["chosen"], s["global_linear_validation"]
    fixed = dict(hidden=8, slope_l2=1e-3, epoch=150)
    l2s = sorted(c.slope_l2.unique())
    look = dict(zip(l2s, [("#1B1B1B", "-", "o"), ("#5E3C99", "--", "s"), ("#2B83BA", "-.", "^"), ("#66A61E", ":", "D")]))
    fig = st.figure(6.5, 5.4)
    axes = fig.subplots(2, 3, sharex=True, sharey="row")
    for r, (key, lab) in enumerate((("val_a_rmse", "Validation absorbance RMSE"), ("val_peak_rmse", "Validation Peak RMSE"))):
        for ci, h in enumerate(sorted(c.hidden.unique())):
            ax, panel = axes[r, ci], f"{'ab'[r]}{ci + 1}"
            ax.axhline(gl[key] * K, color=MODEL_STYLE["Global Linear"]["color"], lw=st.lw * 0.6, ls=(0, (4, 3)), zorder=1)
            for l2 in l2s:
                g = c[(c.hidden == h) & (c.slope_l2 == l2)].sort_values("epoch")
                col, ls, mk = look[l2]
                ax.plot(g.epoch, g[key] * K, color=col, ls=ls, lw=st.lw * 0.6, marker=mk, ms=st.ms * 0.45, zorder=3)
                rows += [row(panel, f"hidden {h}, slope_l2 {l2:g}", int(e), float(v) * K, None, None, key) for e, v in zip(g.epoch, g[key])]
            for cfg, kw, nm in ((ch, dict(mfc=MODEL_STYLE["BiGRU"]["color"]), "selected"), (fixed, dict(mfc="white"), "fixed setting")):
                if cfg["hidden"] == h:
                    v = c[(c.hidden == h) & np.isclose(c.slope_l2, cfg["slope_l2"]) & (c.epoch == cfg["epoch"])][key].iloc[0] * K
                    ax.plot(cfg["epoch"], v, marker="o", ms=st.ms * 1.25, mec=MODEL_STYLE["BiGRU"]["color"], mew=st.lw * 0.8, ls="none", zorder=5, **kw)
                    rows.append(row(panel, nm, int(cfg["epoch"]), float(v), None, None, key))
            rows.append(row(panel, "epoch 0 (identical to Global Linear)", 0, gl[key] * K, None, None, key))
            if r == 0:
                ax.set_title(f"Hidden size {h}", fontsize=st.font)
            if r == 1:
                ax.set_xlabel("Epoch")
            if ci == 0:
                st.ylab(ax, *lab.split(" ", 1), UNIT)
            ax.set_xticks([0, 100, 200, 300])
    oc = MODEL_STYLE["BiGRU"]["color"]
    st.legend_below(fig, [Line2D([], [], color=look[l][0], ls=look[l][1], marker=look[l][2], ms=st.ms * 0.45, lw=st.lw * 0.6, label=f"slope_l2 = {l:g}") for l in l2s] +
                    [Line2D([], [], color=MODEL_STYLE["Global Linear"]["color"], lw=st.lw * 0.6, ls=(0, (4, 3)), label="Epoch 0 (identical to Global Linear)"),
                     Line2D([], [], marker="o", mfc=oc, mec=oc, ls="none", ms=st.ms, label="Selected setting"),
                     Line2D([], [], marker="o", mfc="white", mec=oc, ls="none", ms=st.ms, label="BiGRU (fixed setting)")], ncol=4 if st.pres else 3)
    st.title(fig, "BiGRU tuning: inner validation error of the 84 candidates (Development cohort, 10 subjects)")
    return fig, rows


GROUP_COLOR = {"h* (target)": "#4D4D4D", "h_in (570 nm)": "#B3B3B3", "Soret": "#7B3294", "Band 450-460": "#2C7BB6", "Carotenoid": "#FDAE61",
               "Q-band": "#1A9641", "Red": "#D7191C"}
GROUP_LABEL = {"h* (target)": "Target HbI", "h_in (570 nm)": "Input HbI (570 nm)", "Band 450-460": "Band 450–460"}


def supplementary06(st):
    rows = []
    s, ob = read(SHAP / "shap_summary.csv"), read(SHAP / "shap_input_group_by_output_band.csv")
    models, groups = ["Global Linear", "PLS_v2", "BiGRU", "PLS_v1", "Ridge"], list(GROUP_COLOR)
    fig = st.figure(5.6, 6.4)
    axd = fig.subplot_mosaic([["a", "b1", "b2"]], width_ratios=[1.5, 1, 1]) if st.pres else fig.subplot_mosaic([["a", "a"], ["b1", "b2"]], height_ratios=[1, 1.15])
    ax = axd["a"]
    for mi, m in enumerate(models):
        left = 0.0
        for g in groups:
            v = float(s[(s.model_base == m) & (s.group == g)].share_pct_spectral.iloc[0])
            ax.barh(mi, v, left=left, color=GROUP_COLOR[g], edgecolor="white", lw=st.lw * 0.3, height=0.7)
            if v >= 9:
                ax.text(left + v / 2, mi, f"{v:.0f}", ha="center", va="center", fontsize=st.small() * 0.9, color="white" if g != "h_in (570 nm)" and g != "Carotenoid" else "#000000")
            rows.append(row("a", m, g, v, None, 900, "share of mean |SHAP|, percent"))
            left += v
    ax.set_yticks(range(len(models)), models); ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of mean |SHAP| over 30 output wavelengths (%)")
    ax.tick_params(axis="y", length=0)
    st.panel(ax, "a", "Input group share")
    bands = list(BANDS)
    for key, m in (("b1", "PLS_v2"), ("b2", "BiGRU")):
        ax = axd[key]
        M = np.array([[float(ob[(ob.model_base == m) & (ob.group == g)][f"out|{b}"].iloc[0]) * K for b in bands] for g in groups])
        ax.imshow(M, aspect="auto", cmap="Greys", vmin=0, vmax=M.max() * 1.25)
        for i, g in enumerate(groups):
            for j, b in enumerate(bands):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=st.small() * 0.8, color="white" if M[i, j] > 0.6 * M.max() else "#000000")
                rows.append(row(key, m, f"{g} -> {b}", float(M[i, j]), None, 900, "mean |SHAP|, 10^-3 absorbance"))
        ax.set_xticks(range(len(bands)), [b.replace("Band 450-460", "450–460") for b in bands], rotation=90)
        ax.set_yticks(range(len(groups)), [GROUP_LABEL.get(g, g) for g in groups] if key == "b1" else [""] * len(groups))
        ax.tick_params(length=0)
        ax.set_xlabel("Output band")
        for sp in ax.spines.values():
            sp.set_visible(False)
        st.panel(ax, "b" if key == "b1" else "", m)
    st.legend_below(fig, [Patch(fc=GROUP_COLOR[g], label=GROUP_LABEL.get(g, g)) for g in groups] +
                    [Line2D([], [], ls="none", label=f"(b) rows: input group; cell values: mean |SHAP| {UNIT}")], ncol=4)
    st.title(fig, "Band-level SHAP: share of input groups and input-by-output band map")
    return fig, rows


SPLIT_SETS = [("Development10_Holdout73", HOLD, False), ("Development10_LeaveOneSubjectOut", DEV, False), ("ExternalTest6", EXT, False),
              ("Union16_Holdout73", U16 / "Union16_Holdout73", True), ("Union16_LeaveOneSubjectOut", U16 / "Union16_LeaveOneSubjectOut", True)]


def supplementary07(st):
    rows = []
    fig = st.figure(6.5, 6.6)
    axes = fig.subplots(1, 2, sharey=True)
    ypos, ylab, heads = [], [], []
    y = 0.0
    for a, b in PAIRS:
        heads.append((y - 0.75, f"{a} {MINUS} {b}"))
        for name, _, _ in SPLIT_SETS:
            ypos.append(y); ylab.append(name); y += 1
        y += 1.3
    for ax, p, metric, lab in zip(axes, "ab", ("a_rmse", "peak_rmse"), ("absorbance RMSE", "Peak RMSE")):
        st.zero(ax, horizontal=False)
        i = 0
        for a, b in PAIRS:
            color = MODEL_STYLE[a]["color"]
            for name, folder, supp in SPLIT_SETS:
                q = comp(folder / "comparisons_main.csv", "general", metric, a, b)
                d, se = q.diff_mean * K, q.diff_se * K
                ax.errorbar(d, ypos[i], xerr=se, color=color, elinewidth=st.lw * 0.7, capsize=st.ms * 0.3, ls="none", zorder=3)
                ax.plot(d, ypos[i], marker="o", ms=st.ms * 0.85, mfc="white" if supp else color, mec=color, mew=st.lw * 0.55, ls="none", zorder=4)
                ax.text(1.0, ypos[i], f"n = {int(q.n)}", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=st.small() * 0.85, color="#333333")
                rows.append(row(p, f"{a} minus {b}", name, d, se, int(q.n), "mean", int(q.n_improved)))
                i += 1
        ax.set_yticks(ypos + [yy for yy, _ in heads], ylab + [text for _, text in heads], fontsize=st.small() * 0.9)
        for t in ax.get_yticklabels()[len(ypos):]:              # comparison names as bold row headers
            t.set_fontweight("bold")
        ax.set_ylim(ypos[-1] + 0.7, -1.4)
        ax.tick_params(axis="y", length=0)
        ax.margins(x=0.22)
        ax.set_xlabel(f"Difference in {lab}\n{UNIT}")
        st.panel(ax, p, lab[0].upper() + lab[1:])
    g = "#555555"
    st.legend_below(fig, [Line2D([], [], marker="o", mfc=g, mec=g, ls="none", ms=st.ms * 0.8, label="Development cohort sets and ExternalTest6"),
                          Line2D([], [], marker="o", mfc="white", mec=g, ls="none", ms=st.ms * 0.8, label="Union16 sets (supplementary, fixed hyperparameters)"),
                          Line2D([], [], color=g, marker="|", ms=st.ms, ls="none", label="Mean and standard error across subjects")], ncol=3 if st.pres else 1)
    st.title(fig, "Paired differences across the five split sets")
    return fig, rows


def posthoc01(st):
    rows, models = [], ["Global Linear", VARIANT, "PLS_v2", "BiGRU"]
    rec, sl = read(POST / "records.csv"), read(POST / "slopes.csv")
    set_of = {"Development": "Development10_LeaveOneSubjectOut", "External": "ExternalTest6"}

    def values(ck, m):
        d = rec[(rec.evaluation_set == set_of[ck]) & (rec.metric == "a_rmse") & (rec.model == m)]
        return d.groupby("subject").value.mean().sort_index() * K

    fig = st.figure(6.3, 7.0)
    if st.pres:
        gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1.3, 1])
        axa, axb, axc = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    else:
        gs = fig.add_gridspec(3, 1, height_ratios=[1.9, 1.3, 0.9])
        axa, axb, axc = (fig.add_subplot(gs[i]) for i in range(3))
    cohort_model_panel(st, axa, "a", "general", "a_rmse", models, rows, series_values=values)
    st.ylab(axa, *A_RMSE)
    st.panel(axa, "a", "Absorbance RMSE by subject")
    axa.text(0.5, 0.03, "Post hoc, exploratory (not preregistered)", transform=axa.transAxes, ha="center", va="bottom", fontsize=st.tick, fontweight="bold",
             bbox=dict(fc="white", ec="#000000", lw=st.lw * 0.4, boxstyle="round,pad=0.3"))
    s = sl[sl.evaluation_set == "ExternalTest6"].sort_values("wavelength_nm")
    assert len(s) == 31 and int(s.n_train_subjects.iloc[0]) == 10
    st.shade_bands(axb); st.shade_bands(axc); st.zero(axc)
    w = s.wavelength_nm.to_numpy()
    axb.plot(w, s.pooled, color=MODEL_STYLE["Global Linear"]["color"], lw=st.lw * 0.9, label="Pooled within-subject slope (Global Linear)")
    axb.plot(w, s.mean_of_subject_slopes, color="#000000", ls="--", lw=st.lw * 0.9, label="Mean of subject slopes")
    axc.plot(w, s.difference, color="#000000", lw=st.lw * 0.9)
    rows += [row("b", "pooled slope", int(x), float(v), None, 10, "slope") for x, v in zip(w, s.pooled)]
    rows += [row("b", "mean of subject slopes", int(x), float(v), None, 10, "slope") for x, v in zip(w, s.mean_of_subject_slopes)]
    rows += [row("b", "mean of subject slopes minus pooled", int(x), float(v), None, 10, "slope difference") for x, v in zip(w, s.difference)]
    axb.set_xlim(395, 705); axc.set_xlim(395, 705)
    axb.tick_params(labelbottom=False)
    st.ylab(axb, "Slope b", "(per unit HbI)")
    st.ylab(axc, "Mean " + MINUS + " pooled")
    axc.set_xlabel("Wavelength (nm)")
    st.panel(axb, "b", "Slopes from the 10\nDevelopment cohort subjects" if st.pres else "Slopes from the 10 Development cohort subjects")
    lines = [Line2D([], [], color=MODEL_STYLE["Global Linear"]["color"], lw=st.lw * 0.9, label="(b) Pooled within-subject slope (Global Linear)"),
             Line2D([], [], color="#000000", ls="--", lw=st.lw * 0.9, label="(b) Mean of subject slopes")]
    hv = Line2D([], [], label=VARIANT, **st.marker_kw(VARIANT, 0.8))
    h = [Line2D([], [], marker="s", ls="none", ms=st.ms * 0.8, mfc=MODEL_STYLE[m]["color"], mec=MODEL_STYLE[m]["color"], label=m) for m in ("Global Linear", "PLS_v2", "BiGRU")]
    st.legend_below(fig, h[:1] + [hv] + h[1:] + st.cohort_handles() + [st.mean_handle()] + lines, ncol=3 if st.pres else 2)
    st.title(fig, "Post hoc, exploratory (not preregistered): Global Linear with the mean of subject slopes")
    return fig, rows


# ---------------------------------------------------------------------------
REC = lambda f: f / "records.csv"                                                # noqa: E731
CM = lambda f: f / "comparisons_main.csv"                                        # noqa: E731
FIGURES = {
    "Figure01_DataOverview": (figure01, [UNION_CSV]),
    "Figure02_ModelComparison": (figure02, [REC(DEV), REC(EXT)]),
    "Figure03_PairedDifference_BySubject": (figure03, [REC(DEV), REC(EXT), CM(DEV), CM(EXT)]),
    "Figure04_AccuracyVersusPeakCost": (figure04, [CM(DEV), CM(EXT), DEV / "comparisons_supplementary.csv", EXT / "comparisons_supplementary.csv"]),
    "Figure05_BandError_DifferenceFromGlobalLinear": (figure05, [BAND_DEV / "band_error_comparisons.csv", EXT / "comparisons_band.csv"]),
    "Figure06_ResidualSlope_VersusPersonalReference": (figure06, [BAND_DEV / "residual_slope_per_subject.csv", EXT / "residual_slope_per_subject.csv",
                                                                  BAND_DEV / "residual_slope_summary.csv", EXT / "residual_slope_summary.csv"]),
    "Figure07_ReconstructionExamples_ExternalTest6": (figure07, [EXT / "predictions.json", UNION_CSV]),
    "Figure08_WavelengthError_ExternalTest6": (figure08, [EXT / "predictions.json", UNION_CSV, REC(EXT)]),
    "Figure09_FixedTargetRepeatability": (figure09, [REC(DEV), REC(EXT)]),
    "SupplementaryFigure01_ResidualSlopeSpectrum": (supplementary01, [BAND_DEV / "residual_slope_per_subject.csv", EXT / "residual_slope_per_subject.csv"]),
    "SupplementaryFigure02_ErrorByHbIChangeSize": (supplementary02, [DEV / "comparisons_delta_hbi.csv", EXT / "comparisons_delta_hbi.csv"]),
    "SupplementaryFigure03_PeakRS_ReferenceComparison": (supplementary03, [EXT / "peak_rs_reference.csv"]),
    "SupplementaryFigure04_TrainingSizeEffect": (supplementary04, [U16 / "training_size_effect.csv", REC(EXT), REC(U16 / "Union16_LeaveOneSubjectOut"),
                                                                   REC(U16 / "Development10_LeaveOneSubjectOut_FixedHyperparameters")]),
    "SupplementaryFigure05_BiGRU_TuningSurface": (supplementary05, [TUNE]),
    "SupplementaryFigure06_SHAP_InputGroupShare": (supplementary06, [SHAP / "shap_summary.csv", SHAP / "shap_input_group_by_output_band.csv"]),
    "SupplementaryFigure07_SplitSetConsistency": (supplementary07, [CM(f) for _, f, _ in SPLIT_SETS]),
    "PostHocExploratoryFigure01_SubjectMeanSlopeBaseline": (posthoc01, [POST / "records.csv", POST / "slopes.csv"]),
}


def build(name: str, profiles: list) -> dict | None:
    fn, inputs = FIGURES[name]
    missing = [str(p.relative_to(ROOT)) for p in inputs if not p.exists()]
    if missing:
        print(f"[skipped] {name}: missing input {missing}")
        return None
    info = dict(figure=name, inputs="; ".join(str(p.relative_to(ROOT)).replace("\\", "/") for p in inputs), strings={})
    for prof in profiles:
        st = Style(prof)
        with st.rc():
            fig, rows = fn(st)
            info["strings"][prof] = FS.figure_strings(fig)
            st.save(fig, OUT / prof / f"{name}.jpg")
            w, h = fig.get_size_inches()
            info[f"{prof}_pixels"] = f"{round(w * st.dpi)} x {round(h * st.dpi)}"
            plt.close(fig)
    d = pd.DataFrame(rows, columns=["panel", "series", "x", "y", "error", "n", "kind", "k"])
    (OUT / "plotted_data").mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "plotted_data" / f"{name}.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    print(f"[done] {name}: " + ", ".join(f"{p} {info[f'{p}_pixels']}" for p in profiles))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["both", "presentation", "report"], default="both")
    ap.add_argument("--only", nargs="*", default=None, help="figure name prefixes, for example Figure03")
    a = ap.parse_args(argv)
    profiles = ["presentation", "report"] if a.profile == "both" else [a.profile]
    names = [n for n in FIGURES if a.only is None or any(n.startswith(o) for o in a.only)]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    infos = [i for i in (build(n, profiles) for n in names) if i]
    OUT.mkdir(parents=True, exist_ok=True)
    spath, mpath = OUT / "figure_strings.json", OUT / "figure_manifest.csv"
    strings = json.loads(spath.read_text(encoding="utf-8")) if spath.exists() else {}
    for i in infos:
        strings.setdefault(i["figure"], {}).update(i.pop("strings"))
    spath.write_text(json.dumps(strings, ensure_ascii=False, indent=1), encoding="utf-8")
    old = pd.read_csv(mpath, encoding="utf-8-sig") if mpath.exists() else pd.DataFrame()
    new = pd.DataFrame([dict(i, generated_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), code_commit=commit) for i in infos])
    if len(old):
        keep = old.set_index("figure")
        for col in ("presentation_pixels", "report_pixels"):     # a profile that was not redrawn keeps its recorded size
            if col in keep.columns:
                prev = new.figure.map(keep[col])
                new[col] = new[col].fillna(prev) if col in new.columns else prev
        new = pd.concat([old[~old.figure.isin(new.figure)], new], ignore_index=True)
    new["order"] = new.figure.map({n: k for k, n in enumerate(FIGURES)})
    new.sort_values("order").drop(columns="order").to_csv(mpath, index=False, encoding="utf-8-sig")
    from figure_captions import write_captions
    print("captions:", write_captions().relative_to(ROOT))
    return infos


if __name__ == "__main__":
    main()
