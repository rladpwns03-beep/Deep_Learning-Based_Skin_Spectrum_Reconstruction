"""Document and figures for the BiGRU structure ablation (post hoc, exploratory). Called by scripts/ablation_bigru_structure.py --aggregate.

Writes docs/Results_Ablation_BiGRU_Structure.md and results/ablation_bigru_structure/figures/<profile>/<FigureName>.jpg.
Numbers only; no evaluative wording. Units: 10^-3 absorbance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))
import figure_style as FS                                               # noqa: E402
from figure_style import MINUS, MODEL_STYLE, X3, Style, plt            # noqa: E402
from matplotlib.lines import Line2D                                     # noqa: E402
from models.bigru_variants import REMOVED, STRUCTURAL_IDENTITY, VARIANTS   # noqa: E402
from posthoc_common import cohort_of                                    # noqa: E402

OUT = ROOT / "results" / "ablation_bigru_structure"
DOC = ROOT / "docs" / "Results_Ablation_BiGRU_Structure.md"
K = 1e3
REF = "BiGRU_ResidualSlope"
OTHERS = [v for v in VARIANTS if v != REF]
SETS = ["Development10_LeaveOneSubjectOut", "Union16_LeaveOneSubjectOut"]
SET_LABEL = {"Development10_LeaveOneSubjectOut": "Development10 leave-one-subject-out (n = 10)", "Union16_LeaveOneSubjectOut": "Union16 leave-one-subject-out (n = 16)"}
LABEL = dict(a_rmse="absorbance RMSE", peak_rmse="Peak RMSE", peak_sd="Peak SD at target HbI 0.6", spectral_sd="spectral SD at target HbI 0.6")
VSTYLE = {"BiGRU_ResidualSlope": ("#D55E00", "o", "-"), "BiGRU_FullSlope": ("#7B3294", "s", "--"), "BiGRU_AdditiveCorrection": ("#2B83BA", "^", "-."),
          "BiGRU_DirectOutput": ("#4D4D4D", "D", ":")}
MARK = "Post hoc, exploratory"
EPOCHS = [0, 25, 50, 100, 150, 200, 300, 500, 1000]


def table(header, rows) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def num(v, nd=3):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def cmp_rows(c: pd.DataFrame, extra=None):
    rows = []
    for r in c.itertuples():
        rows.append([r.test_subjects] + ([getattr(r, extra)] if extra else []) + [
            r.comparison, LABEL[r.metric], int(r.n), num(r.mean_a * K), num(r.mean_b * K), f"{r.diff_mean * K:+.3f} ± {r.diff_se * K:.3f}",
            f"{int(r.n_improved)}/{int(r.n)}", num(r.wilcoxon_p), num(r.sign_p)])
    return rows


CMP_HEAD = ["Test subjects", "Comparison", "Metric", "n", "First model", "Second model", "Difference ± standard error", "Difference < 0", "Wilcoxon p", "Sign test p"]


def subject_diff(rec, variant, metric, epoch=200, lr=3e-3):
    g = rec[np.isclose(rec.lr, lr) & (rec.epoch == epoch) & (rec.metric == metric)]
    a = g[g.variant == variant].groupby("subject").value.mean()
    b = g[g.variant == REF].groupby("subject").value.mean()
    return ((a - b) * K).sort_index()


# ---------------------------------------------------------------------------
def figure_curves(st, res):
    fig = st.figure(6.5, 6.0)
    axes = fig.subplots(2, 2, sharex=True, sharey="row")
    x = np.arange(len(EPOCHS))
    for c, s in enumerate(SETS):
        cur = res[s]["curves"]
        for r, met in enumerate(("a_rmse", "peak_rmse")):
            ax = axes[r, c]
            for name, (col, mk, ls) in VSTYLE.items():
                d = cur[(cur.model == name) & np.isclose(cur.lr, 3e-3) & (cur.metric == met)].set_index("epoch").loc[EPOCHS]
                ax.errorbar(x, d["mean"] * K, yerr=d["se"] * K, color=col, marker=mk, ls=ls, lw=st.lw * 0.6, ms=st.ms * 0.55, elinewidth=st.lw * 0.5,
                            capsize=st.ms * 0.25, zorder=3)
            for m in ("Global Linear", "PLS_v2"):
                v = cur[(cur.model == f"{m} (stored result)") & (cur.metric == met)]["mean"].iloc[0] * K
                ax.axhline(v, color=MODEL_STYLE[m]["color"], lw=st.lw * 0.5, ls=(0, (4, 3)), zorder=1)
            ax.set_yscale("log")
            ax.set_xticks(x, [str(e) for e in EPOCHS], rotation=90 if st.pres else 0)
            if r == 0:
                ax.set_title(SET_LABEL[s].replace(" (", "\n(") if st.pres else SET_LABEL[s], fontsize=st.tick)
            if r == 1:
                ax.set_xlabel("Epoch (checkpoint)")
            if c == 0:
                st.ylab(ax, "Absorbance RMSE" if met == "a_rmse" else "Peak RMSE", f"({X3} absorbance,", "log scale)")
    axes[0, 0].text(0.98, 0.95, MARK, transform=axes[0, 0].transAxes, ha="right", va="top", fontsize=st.small(), fontweight="bold",
                    bbox=dict(fc="white", ec="#000000", lw=st.lw * 0.4, boxstyle="round,pad=0.3"))
    h = [Line2D([], [], color=c, marker=m, ls=l, lw=st.lw * 0.6, ms=st.ms * 0.55, label=n) for n, (c, m, l) in VSTYLE.items()]
    h += [Line2D([], [], color=MODEL_STYLE[m]["color"], lw=st.lw * 0.5, ls=(0, (4, 3)), label=f"{m} (stored result)") for m in ("Global Linear", "PLS_v2")]
    h += [Line2D([], [], color="#000000", marker="|", ms=st.ms, ls="none", label="Mean and standard error across subjects (5 seeds averaged per subject)")]
    st.legend_below(fig, h, ncol=3 if st.pres else 2)
    st.title(fig, "Post hoc, exploratory: test error of four BiGRU structures by training epoch")
    return fig


def figure_paired(st, res):
    fig = st.figure(6.5, 5.8)
    axes = fig.subplots(2, 3, sharex=True)
    for r, met in enumerate(("a_rmse", "peak_rmse")):
        for c, v in enumerate(OTHERS):
            ax = axes[r, c]
            st.zero(ax)
            tops = []
            for si, s in enumerate(SETS):
                d = subject_diff(res[s]["rec"], v, met)
                off = np.linspace(-0.24, 0.24, len(d))
                for o, (sid, val) in zip(off, d.items()):
                    ext = cohort_of(sid) == "External cohort"
                    ax.plot(si + o, val, marker="^" if ext else "o", mfc="white" if ext else VSTYLE[v][0], mec=VSTYLE[v][0], mew=st.lw * 0.5, ms=st.ms * 0.6,
                            ls="none", alpha=0.85, zorder=3)
                st.mean_bar(ax, si, d.values)
                tops.append((si, int((d.values < 0).sum()), len(d)))
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.18 * (hi - lo))
            for si, k, n in tops:
                ax.text(si, 0.985, f"{k}/{n} lower", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=st.small())
            ax.set_xlim(-0.6, 1.6)
            ax.set_xticks([0, 1], ["Development10\n(n = 10)", "Union16\n(n = 16)"])
            ax.tick_params(axis="x", length=0)
            if r == 0:
                ax.set_title(f"{v.replace('BiGRU_', '')}\n{MINUS} ResidualSlope", fontsize=st.tick)
            if c == 0:
                st.ylab(ax, "Difference in", "absorbance RMSE" if met == "a_rmse" else "Peak RMSE", f"({X3} absorbance)")
    axes[1, 0].text(0.03, 0.04, MARK, transform=axes[1, 0].transAxes, ha="left", va="bottom", fontsize=st.small(), fontweight="bold",
                    bbox=dict(fc="white", ec="#000000", lw=st.lw * 0.4, boxstyle="round,pad=0.3"))
    g = "#555555"
    st.legend_below(fig, [Line2D([], [], marker="o", mfc=g, mec=g, ls="none", ms=st.ms * 0.7, label="Development cohort test subject"),
                          Line2D([], [], marker="^", mfc="white", mec=g, ls="none", ms=st.ms * 0.7, label="External cohort test subject"), st.mean_handle(),
                          Line2D([], [], ls="none", label="Epoch 200, leave-one-subject-out; column titles name the BiGRU variants")], ncol=3 if st.pres else 2)
    st.title(fig, "Post hoc, exploratory: each BiGRU variant minus BiGRU_ResidualSlope at epoch 200")
    return fig


def make_figures(res) -> dict:
    strings = {}
    for name, fn in (("AblationFigure01_EpochCurves", figure_curves), ("AblationFigure02_PairedDifference_Epoch200", figure_paired)):
        strings[name] = {}
        for prof in ("presentation", "report"):
            st = Style(prof)
            with st.rc():
                fig = fn(st, res)
                strings[name][prof] = FS.figure_strings(fig)
                st.save(fig, OUT / "figures" / prof / f"{name}.jpg")
                plt.close(fig)
    (OUT / "figures" / "figure_strings.json").write_text(json.dumps(strings, ensure_ascii=False, indent=1), encoding="utf-8")
    return strings


# ---------------------------------------------------------------------------
def write(res):
    make_figures(res)
    o = ["# BiGRU structure ablation (post hoc, exploratory)", "",
         "**Status: post hoc, exploratory. Not preregistered; added after the ExternalTest6 results were seen.** The preregistered results "
         "(`results/final_comparison`, `results/external_test`, `results/union16`) are unchanged, and the ExternalTest6 configuration was not used. "
         "This document is generated by `scripts/ablation_bigru_structure.py --aggregate` and reports numbers only.", "",
         "Units: 10^-3 absorbance. Subject value = mean of the metric over seeds. Difference = first model minus second model; "
         "the column \"Difference < 0\" counts subjects. The standard error is the between-subject SD of the paired difference divided by sqrt(n) (ddof = 1). "
         "p values (exact Wilcoxon signed-rank, sign test, two-sided) are descriptive and not corrected for multiple comparisons.", "",
         "## 1. Variants and training", "",
         "All variants use the same encoder (4 inputs per wavelength, one bidirectional GRU layer with hidden 8 per direction, shared Linear(16, 1) "
         "initialised to zero; 689 parameters). Only the assembly of the linear-layer output differs. In all variants A*(570) = h*.", "",
         table(["Variant", "Assembly", "Removed element", "Identity holds by structure"], [
             ["BiGRU_ResidualSlope", "A* = A_in + (h* − h_in)(b + r)", REMOVED["BiGRU_ResidualSlope"], "yes"],
             ["BiGRU_FullSlope", "A* = A_in + (h* − h_in) s", REMOVED["BiGRU_FullSlope"], "yes"],
             ["BiGRU_AdditiveCorrection", "A* = A_in + g", REMOVED["BiGRU_AdditiveCorrection"], "no"],
             ["BiGRU_DirectOutput", "A* = μ + σ f", REMOVED["BiGRU_DirectOutput"], "no"]]),
         "Training (fixed before running): hidden 8, lr 3e-3, clip 1.0, Adam, full batch, subject-weighted MSE over the 30 wavelengths other than 570 nm, "
         "no slope penalty, 1,000 epochs with test metrics recorded at epochs 0, 25, 50, 100, 150, 200, 300, 500, 1000, seeds 0 to 4. "
         "Evaluation sets: Development10_LeaveOneSubjectOut (10 folds) and Union16_LeaveOneSubjectOut (16 folds), read from `data/splits_final/`. "
         "BiGRU_ResidualSlope at epoch 200 equals the frozen BiGRU setting (hidden 8, slope_l2 0, epoch 200).", "",
         "## 2. Main comparison: each variant minus BiGRU_ResidualSlope at epoch 200", ""]
    for s in SETS:
        o += [f"### {s}", "", table(CMP_HEAD, cmp_rows(res[s]["c200"]))]
    o += ["## 3. Supplementary comparison: oracle checkpoint (optimistic for the variant)", "",
          "For each variant the checkpoint with the lowest test mean absorbance RMSE is used and compared with BiGRU_ResidualSlope at epoch 200. "
          "The checkpoint is chosen on the test subjects, so this comparison is optimistic for the variant.", ""]
    for s in SETS:
        o += [f"### {s}", "", "Oracle epoch: " + ", ".join(f"{v} {res[s]['oracle'][v]}" for v in VARIANTS) + ".", "",
              table(CMP_HEAD[:1] + ["Oracle epoch of the variant"] + CMP_HEAD[1:], cmp_rows(res[s]["co"], "oracle_epoch_of_variant"))]
    o += ["## 4. Epoch curves (subject mean ± standard error, lr 3e-3, 5 seeds)", ""]
    for s in SETS:
        cur = res[s]["curves"]
        for met in ("a_rmse", "peak_rmse"):
            rows = []
            for v in VARIANTS:
                d = cur[(cur.model == v) & np.isclose(cur.lr, 3e-3) & (cur.metric == met)].set_index("epoch")
                rows.append([v] + [f"{d.loc[e, 'mean'] * K:.3f} ± {d.loc[e, 'se'] * K:.3f}" for e in EPOCHS])
            for m in ("Global Linear", "PLS_v2"):
                d = cur[(cur.model == f"{m} (stored result)") & (cur.metric == met)].iloc[0]
                rows.append([f"{m} (stored result)"] + [f"{d['mean'] * K:.3f} ± {d['se'] * K:.3f}"] + ["same value"] * (len(EPOCHS) - 1))
            o += [f"{s}, {LABEL[met]}", "", table(["Model"] + [f"epoch {e}" for e in EPOCHS], rows)]
    o += ["The stored Global Linear and PLS_v2 rows are the values of the existing results for the same folds "
          "(Development10: `results/final_comparison/Development10_LeaveOneSubjectOut`; Union16: `results/union16/Union16_LeaveOneSubjectOut`, "
          "where PLS_v2 uses the fixed number of components).", "",
          "## 5. Identity error (target HbI = input HbI)", "",
          "RMSE between output and input over the 30 wavelengths other than 570 nm, for the measurements of each test subject; mean over subjects and seeds. "
          "\"Largest\" is the largest absolute difference over all subjects, seeds and wavelengths.", ""]
    rows = []
    for s in SETS:
        idn = res[s]["ident"]
        for v in VARIANTS:
            d = idn[(idn.variant == v) & np.isclose(idn.lr, 3e-3)]
            r200 = d[(d.epoch == 200) & (d.metric == "identity_rmse")]["mean"].iloc[0]
            m200 = d[(d.epoch == 200) & (d.metric == "identity_max_abs")]["max"].iloc[0]
            mall = d[d.metric == "identity_max_abs"]["max"].max()
            rows.append([s, v, "yes" if STRUCTURAL_IDENTITY[v] else "no", f"{r200 * K:.3f}", f"{m200 * K:.3f}", f"{mall * K:.3f}"])
    o += [table(["Evaluation set", "Variant", "Identity by structure", "RMSE at epoch 200", "Largest at epoch 200", "Largest over all checkpoints"], rows),
          "## 6. Seed variability", "",
          "SD between seeds of the evaluation-set mean absorbance RMSE (each seed: mean over test subjects), lr 3e-3, 5 seeds.", ""]
    rows = []
    for s in SETS:
        sv = res[s]["sv"]
        for v in VARIANTS:
            d = sv[(sv.variant == v) & np.isclose(sv.lr, 3e-3)].set_index("epoch")
            rows.append([s, v] + [f"{d.loc[e, 'mean_over_seeds'] * K:.3f} (SD {d.loc[e, 'sd_between_seeds'] * K:.3f})" for e in (200, 1000)])
    o += [table(["Evaluation set", "Variant", "epoch 200: mean (SD between seeds)", "epoch 1000: mean (SD between seeds)"], rows),
          "## 7. Learning-rate sensitivity (supplementary; Development10_LeaveOneSubjectOut, seeds 0 to 2)", "",
          "The three variants without the Global Linear prior were also trained with lr 1e-3 and 1e-2. All rows of this table use seeds 0 to 2. "
          "The oracle epoch is the checkpoint with the lowest test mean absorbance RMSE.", ""]
    lrs = res["Development10_LeaveOneSubjectOut"]["lrs"].sort_values(["variant", "lr"])
    o += [table(["Variant", "lr", "absorbance RMSE at epoch 200", "Peak RMSE at epoch 200", "Oracle epoch", "absorbance RMSE at oracle epoch", "Peak RMSE at oracle epoch"],
                [[r.variant, f"{r.lr:g}", num(r.a_rmse_epoch200 * K), num(r.peak_rmse_epoch200 * K), "–" if pd.isna(r.oracle_epoch) else int(r.oracle_epoch),
                  num(r.a_rmse_oracle_epoch * K), num(r.peak_rmse_oracle_epoch * K)] for r in lrs.itertuples()]),
          "## 8. Files", "",
          "- Results: `results/ablation_bigru_structure/<evaluation set>/` (`records.csv`, `comparisons_epoch200.csv`, `comparisons_oracle_checkpoint.csv`, "
          "`curves.csv`, `identity_error.csv`, `seed_variability.csv`, `lr_sensitivity.csv`, `config.json`).",
          "- Figures: `results/ablation_bigru_structure/figures/presentation/` and `report/`: `AblationFigure01_EpochCurves` (log-scaled y axis), "
          "`AblationFigure02_PairedDifference_Epoch200`. Both carry the label \"Post hoc, exploratory\".",
          "- Code: `models/bigru_variants.py`, `scripts/ablation_bigru_structure.py`, `scripts/ablation_report.py`, `tests/test_ablation.py`.", ""]
    DOC.write_text("\n".join(o), encoding="utf-8")
    print("written:", DOC.relative_to(ROOT))
