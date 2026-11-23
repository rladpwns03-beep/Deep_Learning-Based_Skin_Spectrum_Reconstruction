"""Document and figures for the repeated experiments (post hoc, exploratory). Called by scripts/run_repeat_experiments.py --aggregate.

Writes docs/Results_Repeat_Experiments.md and results/repeat_experiments/figures/<profile>/<FigureName>.jpg.
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
from matplotlib.patches import Patch                                    # noqa: E402

OUT = ROOT / "results" / "repeat_experiments"
SPLITS = ROOT / "data" / "splits_repeat"
DOC = ROOT / "docs" / "Results_Repeat_Experiments.md"
K = 1e3
LABEL = dict(a_rmse="absorbance RMSE", peak_rmse="Peak RMSE", peak_sd="Peak SD at target HbI 0.6", spectral_sd="spectral SD at target HbI 0.6")
MARK = "Post hoc, exploratory"
CURVE_MODELS = ["PLS_v2", "Ridge", "BiGRU", "BiGRU (fixed setting)"]
ALL_MODELS = ["HbI-binned mean", "Global Linear", "PLS_v2", "Ridge", "BiGRU", "BiGRU (fixed setting)"]
VAR_CMP = ["PLS_v2 minus Global Linear", "BiGRU minus Global Linear", "BiGRU minus PLS_v2"]
DATA_LABEL = {"Development10": "Development cohort data (10 subjects)", "Union16": "Union16 data (16 subjects)"}


def table(header, rows) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def num(v, nd=3):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


# ---------------------------------------------------------------------------
def figure_learning(st, res):
    lc = res["lc"]
    lc = lc[lc.test_subjects == "all test subjects"]
    fig = st.figure(6.5, 5.8)
    axes = fig.subplots(2, 2, sharey="row")
    for c, data_name in enumerate(DATA_LABEL):
        for r, met in enumerate(("a_rmse", "peak_rmse")):
            ax = axes[r, c]
            st.zero(ax)
            d0 = lc[(lc.data == data_name) & (lc.metric == met)]
            sizes = sorted(d0.n_train.unique())
            for mi, m in enumerate(CURVE_MODELS):
                d = d0[d0.model == m].set_index("n_train").loc[sizes]
                kw = {k: v for k, v in st.marker_kw(m, 0.75).items() if k != "ls"}
                ax.errorbar(np.array(sizes) + (mi - 1.5) * 0.09, d.diff_from_global_linear_mean * K, yerr=d.diff_se * K, color=MODEL_STYLE[m]["color"],
                            ls=MODEL_STYLE[m]["ls"], lw=st.lw * 0.6, elinewidth=st.lw * 0.5, capsize=st.ms * 0.25, zorder=3, **kw)
            ax.set_xticks(sizes)
            if r == 0:
                ax.set_title(DATA_LABEL[data_name].replace(" (", "\n(") if st.pres else DATA_LABEL[data_name], fontsize=st.tick)
            if r == 1:
                ax.set_xlabel("Number of training subjects")
            if c == 0:
                st.ylab(ax, "Absorbance RMSE" if met == "a_rmse" else "Peak RMSE", "difference", f"({X3} absorbance)")
    axes[0, 0].text(0.03, 0.05, MARK, transform=axes[0, 0].transAxes, ha="left", va="bottom", fontsize=st.small(), fontweight="bold",
                    bbox=dict(fc="white", ec="#000000", lw=st.lw * 0.4, boxstyle="round,pad=0.3"))
    st.legend_below(fig, [st.model_handle(m, line=True, label=f"{m} {MINUS} Global Linear") for m in CURVE_MODELS] +
                    [Line2D([], [], color="#000000", marker="|", ms=st.ms, ls="none", label="Mean and standard error across test subjects")], ncol=3 if st.pres else 2)
    st.title(fig, "Post hoc, exploratory: difference from Global Linear by number of training subjects")
    return fig


def figure_variability(st, res):
    s120 = "Development10_Holdout73_All120"
    dist, var, old = res["dist"][s120], res["var"], res["old_ids"]
    fig = st.figure(6.5, 5.4)
    axes = fig.subplots(2, 3)
    for r, met in enumerate(("a_rmse", "peak_rmse")):
        for c, cmp_ in enumerate(VAR_CMP):
            ax = axes[r, c]
            g = dist[(dist.comparison == cmp_) & (dist.metric == met)].set_index("split_id").difference * K
            v = var[(var.split_set == s120) & (var.comparison == cmp_) & (var.metric == met)].iloc[0]
            color = MODEL_STYLE[cmp_.split(" minus ")[0]]["color"]
            ax.axvspan(v.random_10_mean_p2_5 * K, v.random_10_mean_p97_5 * K, color="#DDDDDD", lw=0, zorder=0)
            ax.hist(g.values, bins=18, color=color, alpha=0.55, edgecolor="white", lw=st.lw * 0.2, zorder=2)
            st.zero(ax, horizontal=False)
            ax.axvline(g.mean(), color="#000000", lw=st.lw * 0.7, zorder=4)
            ax.axvline(v.existing_10_splits_mean * K, color="#000000", lw=st.lw * 0.7, ls=(0, (2, 2)), zorder=4)
            ax.plot(g.loc[old].values, np.full(10, ax.get_ylim()[1] * 0.96), marker="v", ls="none", ms=st.ms * 0.55, mfc="white", mec="#000000", mew=st.lw * 0.4, zorder=5)
            ax.locator_params(axis="x", nbins=5)
            if r == 0:
                ax.set_title(cmp_.replace(" minus ", f" {MINUS} "), fontsize=st.tick)
            if c == 0:
                st.ylab(ax, "Number of splits", "(absorbance RMSE)" if met == "a_rmse" else "(Peak RMSE)")
            if r == 1:
                ax.set_xlabel(f"Split mean difference\n({X3} absorbance)")
    axes[0, 0].text(0.98, 0.62, MARK, transform=axes[0, 0].transAxes, ha="right", va="top", fontsize=st.small() * 0.9, fontweight="bold",
                    bbox=dict(fc="white", ec="#000000", lw=st.lw * 0.4, boxstyle="round,pad=0.3"))
    st.legend_below(fig, [Patch(fc="#888888", alpha=0.55, label="All 120 splits (7 training and 3 test subjects)"),
                          Line2D([], [], marker="v", mfc="white", mec="#000000", ls="none", ms=st.ms * 0.6, label="The 10 existing splits"),
                          Line2D([], [], color="#000000", lw=st.lw * 0.7, label="Mean of the 120 splits"),
                          Line2D([], [], color="#000000", lw=st.lw * 0.7, ls=(0, (2, 2)), label="Mean of the 10 existing splits"),
                          Patch(fc="#DDDDDD", label="2.5 to 97.5 percentile of the mean of 10 randomly drawn splits")], ncol=3 if st.pres else 2)
    st.title(fig, "Post hoc, exploratory: split-to-split variation over all 120 Development cohort 7:3 splits")
    return fig


def make_figures(res) -> dict:
    strings = {}
    for name, fn in (("RepeatFigure01_LearningCurve", figure_learning), ("RepeatFigure02_SplitSamplingVariability", figure_variability)):
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
def cmp_rows(c: pd.DataFrame):
    return [[r.test_subjects, r.comparison, LABEL[r.metric], int(r.n), num(r.mean_a * K), num(r.mean_b * K), f"{r.diff_mean * K:+.3f} ± {r.diff_se * K:.3f}",
             f"{int(r.n_improved)}/{int(r.n)}", num(r.wilcoxon_p), num(r.sign_p)] for r in c.itertuples()]


CMP_HEAD = ["Test subjects", "Comparison", "Metric", "n", "First model", "Second model", "Difference ± standard error", "Difference < 0", "Wilcoxon p", "Sign test p"]


def write(res):
    make_figures(res)
    lc, var, sels = res["lc"], res["var"], res["sels"]
    o = ["# Repeated experiments (post hoc, exploratory)", "",
         "**Status: post hoc, exploratory. Not preregistered; added after the ExternalTest6 results were seen.** The preregistered results are unchanged and the "
         "ExternalTest6 configuration (training = exactly the 10 Development cohort subjects, evaluation = the 6 External cohort subjects) is not among the splits. "
         "This document is generated by `scripts/run_repeat_experiments.py --aggregate` and reports numbers only.", "",
         "Units: 10^-3 absorbance. Subject value = mean of the metric over all splits of a set in which the subject is a test subject (BiGRU: mean over seeds 0, 1, 2 first). "
         "Difference = first model minus second model; \"Difference < 0\" counts subjects. p values are descriptive, without correction for multiple comparisons. "
         "**Splits share subjects, so the standard deviation between splits is not the error of independent samples.**", "",
         "Models: HbI-binned mean and Global Linear as before; PLS_v2 (number of components up to 10) and Ridge (alpha) selected inside the training subjects of "
         "every split; BiGRU with the frozen values (hidden 8, slope_l2 0, epoch 200) and BiGRU (fixed setting) (hidden 8, slope_l2 0.001, epoch 150), seeds 0, 1, 2. "
         "Intended exception: the two BiGRU settings were determined from results of the 10 Development cohort subjects, so when a Development cohort subject is a "
         "test subject, that subject took part in choosing the setting.", "",
         "## 1. Split sets (`data/splits_repeat/`)", ""]
    rows = []
    for s in res["plans"]:
        m = json.loads((SPLITS / s / "manifest.json").read_text(encoding="utf-8"))
        rows.append([s, m["n_train"], m["n_test"], m["n_splits"], m["n_candidate_test_sets"], m["selection"].split(" from ")[0] + (f" (seed {m['seed']})" if m["seed"] else ""),
                     f"{m['test_count_min']} to {m['test_count_max']}"])
    o += [table(["Split set", "Training subjects", "Test subjects", "Splits", "Candidate test sets", "Selection", "Times a subject is a test subject"], rows)]
    m100 = json.loads((SPLITS / "Union16_Holdout73_Repeat100" / "manifest.json").read_text(encoding="utf-8"))
    m120 = json.loads((SPLITS / "Development10_Holdout73_All120" / "manifest.json").read_text(encoding="utf-8"))
    o += ["- Union16_Holdout73_Repeat100 was drawn uniformly without replacement from the 1,167 test sets that satisfy the stratification rules of Union16_Holdout73; "
          "`union_split_plan` was not used because it fixes the total number of Near-target subjects to 16, which is specific to 10 splits. "
          "Times each subject is a test subject: " + ", ".join(f"{k} {v}" for k, v in m100["test_count_per_subject"].items()) + f" (minimum {m100['test_count_min']}). "
          f"{m100['existing_splits']['n_existing_included']} of the 10 existing Union16_Holdout73 splits are among the 100.",
          f"- Development10_Holdout73_All120 contains all 10 existing Development10_Holdout73 splits: {m120['existing_splits']['all_existing_included']}. "
          "Development10_TrainSize7 uses the same 120 splits and shares its results.",
          "- The Union16 training-size sets are not stratified; the cohort composition of the training set of every split is recorded in `split_plan.csv`.", "",
          "## 2. Training-size tables (General reconstruction)", ""]
    for data_name in DATA_LABEL:
        groups = ["all test subjects"] + (["Development cohort", "External cohort"] if data_name == "Union16" else [])
        for grp in groups:
            for met in ("a_rmse", "peak_rmse"):
                d = lc[(lc.data == data_name) & (lc.test_subjects == grp) & (lc.metric == met)]
                sizes = sorted(d.n_train.unique(), reverse=True)
                rows = []
                for mdl in ALL_MODELS:
                    r = [mdl]
                    for n in sizes:
                        x = d[(d.model == mdl) & (d.n_train == n)]
                        if x.empty:
                            r.append("–")
                            continue
                        x = x.iloc[0]
                        r.append(f"{x['mean'] * K:.3f} ± {x.sd * K:.3f}" if mdl == "Global Linear" else
                                 f"{x['mean'] * K:.3f}; {x.diff_from_global_linear_mean * K:+.3f} ± {x.diff_se * K:.3f} ({int(x.n_improved)}/{int(x.n_subjects)})")
                    rows.append(r)
                o += [f"{DATA_LABEL[data_name]}, test subjects: {grp}, {LABEL[met]}. Cells: subject mean; difference from Global Linear ± standard error "
                      "(subjects with difference < 0 / n). Global Linear row: subject mean ± SD between subjects.", "",
                      table(["Model"] + [f"{n} training subjects" for n in sizes], rows)]
    o += ["## 3. More split repetitions: paired comparisons", ""]
    for s in ("Development10_Holdout73_All120", "Union16_Holdout73_Repeat100"):
        o += [f"### {s}", "", table(CMP_HEAD, cmp_rows(res["cmps"][s]))]
    o += ["## 4. Split sampling variability", "",
          "For every split, the split mean difference is the mean over the test subjects of that split of (first model − second model). "
          "The table gives the mean and the SD between splits of that value.", ""]
    rows = []
    for r in var[var.metric.isin(["a_rmse", "peak_rmse"])].itertuples():
        rows.append([r.split_set, r.comparison, LABEL[r.metric], int(r.n_splits), f"{r.mean_of_split_means * K:+.3f}", num(r.sd_between_splits * K),
                     f"{r.min * K:+.3f} to {r.max * K:+.3f}"])
    o += [table(["Split set", "Comparison", "Metric", "Splits", "Mean of split means", "SD between splits", "Range"], rows),
          "Position of the 10 existing Development10_Holdout73 splits inside all 120 splits. The percentile columns give the share of values that are not above the "
          "mean of the 10 existing splits: first among the 120 single-split values, then among the means of 10 splits drawn at random without replacement "
          "(10,000 draws, fixed seed). The last column is the 2.5 to 97.5 percentile range of those random means.", ""]
    rows = []
    for r in var[(var.split_set == "Development10_Holdout73_All120") & var.metric.isin(["a_rmse", "peak_rmse"])].itertuples():
        rows.append([r.comparison, LABEL[r.metric], f"{r.mean_of_split_means * K:+.3f}", f"{r.existing_10_splits_mean * K:+.3f}",
                     f"{r.existing_10_percentile_among_120_split_values:.1f}", f"{r.existing_10_percentile_among_means_of_10_random_splits:.1f}",
                     f"{r.random_10_mean_p2_5 * K:+.3f} to {r.random_10_mean_p97_5 * K:+.3f}"])
    o += [table(["Comparison", "Metric", "Mean of 120 splits", "Mean of the 10 existing splits", "Percentile among 120 split values",
                 "Percentile among means of 10 random splits", "2.5 to 97.5 percentile of means of 10 random splits"], rows),
          "## 5. Selected hyperparameters per split", ""]
    rows = []
    for s, d in sels.items():
        cnt = d.pls_v2_components.value_counts().reindex(range(1, 11), fill_value=0)
        al = d.ridge_alpha.value_counts().sort_index()
        rows.append([s, len(d)] + [int(cnt[k]) for k in range(1, 11)] + [int((d.pls_v2_components == d.pls_v2_max_candidate).sum()),
                                                                           ", ".join(f"{a:g}: {n}" for a, n in al.items()), int((~d.hbi_binned_mean_fitted).sum())])
    o += [table(["Split set", "Splits"] + [f"PLS_v2 {k}" for k in range(1, 11)] + ["PLS_v2 at the largest candidate", "Ridge alpha: number of splits",
                                                                                      "HbI-binned mean not fitted"], rows),
          "HbI-binned mean: empty HbI bins take the mean spectrum of the nearest filled bin (existing rule of `models/hb_index.py`); the number of empty bins per split is in `selection.csv`.", "",
          "## 6. Agreement with the stored leave-one-subject-out results", "",
          "Development10_TrainSize9 and Union16_TrainSize15 use the same splits as the stored leave-one-subject-out results. The deterministic models were "
          "compared subject by subject (relative tolerance 1e-9).", ""]
    o += [table(["Set, model, metric", "Equal", "Mean (new)", "Mean (stored)"],
                [[k, v["equal"], num(v["mean_new"] * K), num(v["mean_stored"] * K)] for k, v in res["checks"].items()]),
          "Union16_TrainSize15 PLS_v2 is not compared: the stored Union16 result uses a fixed number of components, whereas here the number is selected in every split.", "",
          "## 7. Files", "",
          "- Results: `results/repeat_experiments/<split set>/` (`records.csv`, `summary_by_subject.csv`, `comparisons.csv`, `selection.csv`, `config.json`), "
          "`results/repeat_experiments/learning_curve.csv`, `split_sampling_variability.csv`, `checks.json`.",
          "- Figures: `results/repeat_experiments/figures/presentation/` and `report/`: `RepeatFigure01_LearningCurve`, `RepeatFigure02_SplitSamplingVariability`. "
          "Both carry the label \"Post hoc, exploratory\".",
          "- Code: `common/make_splits_repeat.py`, `scripts/run_repeat_experiments.py`, `scripts/repeat_report.py`, `scripts/posthoc_common.py`, `tests/test_repeat.py`.", ""]
    DOC.write_text("\n".join(o), encoding="utf-8")
    print("written:", DOC.relative_to(ROOT))
