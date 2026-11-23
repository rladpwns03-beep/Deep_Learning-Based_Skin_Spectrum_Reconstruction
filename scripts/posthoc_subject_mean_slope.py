"""Post hoc, exploratory analysis (NOT preregistered; added after the ExternalTest6 results were seen).

Baseline variant: Global Linear with the mean of subject slopes instead of the pooled within-subject slope.

    b_s(lambda)    = sum_i (h_i - mean_h_s)(A_i - mean_A_s) / sum_i (h_i - mean_h_s)^2      (one slope per training subject)
    b_mean(lambda) = mean over training subjects of b_s(lambda),   b_mean(570) = 1
    reconstruction = models.linear.apply_slope (unchanged). No hyperparameter.

Run (project root, conda env skin_spectra):
    python -X utf8 scripts/posthoc_subject_mean_slope.py            # stops if results already exist
    python -X utf8 scripts/posthoc_subject_mean_slope.py --overwrite

Evaluation sets: Development10_LeaveOneSubjectOut (10 folds) and ExternalTest6 (train = Development cohort 10, test = External cohort 6).
Pairs and metrics come from the existing evaluation functions (scripts/final_utils.py, scripts/final_eval.py).
The pooled Global Linear is recomputed here and must equal the stored records before anything is written.
Output: results/posthoc_exploratory/subject_mean_slope_baseline/ only. Nothing is written to results/external_test/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))
from common.data_loader import HBI_INDEX, WL, Spectra                                    # noqa: E402
from final_utils import RES, UNION_CSV, jdump, load_splits, load_spectra, test_pairs      # noqa: E402
from final_eval import comparison_table, split_records                                    # noqa: E402
from models.base import finish, prepare                                                   # noqa: E402
from models.linear import LinearModel, apply_slope, fit_slope                             # noqa: E402

OUT = RES / "posthoc_exploratory" / "subject_mean_slope_baseline"
POOLED, VARIANT = "Global Linear", "Global Linear (mean of subject slopes)"
SETS = {"Development10_LeaveOneSubjectOut": RES / "final_comparison" / "Development10_LeaveOneSubjectOut" / "records.csv",
        "ExternalTest6": RES / "external_test" / "records.csv"}
METRICS = ["a_rmse", "peak_rmse"]


def fit_subject_mean_slope(train: Spectra) -> np.ndarray:
    slope = np.mean([fit_slope(train.select([s])) for s in train.subjects], axis=0)
    slope[HBI_INDEX] = 1.0
    return slope


class SubjectMeanSlopeModel:
    name = VARIANT

    def fit(self, train: Spectra):
        self.slope, self.train_subjects = fit_subject_mean_slope(train), train.subjects
        return self

    def predict(self, a, target):
        a, target, single = prepare(a, target)
        return finish(apply_slope(a, target, self.slope), single)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if OUT.exists() and any(OUT.iterdir()) and not a.overwrite:
        sys.exit(f"[stop] {OUT} already has results. Use --overwrite to replace them.")
    data = load_spectra(UNION_CSV)
    frames, slopes = [], []
    for set_name, stored_path in SETS.items():
        stored = pd.read_csv(stored_path, encoding="utf-8-sig")
        stored = stored[(stored.section == "general") & stored.metric.isin(METRICS)]
        new = []
        for sp in load_splits(set_name):
            tr, te = data.select(sp["train"]), data.select(sp["test"])
            assert not set(sp["train"]) & set(sp["test"]) and all(s < "S100" for s in sp["train"])
            models = {POOLED: [LinearModel().fit(tr)], VARIANT: [SubjectMeanSlopeModel().fit(tr)]}
            ii, jj, subj = test_pairs(set_name, sp["split_id"], data)
            recs = split_records(models, te, data.a[ii], data.a[jj], data.hbi[jj], data.hbi[ii], subj)
            f = pd.DataFrame([r for r in recs if r["section"] == "general" and r["metric"] in METRICS])
            f.insert(0, "split_id", sp["split_id"])
            new.append(f)
            for w, p, v in zip(WL, models[POOLED][0].slope, models[VARIANT][0].slope):
                slopes.append(dict(evaluation_set=set_name, split_id=sp["split_id"], n_train_subjects=len(sp["train"]),
                                   wavelength_nm=int(w), pooled=float(p), mean_of_subject_slopes=float(v), difference=float(v - p)))
        new = pd.concat(new, ignore_index=True)
        # the recomputed pooled Global Linear must equal the stored records
        key = ["split_id", "subject", "metric"]
        chk = new[new.model == POOLED].merge(stored[stored.model == POOLED], on=key, suffixes=("_new", "_stored"))
        if len(chk) != len(new[new.model == POOLED]) or not np.allclose(chk.value_new, chk.value_stored, rtol=1e-9, atol=1e-15):
            sys.exit(f"[stop] {set_name}: recomputed Global Linear differs from the stored records (nothing written)")
        both = pd.concat([new, stored[stored.model.isin(["PLS_v2", "BiGRU"])][new.columns]], ignore_index=True)
        both.insert(0, "evaluation_set", set_name)
        frames.append(both)
    df = pd.concat(frames, ignore_index=True)
    cmp_ = []
    for set_name, g in df.groupby("evaluation_set", sort=False):
        c = comparison_table(g.drop(columns="evaluation_set"), [(VARIANT, POOLED), (VARIANT, "PLS_v2"), (VARIANT, "BiGRU")])
        c.insert(0, "evaluation_set", set_name)
        c = c.drop(columns=["wilcoxon_p", "sign_p"])
        cmp_.append(c)
    OUT.mkdir(parents=True, exist_ok=True)
    f = dict(index=False, encoding="utf-8-sig", float_format="%.10g")
    df.to_csv(OUT / "records.csv", **f)
    pd.concat(cmp_, ignore_index=True).to_csv(OUT / "comparisons.csv", **f)
    pd.DataFrame(slopes).to_csv(OUT / "slopes.csv", **f)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    jdump(OUT / "config.json", dict(
        preregistered=False, computed_after_external_unblinding=True, status="post hoc, exploratory", code_commit=commit,
        variant=VARIANT, definition="b_mean = mean over training subjects of the within-subject least-squares slope b_s; b_mean(570) = 1; "
                                    "reconstruction by models.linear.apply_slope; no hyperparameter",
        evaluation_sets=list(SETS), metrics=METRICS, comparisons="difference = first model minus second model; n_improved = number of subjects with difference < 0",
        pooled_global_linear_equals_stored_records=True,
        note="Not part of the preregistered comparisons. Stored PLS_v2 and BiGRU subject values are reused unchanged."))
    out = pd.concat(cmp_, ignore_index=True)
    out["d"], out["se"] = out.diff_mean * 1e3, out.diff_se * 1e3
    out["a"], out["b"] = out.mean_a * 1e3, out.mean_b * 1e3
    print(out[["evaluation_set", "metric", "comparison", "n", "a", "b", "d", "se", "n_improved"]].round(3).to_string(index=False))
    print("saved:", OUT)


if __name__ == "__main__":
    main()
