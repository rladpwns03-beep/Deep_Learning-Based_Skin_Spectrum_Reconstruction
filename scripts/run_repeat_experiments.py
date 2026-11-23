"""Repeated experiments (post hoc, exploratory; added after the ExternalTest6 results were seen; not preregistered):
more split repetitions and training-size sets, evaluated with the existing metric functions.

Run (project root, conda env skin_spectra):
    python -X utf8 -m common.make_splits_repeat                                   # split sets in data/splits_repeat (once)
    python -X utf8 scripts/run_repeat_experiments.py --verify-parallel --workers 4
    python -X utf8 scripts/run_repeat_experiments.py --run --workers 14           # resumable (finished splits are skipped)
    python -X utf8 scripts/run_repeat_experiments.py --aggregate                  # tables, figures, document

Models (fixed before running)
  HbI-binned mean, Global Linear        as before
  PLS_v2                                B2PLSModel(max_components=10): number of components chosen inside the training subjects of every split
                                        (same code path as pls_v2_k10 of scripts/pls_compare.py)
  Ridge                                 B2RidgeModel(): alpha chosen inside the training subjects of every split
  BiGRU                                 frozen values (hidden 8, slope_l2 0, epoch 200), seeds 0, 1, 2
  BiGRU (fixed setting)                 hidden 8, slope_l2 1e-3, epoch 150, seeds 0, 1, 2
Intended exception: the two BiGRU settings were determined from results of the 10 Development cohort subjects, so in every split
in which a Development cohort subject is evaluated that subject took part in choosing the setting (same treatment as the Union16
supplementary results).

Subject value in a set = mean of the metric over all splits in which the subject is a test subject. Splits share subjects, so the
spread between splits is not the error of independent samples.
The ExternalTest6 configuration (training = exactly the 10 Development cohort subjects, evaluation = the 6 External cohort subjects)
is not among the splits. Writes only under results/repeat_experiments/ (and docs/Results_Repeat_Experiments.md with --aggregate).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))
from final_utils import RES, UNION_CSV, jdump, jload, load_spectra, paired_diff      # noqa: E402
from final_eval import comparison_table, fit_models, split_records, subject_values   # noqa: E402
from models.b2_pls import B2PLSModel                                                 # noqa: E402
from models.b2_ridge import B2RidgeModel                                             # noqa: E402
from models.hb_index import BinnedMeanModel                                          # noqa: E402
from models.linear import LinearModel                                                # noqa: E402
from posthoc_common import STATUS, code_commit, cohort_of, pairs_for, snapshot_existing_results   # noqa: E402

OUT = RES / "repeat_experiments"
SPLITS = ROOT / "data" / "splits_repeat"
PREREG = ROOT / "docs" / "Preregistration_ExternalTest_v1.json"
MODELS = ["HbI-binned mean", "Global Linear", "PLS_v2", "Ridge", "BiGRU", "BiGRU (fixed setting)"]
COMPARISONS = [("PLS_v2", "Global Linear"), ("Ridge", "Global Linear"), ("BiGRU", "Global Linear"), ("BiGRU (fixed setting)", "Global Linear"),
               ("BiGRU", "PLS_v2")]
METRICS = ["a_rmse", "peak_rmse", "peak_sd", "spectral_sd"]
KEEP = {("general", "a_rmse"), ("general", "peak_rmse"), ("repeatability", "peak_sd"), ("repeatability", "spectral_sd")}
SHARED = {"Development10_TrainSize7": "Development10_Holdout73_All120"}             # same 120 splits: computed once
COMPUTED_SETS = ["Union16_TrainSize15", "Union16_TrainSize13", "Union16_Holdout73_Repeat100", "Union16_TrainSize11", "Union16_TrainSize9",
                 "Union16_TrainSize7", "Union16_TrainSize5", "Development10_TrainSize9", "Development10_TrainSize8",
                 "Development10_Holdout73_All120", "Development10_TrainSize6", "Development10_TrainSize5"]
ALL_SETS = COMPUTED_SETS + list(SHARED)
LEARNING = {"Development10": [("Development10_TrainSize9", 9), ("Development10_TrainSize8", 8), ("Development10_TrainSize7", 7),
                              ("Development10_TrainSize6", 6), ("Development10_TrainSize5", 5)],
            "Union16": [("Union16_TrainSize15", 15), ("Union16_TrainSize13", 13), ("Union16_TrainSize11", 11), ("Union16_TrainSize9", 9),
                        ("Union16_TrainSize7", 7), ("Union16_TrainSize5", 5)]}
INTENDED_EXCEPTION = ("intended exception: the BiGRU settings (hidden 8, slope_l2 0, epoch 200; and hidden 8, slope_l2 0.001, epoch 150) were determined from "
                      "results of the 10 Development cohort subjects, so when a Development cohort subject is a test subject that subject took part in "
                      "choosing the setting. PLS_v2 and Ridge are selected inside the training subjects of each split.")
RESAMPLE_SEED, N_RESAMPLE = 20261123, 10000
_STATE = {}


def load_plan(set_name: str) -> list:
    plan = pd.read_csv(SPLITS / set_name / "split_plan.csv")
    return [dict(split_id=r.split_id, train=r.train.split(), test=r.test.split()) for r in plan.itertuples()]


def part_path(set_name: str, split_id: str) -> Path:
    return OUT / set_name / "parts" / f"{split_id}.json"


def _init():
    torch.set_num_threads(1)
    _STATE.update(data=load_spectra(UNION_CSV), hp=jload(PREREG)["hyperparameters"])


def compute(job) -> dict:
    set_name, sp = job
    data, hp = _STATE["data"], _STATE["hp"]
    assert not set(sp["train"]) & set(sp["test"])
    tr, te = data.select(sp["train"]), data.select(sp["test"])
    ii, jj, subj = pairs_for(data, sp["test"])
    models, sel = {}, dict(split_id=sp["split_id"], n_train=len(sp["train"]), hbi_binned_mean_fitted=True, hbi_binned_mean_error="", hbi_binned_mean_empty_bins=None)
    try:
        b = BinnedMeanModel().fit(tr)
        models["HbI-binned mean"] = [b]
        sel["hbi_binned_mean_empty_bins"] = int((b.bin_table.n_measurements == 0).sum())      # empty bins use the nearest filled bin
    except Exception as e:                                        # recorded as missing for this model only; the split continues
        sel.update(hbi_binned_mean_fitted=False, hbi_binned_mean_error=f"{type(e).__name__}: {e}")
    models["Global Linear"] = [LinearModel().fit(tr)]
    pls, ridge = B2PLSModel(max_components=10).fit(tr), B2RidgeModel().fit(tr)
    models["PLS_v2"], models["Ridge"] = [pls], [ridge]
    models.update(fit_models(tr, hp, ["BiGRU", "BiGRU (fixed setting)"]))
    sel.update(pls_v2_components=int(pls.n_components), pls_v2_max_candidate=int(max(pls.cv["components"])), ridge_alpha=float(ridge.alpha))
    recs = [r for r in split_records(models, te, data.a[ii], data.a[jj], data.hbi[jj], data.hbi[ii], subj)
            if (r["section"], r["metric"]) in KEEP and r["model"] != "Raw"]
    for r in recs:
        r["split_id"] = sp["split_id"]
    return dict(records=recs, selection=sel)


def work(job):
    p = part_path(job[0], job[1]["split_id"])
    if p.exists():
        return
    res = compute(job)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(res), encoding="utf-8")
    tmp.replace(p)


def all_jobs():
    return [(s, sp) for s in COMPUTED_SETS for sp in load_plan(s)]


def run(workers: int):
    snapshot_existing_results(OUT)
    jobs = [j for j in all_jobs() if not part_path(j[0], j[1]["split_id"]).exists()]
    print(f"splits to run: {len(jobs)} of {len(all_jobs())}", flush=True)
    t0, done = time.time(), 0
    if workers <= 1:
        _init()
        for j in jobs:
            work(j); done += 1
    else:
        with mp.get_context("spawn").Pool(workers, initializer=_init) as pool:
            for _ in pool.imap_unordered(work, jobs, chunksize=1):
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"finished {done} splits in {time.time() - t0:.0f}s", flush=True)
    (OUT / "run_seconds.txt").open("a", encoding="utf-8").write(f"{len(jobs)} splits, {workers} workers, {time.time() - t0:.0f} s\n")


def verify_parallel(workers: int):
    jobs = [("Development10_TrainSize5", sp) for sp in load_plan("Development10_TrainSize5")[:3]]
    _init()
    single = [compute(j) for j in jobs]
    with mp.get_context("spawn").Pool(max(2, workers), initializer=_init) as pool:
        par = pool.map(compute, jobs, chunksize=1)
    same = json.dumps(single) == json.dumps(par)
    print("parallel result identical to single result:", same)
    if not same:
        sys.exit("[stop] parallel and single results differ")


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def load_set(set_name: str):
    src = SHARED.get(set_name, set_name)
    plan = load_plan(src)
    parts = [jload(part_path(src, sp["split_id"])) for sp in plan if part_path(src, sp["split_id"]).exists()]
    if len(parts) != len(plan):
        sys.exit(f"[stop] {set_name}: {len(parts)} of {len(plan)} splits are finished (run is not complete)")
    rec = pd.DataFrame([r for p in parts for r in p["records"]])[["split_id", "subject", "model", "section", "metric", "value"]]
    rec["test_cohort"] = rec.subject.map(cohort_of)
    return rec, pd.DataFrame([p["selection"] for p in parts]), plan


def compare(rec: pd.DataFrame, cohorts: bool) -> pd.DataFrame:
    out = []
    groups = [("all test subjects", rec)] + ([(c, rec[rec.test_cohort == c]) for c in ("Development cohort", "External cohort")] if cohorts else [])
    for label, g in groups:
        c = comparison_table(g, COMPARISONS, ["general", "repeatability"], METRICS)
        c.insert(0, "test_subjects", label)
        out.append(c)
    return pd.concat(out, ignore_index=True)


def split_level_differences(rec: pd.DataFrame) -> pd.DataFrame:
    """One value per split: mean over the test subjects of the split of (first model - second model)."""
    w = rec[rec.metric.isin(["a_rmse", "peak_rmse"])].pivot_table(index=["split_id", "subject", "metric"], columns="model", values="value")
    rows = []
    for a, b in COMPARISONS:
        d = (w[a] - w[b]).groupby(level=["split_id", "metric"]).mean().reset_index(name="difference")
        d.insert(0, "comparison", f"{a} minus {b}")
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def check_against_stored(recs: dict) -> dict:
    """TrainSize9 and TrainSize15 use the same splits as the stored leave-one-subject-out results: deterministic models must agree."""
    out = {}
    for set_name, path, models in (("Development10_TrainSize9", RES / "final_comparison/Development10_LeaveOneSubjectOut/records.csv", ["HbI-binned mean", "Global Linear", "PLS_v2", "Ridge"]),
                                   ("Union16_TrainSize15", RES / "union16/Union16_LeaveOneSubjectOut/records.csv", ["HbI-binned mean", "Global Linear"])):
        stored = pd.read_csv(path, encoding="utf-8-sig")
        new = recs[set_name]
        for m in models:
            for met in ("a_rmse", "peak_rmse"):
                a = new[(new.model == m) & (new.metric == met) & (new.section == "general")].set_index("subject").value.sort_index()
                b = stored[(stored.model == m) & (stored.metric == met) & (stored.section == "general")].set_index("subject").value.sort_index()
                ok = bool(len(a) == len(b) and np.allclose(a.values, b.values, rtol=1e-9, atol=1e-15))
                out[f"{set_name} | {m} | {met}"] = dict(equal=ok, mean_new=float(a.mean()), mean_stored=float(b.mean()))
    bad = [k for k, v in out.items() if not v["equal"]]
    if bad:
        sys.exit(f"[stop] repeated-experiment results differ from the stored leave-one-subject-out results: {bad}")
    return out


def aggregate() -> dict:
    f = dict(index=False, encoding="utf-8-sig", float_format="%.10g")
    recs, sels, plans, cmps = {}, {}, {}, {}
    for s in ALL_SETS:
        recs[s], sels[s], plans[s] = load_set(s)
    checks = check_against_stored(recs)
    commit = code_commit()
    for s in ALL_SETS:
        out = OUT / s
        out.mkdir(parents=True, exist_ok=True)
        rec, cohorts = recs[s], s.startswith("Union16")
        rec.to_csv(out / "records.csv", **f)
        sv = rec.groupby(["subject", "model", "metric"]).value.agg(value="mean", n_splits_as_test="count").reset_index()
        sv.to_csv(out / "summary_by_subject.csv", **f)
        cmps[s] = compare(rec, cohorts)
        cmps[s].to_csv(out / "comparisons.csv", **f)
        sels[s].to_csv(out / "selection.csv", **f)
        man = jload(SPLITS / s / "manifest.json")
        jdump(out / "config.json", dict(
            preregistered=False, computed_after_external_unblinding=True, status=STATUS, code_commit=commit, split_set=s, n_splits=len(plans[s]),
            n_train=man["n_train"], n_test=man["n_test"], split_selection=man["selection"], split_plan_sha256=man["split_plan_sha256"],
            results_shared_with=SHARED.get(s), models=MODELS, bigru_exception=INTENDED_EXCEPTION,
            subject_value="mean of the metric over all splits in which the subject is a test subject; BiGRU: mean over seeds 0, 1, 2 first",
            paired="difference = first model minus second model; n_improved = subjects with difference < 0; p values are descriptive, without correction "
                   "for multiple comparisons", split_overlap_note="Splits share subjects, so the standard deviation between splits is not the error of independent samples.",
            hbi_binned_mean_missing_splits=int((~sels[s].hbi_binned_mean_fitted).sum()), externaltest6_configuration_used=False))
    # learning curve: subject mean and paired difference from Global Linear by training size
    rows = []
    for data_name, sets in LEARNING.items():
        for s, n_train in sets:
            rec = recs[s]
            groups = [("all test subjects", rec)] + ([(c, rec[rec.test_cohort == c]) for c in ("Development cohort", "External cohort")] if data_name == "Union16" else [])
            for label, g in groups:
                sv = subject_values(g)
                for met in ("a_rmse", "peak_rmse"):
                    sec = "general"
                    gl = sv.loc[(sec, met, "Global Linear")].to_dict()
                    for m in MODELS:
                        if m not in sv.loc[(sec, met)].index.get_level_values(0):
                            continue
                        v = sv.loc[(sec, met, m)]
                        p = paired_diff(v.to_dict(), gl) if m != "Global Linear" else None
                        rows.append(dict(data=data_name, split_set=s, n_train=n_train, n_splits=len(plans[s]), test_subjects=label, model=m, metric=met,
                                         mean=float(v.mean()), sd=float(v.std(ddof=1)), n_subjects=len(v),
                                         diff_from_global_linear_mean=p["diff_mean"] if p else 0.0, diff_se=p["diff_se"] if p else 0.0,
                                         n_improved=p["n_improved"] if p else None, wilcoxon_p=p["wilcoxon_p"] if p else None, sign_p=p["sign_p"] if p else None))
    lc = pd.DataFrame(rows)
    lc.to_csv(OUT / "learning_curve.csv", **f)
    # split sampling variability
    rows, dist = [], {}
    for s in ALL_SETS:
        d = split_level_differences(recs[s])
        dist[s] = d
        for (cmp_, met), g in d.groupby(["comparison", "metric"], sort=False):
            rows.append(dict(split_set=s, comparison=cmp_, metric=met, n_splits=len(g), mean_of_split_means=g.difference.mean(), sd_between_splits=g.difference.std(ddof=1),
                             min=g.difference.min(), max=g.difference.max()))
    var = pd.DataFrame(rows)
    # position of the 10 existing splits inside the 120 exhaustive splits, and the spread of the mean of 10 randomly drawn splits
    s120 = "Development10_Holdout73_All120"
    old = {tuple(sorted(t.split())) for t in pd.read_csv(ROOT / "data/splits_final/Development10_Holdout73/split_plan.csv").test}
    old_ids = [sp["split_id"] for sp in plans[s120] if tuple(sorted(sp["test"])) in old]
    assert len(old_ids) == 10
    rng = np.random.default_rng(RESAMPLE_SEED)
    draws = np.array([rng.choice(120, size=10, replace=False) for _ in range(N_RESAMPLE)])
    extra = []
    for (cmp_, met), g in dist[s120].groupby(["comparison", "metric"], sort=False):
        g = g.set_index("split_id").difference.reindex([sp["split_id"] for sp in plans[s120]])
        means = g.to_numpy()[draws].mean(axis=1)
        ex = float(g.loc[old_ids].mean())
        extra.append(dict(split_set=s120, comparison=cmp_, metric=met, existing_10_splits_mean=ex,
                          existing_10_percentile_among_120_split_values=float((g.to_numpy() <= ex).mean() * 100),
                          existing_10_percentile_among_means_of_10_random_splits=float((means <= ex).mean() * 100),
                          random_10_mean_p2_5=float(np.percentile(means, 2.5)), random_10_mean_p97_5=float(np.percentile(means, 97.5)),
                          n_resamples=N_RESAMPLE, resample_seed=RESAMPLE_SEED))
    var = var.merge(pd.DataFrame(extra), on=["split_set", "comparison", "metric"], how="left")
    var.to_csv(OUT / "split_sampling_variability.csv", **f)
    jdump(OUT / "checks.json", dict(status=STATUS, agreement_with_stored_leave_one_subject_out=checks, existing_10_split_ids_in_all120=old_ids))
    return dict(recs=recs, sels=sels, plans=plans, cmps=cmps, lc=lc, var=var, dist=dist, old_ids=old_ids, checks=checks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--verify-parallel", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.verify_parallel:
        verify_parallel(a.workers)
    if a.run:
        run(a.workers)
    if a.aggregate:
        res = aggregate()
        import repeat_report
        repeat_report.write(res)


if __name__ == "__main__":
    main()
