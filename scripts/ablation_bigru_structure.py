"""BiGRU structure ablation (post hoc, exploratory; added after the ExternalTest6 results were seen; not preregistered).

Run (project root, conda env skin_spectra):
    python -X utf8 scripts/ablation_bigru_structure.py --verify-parallel --workers 4      # one split: parallel result = single result
    python -X utf8 scripts/ablation_bigru_structure.py --run --workers 14                 # resumable (finished jobs are skipped)
    python -X utf8 scripts/ablation_bigru_structure.py --aggregate                        # tables, figures, document

Fixed before running (see models/bigru_variants.py for the four variants):
  hidden 8, lr 3e-3, clip 1.0, Adam, full batch, subject-weighted MSE without 570 nm, no slope penalty (slope_l2 = 0),
  1,000 epochs, test metrics recorded at checkpoints 0, 25, 50, 100, 150, 200, 300, 500, 1000, seeds 0-4,
  evaluation sets Development10_LeaveOneSubjectOut and Union16_LeaveOneSubjectOut (splits read from data/splits_final).
  Learning-rate sensitivity (supplementary): the three variants without the Global Linear prior, lr 1e-3 and 1e-2,
  seeds 0-2, Development10_LeaveOneSubjectOut only.
  Main comparison: each variant minus BiGRU_ResidualSlope at epoch 200. Supplementary: the checkpoint with the lowest test mean
  absorbance RMSE of each variant ("oracle checkpoint (optimistic for the variant)") minus BiGRU_ResidualSlope at epoch 200.

The ExternalTest6 configuration (training = 10 Development cohort subjects, evaluation = 6 External cohort subjects) is not used.
Writes only under results/ablation_bigru_structure/ (and docs/Results_Ablation_BiGRU_Structure.md with --aggregate).
"""
from __future__ import annotations

import argparse
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
from common import config as C                                                     # noqa: E402
from common.data_loader import FREE                                                # noqa: E402
from final_utils import RES, UNION_CSV, jdump, load_splits, load_spectra, test_pairs   # noqa: E402
from final_eval import comparison_table, split_records                             # noqa: E402
from models.bigru_variants import REMOVED, STRUCTURAL_IDENTITY, VARIANTS, BiGRUVariant   # noqa: E402
from posthoc_common import STATUS, code_commit, cohort_of, snapshot_existing_results   # noqa: E402

OUT = RES / "ablation_bigru_structure"
SETS = ["Development10_LeaveOneSubjectOut", "Union16_LeaveOneSubjectOut"]
REF = "BiGRU_ResidualSlope"
OTHERS = [v for v in VARIANTS if v != REF]
EPOCHS, CHECKPOINTS = 1000, (0, 25, 50, 100, 150, 200, 300, 500, 1000)
MAIN_EPOCH, MAIN_LR, SEEDS = 200, 3e-3, (0, 1, 2, 3, 4)
LR_SENSITIVITY, LR_SEEDS, LR_SET = (1e-3, 1e-2), (0, 1, 2), "Development10_LeaveOneSubjectOut"
KEEP = {("general", "a_rmse"), ("general", "peak_rmse"), ("repeatability", "peak_sd"), ("repeatability", "spectral_sd")}
METRICS = ["a_rmse", "peak_rmse", "peak_sd", "spectral_sd"]
STORED = {"Development10_LeaveOneSubjectOut": RES / "final_comparison" / "Development10_LeaveOneSubjectOut" / "records.csv",
          "Union16_LeaveOneSubjectOut": RES / "union16" / "Union16_LeaveOneSubjectOut" / "records.csv"}
_DATA = None


def part_path(set_name, split_id, variant, lr, seed) -> Path:
    return OUT / set_name / "parts" / f"{split_id}__{variant}__lr{lr:g}__seed{seed}.csv"


def all_jobs():
    jobs = []
    for set_name in SETS[::-1]:                                   # longer jobs (15 training subjects) first
        for sp in load_splits(set_name):
            for v in VARIANTS:
                jobs += [(set_name, sp, v, MAIN_LR, s) for s in SEEDS]
    for sp in load_splits(LR_SET):
        for v in OTHERS:
            jobs += [(LR_SET, sp, v, lr, s) for lr in LR_SENSITIVITY for s in LR_SEEDS]
    return jobs


def _init():
    global _DATA
    torch.set_num_threads(1)
    _DATA = load_spectra(UNION_CSV)


def compute(job) -> pd.DataFrame:
    set_name, sp, variant, lr, seed = job
    data = _DATA
    assert not set(sp["train"]) & set(sp["test"])
    tr, te = data.select(sp["train"]), data.select(sp["test"])
    ii, jj, subj = test_pairs(set_name, sp["split_id"], data)
    X, Y, T, h_in = data.a[ii], data.a[jj], data.hbi[jj], data.hbi[ii]

    def evaluate(m):
        rows = [r for r in split_records({m.name: [m]}, te, X, Y, T, h_in, subj) if (r["section"], r["metric"]) in KEEP and r["model"] != "Raw"]
        for s in te.subjects:                                     # identity: target HbI = input HbI
            a = te.a[te.subject == s]
            d = (m.predict(a, a[:, 17]) - a)[:, FREE]
            rows.append(dict(subject=s, model=m.name, section="identity", metric="identity_rmse", value=float(np.sqrt(np.mean(d ** 2)))))
            rows.append(dict(subject=s, model=m.name, section="identity", metric="identity_max_abs", value=float(np.abs(d).max())))
        return rows

    m = BiGRUVariant(variant, seed=seed, lr=lr)
    out = m.fit(tr, EPOCHS, CHECKPOINTS, evaluate)
    assert m.n_params == 689
    frames = []
    for ep, rows in out.items():
        f = pd.DataFrame(rows).rename(columns={"model": "variant"})
        f.insert(1, "epoch", ep)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df.insert(1, "lr", lr); df.insert(2, "seed", seed); df.insert(4, "split_id", sp["split_id"])
    return df[["variant", "lr", "seed", "epoch", "split_id", "subject", "section", "metric", "value"]]


def work(job):
    p = part_path(job[0], job[1]["split_id"], *job[2:])
    if p.exists():
        return 0.0
    t = time.time()
    df = compute(job)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig", float_format="%.10g")
    tmp.replace(p)                                               # a part file appears only when it is complete
    return time.time() - t


def run(workers: int):
    snapshot_existing_results(OUT)
    jobs = [j for j in all_jobs() if not part_path(j[0], j[1]["split_id"], *j[2:]).exists()]
    print(f"jobs to run: {len(jobs)} of {len(all_jobs())}", flush=True)
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
    print(f"finished {done} jobs in {time.time() - t0:.0f}s", flush=True)
    (OUT / "run_seconds.txt").open("a", encoding="utf-8").write(f"{len(jobs)} jobs, {workers} workers, {time.time() - t0:.0f} s\n")


def verify_parallel(workers: int):
    sp = load_splits(SETS[0])[0]
    jobs = [(SETS[0], sp, v, MAIN_LR, s) for v in VARIANTS for s in (0, 1)]
    _init()
    single = [compute(j) for j in jobs]
    with mp.get_context("spawn").Pool(max(2, workers), initializer=_init) as pool:
        par = pool.map(compute, jobs, chunksize=1)
    same = all(a.equals(b) for a, b in zip(single, par))
    print("parallel result identical to single result:", same)
    if not same:
        sys.exit("[stop] parallel and single results differ")


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def load_records(set_name: str) -> pd.DataFrame:
    parts = sorted((OUT / set_name / "parts").glob("*.csv"))
    df = pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in parts], ignore_index=True)
    df["test_cohort"] = df.subject.map(cohort_of)
    return df


def expected_parts(set_name: str) -> int:
    n = len(load_splits(set_name))
    return n * len(VARIANTS) * len(SEEDS) + (n * len(OTHERS) * len(LR_SENSITIVITY) * len(LR_SEEDS) if set_name == LR_SET else 0)


def as_models(df: pd.DataFrame, name_col="variant") -> pd.DataFrame:
    return df.rename(columns={name_col: "model"})[["split_id", "subject", "model", "section", "metric", "value"]]


def compare(df: pd.DataFrame, cohorts: bool) -> pd.DataFrame:
    out = []
    groups = [("all test subjects", df)] + ([(c, df[df.test_cohort == c]) for c in ("Development cohort", "External cohort")] if cohorts else [])
    for label, g in groups:
        c = comparison_table(as_models(g), [(v, REF) for v in OTHERS], ["general", "repeatability"], METRICS)
        c.insert(0, "test_subjects", label)
        out.append(c)
    return pd.concat(out, ignore_index=True)


def aggregate_set(set_name: str) -> dict:
    out = OUT / set_name
    rec = load_records(set_name)
    n_parts = len(list((out / "parts").glob("*.csv")))
    if n_parts != expected_parts(set_name):
        sys.exit(f"[stop] {set_name}: {n_parts} part files, expected {expected_parts(set_name)} (run is not complete)")
    f = dict(index=False, encoding="utf-8-sig", float_format="%.10g")
    rec.to_csv(out / "records.csv", **f)
    main = rec[np.isclose(rec.lr, MAIN_LR)]
    cohorts = set_name.startswith("Union16")
    # 1. main comparison at epoch 200
    c200 = compare(main[main.epoch == MAIN_EPOCH], cohorts)
    c200.to_csv(out / "comparisons_epoch200.csv", **f)
    # 2. oracle checkpoint of each variant (chosen on the test mean absorbance RMSE) against the reference at epoch 200
    sm = main[(main.section == "general") & (main.metric == "a_rmse")].groupby(["variant", "epoch", "subject"]).value.mean()
    by_epoch = sm.groupby(["variant", "epoch"]).mean()
    oracle = {v: int(by_epoch.loc[v].idxmin()) for v in VARIANTS}
    sel = pd.concat([main[(main.variant == v) & (main.epoch == oracle[v])] for v in OTHERS] + [main[(main.variant == REF) & (main.epoch == MAIN_EPOCH)]])
    co = compare(sel, cohorts)
    co.insert(1, "oracle_epoch_of_variant", co.comparison.map(lambda s: oracle[s.split(" minus ")[0]]))
    co.insert(2, "note", "oracle checkpoint (optimistic for the variant)")
    co.to_csv(out / "comparisons_oracle_checkpoint.csv", **f)
    # 3. epoch curves (subject value = mean over seeds; then mean and SE across subjects), plus stored baselines
    rows = []
    sub = rec[rec.metric.isin(METRICS)].groupby(["variant", "lr", "epoch", "metric", "subject"]).value.mean()
    for (v, lr, ep, met), g in sub.groupby(level=[0, 1, 2, 3]):
        rows.append(dict(model=v, lr=lr, epoch=ep, metric=met, mean=g.mean(), sd=g.std(ddof=1), se=g.std(ddof=1) / np.sqrt(len(g)), n=len(g),
                         n_seeds=int(rec[(rec.variant == v) & np.isclose(rec.lr, lr)].seed.nunique())))
    stored = pd.read_csv(STORED[set_name], encoding="utf-8-sig")
    for m in ("Global Linear", "PLS_v2"):
        for met in METRICS:
            g = stored[(stored.model == m) & (stored.metric == met) & stored.section.isin(["general", "repeatability"])].groupby("subject").value.mean()
            rows.append(dict(model=f"{m} (stored result)", lr=np.nan, epoch=np.nan, metric=met, mean=g.mean(), sd=g.std(ddof=1), se=g.std(ddof=1) / np.sqrt(len(g)),
                             n=len(g), n_seeds=np.nan))
    curves = pd.DataFrame(rows)
    curves.to_csv(out / "curves.csv", **f)
    # 4. identity error (target HbI = input HbI): mean over subjects of the seed-mean RMSE, and the largest absolute difference
    idn = rec[rec.section == "identity"]
    ident = idn.groupby(["variant", "lr", "epoch", "metric"]).value.agg(["mean", "max"]).reset_index()
    ident["structural_identity"] = ident.variant.map(STRUCTURAL_IDENTITY)
    ident.to_csv(out / "identity_error.csv", **f)
    # 5. seed variability of the evaluation-set mean absorbance RMSE
    per_seed = rec[(rec.section == "general") & (rec.metric == "a_rmse")].groupby(["variant", "lr", "epoch", "seed"]).value.mean()
    sv = per_seed.groupby(level=[0, 1, 2]).agg(mean_over_seeds="mean", sd_between_seeds=lambda x: x.std(ddof=1), n_seeds="count").reset_index()
    sv.to_csv(out / "seed_variability.csv", **f)
    # learning-rate sensitivity (seeds 0-2 for every learning rate so that the comparison uses the same seeds)
    lrs = None
    if set_name == LR_SET:
        g = rec[rec.seed.isin(LR_SEEDS) & rec.variant.isin(OTHERS) & (rec.section == "general")]
        s3 = g.groupby(["variant", "lr", "epoch", "metric", "subject"]).value.mean().groupby(level=[0, 1, 2, 3]).mean().unstack("metric").reset_index()
        rows = []
        for (v, lr), d in s3.groupby(["variant", "lr"]):
            at200, best = d[d.epoch == MAIN_EPOCH].iloc[0], d.loc[d.a_rmse.idxmin()]
            rows.append(dict(variant=v, lr=lr, a_rmse_epoch200=at200.a_rmse, peak_rmse_epoch200=at200.peak_rmse, oracle_epoch=int(best.epoch),
                             a_rmse_oracle_epoch=best.a_rmse, peak_rmse_oracle_epoch=best.peak_rmse))
        ref3 = rec[rec.seed.isin(LR_SEEDS) & (rec.variant == REF) & (rec.section == "general") & (rec.epoch == MAIN_EPOCH)]
        r = ref3.groupby(["metric", "subject"]).value.mean().groupby(level=0).mean()
        rows.append(dict(variant=REF, lr=MAIN_LR, a_rmse_epoch200=r["a_rmse"], peak_rmse_epoch200=r["peak_rmse"], oracle_epoch=np.nan,
                         a_rmse_oracle_epoch=np.nan, peak_rmse_oracle_epoch=np.nan))
        lrs = pd.DataFrame(rows)
        lrs.to_csv(out / "lr_sensitivity.csv", **f)
    jdump(out / "config.json", dict(
        preregistered=False, computed_after_external_unblinding=True, status=STATUS, code_commit=code_commit(), evaluation_set=set_name,
        variants={v: dict(removed=REMOVED[v], structural_identity=STRUCTURAL_IDENTITY[v]) for v in VARIANTS}, n_parameters=689,
        training=dict(hidden=8, lr=MAIN_LR, clip=1.0, optimizer="Adam", batch="full", loss="subject-weighted MSE over the 30 wavelengths other than 570 nm",
                      slope_l2=0.0, epochs=EPOCHS, checkpoints=list(CHECKPOINTS), seeds=list(SEEDS)),
        lr_sensitivity=dict(variants=OTHERS, lr=list(LR_SENSITIVITY), seeds=list(LR_SEEDS), evaluation_set=LR_SET),
        main_comparison=f"each variant minus {REF} at epoch {MAIN_EPOCH}", oracle_epoch=oracle,
        subject_value="mean of the metric over seeds", paired="difference = first model minus second model; n_improved = subjects with difference < 0; "
                                                             "p values are descriptive, without correction for multiple comparisons",
        externaltest6_configuration_used=False, n_part_files=n_parts))
    return dict(rec=rec, c200=c200, co=co, curves=curves, ident=ident, sv=sv, lrs=lrs, oracle=oracle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--verify-parallel", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(1)
    assert C.TARGET_HBI == 0.6
    if a.verify_parallel:
        verify_parallel(a.workers)
    if a.run:
        run(a.workers)
    if a.aggregate:
        res = {s: aggregate_set(s) for s in SETS}
        import ablation_report
        ablation_report.write(res)


if __name__ == "__main__":
    main()
