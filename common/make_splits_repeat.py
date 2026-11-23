"""Split sets for the repeated experiments (post hoc, exploratory). Writes only data/splits_repeat/; data/splits_final is read only.

Run (project root):  python -X utf8 -m common.make_splits_repeat        # stops if data/splits_repeat already exists

More repetitions of the 7:3 type splits
  Development10_Holdout73_All120   every choice of 3 test subjects out of the 10 Development cohort subjects (120, exhaustive)
  Union16_Holdout73_Repeat100      100 test sets of 5 drawn uniformly without replacement (fixed seed) from the test sets that satisfy
                                   the stratification rules of Union16_Holdout73 (common.make_splits_final.union_valid_sets).
                                   union_split_plan itself is specific to 10 splits (total of Near-target subjects fixed to 16), so it is not used.
Training-size sets (test subjects = all subjects of the cohort data set that are not in training)
  Development10_TrainSize9, 8      exhaustive (10, 45).  Development10_TrainSize7 = the same 120 splits as All120 (results are shared).
  Development10_TrainSize6, 5      120 of 210 / 252, uniform without replacement, fixed seed.
  Union16_TrainSize15              exhaustive (16).  Union16_TrainSize13, 11, 9, 7, 5: 100 each, no stratification, fixed seed.

Each set folder: split_plan.csv (split_id, train, test, n_train_development, n_train_external) and manifest.json.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.data_loader import ROOT, load_spectra
from common.make_splits_final import UNION_CSV, cohort, union_valid_sets

OUT = ROOT / "data" / "splits_repeat"
FINAL = ROOT / "data" / "splits_final"
BASE_SEED = 20261123
N_DEV_SAMPLE, N_UNION_SAMPLE = 120, 100


def sample(combos: list, n: int, seed: int) -> list:
    """n combinations drawn uniformly without replacement; returned in the (lexicographic) order of the full list."""
    idx = np.sort(np.random.default_rng(seed).choice(len(combos), size=n, replace=False))
    return [combos[i] for i in idx]


def write_set(name: str, universe: list, tests: list, how: str, seed, n_candidates: int, extra: dict | None = None) -> dict:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=False)
    rows = []
    for k, t in enumerate(tests):
        t = sorted(t)
        train = [s for s in universe if s not in t]
        assert not set(train) & set(t) and len(train) + len(t) == len(universe)
        rows.append(dict(split_id=f"split_{k:03d}", train=" ".join(train), test=" ".join(t),
                         n_train_development=sum(cohort(s) == "development" for s in train), n_train_external=sum(cohort(s) == "external" for s in train)))
    plan = pd.DataFrame(rows)
    assert plan.test.nunique() == len(plan), "duplicate test sets"
    plan.to_csv(d / "split_plan.csv", index=False, encoding="utf-8")
    counts = pd.Series(" ".join(plan.test).split()).value_counts().reindex(universe, fill_value=0)
    m = dict(split_set=name, status="post hoc, exploratory", generator="common/make_splits_repeat.py", n_splits=len(plan), n_train=len(universe) - len(tests[0]),
             n_test=len(tests[0]), selection=how, seed=seed, n_candidate_test_sets=n_candidates,
             test_count_per_subject={s: int(v) for s, v in counts.items()}, test_count_min=int(counts.min()), test_count_max=int(counts.max()),
             train_cohort_composition=plan.groupby(["n_train_development", "n_train_external"]).size().reset_index(name="n_splits").to_dict(orient="records"),
             split_plan_sha256=hashlib.sha256((d / "split_plan.csv").read_bytes()).hexdigest(), **(extra or {}))
    (d / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    return m


def existing_tests(name: str) -> set:
    return {tuple(sorted(t.split())) for t in pd.read_csv(FINAL / name / "split_plan.csv").test}


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"[stop] {OUT} already exists. Nothing is overwritten.")
    data = load_spectra(UNION_CSV)
    subs = data.subjects
    dev = [s for s in subs if cohort(s) == "development"]
    assert len(dev) == 10 and len(subs) == 16
    out, k = {}, 0

    def seed():
        nonlocal k
        k += 1
        return BASE_SEED + k

    def included(tests, name):
        ex = existing_tests(name)
        have = {tuple(sorted(t)) for t in tests}
        return dict(existing_split_set=name, n_existing=len(ex), n_existing_included=len(ex & have), all_existing_included=ex <= have)

    all3 = list(itertools.combinations(dev, 3))
    out["Development10_Holdout73_All120"] = write_set("Development10_Holdout73_All120", dev, all3, "exhaustive", None, len(all3),
                                                      dict(existing_splits=included(all3, "Development10_Holdout73")))
    for n_test in (1, 2, 3, 4, 5):
        combos = list(itertools.combinations(dev, n_test))
        name = f"Development10_TrainSize{10 - n_test}"
        if n_test <= 3:
            extra = dict(shares_results_with="Development10_Holdout73_All120") if n_test == 3 else \
                (dict(existing_splits=included(combos, "Development10_LeaveOneSubjectOut")) if n_test == 1 else None)
            out[name] = write_set(name, dev, combos, "exhaustive", None, len(combos), extra)
        else:
            sd = seed()
            out[name] = write_set(name, dev, sample(combos, N_DEV_SAMPLE, sd), "uniform without replacement", sd, len(combos))
    valid = union_valid_sets(data)
    sd = seed()
    rep = sample(valid, N_UNION_SAMPLE, sd)
    out["Union16_Holdout73_Repeat100"] = write_set(
        "Union16_Holdout73_Repeat100", subs, rep, "uniform without replacement from the test sets valid under the Union16_Holdout73 stratification rules "
        "(3 Development cohort + 2 External cohort subjects, 1 or 2 Near-target subjects, 1 to 3 In-range subjects)", sd, len(valid),
        dict(existing_splits=included(rep, "Union16_Holdout73"), union_split_plan_used=False,
             reason="union_split_plan fixes the total number of Near-target subjects to 16, which is specific to 10 splits"))
    for n_test in (1, 3, 5, 7, 9, 11):
        combos = list(itertools.combinations(subs, n_test))
        name = f"Union16_TrainSize{16 - n_test}"
        if n_test == 1:
            out[name] = write_set(name, subs, combos, "exhaustive", None, len(combos), dict(existing_splits=included(combos, "Union16_LeaveOneSubjectOut")))
        else:
            sd = seed()
            out[name] = write_set(name, subs, sample(combos, N_UNION_SAMPLE, sd), "uniform without replacement, no stratification", sd, len(combos))
    for name, m in out.items():
        print(f"{name}: {m['n_splits']} splits, train {m['n_train']} / test {m['n_test']}, {m['selection'][:40]}, test count per subject {m['test_count_min']}-{m['test_count_max']}")


if __name__ == "__main__":
    main()
