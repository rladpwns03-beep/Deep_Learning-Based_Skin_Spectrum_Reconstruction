"""Checks for the repeated experiments (post hoc, exploratory).

Run (project root):  python -X utf8 -m unittest tests.test_repeat -v

Split checks need data/splits_repeat; result checks need results/repeat_experiments (skipped when absent).
"""
from __future__ import annotations

import itertools
import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))

from final_utils import UNION_CSV, load_splits, load_spectra, test_pairs       # noqa: E402
from posthoc_common import FORBIDDEN_WORDS, SNAPSHOT_NAME, changed_existing_results, pairs_for   # noqa: E402

torch.set_num_threads(1)
SPLITS = ROOT / "data" / "splits_repeat"
FINAL = ROOT / "data" / "splits_final"
OUT = ROOT / "results" / "repeat_experiments"
HAVE_SPLITS = (SPLITS / "Union16_TrainSize5" / "split_plan.csv").exists()
HAVE = (OUT / "learning_curve.csv").exists()
EXPECT = {"Development10_Holdout73_All120": (120, 7, 3), "Development10_TrainSize9": (10, 9, 1), "Development10_TrainSize8": (45, 8, 2),
          "Development10_TrainSize7": (120, 7, 3), "Development10_TrainSize6": (120, 6, 4), "Development10_TrainSize5": (120, 5, 5),
          "Union16_Holdout73_Repeat100": (100, 11, 5), "Union16_TrainSize15": (16, 15, 1), "Union16_TrainSize13": (100, 13, 3),
          "Union16_TrainSize11": (100, 11, 5), "Union16_TrainSize9": (100, 9, 7), "Union16_TrainSize7": (100, 7, 9), "Union16_TrainSize5": (100, 5, 11)}
DEV = [f"S{k:03d}" for k in range(1, 11)]
EXT = [f"S10{k}" for k in range(1, 7)]


class TestPairs(unittest.TestCase):
    def test_pairs_equal_final_utils_test_pairs_in_all_five_split_sets(self):
        data = load_spectra(UNION_CSV)
        n = 0
        for d in sorted(FINAL.iterdir()):
            for sp in load_splits(d.name):
                a = test_pairs(d.name, sp["split_id"], data)
                b = pairs_for(data, sp["test"])
                for x, y in zip(a, b):
                    np.testing.assert_array_equal(x, y, err_msg=f"{d.name}/{sp['split_id']}")
                n += 1
        self.assertEqual(n, 10 + 10 + 10 + 16 + 1)


@unittest.skipUnless(HAVE_SPLITS, "data/splits_repeat not found")
class TestSplits(unittest.TestCase):
    def test_counts_uniqueness_and_no_overlap(self):
        self.assertEqual(sorted(p.name for p in SPLITS.iterdir()), sorted(EXPECT))
        for name, (n, n_train, n_test) in EXPECT.items():
            plan = pd.read_csv(SPLITS / name / "split_plan.csv")
            universe = set(DEV) if name.startswith("Development10") else set(DEV + EXT)
            self.assertEqual(len(plan), n, name)
            self.assertEqual(plan.test.nunique(), n, name)
            for r in plan.itertuples():
                tr, te = set(r.train.split()), set(r.test.split())
                self.assertFalse(tr & te)
                self.assertEqual(tr | te, universe)
                self.assertEqual((len(tr), len(te)), (n_train, n_test))
                self.assertFalse(tr == set(DEV) and te == set(EXT), "the ExternalTest6 configuration must not appear")
            m = json.loads((SPLITS / name / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(m["n_splits"], n)
            import hashlib
            self.assertEqual(m["split_plan_sha256"], hashlib.sha256((SPLITS / name / "split_plan.csv").read_bytes()).hexdigest())

    def test_exhaustive_sets_and_shared_set(self):
        t3 = {tuple(t.split()) for t in pd.read_csv(SPLITS / "Development10_Holdout73_All120" / "split_plan.csv").test}
        self.assertEqual(t3, set(itertools.combinations(DEV, 3)))
        a = pd.read_csv(SPLITS / "Development10_Holdout73_All120" / "split_plan.csv")
        b = pd.read_csv(SPLITS / "Development10_TrainSize7" / "split_plan.csv")
        self.assertTrue(a.equals(b))
        old = {tuple(sorted(t.split())) for t in pd.read_csv(FINAL / "Development10_Holdout73" / "split_plan.csv").test}
        self.assertLessEqual(old, t3)

    def test_union_repeat100_satisfies_stratification_rules(self):
        from common.make_splits_final import union_valid_sets
        valid = {tuple(sorted(v)) for v in union_valid_sets(load_spectra(UNION_CSV))}
        self.assertEqual(len(valid), 1167)
        plan = pd.read_csv(SPLITS / "Union16_Holdout73_Repeat100" / "split_plan.csv")
        self.assertTrue(all(tuple(sorted(t.split())) in valid for t in plan.test))
        m = json.loads((SPLITS / "Union16_Holdout73_Repeat100" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(sum(m["test_count_per_subject"].values()), 500)
        self.assertEqual(m["test_count_min"], min(m["test_count_per_subject"].values()))

    def test_sampling_is_deterministic(self):
        from common.make_splits_repeat import sample
        combos = list(itertools.combinations(DEV, 4))
        self.assertEqual(sample(combos, 120, 7), sample(combos, 120, 7))
        self.assertNotEqual(sample(combos, 120, 7), sample(combos, 120, 8))


@unittest.skipUnless(HAVE, "results/repeat_experiments not found")
class TestRepeatResults(unittest.TestCase):
    def test_agreement_with_stored_leave_one_subject_out(self):
        ref = [("Development10_TrainSize9", "Global Linear", 12.126), ("Development10_TrainSize9", "PLS_v2", 11.057), ("Union16_TrainSize15", "Global Linear", 11.934)]
        for s, model, value in ref:
            r = pd.read_csv(OUT / s / "records.csv", encoding="utf-8-sig")
            v = r[(r.model == model) & (r.section == "general") & (r.metric == "a_rmse")].groupby("subject").value.mean().mean() * 1e3
            self.assertLessEqual(abs(round(v, 3) - value), 0.001 + 1e-12, (s, model, v))
        chk = json.loads((OUT / "checks.json").read_text(encoding="utf-8"))["agreement_with_stored_leave_one_subject_out"]
        self.assertTrue(all(v["equal"] for v in chk.values()))

    def test_rerun_of_one_split_reproduces_the_stored_part(self):
        import run_repeat_experiments as R
        R._init()
        sp = R.load_plan("Development10_TrainSize5")[0]
        new = R.compute(("Development10_TrainSize5", sp))
        old = json.loads(R.part_path("Development10_TrainSize5", sp["split_id"]).read_text(encoding="utf-8"))
        self.assertEqual(json.loads(json.dumps(new)), old)

    def test_all_sets_complete_and_flagged(self):
        import run_repeat_experiments as R
        for s in R.ALL_SETS:
            cfg = json.loads((OUT / s / "config.json").read_text(encoding="utf-8"))
            self.assertIs(cfg["preregistered"], False)
            self.assertEqual(cfg["status"], "post hoc, exploratory")
            self.assertIn("intended exception", cfg["bigru_exception"])
            r = pd.read_csv(OUT / s / "records.csv", encoding="utf-8-sig")
            self.assertEqual(r.split_id.nunique(), EXPECT[s][0], s)
            self.assertEqual(len(pd.read_csv(OUT / s / "selection.csv", encoding="utf-8-sig")), EXPECT[s][0])
        v = pd.read_csv(OUT / "split_sampling_variability.csv", encoding="utf-8-sig")
        self.assertEqual(int(v[v.split_set == "Development10_Holdout73_All120"].n_resamples.iloc[0]), 10000)

    def test_existing_results_unchanged(self):
        before = json.loads((OUT / SNAPSHOT_NAME).read_text(encoding="utf-8"))
        self.assertGreater(len(before), 100)
        self.assertEqual(changed_existing_results(before), [])
        import hashlib
        for d in sorted(FINAL.iterdir()):                         # data/splits_final manifests as frozen in the preregistration
            pre = json.loads((ROOT / "docs" / "Preregistration_ExternalTest_v1.json").read_text(encoding="utf-8"))["split_manifest_sha256"]
            self.assertEqual(hashlib.sha256((d / "manifest.json").read_bytes()).hexdigest(), pre[d.name])

    def test_wording(self):
        texts = [(ROOT / "docs" / "Results_Repeat_Experiments.md").read_text(encoding="utf-8")]
        strings = json.loads((OUT / "figures" / "figure_strings.json").read_text(encoding="utf-8"))
        texts += [s for f in strings.values() for p in f.values() for s in p]
        for t in texts:
            self.assertNotIn("·", t)
            for w in FORBIDDEN_WORDS:
                self.assertIsNone(re.search(rf"\b{w}\b", t, re.IGNORECASE), f"'{w}' in: {t[:80]}")
        for f in strings.values():
            for p in f.values():
                self.assertTrue(any("Post hoc, exploratory" in s for s in p))
        from PIL import Image
        for prof, width in (("presentation", 2400), ("report", 3780)):
            for name in strings:
                with Image.open(OUT / "figures" / prof / f"{name}.jpg") as im:
                    self.assertEqual(im.size[0], width)


if __name__ == "__main__":
    unittest.main(verbosity=2)
