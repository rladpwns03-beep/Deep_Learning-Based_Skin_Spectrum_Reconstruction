"""Checks for the BiGRU structure ablation (post hoc, exploratory).

Run (project root):  python -X utf8 -m unittest tests.test_ablation -v

Model checks use synthetic spectra. Checks that read results/ablation_bigru_structure are skipped when that folder is absent.
"""
from __future__ import annotations

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
sys.path.insert(2, str(ROOT / "tests"))

from common.data_loader import FREE, HBI_INDEX                         # noqa: E402
from models.bigru import BiGRUModel                                    # noqa: E402
from models.bigru_variants import VARIANTS, BiGRUVariant               # noqa: E402
from models.linear import LinearModel                                  # noqa: E402
from posthoc_common import FORBIDDEN_WORDS, SNAPSHOT_NAME, changed_existing_results   # noqa: E402
from test_models import QUERY, TRAIN                                   # noqa: E402  (synthetic data)

torch.set_num_threads(1)
OUT = ROOT / "results" / "ablation_bigru_structure"
HAVE = (OUT / "Union16_LeaveOneSubjectOut" / "records.csv").exists()
EPOCHS = 12


def fitted(name, seed=0, epochs=EPOCHS, lr=3e-3):
    m = BiGRUVariant(name, seed=seed, lr=lr)
    m.fit(TRAIN, epochs)
    return m


class TestVariants(unittest.TestCase):
    def test_1_residual_slope_equals_existing_bigru(self):
        for seed in (0, 3):
            ref = BiGRUModel(seed=seed, epochs=EPOCHS, cfg=dict(hidden=8, lr=3e-3, slope_l2=0.0, clip=1.0)).fit(TRAIN)
            new = fitted("BiGRU_ResidualSlope", seed)
            for t in (0.6, QUERY.hbi[::-1]):
                self.assertLessEqual(float(np.abs(new.predict(QUERY.a, t) - ref.predict(QUERY.a, t)).max()), 1e-6)

    def test_3_parameter_count(self):
        for name in VARIANTS:
            self.assertEqual(fitted(name, epochs=0).n_params, 689, name)

    def test_4_target_hbi_fixed_at_570(self):
        for name in VARIANTS:
            m = fitted(name)
            for t in (0.45, 0.6, np.linspace(0.45, 0.7, len(QUERY))):
                np.testing.assert_array_equal(m.predict(QUERY.a, t)[:, HBI_INDEX], np.broadcast_to(t, (len(QUERY),)))
            out, _ = m.net(torch.tensor(QUERY.a, dtype=torch.float32), torch.full((len(QUERY),), 0.6))      # also inside the training forward pass
            np.testing.assert_allclose(out[:, HBI_INDEX].detach().numpy(), 0.6, atol=1e-7)

    def test_5_epoch0_prediction(self):
        t = 0.6
        gl = LinearModel().fit(TRAIN).predict(QUERY.a, t)
        inp = QUERY.a.copy(); inp[:, HBI_INDEX] = t
        mu = np.tile(TRAIN.a.mean(0), (len(QUERY), 1)); mu[:, HBI_INDEX] = t
        expect = {"BiGRU_ResidualSlope": gl, "BiGRU_FullSlope": inp, "BiGRU_AdditiveCorrection": inp, "BiGRU_DirectOutput": mu}
        for name, e in expect.items():
            snap = BiGRUVariant(name, seed=0).fit(TRAIN, 2, [0], lambda m: m.predict(QUERY.a, t).copy())
            np.testing.assert_array_equal(snap[0], e, err_msg=name)

    def test_6_identity(self):
        for name in ("BiGRU_ResidualSlope", "BiGRU_FullSlope"):
            np.testing.assert_array_equal(fitted(name).predict(QUERY.a, QUERY.hbi), QUERY.a, err_msg=name)
        for name in ("BiGRU_AdditiveCorrection", "BiGRU_DirectOutput"):       # not guaranteed by structure: measured, not asserted to be zero
            d = (fitted(name).predict(QUERY.a, QUERY.hbi) - QUERY.a)[:, FREE]
            self.assertTrue(np.isfinite(d).all())
            self.assertGreater(float(np.abs(d).max()), 0.0)

    def test_7_same_setting_twice(self):
        for name in VARIANTS:
            a, b = fitted(name, 1), fitted(name, 1)
            np.testing.assert_array_equal(a.predict(QUERY.a, 0.6), b.predict(QUERY.a, 0.6))
            self.assertEqual(a.history, b.history)
        self.assertFalse(np.array_equal(fitted("BiGRU_FullSlope", 0).predict(QUERY.a, 0.6), fitted("BiGRU_FullSlope", 1).predict(QUERY.a, 0.6)))

    def test_same_seed_gives_same_initial_gru_weights_in_all_variants(self):
        for seed in (0, 1, 4):
            nets = {name: fitted(name, seed, epochs=0).net for name in VARIANTS}
            ref = BiGRUModel(seed=seed, epochs=0, cfg=dict(hidden=8, lr=3e-3, slope_l2=0.0, clip=1.0)).fit(TRAIN).net
            base = {k: v for k, v in ref.gru.state_dict().items()}
            for name, net in nets.items():
                for k, v in net.gru.state_dict().items():
                    self.assertTrue(torch.equal(v, base[k]), f"seed {seed}, {name}, {k}")
                self.assertTrue(bool((net.head.weight == 0).all() and (net.head.bias == 0).all()))


@unittest.skipUnless(HAVE, "results/ablation_bigru_structure not found")
class TestAblationResults(unittest.TestCase):
    def test_2_reference_variant_reproduces_stored_bigru(self):
        ref = {"Union16_LeaveOneSubjectOut": 11.361, "Development10_LeaveOneSubjectOut": 11.141}
        stored = {"Union16_LeaveOneSubjectOut": ROOT / "results/union16/Union16_LeaveOneSubjectOut/records.csv",
                  "Development10_LeaveOneSubjectOut": ROOT / "results/union16/Development10_LeaveOneSubjectOut_FixedHyperparameters/records.csv"}
        for s, value in ref.items():
            r = pd.read_csv(OUT / s / "records.csv", encoding="utf-8-sig")
            g = r[(r.variant == "BiGRU_ResidualSlope") & np.isclose(r.lr, 3e-3) & (r.epoch == 200) & r.seed.isin([0, 1, 2]) & (r.metric == "a_rmse")]
            sub = g.groupby("subject").value.mean()
            self.assertLessEqual(abs(round(sub.mean() * 1e3, 3) - value), 0.001 + 1e-12, s)
            old = pd.read_csv(stored[s], encoding="utf-8-sig")
            old = old[(old.model == "BiGRU") & (old.section == "general") & (old.metric == "a_rmse")].set_index("subject").value
            np.testing.assert_allclose(sub.sort_index().values, old.sort_index().values, rtol=1e-6)

    def test_run_is_complete_and_flagged(self):
        import ablation_bigru_structure as A
        for s in A.SETS:
            self.assertEqual(len(list((OUT / s / "parts").glob("*.csv"))), A.expected_parts(s))
            cfg = json.loads((OUT / s / "config.json").read_text(encoding="utf-8"))
            self.assertIs(cfg["preregistered"], False)
            self.assertEqual(cfg["status"], "post hoc, exploratory")
            self.assertIs(cfg["externaltest6_configuration_used"], False)
            r = pd.read_csv(OUT / s / "records.csv", encoding="utf-8-sig")
            self.assertEqual(sorted(r.epoch.unique()), [0, 25, 50, 100, 150, 200, 300, 500, 1000])
            idn = pd.read_csv(OUT / s / "identity_error.csv", encoding="utf-8-sig")
            for v in ("BiGRU_ResidualSlope", "BiGRU_FullSlope"):
                self.assertEqual(float(idn[idn.variant == v]["max"].max()), 0.0, v)

    def test_existing_results_unchanged(self):
        before = json.loads((OUT / SNAPSHOT_NAME).read_text(encoding="utf-8"))
        self.assertGreater(len(before), 100)
        self.assertEqual(changed_existing_results(before), [])

    def test_wording(self):
        texts = [(ROOT / "docs" / "Results_Ablation_BiGRU_Structure.md").read_text(encoding="utf-8")]
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
