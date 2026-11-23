"""재학습·튜닝 실험(2026-11) 새 스크립트의 핵심 검사.

실행 (code 폴더):  python -X utf8 -m unittest discover -s tests -v

학습이 들어가는 테스트는 합성 스펙트럼만 쓴다. 분할 파일 검사는 data/splits_final 이 있을 때만 실행한다 (모델 없음).
"""
from __future__ import annotations

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

from common.data_loader import FREE, HBI_INDEX, WL                    # noqa: E402
from models.bigru import BiGRUModel                                   # noqa: E402
from models.linear import LinearModel, apply_slope, fit_slope          # noqa: E402
from test_models import QUERY, TRAIN                                  # noqa: E402  (합성 데이터)
import final_utils as U                                               # noqa: E402
import shap_analysis_bands as SB                                      # noqa: E402
import tune_bigru as TB                                               # noqa: E402

torch.set_num_threads(1)
SPLITS = ROOT / "data" / "splits_final"


@unittest.skipUnless(SPLITS.exists(), "data/splits_final 없음")
class TestSplitLeakage(unittest.TestCase):
    def test_no_subject_overlap_and_pairs_within_subject(self):
        n = 0
        for plan in sorted(SPLITS.glob("*/split_plan.csv")):
            for r in pd.read_csv(plan).itertuples():
                d = plan.parent / r.split_id
                train, test = set(r.train.split()), set(r.test.split())
                self.assertFalse(train & test, f"{d}: train/test 피험자 중복")
                tr, te = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
                self.assertEqual(set(tr.subject), train)
                self.assertEqual(set(te.subject), test)
                self.assertLessEqual(set(pd.read_csv(d / "train_pairs.csv").subject), train)
                self.assertLessEqual(set(pd.read_csv(d / "test_pairs.csv").subject), test)
                n += 1
        self.assertEqual(n, 10 + 10 + 10 + 16 + 1)

    def test_development_sets_have_no_external_subject(self):
        for name in ("Development10_Holdout73", "Development10_LeaveOneSubjectOut"):
            for r in pd.read_csv(SPLITS / name / "split_plan.csv").itertuples():
                self.assertTrue(all(s < "S100" for s in (r.train + " " + r.test).split()))

    def test_leave_one_subject_out_each_subject_once(self):
        for name, n in (("Development10_LeaveOneSubjectOut", 10), ("Union16_LeaveOneSubjectOut", 16)):
            tests = pd.read_csv(SPLITS / name / "split_plan.csv").test.tolist()
            self.assertEqual(len(tests), n)
            self.assertEqual(len(set(tests)), n)


class TestTuningLoop(unittest.TestCase):
    EPOCHS = 12

    def test_history_and_weights_equal_bigru_fit(self):
        for hidden, l2 in ((4, 0.0), (16, 1e-2)):
            m, _ = TB.train_with_checkpoints(TRAIN, hidden, l2, 0, [self.EPOCHS])
            ref = BiGRUModel(seed=0, epochs=self.EPOCHS, cfg=TB.make_cfg(hidden, l2)).fit(TRAIN)
            np.testing.assert_array_equal(np.array(m.history), np.array(ref.history))
            for a, b in zip(m.net.state_dict().values(), ref.net.state_dict().values()):
                self.assertTrue(torch.equal(a, b))

    def test_checkpoint_equals_shorter_training(self):
        snaps = TB.train_with_checkpoints(TRAIN, 8, 1e-3, 0, [0, 5, self.EPOCHS], lambda m: m.predict(QUERY.a, 0.6).copy())[1]
        for e in (5, self.EPOCHS):
            ref = BiGRUModel(seed=0, epochs=e, cfg=TB.make_cfg(8, 1e-3)).fit(TRAIN)
            np.testing.assert_array_equal(snaps[e], ref.predict(QUERY.a, 0.6))

    def test_epoch0_equals_global_linear(self):
        sl = fit_slope(TRAIN)
        ref = TB.validation_sums(lambda X, T: apply_slope(X, T, sl), QUERY)
        for hidden, l2 in ((4, 1e-1), (16, 0.0)):
            out = TB.train_with_checkpoints(TRAIN, hidden, l2, 0, [0, 2], lambda m: TB.validation_sums(m.predict, QUERY))[1]
            self.assertEqual(out[0], ref)
            self.assertNotEqual(out[2], ref)
            m0 = BiGRUModel(seed=0, epochs=0, cfg=TB.make_cfg(hidden, l2)).fit(TRAIN)
            np.testing.assert_array_equal(m0.predict(QUERY.a, 0.6), LinearModel().fit(TRAIN).predict(QUERY.a, 0.6))

    def test_selection_rule_and_tie_break(self):
        c = lambda h, l2, e, a, p: dict(hidden=h, slope_l2=l2, epoch=e, val_a_rmse=a, val_peak_rmse=p)   # noqa: E731
        cands = [c(h, l2, 0, 1.0, 0.5) for h in TB.GRID_HIDDEN for l2 in TB.GRID_SLOPE_L2]
        ch = TB.select(cands)                                # 전부 동률 -> 작은 hidden, 큰 slope_l2, 적은 epoch
        self.assertEqual((ch["hidden"], ch["slope_l2"], ch["epoch"]), (4, 0.1, 0))
        self.assertTrue(ch["epoch0_global_linear"])
        cands += [c(8, 1e-3, 100, 0.8, 0.9), c(16, 0.0, 50, 0.9, 0.4)]
        self.assertEqual(TB.select(cands)["epoch"], 100)     # absorbance RMSE 최소
        pc = TB.select_peak_constrained(cands)               # Peak RMSE <= epoch 0 값인 후보 중 최소
        self.assertEqual((pc["hidden"], pc["epoch"]), (16, 50))
        self.assertEqual(len(TB.GRID_HIDDEN) * len(TB.GRID_SLOPE_L2) * len(TB.GRID_EPOCH), 84)


class TestBandsAndGroups(unittest.TestCase):
    def test_band_wavelengths_sum_to_30(self):
        self.assertEqual({b: len(v) for b, v in U.BANDS.items()},
                         {"Soret": 5, "Band 450-460": 2, "Carotenoid": 5, "Q-band": 7, "Red": 11})
        allw = sum(U.BANDS.values(), [])
        self.assertEqual(len(allw), 30)
        self.assertEqual(sorted(allw), WL[FREE].tolist())
        self.assertNotIn(570, allw)
        self.assertNotIn("Carotenoid", U.SLOPE_BANDS)
        self.assertEqual(len(U.SLOPE_BANDS), 4)

    def test_shap_groups_cover_32_features_once(self):
        self.assertEqual(len(SB.GROUPS), 7)
        flat = sum(SB.GROUPS.values(), [])
        self.assertEqual(sorted(flat), list(range(32)))
        self.assertEqual(SB.GROUPS["h_in (570 nm)"], [HBI_INDEX])
        self.assertEqual(SB.GROUPS["h* (target)"], [31])
        bad = dict(SB.GROUPS); bad["Red"] = bad["Red"][:-1]
        with self.assertRaises(ValueError):
            SB.check_groups(bad)

    def test_shapley_matches_old_implementation_and_efficiency(self):
        m = LinearModel().fit(TRAIN)
        X = np.column_stack([QUERY.a[:4], QUERY.hbi[4:8]])
        BG = np.column_stack([TRAIN.a[:6], TRAIN.hbi[6:12]])
        ref, new = SB.old.shapley(m, X, BG), SB.shapley(m, X, BG, SB.old.GROUPS)
        for a, b in zip(ref, new):
            np.testing.assert_array_equal(a, b)
        phi, base, fx = SB.shapley(m, X, BG, SB.GROUPS)
        np.testing.assert_allclose(base[None] + phi.sum(1), fx, atol=1e-12)


class TestPairedDiff(unittest.TestCase):
    def test_sign_and_improvement(self):
        a = {"s1": 1.0, "s2": 2.0, "s3": 3.0, "s4": float("nan")}
        b = {"s1": 2.0, "s2": 2.5, "s3": 2.0, "s4": 1.0}
        p = U.paired_diff(a, b)
        self.assertEqual(p["n"], 3)                           # 값이 없는 피험자는 n에서 빠짐
        self.assertAlmostEqual(p["diff_mean"], (-1.0 - 0.5 + 1.0) / 3)
        self.assertEqual(p["n_improved"], 2)                  # 차이 < 0
        self.assertAlmostEqual(p["diff_se"], np.std([-1.0, -0.5, 1.0], ddof=1) / np.sqrt(3))

    def test_abs_judged(self):
        p = U.paired_diff({"s1": -0.1, "s2": 0.5}, {"s1": 0.3, "s2": -0.2}, lower_is_better_abs=True)
        self.assertEqual(p["n_improved"], 1)                  # |-0.1| < |0.3| 만 개선


if __name__ == "__main__":
    unittest.main(verbosity=2)
