"""Phase 1 단위 테스트: B1, B2, BiGRU 구현과 공통 인터페이스.

실행 (code 폴더, torch가 있는 환경):  python -m unittest tests.test_models -v

실제 측정 데이터로 학습·평가하지 않는다. 모든 테스트는 합성(synthetic) 스펙트럼만 사용한다.
기존 구현과의 일치는 Model_Ablation/ablation.py 를 같은 합성 데이터에 적용해 비교한다.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "Model_Ablation"))

import ablation as legacy                                          # noqa: E402  기존 구현 (참고용, 수정하지 않음)
from common.data_loader import HBI_INDEX, WL, Spectra, make_pairs  # noqa: E402
from models.bigru import BiGRUModel, _objective                     # noqa: E402
from models.hb_index import BIN_EDGES, BinnedMeanModel, bin_index   # noqa: E402
from models.linear import LinearModel                               # noqa: E402

torch.set_num_threads(1)


def synthetic(n_subj=4, n_trial=8, seed=0) -> Spectra:
    """피험자별 기저 스펙트럼 + HbI에 따른 (약간 비선형) 변화 + 잡음. 570 nm = HbI."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, len(WL))
    slope = 0.6 * np.exp(-((x - 0.55) / 0.12) ** 2) + 0.3 * np.exp(-((x - 0.05) / 0.08) ** 2)
    A, S, T = [], [], []
    for s in range(n_subj):
        base = 0.35 + 0.25 * np.exp(-x * 3) + rng.normal(0, 0.02, len(WL))
        h = np.sort(rng.uniform(0.44, 0.71, n_trial))
        a = base + np.outer(h - 0.55, slope) + 0.3 * np.outer((h - 0.55) ** 2, slope) \
            + rng.normal(0, 0.003, (n_trial, len(WL)))
        a[:, HBI_INDEX] = h
        A.append(a); S += [f"X{s}"] * n_trial; T += list(range(1, n_trial + 1))
    return Spectra(np.vstack(A), np.array(S), np.array(T))


TRAIN = synthetic()
QUERY = synthetic(n_subj=2, seed=7)           # 학습에 쓰지 않은 합성 피험자


class CommonInterface:
    """세 모델이 모두 통과해야 하는 공통 조건."""
    def make(self):
        raise NotImplementedError

    def setUp(self):
        self.m = self.make()

    def test_batch_shape(self):
        out = self.m.predict(QUERY.a, 0.6)
        self.assertEqual(out.shape, QUERY.a.shape)

    def test_single_shape(self):
        out = self.m.predict(QUERY.a[0], 0.6)
        self.assertEqual(out.shape, (len(WL),))
        np.testing.assert_array_equal(out, self.m.predict(QUERY.a[:1], 0.6)[0])

    def test_per_row_target(self):
        t = np.linspace(0.45, 0.70, len(QUERY))
        out = self.m.predict(QUERY.a, t)
        np.testing.assert_array_equal(out[:, HBI_INDEX], t)

    def test_target_constraint(self):
        for t in (0.45, 0.6, 0.6123, 0.72):
            np.testing.assert_array_equal(self.m.predict(QUERY.a, t)[:, HBI_INDEX], t)

    def test_identity(self):
        np.testing.assert_array_equal(self.m.predict(QUERY.a, QUERY.hbi), QUERY.a)

    def test_non_contiguous_input(self):
        # 역순·건너뛰기 슬라이스(음수/비연속 stride) 입력도 같은 결과
        np.testing.assert_array_equal(self.m.predict(QUERY.a[::-1], 0.6), self.m.predict(QUERY.a[::-1].copy(), 0.6))
        np.testing.assert_array_equal(self.m.predict(QUERY.a[::2], 0.6), self.m.predict(QUERY.a[::2].copy(), 0.6))

    def test_finite(self):
        self.assertTrue(np.isfinite(self.m.predict(QUERY.a, 0.6)).all())

    def test_rejects_nan_inf_and_bad_shape(self):
        bad = QUERY.a.copy(); bad[0, 3] = np.nan
        with self.assertRaises(ValueError):
            self.m.predict(bad, 0.6)
        with self.assertRaises(ValueError):
            self.m.predict(QUERY.a, np.inf)
        with self.assertRaises(ValueError):
            self.m.predict(QUERY.a[:, :30], 0.6)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.bin"
            self.m.save(p)
            m2 = type(self.m).load(p)
        np.testing.assert_array_equal(m2.predict(QUERY.a, 0.6), self.m.predict(QUERY.a, 0.6))
        self.assertEqual(m2.train_subjects, TRAIN.subjects)


class TestB2(CommonInterface, unittest.TestCase):
    def make(self):
        return LinearModel().fit(TRAIN)

    def test_name(self):
        self.assertEqual(self.m.name, "B2")

    def test_matches_legacy(self):
        pre = legacy.fit_pre(TRAIN.a, TRAIN.subject)
        np.testing.assert_array_equal(self.m.slope, pre["b1"])
        for t in (0.6, QUERY.hbi[::-1]):
            np.testing.assert_array_equal(self.m.predict(QUERY.a, t), legacy.b1_pred(QUERY.a, t, pre))

    def test_formula(self):
        t = 0.6
        exp = QUERY.a + (t - QUERY.hbi)[:, None] * self.m.slope
        exp[:, HBI_INDEX] = t
        np.testing.assert_allclose(self.m.predict(QUERY.a, t), exp, rtol=0, atol=1e-15)
        self.assertEqual(self.m.slope[HBI_INDEX], 1.0)


class TestB1(CommonInterface, unittest.TestCase):
    def make(self):
        return BinnedMeanModel().fit(TRAIN)

    def test_name(self):
        self.assertEqual(self.m.name, "B1")

    def test_bin_edges_fixed(self):
        np.testing.assert_allclose(BIN_EDGES, [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75])

    def test_bin_boundaries(self):
        b, out = bin_index([0.40, 0.4499, 0.45, 0.5999, 0.60, 0.7499, 0.75, 0.39])
        np.testing.assert_array_equal(b, [0, 0, 1, 3, 4, 6, 6, 0])
        np.testing.assert_array_equal(out, [0, 0, 0, 0, 0, 0, 1, 1])

    def test_formula_and_bin_means_from_train_only(self):
        bi, _ = bin_index(TRAIN.hbi)
        x, t = QUERY.a[:1], 0.62
        ki, kt = bin_index(x[:, HBI_INDEX])[0][0], bin_index(t)[0]
        mean = lambda k: TRAIN.a[bi == self.m.source[k]].mean(0)
        exp = x + mean(kt) - mean(ki); exp[:, HBI_INDEX] = t
        np.testing.assert_allclose(self.m.predict(x, t), exp, rtol=0, atol=1e-15)

    def test_bin_table_counts(self):
        bi, _ = bin_index(TRAIN.hbi)
        for r in self.m.bin_table.itertuples():
            m = bi == r.bin
            self.assertEqual(r.n_measurements, int(m.sum()))
            self.assertEqual(r.n_subjects, len(set(TRAIN.subject[m])))
        self.assertEqual(self.m.bin_table.n_measurements.sum(), len(TRAIN))

    def test_empty_bin_fallback(self):
        # 0.55-0.60 bin(3)을 비운 학습 데이터 -> 가장 가까운 유효 bin (거리 같으면 낮은 쪽 bin 2)
        keep = bin_index(TRAIN.hbi)[0] != 3
        m = BinnedMeanModel().fit(Spectra(TRAIN.a[keep], TRAIN.subject[keep], TRAIN.trial[keep]))
        self.assertEqual(m.bin_table.loc[3, "n_measurements"], 0)
        self.assertEqual(m.source[3], 2)
        self.assertEqual(m.bin_table.loc[3, "fallback_from"], 2)
        np.testing.assert_array_equal(m.means[3], m.means[2])
        m.reset_counters()
        x = QUERY.a[bin_index(QUERY.hbi)[0] == 2][:1]       # 입력 bin 2 (유효), 목표 0.58 (빈 bin 3)
        m.predict(x, 0.58)
        self.assertEqual(m.counters(), dict(fallback_uses=1, n_lookups=2, out_of_range=0))

    def test_same_bin_only_570_changes(self):
        # 한계(경계 불연속성) 확인: 같은 bin 안에서는 570 nm 외 보정량이 0
        x = QUERY.a[:1].copy(); x[0, HBI_INDEX] = 0.61
        out = self.m.predict(x, 0.64)
        free = np.arange(len(WL)) != HBI_INDEX
        np.testing.assert_array_equal(out[:, free], x[:, free])
        self.assertEqual(out[0, HBI_INDEX], 0.64)


class TestBiGRU(CommonInterface, unittest.TestCase):
    EPOCHS = 5

    def make(self):
        return BiGRUModel(seed=0, epochs=self.EPOCHS).fit(TRAIN)

    def test_name_and_size(self):
        self.assertEqual(self.m.name, "BiGRU")
        self.assertEqual(self.m.n_params, 689)                 # 기존 BiGRU8과 같은 크기

    def test_config_fixed(self):
        from common.config import BIGRU, BIGRU_SEEDS
        self.assertEqual(BIGRU["epochs"], 150)
        self.assertEqual(tuple(BIGRU_SEEDS), (0, 1, 2))
        self.assertEqual(BiGRUModel().epochs, 150)

    def test_epoch0_equals_b2(self):
        m0 = BiGRUModel(seed=0, epochs=0).fit(TRAIN)
        np.testing.assert_array_equal(m0.predict(QUERY.a, 0.6), LinearModel().fit(TRAIN).predict(QUERY.a, 0.6))

    def test_matches_legacy(self):
        spec = dict(cell="GRU", hidden=8, bi=True, name="BiGRU8")
        for seed in (0, 1):
            new = BiGRUModel(seed=seed, epochs=self.EPOCHS).fit(TRAIN)
            om, opre, _ = legacy.fit(TRAIN.a, TRAIN.subject, spec, seed, epochs=self.EPOCHS)
            np.testing.assert_array_equal(new.predict(QUERY.a, 0.6), legacy.predict(om, opre, QUERY.a, 0.6)[0])

    def test_forward_and_gradient(self):
        m = BiGRUModel(seed=0, epochs=0).fit(TRAIN)
        X, Y, W, _, _ = make_pairs(TRAIN)
        batch = tuple(torch.tensor(v, dtype=torch.float32) for v in (X, Y, W))
        out, r = m.net(batch[0], batch[1][:, HBI_INDEX])
        self.assertEqual(tuple(out.shape), (len(X), len(WL)))
        loss, _ = _objective(m.net, batch, 1e-3)
        loss.backward()
        grads = {n: p.grad for n, p in m.net.named_parameters()}
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads.values()))
        self.assertGreater(grads["head.weight"].abs().sum().item(), 0)   # 0 초기화된 head로 기울기가 흐름
        # 한 step 뒤에는 GRU 파라미터에도 기울기가 전달됨
        opt = torch.optim.Adam(m.net.parameters(), lr=3e-3); opt.step(); opt.zero_grad()
        _objective(m.net, batch, 1e-3)[0].backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for n, p in m.net.named_parameters() if n.startswith("gru")), 0)

    def test_reproducible_and_seed_sensitive(self):
        a = BiGRUModel(seed=0, epochs=self.EPOCHS).fit(TRAIN).predict(QUERY.a, 0.6)
        b = BiGRUModel(seed=0, epochs=self.EPOCHS).fit(TRAIN).predict(QUERY.a, 0.6)
        c = BiGRUModel(seed=1, epochs=self.EPOCHS).fit(TRAIN).predict(QUERY.a, 0.6)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_history_finite(self):
        self.assertEqual(len(self.m.history), self.EPOCHS)
        self.assertTrue(np.isfinite(np.array(self.m.history)).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
