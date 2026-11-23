"""Checks for scripts/make_figures.py (figures are drawn from stored results only).

Run (project root):  python -X utf8 -m unittest tests.test_figures -v

The module regenerates all figures once in both profiles (and the presentation profile a second time), so it takes a few minutes.
It is skipped when the stored results are not present.

  1. read-only: sha256 of every existing file under results/ (outside figures_final and posthoc_exploratory) is unchanged
  2. numbers: plotted_data values agree with the reference values of the figure specification (last digit +/- 1)
  3. recomputation: Figure03 differences and Figure06 Pearson r agree with the stored tables (relative 1e-6)
  4. files: 34 JPEG files open, RGB, 2400 px (presentation) and 3780 px (report) wide
  5. strings: no evaluative words, no made-up abbreviations, no middle dot in figure strings and captions
  6. determinism: plotted_data is byte-identical between two runs
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))
RES = ROOT / "results"
OUT = RES / "figures_final"
HAVE = (RES / "external_test" / "records.csv").exists() and (RES / "posthoc_exploratory" / "subject_mean_slope_baseline" / "records.csv").exists()
STATE = {}
FORBIDDEN_WORDS = ["significant", "significantly", "outperform", "outperforms", "outperformed", "superior", "best", "better", "worse", "novel",
                   "robust", "clearly"]
FORBIDDEN_ABBREVIATIONS = ["GL", "LOSO", "D10", "U16", "Ext", "Dev"]
D, E, B = "Development", "External", "band_analysis"


def sha_tree() -> dict:
    skip = {"figures_final", "posthoc_exploratory"}
    return {str(p.relative_to(RES)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(RES.rglob("*")) if p.is_file() and p.relative_to(RES).parts[0] not in skip}


def plotted_bytes() -> dict:
    return {p.name: p.read_bytes() for p in sorted((OUT / "plotted_data").glob("*.csv"))}


def setUpModule():
    if not HAVE:
        return
    import make_figures
    STATE["before"] = sha_tree()
    STATE["infos"] = make_figures.main(["--profile", "both"])
    STATE["first"] = plotted_bytes()
    STATE["jpg_first"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUT / "presentation").glob("*.jpg"))}
    make_figures.main(["--profile", "presentation", "--textless-panels"])
    STATE["jpg_second"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUT / "presentation").glob("*.jpg"))}
    STATE["second"] = plotted_bytes()
    STATE["after"] = sha_tree()
    STATE["names"] = list(make_figures.FIGURES)


def data(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / "plotted_data" / f"{name}.csv", encoding="utf-8-sig")


def one(df, **kw):
    for k, v in kw.items():
        df = df[df[k].astype(str) == str(v)]
    assert len(df) == 1, (kw, len(df))
    return df.iloc[0]


def close(value, ref, nd=3):
    return abs(round(float(value), nd) - ref) <= 10 ** (-nd) + 1e-12


@unittest.skipUnless(HAVE, "stored results not found")
class TestFigureFiles(unittest.TestCase):
    def test_1_results_are_read_only(self):
        self.assertEqual(STATE["before"], STATE["after"])
        self.assertGreater(len(STATE["before"]), 100)

    def test_4_jpeg_files(self):
        from PIL import Image
        self.assertEqual(len(STATE["infos"]), 17)
        n = 0
        for prof, width in (("presentation", 2400), ("report", 3780)):
            for name in STATE["names"]:
                with Image.open(OUT / prof / f"{name}.jpg") as im:
                    im.load()
                    self.assertEqual(im.format, "JPEG")
                    self.assertEqual(im.mode, "RGB")
                    self.assertEqual(im.size[0], width, f"{prof}/{name}")      # no tight bounding box: exact width
                    self.assertLessEqual(im.size[1], 1300 if prof == "presentation" else 5100)
                n += 1
        self.assertEqual(n, 34)
        m = pd.read_csv(OUT / "figure_manifest.csv", encoding="utf-8-sig")
        self.assertEqual(sorted(m.figure), sorted(STATE["names"]))

    def test_5_strings(self):
        strings = json.loads((OUT / "figure_strings.json").read_text(encoding="utf-8"))
        texts = [s for f in strings.values() for prof in f.values() for s in prof]
        self.assertGreater(len(texts), 500)
        caption = (ROOT / "docs" / "Figure_Captions.md").read_text(encoding="utf-8")
        caption = caption.split("(peak_rs_reference): ")[0] + caption.split("(peak_rs_reference): ")[1].split("\n", 1)[1]   # quoted config string excluded
        for text in texts + [caption] + STATE["names"]:
            self.assertNotIn("·", text)
            for w in FORBIDDEN_WORDS:
                self.assertIsNone(re.search(rf"\b{w}\b", text, re.IGNORECASE), f"'{w}' in: {text[:80]}")
            for a in FORBIDDEN_ABBREVIATIONS:
                self.assertIsNone(re.search(rf"(?<![A-Za-z0-9]){a}(?![A-Za-z0-9])", text), f"'{a}' in: {text[:80]}")
        for s in texts:                                           # English only inside figures
            self.assertIsNone(re.search(r"[가-힣]", s), s)
        self.assertIn("Post hoc, exploratory (not preregistered)", strings["PostHocExploratoryFigure01_SubjectMeanSlopeBaseline"]["report"])

    def test_textless_split_panels(self):
        from PIL import Image
        self.assertEqual(STATE["jpg_first"], STATE["jpg_second"])            # the versions with text are not changed by the extra output
        root = OUT / "presentation_textless_panels"
        idx = pd.read_csv(root / "panel_index.csv", encoding="utf-8-sig")
        self.assertEqual(sorted(idx.figure.unique()), sorted(STATE["names"]))
        self.assertEqual(len(idx), 54)
        for r in idx.itertuples():
            with Image.open(root / r.file) as im:
                im.load()
                self.assertEqual(im.mode, "RGB")
                self.assertAlmostEqual(im.size[0] / 200, r.width_in, delta=0.02)
        for name in STATE["names"]:
            t = pd.read_csv(root / name / "text_labels.csv", encoding="utf-8-sig")
            self.assertGreater(len(t), 5)
            self.assertTrue(set(t.panel) - {"figure"} <= set(idx[idx.figure == name].panel))
            self.assertTrue((root / name / f"{name}__legend_keys.jpg").exists())

    def test_6_deterministic_plotted_data(self):
        self.assertEqual(STATE["first"], STATE["second"])
        self.assertEqual(len(STATE["first"]), 17)


@unittest.skipUnless(HAVE, "stored results not found")
class TestPlottedNumbers(unittest.TestCase):
    def test_2_r1_subject_means(self):
        f2, f9 = data("Figure02_ModelComparison"), data("Figure09_FixedTargetRepeatability")
        ref = [(f2, "a", "HbI-binned mean", D, 17.414), (f2, "a", "Global Linear", D, 12.126), (f2, "a", "PLS_v2", D, 11.057), (f2, "a", "BiGRU", D, 11.301),
               (f2, "b", "Global Linear", D, 0.716), (f2, "b", "PLS_v2", D, 0.684), (f2, "b", "BiGRU", D, 0.949),
               (f9, "a", "Raw", D, 2.202), (f9, "a", "Global Linear", D, 0.507), (f9, "a", "BiGRU", D, 0.686),
               (f2, "a", "HbI-binned mean", E, 18.718), (f2, "a", "Global Linear", E, 12.910), (f2, "a", "PLS_v2", E, 12.009), (f2, "a", "BiGRU", E, 11.862),
               (f2, "b", "HbI-binned mean", E, 2.651), (f2, "b", "Global Linear", E, 0.893), (f2, "b", "PLS_v2", E, 0.921), (f2, "b", "BiGRU", E, 1.087),
               (f9, "a", "Raw", E, 2.301), (f9, "a", "Global Linear", E, 0.631), (f9, "a", "PLS_v2", E, 0.665), (f9, "a", "BiGRU", E, 0.798),
               (f9, "b", "Raw", E, 18.318), (f9, "b", "PLS_v2", E, 7.410), (f9, "b", "PLS_v1", E, 4.196)]
        for df, panel, model, cohort, value in ref:
            r = one(df, panel=panel, series=model, x=cohort, kind="mean")
            self.assertTrue(close(r.y, value), (panel, model, cohort, r.y, value))
        u = pd.read_csv(RES / "union16/Union16_LeaveOneSubjectOut/summary.csv", encoding="utf-8-sig")       # not drawn as means; checked in the CSV
        for model, value in (("Global Linear", 11.934), ("PLS_v2", 11.275), ("BiGRU", 11.361)):
            self.assertTrue(close(one(u, section="general", metric="a_rmse", model=model)["mean"] * 1e3, value))

    def test_2_r2_paired_differences(self):
        f3, s7 = data("Figure03_PairedDifference_BySubject"), data("SupplementaryFigure07_SplitSetConsistency")
        f5, s2 = data("Figure05_BandError_DifferenceFromGlobalLinear"), data("SupplementaryFigure02_ErrorByHbIChangeSize")
        pg, bg, bp = "PLS_v2 minus Global Linear", "BiGRU minus Global Linear", "BiGRU minus PLS_v2"
        ref = [(f3, "a1", pg, D, -1.068, 0.639, 8, 10), (f3, "a2", bg, D, -0.824, 0.417, 8, 10), (f3, "a3", bp, D, 0.244, 0.582, 6, 10),
               (f3, "b1", pg, D, -0.032, 0.040, 4, 10), (f3, "b2", bg, D, 0.233, 0.129, 2, 10),
               (s7, "a", pg, "Development10_Holdout73", -0.185, 0.441, 7, 10), (s7, "a", bg, "Development10_Holdout73", -0.467, 0.312, 7, 10),
               (s7, "b", bg, "Development10_Holdout73", 0.286, 0.109, 1, 10),
               (f3, "a1", pg, E, -0.901, 0.652, 4, 6), (f3, "a2", bg, E, -1.047, 0.644, 4, 6), (f3, "a3", bp, E, -0.146, 1.136, 2, 6),
               (f3, "b1", pg, E, 0.029, 0.027, 3, 6), (f3, "b2", bg, E, 0.195, 0.091, 1, 6), (f3, "b3", bp, E, 0.166, 0.076, 1, 6),
               (s7, "a", pg, "Union16_Holdout73", -0.065, 0.571, 5, 16), (s7, "a", bg, "Union16_Holdout73", -0.284, 0.354, 8, 16),
               (s7, "a", pg, "Union16_LeaveOneSubjectOut", -0.659, 0.551, 9, 16), (s7, "a", bg, "Union16_LeaveOneSubjectOut", -0.574, 0.315, 10, 16),
               (s7, "b", pg, "Union16_LeaveOneSubjectOut", 0.089, 0.054, 5, 16),
               (f5, "a", pg, f"Soret|{D}", -2.566, 1.483, 6, 10), (f5, "a", pg, f"Band 450-460|{D}", -2.607, 1.457, 7, 10), (f5, "b", bg, f"Red|{D}", -1.258, 0.988, 8, 10),
               (f5, "a", pg, f"Soret|{E}", 0.611, 0.884, 2, 6), (f5, "a", pg, f"Red|{E}", -1.968, 1.264, 4, 6), (f5, "b", bg, f"Soret|{E}", 1.027, 1.497, 2, 6),
               (f5, "b", bg, f"Q-band|{E}", 0.325, 0.505, 1, 6), (f5, "b", bg, f"Red|{E}", -3.223, 1.511, 4, 6),
               (s2, "b2", bg, f"0.100-0.300|{E}", 0.581, 0.241, 0, 5), (s2, "a2", pg, f"0.050-0.100|{E}", -1.767, 1.191, 4, 6)]
        for df, panel, series, x, d, se, k, n in ref:
            r = one(df, panel=panel, series=series, x=x, kind="mean")
            self.assertTrue(close(r.y, d) and close(r.error, se), (panel, series, x, r.y, r.error))
            self.assertEqual((int(r.k), int(r.n)), (k, n), (panel, series, x))
        c = pd.read_csv(RES / "external_test/comparisons_main.csv", encoding="utf-8-sig")                    # not drawn as a difference; checked in the CSV
        r = one(c, section="repeatability", metric="peak_sd", comparison=bg)
        self.assertTrue(close(r.diff_mean * 1e3, 0.167) and close(r.diff_se * 1e3, 0.056) and (int(r.n_improved), int(r.n)) == (0, 6))

    def test_2_r3_other_values(self):
        f6 = data("Figure06_ResidualSlope_VersusPersonalReference")
        bands = ["Soret", "Band 450-460", "Q-band", "Red"]
        for model, cohort, rs in (("PLS_v2", D, [0.722, 0.804, 0.150, 0.311]), ("BiGRU", D, [0.255, 0.402, 0.670, 0.635]),
                                  ("PLS_v2", E, [-0.037, 0.150, -0.163, -0.073]), ("BiGRU", E, [0.319, -0.149, -0.103, 0.204])):
            for b, r in zip(bands, rs):
                self.assertTrue(close(one(f6, series=f"{model} | {b}", x=cohort, kind="pearson_r").y, r), (model, cohort, b))
        for b, v in zip(bands, [-0.120, -0.132, -0.050, -0.218]):
            rows = f6[(f6.series == f"Personal reference | {b}") & (f6.x == E)]
            self.assertTrue(all(close(y, v) for y in rows.y) and len(rows) == 2)
        for model, v in (("PLS_v2", -0.103), ("BiGRU", -0.114)):
            self.assertTrue(close(one(f6, series=f"{model} | Red", x=E, kind="model mean residual slope").y, v))
        s3 = data("SupplementaryFigure03_PeakRS_ReferenceComparison")
        for subj, vals in (("S101", [2.79, 2.93, 4.18]), ("S104", [28.39, 27.96, 35.42]), ("S106", [-2.08, -3.72, -2.52])):
            for model, v in zip(["Global Linear", "PLS_v2", "BiGRU"], vals):
                self.assertTrue(close(one(s3, series=model, x=subj).y, v, 2), (subj, model))
        self.assertTrue(close(one(s3, series="regression slope (AU per HbI)", x="S104").y, -646.6, 1))
        s4 = data("SupplementaryFigure04_TrainingSizeEffect")
        for model, lo, hi, d, se, k in (("Global Linear", 12.910, 11.383, -1.527, 0.154, 6), ("PLS_v2", 12.009, 11.297, -0.712, 0.252, 5),
                                        ("BiGRU", 11.862, 11.039, -0.823, 0.205, 6)):
            self.assertTrue(close(one(s4, panel="a", series=model, x="10", kind="mean").y, lo) and close(one(s4, panel="a", series=model, x="15", kind="mean").y, hi))
            r = one(s4, panel="a", series=model, kind="mean difference")
            self.assertTrue(close(r.y, d) and close(r.error, se) and int(r.k) == k)
        for model, d, se in (("Global Linear", 0.139, 0.514), ("PLS_v2", 0.835, 0.590), ("BiGRU", 0.412, 0.493)):
            r = one(s4, panel="b", series=model, kind="mean difference")
            self.assertTrue(close(r.y, d) and close(r.error, se))
        s5 = data("SupplementaryFigure05_BiGRU_TuningSurface")
        cand = s5[s5.series.str.startswith("hidden")]
        self.assertEqual((cand.kind == "val_a_rmse").sum(), 84)
        self.assertEqual((cand.kind == "val_peak_rmse").sum(), 84)
        self.assertTrue(close(one(s5, panel="a2", series="selected").y, 11.123) and close(one(s5, panel="b2", series="selected").y, 1.163))
        self.assertEqual(int(one(s5, panel="a2", series="selected").x), 200)
        self.assertTrue(close(one(s5, panel="a1", series="epoch 0 (identical to Global Linear)").y, 12.505))
        self.assertTrue(close(one(s5, panel="b1", series="epoch 0 (identical to Global Linear)").y, 0.742))
        s6 = data("SupplementaryFigure06_SHAP_InputGroupShare")
        for model, group, v in (("PLS_v2", "h* (target)", 10.28), ("PLS_v2", "h_in (570 nm)", 44.16), ("PLS_v2", "Q-band", 40.47),
                                ("BiGRU", "h* (target)", 44.21), ("BiGRU", "h_in (570 nm)", 46.65)):
            self.assertTrue(close(one(s6, panel="a", series=model, x=group).y, v, 2), (model, group))

    def test_2_posthoc_values(self):
        p = data("PostHocExploratoryFigure01_SubjectMeanSlopeBaseline")
        v = "Global Linear (mean of subject slopes)"
        for cohort, pooled, variant in ((D, 12.126, 12.264), (E, 12.910, 11.097)):
            self.assertTrue(close(one(p, panel="a", series="Global Linear", x=cohort, kind="mean").y, pooled))
            self.assertTrue(close(one(p, panel="a", series=v, x=cohort, kind="mean").y, variant))
        c = pd.read_csv(RES / "posthoc_exploratory/subject_mean_slope_baseline/comparisons.csv", encoding="utf-8-sig")
        for es, metric, d, se, k, n in (("Development10_LeaveOneSubjectOut", "a_rmse", 0.139, 0.435, 6, 10), ("Development10_LeaveOneSubjectOut", "peak_rmse", 0.017, 0.029, 5, 10),
                                        ("ExternalTest6", "a_rmse", -1.813, 0.211, 6, 6), ("ExternalTest6", "peak_rmse", -0.031, 0.045, 4, 6)):
            r = one(c, evaluation_set=es, metric=metric, comparison=f"{v} minus Global Linear")
            self.assertTrue(close(r.diff_mean * 1e3, d) and close(r.diff_se * 1e3, se) and (int(r.n_improved), int(r.n)) == (k, n), (es, metric))
        cfg = json.loads((RES / "posthoc_exploratory/subject_mean_slope_baseline/config.json").read_text(encoding="utf-8"))
        self.assertIs(cfg["preregistered"], False)
        self.assertIs(cfg["computed_after_external_unblinding"], True)
        self.assertEqual(cfg["status"], "post hoc, exploratory")

    def test_3_recomputed_values_match_stored_tables(self):
        f3 = data("Figure03_PairedDifference_BySubject")
        folder = {D: RES / "final_comparison/Development10_LeaveOneSubjectOut", E: RES / "external_test"}
        metric = {"a": "a_rmse", "b": "peak_rmse"}
        for r in f3[f3.kind == "mean"].itertuples():
            c = pd.read_csv(folder[r.x] / "comparisons_main.csv", encoding="utf-8-sig")
            s = one(c, section="general", metric=metric[r.panel[0]], comparison=r.series)
            sub = f3[(f3.panel == r.panel) & (f3.x == r.x) & f3.kind.str.startswith("subject")].y.to_numpy()
            np.testing.assert_allclose([sub.mean(), sub.std(ddof=1) / np.sqrt(len(sub))], [s.diff_mean * 1e3, s.diff_se * 1e3], rtol=1e-6)
            np.testing.assert_allclose([r.y, r.error], [s.diff_mean * 1e3, s.diff_se * 1e3], rtol=1e-6)
            self.assertEqual((int((sub < 0).sum()), len(sub)), (int(s.n_improved), int(s.n)))
        f6 = data("Figure06_ResidualSlope_VersusPersonalReference")
        summ = {D: RES / "band_analysis/Development10_LeaveOneSubjectOut/residual_slope_summary.csv", E: RES / "external_test/residual_slope_summary.csv"}
        for r in f6[f6.kind == "pearson_r"].itertuples():
            model, band = r.series.split(" | ")
            pts = f6[(f6.panel == r.panel) & f6.kind.str.endswith(f"|{r.x}")]
            s = one(pd.read_csv(summ[r.x], encoding="utf-8-sig"), item=band, model=model)
            np.testing.assert_allclose(np.corrcoef(pts.x.astype(float), pts.y)[0, 1], s.pearson_with_personal_reference, rtol=1e-6)
            np.testing.assert_allclose(r.y, s.pearson_with_personal_reference, rtol=1e-6)

    def test_figure_data_shapes(self):
        f1 = data("Figure01_DataOverview")
        self.assertEqual((f1.kind == "measurement").sum(), 160)
        self.assertEqual(f1[f1.kind == "measurement"].x.nunique(), 16)
        f7 = data("Figure07_ReconstructionExamples_ExternalTest6")
        self.assertEqual(sorted(f7[f7.series == "selected pair"].panel), [f"S10{k}" for k in range(1, 7)])
        f8 = data("Figure08_WavelengthError_ExternalTest6")
        self.assertNotIn(570, f8.x.astype(int).tolist())
        self.assertEqual(len(f8[(f8.panel == "a") & (f8.series == "BiGRU")]), 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
