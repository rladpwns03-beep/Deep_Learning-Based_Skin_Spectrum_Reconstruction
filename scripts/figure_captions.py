"""English captions for the figures made by scripts/make_figures.py. write_captions() creates docs/Figure_Captions.md.

Each caption gives: one-sentence summary (what is drawn, not a conclusion), panels, data (split set, n, unit of analysis),
markers / error bars / units, notes, and the input files. No evaluative wording and no p values.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "Figure_Captions.md"
PEAK_RS_CONFIG = ROOT / "results" / "external_test" / "config.json"

ERR = ("Points: individual subjects. Centre marker and bars: mean and standard error across subjects "
       "(between-subject SD / sqrt(n), ddof = 1).")
UNIT = "Units: 10^-3 absorbance."
DIFF = "Difference = first model minus second model; negative values indicate lower error for the first model."
COHORTS = ("Development cohort: leave-one-subject-out, n = 10 subjects, hyperparameters tuned inside the training subjects of each fold. "
           "External cohort: n = 6 subjects, evaluated once with frozen settings (training = all 10 Development cohort subjects). "
           "Unit of analysis: subject.")
SEED = "BiGRU subject values are the mean of the metric over seeds 0, 1 and 2."
DEV = "results/final_comparison/Development10_LeaveOneSubjectOut"
EXT = "results/external_test"

CAPTIONS = {
    "Figure01_DataOverview": dict(
        summary="Measured HbI values and mean absorbance spectra of the 16 subjects (160 measurements).",
        panels="(a) HbI (absorbance at 570 nm) of the 10 measurements of each subject; dashed line: target HbI 0.6; grey band: 0.6 +/- 0.02. "
               "(b) Mean absorbance spectrum of each subject (thin lines) and of each cohort (thick lines), 400 to 700 nm; shading: the five "
               "wavelength bands (Soret 400-440, Band 450-460, Carotenoid 470-510, Q-band 520-590 without 570, Red 600-700 nm); dotted lines: 490 nm and 570 nm.",
        data="Development cohort S001 to S010 (n = 10) and External cohort S101 to S106 (n = 6), 10 measurements per subject.",
        marks="Filled circles and solid lines: Development cohort. Open triangles and dashed lines: External cohort. Absorbance = -log10(reflectance).",
        notes="Descriptive figure; no model is involved.",
        source="data/processed/spectra_union16.csv"),
    "Figure02_ModelComparison": dict(
        summary="General reconstruction error of HbI-binned mean, Global Linear, PLS_v2 and BiGRU in the two cohorts.",
        panels="(a) Absorbance RMSE over the 30 wavelengths other than 570 nm. (b) Peak RMSE, Peak = A(490) - (A(470) + A(510)) / 2. "
               "The y axes start at 0.",
        data=COHORTS + " Each subject value is computed over the 90 ordered measurement pairs of that subject.",
        marks=ERR + " Filled circles: Development cohort; open triangles: External cohort; colour: model. " + UNIT + " " + SEED,
        notes="Baselines: HbI-binned mean and Global Linear. Proposals: PLS_v2 and BiGRU.",
        source=f"{DEV}/records.csv; {EXT}/records.csv"),
    "Figure03_PairedDifference_BySubject": dict(
        summary="Paired differences in absorbance RMSE (570 nm excluded) and Peak RMSE between models, by subject.",
        panels="Rows: absorbance RMSE (top) and Peak RMSE (bottom). Columns: PLS_v2 minus Global Linear, BiGRU minus Global Linear, BiGRU minus PLS_v2. "
               "Dashed line: zero difference. The label k/n lower gives the number of subjects with a difference below zero. "
               "In each cohort the two subjects with the largest absolute difference are labelled.",
        data=COHORTS,
        marks=ERR + " " + DIFF + " " + UNIT + " " + SEED,
        notes="The plotted means, standard errors and k/n were recomputed from the subject records and agree with the stored comparison tables.",
        source=f"{DEV}/records.csv; {EXT}/records.csv; {DEV}/comparisons_main.csv; {EXT}/comparisons_main.csv"),
    "Figure04_AccuracyVersusPeakCost": dict(
        summary="Mean difference from Global Linear in absorbance RMSE (x axis) against the mean difference in Peak RMSE (y axis) for six models.",
        panels="(a) Development cohort. (b) External cohort. Both panels share the same axis ranges. The origin is Global Linear.",
        data=COHORTS,
        marks="Markers: mean paired difference across subjects; bars in both directions: standard error across subjects (SD / sqrt(n), ddof = 1). "
              "Filled markers: PLS_v2 and BiGRU. Open markers: supplementary models (PLS_v1, Ridge, BiGRU (fixed setting), BiGRU (Peak-constrained selection)). "
              + DIFF + " " + UNIT,
        notes="In the External cohort BiGRU (Peak-constrained selection) was selected at epoch 0, which is identical to Global Linear, so it lies at the origin. "
              "The variants limited to 5 components are not drawn.",
        source=f"{DEV}/comparisons_main.csv; {DEV}/comparisons_supplementary.csv; {EXT}/comparisons_main.csv; {EXT}/comparisons_supplementary.csv"),
    "Figure05_BandError_DifferenceFromGlobalLinear": dict(
        summary="Difference from Global Linear in absorbance RMSE within each of the five wavelength bands.",
        panels="(a) PLS_v2 minus Global Linear. (b) BiGRU minus Global Linear. Bands are ordered by wavelength; Q-band excludes 570 nm, "
               "which is fixed to the target HbI in every model.",
        data=COHORTS,
        marks="Bars: mean paired difference across subjects; error bars: standard error across subjects (SD / sqrt(n), ddof = 1). "
              "Filled bars: Development cohort; hatched bars: External cohort. k/n: subjects with a difference below zero. " + DIFF + " " + UNIT,
        notes="Band definitions were fixed before the results were computed (scripts/final_utils.py, BANDS).",
        source="results/band_analysis/Development10_LeaveOneSubjectOut/band_error_comparisons.csv; results/external_test/comparisons_band.csv"),
    "Figure06_ResidualSlope_VersusPersonalReference": dict(
        summary="Band-mean residual slope of each model against the personal reference residual slope of the same test subject.",
        panels="Rows: PLS_v2 (top) and BiGRU (bottom). Columns: Soret, Band 450-460, Q-band (570 nm excluded) and Red. "
               "Dotted line: y = x. Dashed lines: zero. Each panel lists the Pearson r and n for each cohort.",
        data=COHORTS + " Residual slope of a model = (model prediction - Global Linear prediction) / change in HbI, averaged over the test pairs of a subject "
             "(pairs with zero change excluded). Personal reference = within-subject least-squares slope of the test subject minus the Global Linear slope; "
             "it is used for evaluation only.",
        marks="Points: individual subjects (filled circles: Development cohort; open triangles: External cohort). Units: absorbance per unit HbI. " + SEED,
        notes="With n = 6 in the External cohort the uncertainty of r is large. The recomputed r values agree with the stored summary tables.",
        source="results/band_analysis/Development10_LeaveOneSubjectOut/residual_slope_per_subject.csv; results/external_test/residual_slope_per_subject.csv; "
               "the residual_slope_summary.csv files in the same folders"),
    "Figure07_ReconstructionExamples_ExternalTest6": dict(
        summary="One reconstruction example per External cohort subject, chosen by a fixed rule.",
        panels="Six cells (S101 to S106). Upper part of each cell: absorbance of the measured input, the measured target and the reconstructions by "
               "Global Linear, PLS_v2 and BiGRU. Lower part: reconstruction minus measured target, with the five wavelength bands shaded. "
               "The cell title gives the input HbI and the target HbI; in-range subjects (HbI range contains 0.6) are marked.",
        data="External cohort, frozen settings. Selection rule: the 90 pairs of a subject are sorted by (Global Linear absorbance RMSE without 570 nm, "
             "input trial, target trial) in ascending order and the 45th pair is shown.",
        marks="Grey dashed line: measured input. Black line: measured target. Coloured lines: models. Error in 10^-3 absorbance. "
              "The BiGRU curve is the mean of the predictions of the three seeds, whereas reported BiGRU metrics are the mean of the per-seed metrics.",
        notes="Examples only; the selected pairs are listed in the plotted data file.",
        source="results/external_test/predictions.json; data/processed/spectra_union16.csv"),
    "Figure08_WavelengthError_ExternalTest6": dict(
        summary="Reconstruction RMSE at each wavelength in the External cohort.",
        panels="(a) RMSE at each wavelength for HbI-binned mean, Global Linear, PLS_v2 and BiGRU. (b) Difference from Global Linear for PLS_v2 and BiGRU. "
               "Dotted line: 570 nm, which is fixed to the target HbI (error 0) and is left out of the curves.",
        data="External cohort, n = 6 subjects, frozen settings. For each subject the RMSE at a wavelength is taken over that subject's 90 pairs.",
        marks="Lines: mean across the 6 subjects. Shaded bands: mean +/- standard error across subjects (SD / sqrt(n), ddof = 1). " + UNIT +
              " BiGRU: the RMSE is computed per seed and then averaged over seeds.",
        notes="The subject-level absorbance RMSE recombined from the wavelength values agrees with the stored records.",
        source="results/external_test/predictions.json; data/processed/spectra_union16.csv; results/external_test/records.csv"),
    "Figure09_FixedTargetRepeatability": dict(
        summary="Within-subject variation after every test measurement is reconstructed to the fixed target HbI 0.6.",
        panels="(a) Peak SD at target HbI 0.6. (b) Spectral SD (mean over the 30 wavelengths other than 570 nm of the SD across the 10 reconstructions).",
        data=COHORTS + " No measured spectrum at exactly HbI 0.6 exists, so this figure shows repeatability, not accuracy.",
        marks=ERR + " Filled markers: Development cohort; open triangles: External cohort; open diamonds: PLS_v1 in the Development cohort. "
              "Raw: measured spectra without correction. " + UNIT + " " + SEED,
        notes="PLS_v1 is a supplementary model without the identity constraint.",
        source=f"{DEV}/records.csv; {EXT}/records.csv"),
    "SupplementaryFigure01_ResidualSlopeSpectrum": dict(
        summary="Residual slope at each wavelength for PLS_v2, BiGRU and the personal reference.",
        panels="(a) Development cohort. (b) External cohort. Dashed line: zero (Global Linear). Shading: the five wavelength bands. 570 nm is excluded.",
        data=COHORTS + " Definitions as in Figure06.",
        marks="Lines: mean across subjects. Shaded bands: mean +/- standard error across subjects (SD / sqrt(n), ddof = 1). Units: absorbance per unit HbI. " + SEED,
        notes="The personal reference is computed from the test subject's own 10 measurements and is used for evaluation only.",
        source="results/band_analysis/Development10_LeaveOneSubjectOut/residual_slope_per_subject.csv; results/external_test/residual_slope_per_subject.csv"),
    "SupplementaryFigure02_ErrorByHbIChangeSize": dict(
        summary="Difference from Global Linear by the size of the HbI change between input and target.",
        panels="Rows: absorbance RMSE difference (top) and Peak RMSE difference (bottom). Columns: Development cohort and External cohort. "
               "x axis: four bins of the absolute HbI change (both directions pooled).",
        data=COHORTS + " A subject without pairs in a bin is left out of that bin; n under each bin is the number of subjects that remain.",
        marks="Markers: mean paired difference across subjects; bars: standard error across subjects (SD / sqrt(n), ddof = 1). " + DIFF + " " + UNIT,
        notes="Bins were fixed before the evaluation (0 to 0.025, 0.025 to 0.05, 0.05 to 0.10, 0.10 to 0.30).",
        source=f"{DEV}/comparisons_delta_hbi.csv; {EXT}/comparisons_delta_hbi.csv"),
    "SupplementaryFigure03_PeakRS_ReferenceComparison": dict(
        summary="Mean Peak_RS of the 10 spectra reconstructed to HbI 0.6 minus a within-subject regression reference, for each External cohort subject.",
        panels="Single panel. x axis: subjects S101 to S106; the second line of each tick label is the within-subject regression slope of Peak_RS on HbI "
               "(AU per HbI). Grey band around zero: +/- standard error of the regression reference of that subject.",
        data="External cohort, n = 6 subjects, frozen settings. Reference: linear regression of Peak on HbI over the subject's 10 measurements, evaluated at HbI 0.6.",
        marks="Markers: HbI-binned mean, Global Linear, PLS_v2, BiGRU. Filled markers: in-range subjects (S101, S106). Open markers: subjects whose HbI range "
              "does not contain 0.6, for which the reference is an extrapolation. Units: Peak_RS in AU.",
        notes="The reference is an estimate, not an independent ground truth. Peak_RS conversion as recorded in results/external_test/config.json (peak_rs_reference): {PEAK_RS}",
        source="results/external_test/peak_rs_reference.csv; results/external_test/config.json"),
    "SupplementaryFigure04_TrainingSizeEffect": dict(
        summary="Absorbance RMSE of the same test subjects when the models are trained on fewer subjects and on 15 subjects.",
        panels="(a) External cohort, n = 6: training on 10 subjects (ExternalTest6) and on 15 subjects (Union16 leave-one-subject-out fold of that subject). "
               "(b) Development cohort, n = 10: training on 9 subjects and on 15 subjects, with the same fixed hyperparameters in both cases.",
        data="Models: Global Linear, PLS_v2, BiGRU. Unit of analysis: subject. All runs use the frozen hyperparameters.",
        marks="Thin lines: individual subjects. Thick lines with markers: mean across subjects. " + UNIT + " " + SEED,
        notes="Supplementary result with fixed hyperparameters. The fixed hyperparameters were selected on all 10 Development cohort subjects, so in panel (b) "
              "each test subject took part in that selection. These results are not independent of ExternalTest6.",
        source="results/union16/training_size_effect.csv; results/external_test/records.csv; results/union16/Union16_LeaveOneSubjectOut/records.csv; "
               "results/union16/Development10_LeaveOneSubjectOut_FixedHyperparameters/records.csv"),
    "SupplementaryFigure05_BiGRU_TuningSurface": dict(
        summary="Inner validation error of the 84 BiGRU candidates when all 10 Development cohort subjects are the training set.",
        panels="Rows: validation absorbance RMSE (top) and validation Peak RMSE (bottom). Columns: hidden size 4, 8 and 16. x axis: epoch. "
               "Lines: the four values of slope_l2. Horizontal dashed line: epoch 0, which is identical to Global Linear.",
        data="Development cohort, 10 subjects, leave-one-subject-out inside the training subjects, selection seed 0. Each value pools all inner validation pairs.",
        marks="Filled circle: selected setting (hidden 8, slope_l2 0, epoch 200). Open circle: BiGRU (fixed setting) (hidden 8, slope_l2 0.001, epoch 150). " + UNIT,
        notes="The selection criterion was the validation absorbance RMSE only. The External cohort was not used.",
        source="results/bigru_tuned/Development10_AllSubjects/selection.json"),
    "SupplementaryFigure06_SHAP_InputGroupShare": dict(
        summary="Band-level SHAP values: share of seven input groups and the input group by output band map.",
        panels="(a) Share of the mean absolute SHAP value over the 30 output wavelengths for Global Linear, PLS_v2, BiGRU, PLS_v1 and Ridge. "
               "(b) Mean absolute SHAP value of each input group for each output band, for PLS_v2 and BiGRU; each map has its own grey scale.",
        data="Computed only on Development10_LeaveOneSubjectOut (900 test pairs, 10 folds). Exact Shapley values over 7 input groups (128 coalitions), "
             "background = 100 training pairs of the fold. BiGRU: computed per seed and then averaged.",
        marks="Panel (a): percent. Panel (b): 10^-3 absorbance.",
        notes="PLS_v2 and Ridge take the input in the form (change in HbI) x (A_in - A_in(570)), so the contributions of the input HbI group and of the "
              "spectral groups can cancel with opposite signs; group shares alone are not read as importance. Mixing groups from different measurements "
              "can create spectra that do not occur (a limit of interventional SHAP).",
        source="results/shap_bands/Development10_LeaveOneSubjectOut/shap_summary.csv; results/shap_bands/Development10_LeaveOneSubjectOut/shap_input_group_by_output_band.csv"),
    "SupplementaryFigure07_SplitSetConsistency": dict(
        summary="Paired differences between models in the five split sets.",
        panels="(a) Absorbance RMSE. (b) Peak RMSE. Rows: three comparisons (PLS_v2 minus Global Linear, BiGRU minus Global Linear, BiGRU minus PLS_v2), "
               "each in Development10_Holdout73, Development10_LeaveOneSubjectOut, ExternalTest6, Union16_Holdout73 and Union16_LeaveOneSubjectOut. Dashed line: zero.",
        data="Unit of analysis: subject; n is printed in each row. In the two 7:3 split sets a subject appears in several splits and its values are "
             "averaged over splits before the paired difference is taken.",
        marks="Markers: mean paired difference; bars: standard error across subjects (SD / sqrt(n), ddof = 1). " + DIFF + " " + UNIT,
        notes="Union16 sets (open markers): supplementary, fixed hyperparameters; Development cohort test subjects in these sets took part in the selection of those "
              "hyperparameters. The split sets overlap in subjects and are not independent of each other.",
        source="comparisons_main.csv in results/final_comparison/Development10_Holdout73, results/final_comparison/Development10_LeaveOneSubjectOut, "
               "results/external_test, results/union16/Union16_Holdout73 and results/union16/Union16_LeaveOneSubjectOut"),
    "PostHocExploratoryFigure01_SubjectMeanSlopeBaseline": dict(
        summary="Post hoc, exploratory (not preregistered): Global Linear with the mean of subject slopes in place of the pooled within-subject slope.",
        panels="(a) Absorbance RMSE by subject for Global Linear (pooled slope), Global Linear (mean of subject slopes), PLS_v2 and BiGRU in the two cohorts. "
               "(b) The two slopes b estimated from the 10 Development cohort subjects (upper part) and their difference, mean of subject slopes minus pooled (lower part).",
        data=COHORTS + " The variant has no hyperparameter. PLS_v2 and BiGRU values are the stored subject values.",
        marks=ERR + " Open stars: Global Linear (mean of subject slopes). " + UNIT + " Slopes in absorbance per unit HbI.",
        notes="This analysis was not preregistered and was computed after the external results were seen. It is kept apart from the preregistered figures.",
        source="results/posthoc_exploratory/subject_mean_slope_baseline/records.csv; results/posthoc_exploratory/subject_mean_slope_baseline/slopes.csv"),
}


def write_captions() -> Path:
    peak_rs = json.loads(PEAK_RS_CONFIG.read_text(encoding="utf-8"))["peak_rs_reference"] if PEAK_RS_CONFIG.exists() else "(config file not found)"
    out = ["# Figure captions", "",
           "Generated by `scripts/make_figures.py` (`scripts/figure_captions.py`). Figures: `results/figures_final/presentation/` (12.0 in wide, 200 dpi, with title) "
           "and `results/figures_final/report/` (6.3 in wide, 600 dpi, no title). The numbers drawn in each figure are in `results/figures_final/plotted_data/`.", "",
           "Conventions used in every figure: a point is one subject, the centre marker is the mean across subjects, and error bars are the standard error "
           "(between-subject SD / sqrt(n), ddof = 1). Difference = first model minus second model. No p values are shown.", ""]
    for name, c in CAPTIONS.items():
        out += [f"## {name}", "", c["summary"], "",
                f"- Panels: {c['panels']}", f"- Data: {c['data']}", f"- Markers, error bars, units: {c['marks']}",
                f"- Notes: {c['notes'].replace('{PEAK_RS}', peak_rs)}", f"- Source: {c['source']}", ""]
    DOC.parent.mkdir(exist_ok=True)
    DOC.write_text("\n".join(out), encoding="utf-8")
    return DOC
