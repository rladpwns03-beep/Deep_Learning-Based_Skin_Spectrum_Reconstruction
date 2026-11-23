# Post hoc exploratory analysis: Global Linear with the mean of subject slopes

**Status: not preregistered, computed after the external results were seen.** This analysis is not part of the preregistered comparisons (`docs/Preregistration_ExternalTest_v1.md`) and is kept apart from them. No evaluative statement is made here.

## Definition

The preregistered Global Linear (`models/linear.py`, `fit_slope`) centres absorbance and HbI within each training subject, pools all subjects, and fits one slope per wavelength (pooled within-subject slope). The variant first fits one slope per training subject and then takes the simple mean over subjects:

- b_s(λ) = Σ_{i∈s} (h_i − mean_h_s)(A_{i,λ} − mean_A_{s,λ}) / Σ_{i∈s} (h_i − mean_h_s)²
- b_mean(λ) = (1/S) Σ_s b_s(λ), with b_mean(570) = 1

Reconstruction uses `models/linear.py` `apply_slope` unchanged. The variant has no hyperparameter. In the figures and tables it is named `Global Linear (mean of subject slopes)`.

## How it was run

- Script: `scripts/posthoc_subject_mean_slope.py`. Output: `results/posthoc_exploratory/subject_mean_slope_baseline/` (`records.csv`, `comparisons.csv`, `slopes.csv`, `config.json`). Nothing was written to `results/external_test/`.
- Evaluation sets: Development10_LeaveOneSubjectOut (10 folds, 9 training subjects per fold) and ExternalTest6 (training = all 10 Development cohort subjects, evaluation = 6 External cohort subjects). Splits are read from `data/splits_final/`.
- Test pairs and metrics come from the existing evaluation functions (`scripts/final_utils.py`, `scripts/final_eval.py`, `common/metrics.py`).
- The pooled Global Linear was recomputed in the same script and agreed with the stored `records.csv` of both evaluation sets before anything was written.
- The stored subject values of PLS_v2 and BiGRU are reused unchanged for the comparisons against those models.
- `config.json` records `"preregistered": false`, `"computed_after_external_unblinding": true`, `"status": "post hoc, exploratory"` and the code commit.

## Results (General reconstruction; units 10^-3 absorbance)

Difference = variant minus the other model. The standard error is the between-subject SD of the paired difference divided by sqrt(n) (ddof = 1). The last column counts subjects with a difference below zero.

| Evaluation set | Metric | Compared with | Variant | Other model | Difference ± standard error | Difference < 0 |
|---|---|---|---|---|---|---|
| Development10_LeaveOneSubjectOut | absorbance RMSE | Global Linear (pooled) | 12.264 | 12.126 | +0.139 ± 0.435 | 6/10 |
| Development10_LeaveOneSubjectOut | absorbance RMSE | PLS_v2 | 12.264 | 11.057 | +1.207 ± 0.886 | 3/10 |
| Development10_LeaveOneSubjectOut | absorbance RMSE | BiGRU | 12.264 | 11.301 | +0.963 ± 0.761 | 2/10 |
| Development10_LeaveOneSubjectOut | Peak RMSE | Global Linear (pooled) | 0.733 | 0.716 | +0.017 ± 0.029 | 5/10 |
| Development10_LeaveOneSubjectOut | Peak RMSE | PLS_v2 | 0.733 | 0.684 | +0.049 ± 0.060 | 6/10 |
| Development10_LeaveOneSubjectOut | Peak RMSE | BiGRU | 0.733 | 0.949 | −0.216 ± 0.119 | 7/10 |
| ExternalTest6 | absorbance RMSE | Global Linear (pooled) | 11.097 | 12.910 | −1.813 ± 0.211 | 6/6 |
| ExternalTest6 | absorbance RMSE | PLS_v2 | 11.097 | 12.009 | −0.912 ± 0.623 | 4/6 |
| ExternalTest6 | absorbance RMSE | BiGRU | 11.097 | 11.862 | −0.766 ± 0.577 | 5/6 |
| ExternalTest6 | Peak RMSE | Global Linear (pooled) | 0.862 | 0.893 | −0.031 ± 0.045 | 4/6 |
| ExternalTest6 | Peak RMSE | PLS_v2 | 0.862 | 0.921 | −0.060 ± 0.021 | 5/6 |
| ExternalTest6 | Peak RMSE | BiGRU | 0.862 | 1.087 | −0.226 ± 0.073 | 6/6 |

## Reading notes

- The variant was defined and evaluated after the ExternalTest6 results were known, so the ExternalTest6 rows are not a preregistered test of the variant.
- The sign of the difference from the pooled Global Linear differs between the two evaluation sets (positive mean in Development10_LeaveOneSubjectOut, negative mean in ExternalTest6).
- Figure: `PostHocExploratoryFigure01_SubjectMeanSlopeBaseline` (`results/figures_final/`), with the label "Post hoc, exploratory (not preregistered)" inside the figure. The variant is not drawn in any other figure.
