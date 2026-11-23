"""ExternalTest6: 학습 = Development cohort 10명 전체, 평가 = External cohort 6명 (S101-S106). 사용자 승인 후 1회만 실행.

실행 (code 폴더, conda 환경 skin_spectra):
    python -X utf8 scripts/external_test.py --approved      # results/external_test/ (이미 결과가 있으면 중단, 다시 실행하지 않음)

  - 하이퍼파라미터는 docs/Preregistration_ExternalTest_v1.json (동결 문서와 같은 값)에서 읽어 고정한다. 다시 고르지 않는다.
  - 테스트 쌍: data/splits_final/ExternalTest6 (피험자당 90쌍). BiGRU seed 0, 1, 2 (피험자 값 = seed별 지표의 평균).
  - 주 표와 사전 지정 비교: Raw, HbI-binned mean, Global Linear, PLS_v2, BiGRU.
    보조 표: PLS_v1, Ridge, BiGRU (fixed setting), BiGRU (Peak-constrained selection).
  - 실험: General reconstruction / Fixed-target repeatability (In-range subject = S101, S106) /
          Near-target accuracy (0.6 ± 0.02 실측이 S101, S105에 1개씩뿐이라 기술 통계만).
  - 구간별 분석: 파장 구간별 복원 오차, 잔차 기울기 4구간, HbI 변화 크기 구간 (Development cohort 분석과 같은 정의·같은 함수).
  - 추가 기준값: 피험자별 10회 측정의 Peak 대 HbI 선형회귀로 얻은 HbI 0.6에서의 값과 그 표준오차 (Peak_RS 단위).
    Peak_RS = 8387.2 x Peak + 83.495 (Hwang et al. 2023, Sensors 23(17):7654 식 (2)).
    이 기준값은 추정값이며 독립 정답이 아니다. HbI 범위가 0.6을 포함하지 않는 피험자에서는 외삽이다.
  - 쌍체 비교: 피험자 6명. 차이 = 앞 모델 - 뒤 모델, 개선 = 차이 < 0. n = 6에서 부호검정 양측 p의 최솟값은 0.031.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))
from common import config as C                                                 # noqa: E402
from common.data_loader import HBI_INDEX                                       # noqa: E402
from common.metrics import peak                                                # noqa: E402
from band_analysis import residual_slope_rows, residual_slope_summary          # noqa: E402
from final_utils import (BAND_COMPARISONS, MAIN_COMPARISONS, MAIN_MODELS, P_NOTE, RES, ROOT as _R, UNION_CSV, jdump, jload,   # noqa: E402
                         load_splits, load_spectra, test_pairs)
from final_eval import comparison_table, fit_models, split_records, summary_table   # noqa: E402

OUT = RES / "external_test"
PREREG = ROOT / "docs" / "Preregistration_ExternalTest_v1.json"
SPLIT_SET = "ExternalTest6"
SUPP = ["PLS_v1", "Ridge", "BiGRU (fixed setting)", "BiGRU (Peak-constrained selection)"]
SUPP_COMPARISONS = [(m, "Global Linear") for m in SUPP] + [("BiGRU (Peak-constrained selection)", "BiGRU")]
PEAK_RS_SCALE, PEAK_RS_OFFSET = 8387.2, 83.495


def peak_reference(te, models: dict) -> pd.DataFrame:
    """피험자별 Peak 대 HbI 선형회귀의 HbI 0.6 값(과 표준오차) 대 모델로 0.6 복원한 10개 스펙트럼의 Peak 평균 (Peak_RS 단위)."""
    rows = []
    for s in te.subjects:
        A0 = te.a[te.subject == s]
        h, p = A0[:, HBI_INDEX], peak(A0)
        n, hm = len(h), h.mean()
        sxx = float(np.sum((h - hm) ** 2))
        slope = float(np.sum((h - hm) * (p - p.mean())) / sxx)
        icpt = float(p.mean() - slope * hm)
        resid = p - (icpt + slope * h)
        s_err = float(np.sqrt(np.sum(resid ** 2) / (n - 2)))
        ref = icpt + slope * C.TARGET_HBI
        se = s_err * float(np.sqrt(1 / n + (C.TARGET_HBI - hm) ** 2 / sxx))
        in_range = bool(h.min() <= C.TARGET_HBI <= h.max())
        for name, ms in models.items():
            pm = float(np.mean([peak(m.predict(A0, C.TARGET_HBI)).mean() for m in ms]))
            rows.append(dict(subject=s, model=name, in_range_subject=in_range, extrapolated=not in_range,
                             reference_peak_rs=PEAK_RS_SCALE * ref + PEAK_RS_OFFSET, reference_se_peak_rs=PEAK_RS_SCALE * se,
                             model_peak_rs=PEAK_RS_SCALE * pm + PEAK_RS_OFFSET,
                             model_minus_reference_peak_rs=PEAK_RS_SCALE * (pm - ref),
                             regression_slope_peak_rs_per_hbi=PEAK_RS_SCALE * slope))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approved", action="store_true", help="사용자가 동결 문서를 승인한 뒤에만 지정")
    a = ap.parse_args()
    if not a.approved:
        sys.exit("[중단] 사용자 승인 전에는 실행하지 않는다 (--approved).")
    if OUT.exists() and any(OUT.iterdir()):
        sys.exit(f"[중단] {OUT} 에 이미 결과가 있음. ExternalTest6은 1회만 실행한다.")
    torch.set_num_threads(1)
    pre = jload(PREREG)
    hp = pre["hyperparameters"]
    data = load_spectra(UNION_CSV)
    sp = load_splits(SPLIT_SET)[0]
    assert all(s < "S100" for s in sp["train"]) and len(sp["train"]) == 10
    assert sp["test"] == [f"S10{k}" for k in range(1, 7)]
    tr, te = data.select(sp["train"]), data.select(sp["test"])
    models = fit_models(tr, hp, MAIN_MODELS + SUPP)
    ii, jj, subj = test_pairs(SPLIT_SET, sp["split_id"], data)
    X, Y, T, h_in = data.a[ii], data.a[jj], data.hbi[jj], data.hbi[ii]
    assert len(ii) == 540
    df = pd.DataFrame(split_records(models, te, X, Y, T, h_in, subj))
    df.insert(0, "split_id", sp["split_id"])
    slope_rows, slope_checks = residual_slope_rows(models, models["Global Linear"][0], te, X, T, h_in, subj, sp["split_id"])
    sl = pd.DataFrame(slope_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    f = dict(index=False, encoding="utf-8-sig", float_format="%.10g")
    df.to_csv(OUT / "records.csv", **f)
    summary_table(df, False).to_csv(OUT / "summary.csv", **f)
    core = ["general", "repeatability"]
    comparison_table(df, MAIN_COMPARISONS, core).to_csv(OUT / "comparisons_main.csv", **f)
    comparison_table(df, SUPP_COMPARISONS, core).to_csv(OUT / "comparisons_supplementary.csv", **f)
    comparison_table(df, BAND_COMPARISONS, ["dh|"], ["a_rmse", "r_rmse_pp", "peak_rmse"]).to_csv(OUT / "comparisons_delta_hbi.csv", **f)
    comparison_table(df, BAND_COMPARISONS, ["band|"]).to_csv(OUT / "comparisons_band.csv", **f)
    sl.to_csv(OUT / "residual_slope_per_subject.csv", **f)
    residual_slope_summary(sl).to_csv(OUT / "residual_slope_summary.csv", **f)
    peak_reference(te, models).to_csv(OUT / "peak_rs_reference.csv", **f)
    batch = {(name if len(ms) == 1 else f"{name}|seed{k}"): m.predict(X, T) for name, ms in models.items() for k, m in enumerate(ms)}
    preds = [dict(subject=subj[n], input_trial=int(data.trial[ii[n]]), target_trial=int(data.trial[jj[n]]), target_hbi=float(T[n]),
                  predictions={k: v[n].tolist() for k, v in batch.items()}) for n in range(len(ii))]
    jdump(OUT / "predictions.json", dict(general_reconstruction=preds), indent=None)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    jdump(OUT / "config.json", dict(
        split_set=SPLIT_SET, train=sp["train"], test=sp["test"], hyperparameters=hp,
        preregistration=dict(file=str(PREREG.relative_to(ROOT)), sha256=hashlib.sha256(PREREG.read_bytes()).hexdigest()),
        code_commit=commit, main_models=["Raw"] + MAIN_MODELS, supplementary_models=SUPP,
        near_target="0.6 ± 0.02 실측이 S101과 S105에 1개씩뿐이므로 기술 통계로만 보고",
        in_range_subjects=["S101", "S106"],
        paired_comparison="피험자 6명 쌍체 차이. 차이 = 앞 모델 - 뒤 모델, 개선 = 차이 < 0. " + P_NOTE +
                          " n = 6에서 부호검정 양측 p의 최솟값은 0.031.",
        peak_rs_reference="Peak_RS = 8387.2 x Peak + 83.495 (Hwang et al. 2023, Sensors 23(17):7654 식 (2)). "
                          "기준값은 피험자 내 선형회귀의 추정값이며 독립 정답이 아니다. In-range subject가 아니면 외삽.",
        residual_slope_checks=slope_checks, rerun_policy="평가 후 설정을 바꿔 다시 실행하지 않는다"))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
