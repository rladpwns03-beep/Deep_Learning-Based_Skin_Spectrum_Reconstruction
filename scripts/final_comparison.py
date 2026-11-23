"""주 비교 (Development cohort만 사용): General reconstruction, Near-target accuracy, Fixed-target repeatability,
HbI 변화 크기 구간별 오차, Peak-constrained selection 평가, 그리고 Development cohort 10명 전체의 PLS/Ridge 선택.

실행 (code 폴더, conda 환경 skin_spectra):
    python -X utf8 scripts/final_comparison.py                              # 두 분할 세트 (이미 결과가 있으면 그 세트는 중단)
    python -X utf8 scripts/final_comparison.py --all-subjects-selection     # 10명 전체에서 PLS 성분 수, Ridge alpha 선택 (ExternalTest6용)

설계 (결과를 보기 전에 고정)
  - 분할 세트: Development10_LeaveOneSubjectOut (주 프로토콜), Development10_Holdout73 (보조 프로토콜).
  - 모델은 저장된 것을 불러와 쓴다 (재학습 없음). 테스트 쌍, 목표 HbI, 정답이 기존 저장 예측과 같은지,
    불러온 모델의 예측이 저장 예측과 같은지 검사한다.
  - 집계: LeaveOneSubjectOut = 피험자 10명의 평균 ± 표준편차.
          Holdout73 = 피험자 -> 분할 안 평균 -> 분할 10개의 평균 ± 표준편차 (분할끼리 피험자가 겹치므로 분할 간 표준편차를
          독립 표본의 오차로 해석하지 않는다).
  - 쌍체 비교: 분석 단위 = 피험자 (Holdout73은 피험자별로 분할 평균을 낸 뒤 n = 10). 차이 = 앞 모델 - 뒤 모델, 개선 = 차이 < 0.
    Peak bias는 |bias| 감소를 개선으로 본다. p 값은 다중 비교 보정이 없는 기술적 참고값.
    Holdout73에서는 주 표의 평균(분할 평균의 평균)과 쌍체 표의 평균(피험자 평균)이 다를 수 있다.
  - Near-target accuracy 허용 폭 0.02: 대상 피험자가 적어 기술 통계만. 허용 폭 0.03: 쌍체 차이를 보조로 보고.
  - HbI 변화 크기 구간: 해당 구간에 쌍이 없는 피험자는 그 구간의 n에서 빠진다.
  - 판정 문구는 넣지 않는다 (평균 차이, 표준편차, 표준오차, k/n, p 값만).

저장: results/final_comparison/<분할 세트>/  records.csv, summary.csv, summary_by_subject.csv,
      comparisons_main.csv, comparisons_supplementary.csv, comparisons_delta_hbi.csv, metrics.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))
from final_utils import (BAND_COMPARISONS, DEV_CSV, MAIN_COMPARISONS, P_NOTE, RES, SUPP_COMPARISONS, jdump, jload,   # noqa: E402
                         load_splits, load_spectra, test_pairs)
from final_eval import SCHEME_DIR, comparison_table, load_dev_models, split_records, summary_table   # noqa: E402
from models.b2_pls import B2PLSModel                                           # noqa: E402
from models.b2_ridge import B2RidgeModel                                       # noqa: E402
from models.pls import PLSModel                                                # noqa: E402

OUT = RES / "final_comparison"
SPLIT_SETS = ["Development10_LeaveOneSubjectOut", "Development10_Holdout73"]
STORED_KEYS = {"HbI-binned mean": ["B1"], "Global Linear": ["B2"], "BiGRU (fixed setting)": [f"BiGRU_seed{k}" for k in (0, 1, 2)]}
PLS_DIRS = {"PLS_v2": "pls_v2_k10", "PLS_v1": "pls_v1_k10", "Ridge": "ridge",
            "PLS_v2 (max 5 components)": "pls_v2", "PLS_v1 (max 5 components)": "pls_v1"}
EXPECTED_NEAR_003 = ["S001", "S002", "S003", "S004", "S005", "S008", "S009", "S010"]


def stored_predictions(split_set):
    """기존 저장 예측: {(split, subject, input_trial, target_trial): dict(actual, target_hbi, {모델: [seed별 예측]})}"""
    sc = SCHEME_DIR[split_set]
    path, key = (RES / "holdout/predictions/predictions.json", "split_id") if sc == "holdout" else \
        (RES / "loso/predictions/loso_predictions.json", "fold_id")
    out = {}
    for r in jload(path)["general_reconstruction"]:
        k = (r[key], r["subject"], r["input_trial"], r["target_trial"])
        out[k] = dict(actual=r["actual"], target_hbi=r["target_hbi"],
                      pred={n: [r["predictions"][c] for c in cols] for n, cols in STORED_KEYS.items()})
    for n, d in PLS_DIRS.items():
        for r in jload(RES / d / sc / "predictions.json")["general_reconstruction"]:
            out[(r["split"], r["subject"], r["input_trial"], r["target_trial"])]["pred"][n] = [r["pred"]]
    for n, sub in (("BiGRU", ""), ("BiGRU (Peak-constrained selection)", "peak_constrained")):
        for r in jload(RES / "bigru_tuned" / split_set / sub / "predictions.json")["general_reconstruction"]:
            out[(r["split_id"], r["subject"], r["input_trial"], r["target_trial"])]["pred"][n] = \
                [r["predictions"][f"seed{k}"] for k in (0, 1, 2)]
    return out


def run(split_set):
    out = OUT / split_set
    if out.exists() and any(out.iterdir()):
        print(f"[건너뜀] {out} 에 이미 결과가 있음. 덮어쓰지 않음.")
        return
    data = load_spectra(DEV_CSV)
    stored = stored_predictions(split_set)
    frames, max_diff = [], {}
    for sp in load_splits(split_set):
        sid = sp["split_id"]
        models = load_dev_models(split_set, sid)
        te = data.select(sp["test"])
        ii, jj, subj = test_pairs(split_set, sid, data)
        X, Y, T = data.a[ii], data.a[jj], data.hbi[jj]
        for n_, ms in models.items():
            assert all(not set(m.train_subjects) & set(sp["test"]) and sorted(m.train_subjects) == sorted(sp["train"]) for m in ms), \
                f"{sid} {n_}: 저장 모델의 학습 피험자가 분할과 다름"
        # 저장 예측과 같은 쌍·정답·목표인지, 불러온 모델의 예측이 저장 예측과 같은지
        st = [stored[(sid, s, int(data.trial[i]), int(data.trial[j]))] for s, i, j in zip(subj, ii, jj)]
        assert np.allclose(Y, np.array([r["actual"] for r in st]), atol=1e-12)
        assert np.allclose(T, np.array([r["target_hbi"] for r in st]), atol=1e-12)
        for n_, ms in models.items():
            for k, m in enumerate(ms):
                d = float(np.abs(m.predict(X, T) - np.array([r["pred"][n_][k] for r in st])).max())
                max_diff[n_] = max(max_diff.get(n_, 0.0), d)
        f = pd.DataFrame(split_records(models, te, X, Y, T, data.hbi[ii], subj))
        f.insert(0, "split_id", sid)
        frames.append(f)
        print(f"[{split_set} {sid}] test {sp['test']} 완료", flush=True)
    if max(max_diff.values()) > 1e-9:
        raise RuntimeError(f"불러온 모델의 예측이 저장 예측과 다름: {max_diff}")
    df = pd.concat(frames, ignore_index=True)
    by_split = SCHEME_DIR[split_set] == "holdout"
    near003 = sorted(df[df.section == "near_0.03"].subject.unique())
    near002 = sorted(df[df.section == "near_0.02"].subject.unique())

    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "records.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    summary_table(df, by_split).to_csv(out / "summary.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    summary_table(df, False).to_csv(out / "summary_by_subject.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    core = ["general", "near_0.03", "repeatability"]            # repeatability_in_range / _other 포함 (접두 일치)
    comparison_table(df, MAIN_COMPARISONS, core).to_csv(out / "comparisons_main.csv", index=False, encoding="utf-8-sig",
                                                        float_format="%.10g")
    comparison_table(df, SUPP_COMPARISONS, core).to_csv(out / "comparisons_supplementary.csv", index=False,
                                                        encoding="utf-8-sig", float_format="%.10g")
    comparison_table(df, BAND_COMPARISONS, ["dh|"], ["a_rmse", "r_rmse_pp", "peak_rmse"]).to_csv(
        out / "comparisons_delta_hbi.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    jdump(out / "metrics.json", dict(
        split_set=split_set, data=str(DEV_CSV.relative_to(ROOT)), external_cohort_used=False,
        aggregation=("피험자 -> 분할 안 평균 -> 분할 10개의 평균 ± 표준편차. 분할끼리 피험자가 겹치므로 분할 간 표준편차를 "
                     "독립 표본의 오차로 해석하지 않는다." if by_split else "피험자 10명의 평균 ± 표준편차 (ddof=1)."),
        paired_comparison=("분석 단위 = 피험자" + (" (피험자별로 분할 평균을 낸 뒤 n = 10; 주 표의 평균과 쌍체 표의 평균이 다를 수 있음)"
                                               if by_split else "") +
                           ". 차이 = 앞 모델 - 뒤 모델, 개선 = 차이 < 0. peak_bias와 bias는 |값| 감소를 개선으로 판정. " + P_NOTE),
        units="a_rmse, peak_rmse, peak_bias, peak_sd, spectral_sd: absorbance / r_rmse_pp: 반사율 %p",
        bigru_subject_value="seed 0, 1, 2 지표의 평균",
        near_target=dict(tol_0p02_subjects=near002, tol_0p03_subjects=near003,
                         tol_0p03_expected=EXPECTED_NEAR_003, tol_0p03_as_expected=near003 == EXPECTED_NEAR_003,
                         note="허용 폭 0.02는 기술 통계만. 허용 폭 0.03은 쌍체 차이를 보조로 보고."),
        delta_hbi_note="해당 구간에 쌍이 없는 피험자는 그 구간의 n에서 빠진다.",
        checks=dict(pairs_actual_target_equal_stored=True, loaded_models_match_stored_predictions_max_abs_diff=max_diff,
                    band_sse_recombines_to_total_rmse=True, models_trained_without_test_subjects=True),
        n_records=int(len(df))))
    print(f"저장: {out} | near 0.03 대상 {near003} (기대와 같음: {near003 == EXPECTED_NEAR_003})")


def all_subjects_selection():
    path = OUT / "Development10_AllSubjects_selection.json"
    if path.exists():
        sys.exit(f"[중단] {path} 이 이미 있음. 덮어쓰지 않음.")
    tr = load_spectra(DEV_CSV)
    assert len(tr.subjects) == 10
    v2, v1, rg = B2PLSModel(max_components=10).fit(tr), PLSModel(max_components=10).fit(tr), B2RidgeModel().fit(tr)
    res = dict(
        unit="Development10_AllSubjects", train=tr.subjects, external_cohort_used=False,
        method="학습 피험자(10명) 안 Leave-One-Subject-Out, pls_compare.py 의 pls_v2_k10 / pls_v1_k10 / ridge 와 같은 절차",
        PLS_v2=dict(n_components=v2.n_components, max_components=10, at_upper_limit=v2.n_components == max(v2.cv["components"]),
                    grid=v2.cv["components"], val_a_rmse=v2.cv["rmse"]),
        PLS_v1=dict(n_components=v1.n_components, max_components=10, at_upper_limit=v1.n_components == max(v1.cv["components"]),
                    grid=v1.cv["components"], val_a_rmse=np.sqrt(v1.cv["mean_mse"]).tolist()),
        Ridge=dict(alpha=rg.alpha, grid=rg.cv["alphas"], val_a_rmse=rg.cv["rmse"], extended=rg.cv["extended"],
                   at_boundary_after_extension=rg.cv["at_boundary_after_extension"]),
        expected=dict(PLS_v2=6, PLS_v1=6), as_expected=bool(v2.n_components == 6 and v1.n_components == 6))
    jdump(path, res)
    print(f"PLS_v2 성분 수 {v2.n_components}, PLS_v1 성분 수 {v1.n_components}, Ridge alpha {rg.alpha:g} | 기대(6, 6)와 같음: {res['as_expected']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-set", choices=SPLIT_SETS, nargs="*", default=SPLIT_SETS)
    ap.add_argument("--all-subjects-selection", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.all_subjects_selection:
        all_subjects_selection()
        return
    for s in a.split_set:
        run(s)


if __name__ == "__main__":
    main()
