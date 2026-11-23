"""Union16 분할 (보조, 고정 설정): 16명 Leave-One-Subject-Out과 11:5 분할에서 모든 모델을 다시 학습. Phase 6 뒤에만 실행.

실행 (code 폴더, conda 환경 skin_spectra):
    python -X utf8 scripts/run_union16.py --approved --workers 8     # results/union16/ (이미 결과가 있으면 그 세트는 건너뜀)

  - 하이퍼파라미터는 docs/Preregistration_ExternalTest_v1.json 의 값으로 고정 (PLS는 n_components를 명시 전달, 다시 고르지 않음).
  - 규칙 3의 의도된 예외: 고정값은 Development cohort 10명 전체로 고른 값이므로, Development cohort 피험자가 test인 분할에서는
    그 피험자가 하이퍼파라미터 선택에 쓰였다. External cohort 피험자가 test인 분할에는 이 문제가 없다. Union16은 보조 결과다.
  - 학습 피험자 수 효과 (쌍체 차이, 같은 고정값끼리 비교)
      External cohort 6명:     10명 학습 (ExternalTest6) 대 15명 학습 (Union16_LeaveOneSubjectOut 의 해당 fold)
      Development cohort 10명:  9명 학습 (Development10_LeaveOneSubjectOut, 같은 고정값으로 다시 학습) 대 15명 학습
      참고 행: 9명 학습을 fold별 튜닝값으로 한 기존 결과 (results/final_comparison) 대 15명 학습
    이 결과는 ExternalTest6과 독립이 아니므로 일반화 성능 추정치가 아니라 데이터 증가 효과의 추정으로만 표기한다.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))
from final_utils import (MAIN_COMPARISONS, MAIN_MODELS, P_NOTE, RES, UNION_CSV, jdump, jload, load_splits, load_spectra,   # noqa: E402
                         paired_diff, test_pairs)
from final_eval import comparison_table, fit_models, split_records, subject_values, summary_table   # noqa: E402

OUT = RES / "union16"
PREREG = ROOT / "docs" / "Preregistration_ExternalTest_v1.json"
SUPP = ["PLS_v1", "Ridge", "BiGRU (fixed setting)", "BiGRU (Peak-constrained selection)"]
SUPP_COMPARISONS = [(m, "Global Linear") for m in SUPP] + [("BiGRU (Peak-constrained selection)", "BiGRU")]
MODELS = MAIN_MODELS + SUPP
# (출력 폴더, 테스트 쌍을 읽을 분할 세트)
SETS = [("Union16_LeaveOneSubjectOut", "Union16_LeaveOneSubjectOut"), ("Union16_Holdout73", "Union16_Holdout73"),
        ("Development10_LeaveOneSubjectOut_FixedHyperparameters", "Development10_LeaveOneSubjectOut")]
RULE3_NOTE = ("규칙 3의 의도된 예외: 고정 하이퍼파라미터는 Development cohort 10명 전체로 고른 값이므로 Development cohort 피험자가 "
              "test인 분할에서는 그 피험자가 선택에 쓰였다. External cohort 피험자가 test인 분할에는 이 문제가 없다.")
_STATE = {}


def _init(hp):
    torch.set_num_threads(1)
    _STATE.update(data=load_spectra(UNION_CSV), hp=hp)


def _work(job):
    split_set, sp = job
    data, hp = _STATE["data"], _STATE["hp"]
    assert not set(sp["train"]) & set(sp["test"])
    models = fit_models(data.select(sp["train"]), hp, MODELS)
    ii, jj, subj = test_pairs(split_set, sp["split_id"], data)
    recs = split_records(models, data.select(sp["test"]), data.a[ii], data.a[jj], data.hbi[jj], data.hbi[ii], subj)
    for r in recs:
        r["split_id"] = sp["split_id"]
    return recs


def run_set(name, split_set, hp, workers):
    out = OUT / name
    if out.exists() and any(out.iterdir()):
        print(f"[건너뜀] {out} 에 이미 결과가 있음.")
        return pd.read_csv(out / "records.csv")
    jobs = [(split_set, sp) for sp in load_splits(split_set)]
    if workers <= 1:
        _init(hp)
        res = [_work(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(hp,)) as pool:
            res = pool.map(_work, jobs, chunksize=1)
    df = pd.DataFrame([r for rs in res for r in rs])
    by_split = name.endswith("Holdout73")
    out.mkdir(parents=True, exist_ok=True)
    f = dict(index=False, encoding="utf-8-sig", float_format="%.10g")
    df.to_csv(out / "records.csv", **f)
    summary_table(df, by_split).to_csv(out / "summary.csv", **f)
    summary_table(df, False).to_csv(out / "summary_by_subject.csv", **f)
    core = ["general", "near_0.03", "repeatability"]
    comparison_table(df, MAIN_COMPARISONS, core).to_csv(out / "comparisons_main.csv", **f)
    comparison_table(df, SUPP_COMPARISONS, core).to_csv(out / "comparisons_supplementary.csv", **f)
    jdump(out / "config.json", dict(
        name=name, split_set=split_set, hyperparameters=hp, models=MODELS, status="보조 결과 (고정 설정)",
        rule3_exception=RULE3_NOTE,
        aggregation="피험자 -> 분할 안 평균 -> 분할 평균 ± 표준편차" if by_split else "피험자 평균 ± 표준편차",
        paired_comparison="분석 단위 = 피험자. 차이 = 앞 모델 - 뒤 모델, 개선 = 차이 < 0. " + P_NOTE))
    print(f"저장: {out}")
    return df


def size_effect(u16, dev9_fixed):
    """학습 피험자 수 효과: '15명 학습 minus 더 적은 학습'의 피험자 쌍체 차이 (General reconstruction)."""
    def vals(df, model, metric):
        return subject_values(df[(df.section == "general") & (df.metric == metric)]).loc[("general", metric, model)].to_dict()

    ext10 = pd.read_csv(RES / "external_test" / "records.csv")
    dev9_tuned = pd.read_csv(RES / "final_comparison" / "Development10_LeaveOneSubjectOut" / "records.csv")
    rows = []
    cases = [("External cohort 6명", "15 train (Union16) minus 10 train (ExternalTest6)", ext10, "주 비교 (같은 고정값)"),
             ("Development cohort 10명", "15 train (Union16) minus 9 train (same fixed hyperparameters)", dev9_fixed, "주 비교 (같은 고정값)"),
             ("Development cohort 10명", "15 train (Union16) minus 9 train (fold-wise tuned)", dev9_tuned, "참고 행 (설정 차이가 섞임)")]
    for cohort, label, small, role in cases:
        for model in MODELS:
            for metric in ("a_rmse", "r_rmse_pp", "peak_rmse"):
                b = vals(small, model, metric)
                a = {s: v for s, v in vals(u16, model, metric).items() if s in b}
                p = paired_diff(a, b)
                p.pop("per_subject_diff"); p.pop("subjects")
                rows.append(dict(cohort=cohort, role=role, model=model, metric=metric, comparison=label, **p))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approved", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if not a.approved or not (RES / "external_test" / "records.csv").exists():
        sys.exit("[중단] Phase 6 (ExternalTest6) 이후, 사용자 승인 뒤에만 실행한다 (--approved).")
    hp = jload(PREREG)["hyperparameters"]
    dfs = {name: run_set(name, split_set, hp, a.workers) for name, split_set in SETS}
    eff = size_effect(dfs["Union16_LeaveOneSubjectOut"], dfs["Development10_LeaveOneSubjectOut_FixedHyperparameters"])
    eff.to_csv(OUT / "training_size_effect.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    jdump(OUT / "training_size_effect_note.json", dict(
        note="ExternalTest6과 독립이 아니므로 일반화 성능 추정치가 아니라 데이터 증가 효과의 추정으로만 표기한다.",
        rule3_exception=RULE3_NOTE, paired="차이 = 15명 학습 - 더 적은 학습, 개선 = 차이 < 0. " + P_NOTE))
    print(eff[eff.metric == "a_rmse"][["cohort", "role", "model", "n", "diff_mean", "diff_se", "n_improved"]].to_string(index=False))


if __name__ == "__main__":
    main()
