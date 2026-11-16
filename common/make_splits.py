"""Experiment 1, 2에 쓸 데이터 분할을 만들어 data/splits/ 에 저장한다 (모델 학습 없음).

실행:  python -m common.make_splits   (code 폴더에서)

분할 원칙 (common/data_loader.py, 설정은 common/config.py)
  - 피험자 단위 7:3 제약 무작위 분할을 서로 다른 10개 (rep00 ~ rep09)
  - 각 rep 안에서 세 모델(B1, B2, BiGRU)은 같은 학습/테스트 데이터를 쓴다
  - 정답(Ground Truth)은 항상 같은 피험자의 '실제 측정 스펙트럼'
  - 0.6 근처 = |HbI - 0.6| <= 0.02 (주 분석) / 0.03 (보조 분석)

출력
  data/splits/split_plan.csv                    rep별 테스트 피험자와 0.6 근처 실측 수 요약
  data/splits/rep00 ~ rep09/
    subjects.csv                    피험자별 역할(train/test)과 HbI 범위
    measurements.csv                측정 100개 각각의 역할, HbI, 0.6 근처 여부
    train_pairs.csv                 학습용 측정 쌍 (학습 피험자 안의 i -> j, 7 x 90 = 630)
    exp1_test_pairs.csv             Exp 1: 테스트 피험자 안의 모든 i -> j (3 x 90 = 270), 정답 = 실제 측정 j
    exp2_accuracy_pairs_tol0.02.csv Exp 2 정확도 (주):   정답 = |HbI-0.6|<=0.02 테스트 측정
    exp2_accuracy_pairs_tol0.03.csv Exp 2 정확도 (보조): 정답 = |HbI-0.6|<=0.03 테스트 측정
    exp2_repeatability_inputs.csv   Exp 2 반복성: 테스트 측정 30개 전부를 HbI 0.6으로 복원할 입력 목록
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import NEAR_TOL_MAIN, NEAR_TOL_SUPP, SPLIT_SEED, TARGET_HBI
from common.data_loader import Spectra, load_spectra, near_target_subjects, repeated_splits

OUT = Path(__file__).resolve().parents[1] / "data" / "splits"


def pair_table(part: Spectra, role: str) -> pd.DataFrame:
    """같은 피험자 안의 모든 순서쌍 i != j. 정답 = 실제 측정 j."""
    rows = []
    for s in part.subjects:
        ix = np.flatnonzero(part.subject == s)
        for i in ix:
            for j in ix:
                if i != j:
                    hi, hj = part.hbi[i], part.hbi[j]
                    rows.append(dict(role=role, subject=s, input_trial=part.trial[i], target_trial=part.trial[j],
                                     input_hbi=hi, target_hbi=hj, delta_hbi=hj - hi))
    return pd.DataFrame(rows)


def accuracy_pairs(exp1: pd.DataFrame, meas: pd.DataFrame, tol: float) -> pd.DataFrame:
    """Exp 2 정확도: 정답 = |HbI-0.6|<=tol 인 테스트 측정, 입력 = 같은 피험자의 나머지 측정.
    복원 목표는 정답 측정의 실제 HbI (예: 0.5989); 0.6과의 차이는 target_minus_06 열에 기록."""
    gt = meas[(meas.role == "test") & ((meas.hbi - TARGET_HBI).abs() <= tol)]
    acc = exp1.merge(gt[["subject", "trial"]].rename(columns={"trial": "target_trial"}), on=["subject", "target_trial"])
    acc["target_minus_06"] = acc.target_hbi - TARGET_HBI
    acc["input_is_near_06"] = (acc.input_hbi - TARGET_HBI).abs() <= tol
    return acc


def write_rep(data: Spectra, rep: int, train: Spectra, test: Spectra, out: Path) -> dict:
    near_subj = set(near_target_subjects(data))
    subj = []
    for s in data.subjects:
        h = data.hbi[data.subject == s]
        subj.append(dict(subject=s, role="test" if s in test.subjects else "train", n_trials=len(h),
                         hbi_min=h.min(), hbi_max=h.max(),
                         target_in_range=bool(h.min() <= TARGET_HBI <= h.max()), has_near_06=s in near_subj))
    subj = pd.DataFrame(subj)
    meas = pd.DataFrame(dict(subject=data.subject, trial=data.trial, hbi=data.hbi))
    meas["role"] = np.where(meas.subject.isin(test.subjects), "test", "train")
    meas["near_06_main"] = (meas.hbi - TARGET_HBI).abs() <= NEAR_TOL_MAIN
    meas["near_06_supp"] = (meas.hbi - TARGET_HBI).abs() <= NEAR_TOL_SUPP

    train_pairs = pair_table(train, "train")
    exp1 = pair_table(test, "test")
    acc = {tol: accuracy_pairs(exp1, meas, tol) for tol in (NEAR_TOL_MAIN, NEAR_TOL_SUPP)}

    repeat = meas[meas.role == "test"][["subject", "trial", "hbi"]].rename(columns={"hbi": "input_hbi"}).copy()
    repeat["target_hbi"] = TARGET_HBI
    repeat["delta_hbi"] = TARGET_HBI - repeat.input_hbi
    repeat["target_in_subject_range"] = repeat.subject.map(subj.set_index("subject").target_in_range)

    out.mkdir(parents=True, exist_ok=True)
    files = [("subjects", subj), ("measurements", meas), ("train_pairs", train_pairs),
             ("exp1_test_pairs", exp1), ("exp2_repeatability_inputs", repeat)]
    files += [(f"exp2_accuracy_pairs_tol{tol:.2f}", df) for tol, df in acc.items()]
    for name, df in files:
        df.to_csv(out / f"{name}.csv", index=False, float_format="%.6f")

    tm = meas[meas.role == "train"]
    gt = lambda tol: meas[(meas.role == "test") & ((meas.hbi - TARGET_HBI).abs() <= tol)]
    return dict(rep=rep, test=" ".join(test.subjects), train=" ".join(train.subjects),
                train_near_main=int(tm.near_06_main.sum()), train_near_supp=int(tm.near_06_supp.sum()),
                test_gt_main=len(gt(NEAR_TOL_MAIN)), test_gt_main_subjects=gt(NEAR_TOL_MAIN).subject.nunique(),
                test_gt_supp=len(gt(NEAR_TOL_SUPP)), test_gt_supp_subjects=gt(NEAR_TOL_SUPP).subject.nunique(),
                acc_pairs_main=len(acc[NEAR_TOL_MAIN]), acc_pairs_supp=len(acc[NEAR_TOL_SUPP]),
                test_06_in_range=" ".join(sorted(subj[(subj.role == "test") & subj.target_in_range].subject)),
                n_train_pairs=len(train_pairs), n_exp1_pairs=len(exp1), n_repeat_inputs=len(repeat))


def main():
    if OUT.exists():                      # 이전 버전(단일 분할) 파일 정리: 이 스크립트가 만든 폴더
        shutil.rmtree(OUT)
    data = load_spectra()
    rows = [write_rep(data, rep, tr, te, OUT / f"rep{rep:02d}") for rep, tr, te in repeated_splits(data)]
    plan = pd.DataFrame(rows)
    plan.to_csv(OUT / "split_plan.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"분할 seed {SPLIT_SEED} | 0.6 근처: 주 ±{NEAR_TOL_MAIN}, 보조 ±{NEAR_TOL_SUPP}")
    print(plan[["rep", "test", "train_near_main", "test_gt_main", "test_gt_supp", "test_gt_supp_subjects",
                "test_06_in_range"]].to_string(index=False))
    cnt = pd.Series(" ".join(plan.test).split()).value_counts().reindex(data.subjects, fill_value=0)
    print("\n피험자별 테스트 횟수:", dict(cnt))
    print(f"rep마다: 학습 쌍 {plan.n_train_pairs[0]}, Exp1 테스트 쌍 {plan.n_exp1_pairs[0]}, "
          f"Exp2 반복성 입력 {plan.n_repeat_inputs[0]}")


if __name__ == "__main__":
    main()
