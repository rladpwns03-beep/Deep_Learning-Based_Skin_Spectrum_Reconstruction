"""분할 세트 5개를 data/splits_final/ 에 산출한다 (모델 학습 없음, 기존 data/splits 는 읽기만 함).

실행:  python -X utf8 -m common.make_splits_final   (code 폴더에서)

  Development10_Holdout73            Development cohort 10명의 7:3 분할 10개 (data_loader.split_plan 그대로)
  Development10_LeaveOneSubjectOut   Development cohort 10명, 1명 test / 9명 train
  Union16_Holdout73                  16명의 11:5 분할 10개 (아래 규칙)
  Union16_LeaveOneSubjectOut         16명, 1명 test / 15명 train
  ExternalTest6                      train = Development cohort 10명, test = S101-S106

Union16_Holdout73 규칙 (모델 결과를 보기 전에 고정)
  - test 5명 = Development cohort 3명 + External cohort 2명
  - test의 Near-target subject 1명 또는 2명, 10개 분할 합계 16명
  - test의 In-range subject 1명 이상 3명 이하, 10개 분할 합계 18-20명
  - 피험자별 test 횟수: Development cohort 각각 정확히 3회, External cohort 3회 또는 4회 (4회는 2명)
  - 후보 순서: Development cohort 3명 조합(사전순) x External cohort 2명 조합(사전순) 순서로 나열한 유효 조합을
    numpy default_rng(0).permutation 으로 섞음. 그 순서로 깊이 우선 탐색 (data_loader._split_plan_cached 와 같은 방식).
  - 조건을 만족하는 조합을 찾지 못하면 규칙을 완화하지 않고 오류로 멈춘다.

분할 폴더마다 train.csv, test.csv, train_pairs.csv, test_pairs.csv, subjects.csv. 세트 루트에 split_plan.csv, manifest.json.
이미 data/splits_final 이 있으면 덮어쓰지 않고 멈춘다.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import NEAR_TOL_MAIN, SPLIT_SEED, TARGET_HBI
from common.data_loader import ROOT, SPECTRA_CSV, WL, Spectra, load_spectra, near_target_subjects, split_plan

OUT = ROOT / "data" / "splits_final"
UNION_CSV = ROOT / "data" / "processed" / "spectra_union16.csv"
WL_COLS = [f"{w}nm" for w in WL]
N_UNION_SPLITS = 10
UNION_TEST_DEV, UNION_TEST_EXT = 3, 2
UNION_NEAR_PER_SPLIT, UNION_NEAR_TOTAL = (1, 2), 16
UNION_INRANGE_PER_SPLIT, UNION_INRANGE_TOTAL = (1, 3), (18, 20)
EXPECTED_VALID_UNION_SETS = 1167


def cohort(s: str) -> str:
    return "external" if s >= "S100" else "development"


def in_range_subjects(data: Spectra) -> list[str]:
    out = []
    for s in data.subjects:
        h = data.hbi[data.subject == s]
        if h.min() <= TARGET_HBI <= h.max():
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Union16_Holdout73
# ---------------------------------------------------------------------------
def union_valid_sets(data: Spectra) -> list[tuple[str, ...]]:
    near, inr = set(near_target_subjects(data)), set(in_range_subjects(data))
    dev = [s for s in data.subjects if cohort(s) == "development"]
    ext = [s for s in data.subjects if cohort(s) == "external"]
    valid = []
    for d in itertools.combinations(dev, UNION_TEST_DEV):
        for e in itertools.combinations(ext, UNION_TEST_EXT):
            t = set(d + e)
            if (UNION_NEAR_PER_SPLIT[0] <= len(t & near) <= UNION_NEAR_PER_SPLIT[1] and
                    UNION_INRANGE_PER_SPLIT[0] <= len(t & inr) <= UNION_INRANGE_PER_SPLIT[1]):
                valid.append(d + e)
    return valid


def union_split_plan(data: Spectra, n_splits: int = N_UNION_SPLITS, seed: int = SPLIT_SEED) -> list[list[str]]:
    near, inr = set(near_target_subjects(data)), set(in_range_subjects(data))
    dev = [s for s in data.subjects if cohort(s) == "development"]
    ext = [s for s in data.subjects if cohort(s) == "external"]
    valid = union_valid_sets(data)
    if len(valid) != EXPECTED_VALID_UNION_SETS:
        raise RuntimeError(f"유효 테스트 조합 {len(valid)}개 (기대 {EXPECTED_VALID_UNION_SETS})")
    dq, dr = divmod(n_splits * UNION_TEST_DEV, len(dev))          # 3, 0
    eq, er = divmod(n_splits * UNION_TEST_EXT, len(ext))          # 3, 2
    if dr != 0:
        raise RuntimeError("Development cohort 테스트 횟수를 똑같이 나눌 수 없음")
    rng = np.random.default_rng(seed)
    cands = [valid[i] for i in rng.permutation(len(valid))]
    n_near = [len(set(c) & near) for c in cands]
    count = {s: 0 for s in data.subjects}
    plan, state = [], dict(near=0)

    def ok():
        left = n_splits - len(plan) - 1                           # 이 후보를 넣은 뒤 남는 분할 수
        if any(count[s] > dq for s in dev):
            return False
        if any(count[s] > eq + 1 for s in ext) or sum(count[s] > eq for s in ext) > er:
            return False
        return (state["near"] + left * UNION_NEAR_PER_SPLIT[0] <= UNION_NEAR_TOTAL <=
                state["near"] + left * UNION_NEAR_PER_SPLIT[1])

    def dfs(start):
        if len(plan) == n_splits:
            return True
        for k in range(start, len(cands)):
            c = cands[k]
            for s in c:
                count[s] += 1
            state["near"] += n_near[k]
            if ok():
                plan.append(c)
                if dfs(k + 1):
                    return True
                plan.pop()
            for s in c:
                count[s] -= 1
            state["near"] -= n_near[k]
        return False

    if not dfs(0):
        raise RuntimeError("Union16_Holdout73: 모든 조건을 만족하는 분할 조합을 찾지 못함 (규칙을 완화하지 않음)")
    total_inr = sum(len(set(c) & inr) for c in plan)
    assert len(set(plan)) == n_splits
    assert UNION_INRANGE_TOTAL[0] <= total_inr <= UNION_INRANGE_TOTAL[1], total_inr
    return [sorted(p) for p in plan]


# ---------------------------------------------------------------------------
# 파일 쓰기
# ---------------------------------------------------------------------------
def pair_table(part: Spectra) -> pd.DataFrame:
    """같은 피험자 안의 모든 순서쌍 i != j (make_pairs 와 같은 순서)."""
    rows = []
    for s in part.subjects:
        ix = np.flatnonzero(part.subject == s)
        for i in ix:
            for j in ix:
                if i != j:
                    rows.append(dict(subject=s, input_trial=int(part.trial[i]), target_trial=int(part.trial[j]),
                                     input_hbi=part.hbi[i], target_hbi=part.hbi[j], delta_hbi=part.hbi[j] - part.hbi[i]))
    return pd.DataFrame(rows)


def measurement_table(raw: pd.DataFrame, data: Spectra, subjects) -> pd.DataFrame:
    """측정 행: subject, trial, hbi, 400nm ... 700nm (반사율 %, 원본 값 그대로)."""
    d = raw[["subject", "trial"] + WL_COLS].copy()
    d.insert(2, "hbi", data.hbi)
    return d[d.subject.isin(list(subjects))].reset_index(drop=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_set(name: str, raw: pd.DataFrame, data: Spectra, splits: list[dict], seven_three: bool, extra_checks: dict,
              notes: list[str], expect: dict) -> dict:
    """splits: [dict(split_id, train, test)]. expect: 분할당 행 수와 쌍 수."""
    root = OUT / name
    near, inr = set(near_target_subjects(data)), set(in_range_subjects(data))
    universe = sorted({s for sp in splits for s in sp["train"] + sp["test"]})
    hashes, plan_rows = {}, []
    chk = dict(no_subject_overlap=True, pairs_within_subject=True, row_counts=True, pair_counts=True)
    for sp in splits:
        d = root / sp["split_id"]
        d.mkdir(parents=True, exist_ok=False)
        train, test = sorted(sp["train"]), sorted(sp["test"])
        tr, te = data.select(train), data.select(test)
        subj = []
        for s in universe:
            h = data.hbi[data.subject == s]
            subj.append(dict(subject=s, role="train" if s in train else "test" if s in test else "unused",
                             cohort=cohort(s), n_trials=len(h), hbi_min=h.min(), hbi_max=h.max(),
                             in_range_subject=s in inr, near_target_subject=s in near))
        tables = dict(train=measurement_table(raw, data, train), test=measurement_table(raw, data, test),
                      train_pairs=pair_table(tr), test_pairs=pair_table(te), subjects=pd.DataFrame(subj))
        for k, t in tables.items():
            t.to_csv(d / f"{k}.csv", index=False, encoding="utf-8")
            hashes[f"{sp['split_id']}/{k}.csv"] = sha256(d / f"{k}.csv")
        chk["no_subject_overlap"] &= not set(train) & set(test)
        chk["pairs_within_subject"] &= (set(tables["train_pairs"].subject) <= set(train) and
                                        set(tables["test_pairs"].subject) <= set(test) and
                                        set(tables["train"].subject) == set(train) and set(tables["test"].subject) == set(test))
        chk["row_counts"] &= (len(tables["train"]), len(tables["test"])) == expect["rows"]
        chk["pair_counts"] &= (len(tables["train_pairs"]), len(tables["test_pairs"])) == expect["pairs"]
        row = dict(split_id=sp["split_id"], train=" ".join(train), test=" ".join(test))
        if seven_three:
            for part, subs in (("test", test), ("train", train)):
                row[f"{part}_n_near_target"] = len(set(subs) & near)
                row[f"{part}_pct_near_target"] = round(100 * len(set(subs) & near) / len(subs), 1)
                row[f"{part}_n_in_range"] = len(set(subs) & inr)
                row[f"{part}_pct_in_range"] = round(100 * len(set(subs) & inr) / len(subs), 1)
        plan_rows.append(row)
    plan = pd.DataFrame(plan_rows)
    plan.to_csv(root / "split_plan.csv", index=False, encoding="utf-8")
    hashes["split_plan.csv"] = sha256(root / "split_plan.csv")
    chk["n_splits"] = len(splits) == expect["n_splits"]
    chk.update(extra_checks)
    chk = {k: bool(v) for k, v in chk.items()}
    counts = pd.Series([s for sp in splits for s in sp["test"]]).value_counts().reindex(universe, fill_value=0)
    manifest = dict(split_set=name, n_splits=len(splits), rows_per_split=dict(train=expect["rows"][0], test=expect["rows"][1]),
                    pairs_per_split=dict(train=expect["pairs"][0], test=expect["pairs"][1]),
                    target_hbi=TARGET_HBI, near_tolerance=NEAR_TOL_MAIN, split_seed=SPLIT_SEED,
                    near_target_subjects=sorted(near & set(universe)), in_range_subjects=sorted(inr & set(universe)),
                    test_count_per_subject={s: int(v) for s, v in counts.items()},
                    checks=chk, all_checks_passed=all(chk.values()), notes=notes, sha256=hashes)
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    if not manifest["all_checks_passed"]:
        raise RuntimeError(f"{name}: 검사 실패 {[k for k, v in chk.items() if not v]}")
    return dict(manifest=manifest, plan=plan)


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"[중단] {OUT} 에 이미 파일이 있음. 덮어쓰지 않음.")
    dev_raw, uni_raw = pd.read_csv(SPECTRA_CSV), pd.read_csv(UNION_CSV)
    dev, uni = load_spectra(SPECTRA_CSV), load_spectra(UNION_CSV)
    assert dev.subject.tolist() == dev_raw.subject.tolist() and uni.subject.tolist() == uni_raw.subject.tolist()
    dev_subjects = dev.subjects
    ext_subjects = [s for s in uni.subjects if cohort(s) == "external"]
    assert len(dev_subjects) == 10 and len(ext_subjects) == 6 and dev_subjects == [s for s in uni.subjects if cohort(s) == "development"]
    assert np.array_equal(uni.select(dev_subjects).a, dev.a)
    inr_dev = set(in_range_subjects(dev))
    results = {}

    # ---- Development10_Holdout73 -------------------------------------------
    tests = split_plan(dev)
    splits = [dict(split_id=f"split_{k:02d}", test=t, train=[s for s in dev_subjects if s not in t]) for k, t in enumerate(tests)]
    old = pd.read_csv(ROOT / "data" / "splits" / "split_plan.csv")
    same_old = [sorted(r.test.split()) for r in old.itertuples()] == [sorted(t) for t in tests]
    no_inr = [sp["split_id"] for sp in splits if not set(sp["test"]) & inr_dev]
    results["Development10_Holdout73"] = write_set(
        "Development10_Holdout73", dev_raw, dev, splits, True,
        dict(test_sets_equal_data_splits_split_plan=same_old,
             near_target_exactly_one_in_test=all(len(set(sp["test"]) & set(near_target_subjects(dev))) == 1 for sp in splits),
             first_split_fixed=sorted(tests[0]) == ["S002", "S008", "S009"]),
        ["기존 코드(common/data_loader.split_plan)의 분할 그대로.",
         f"In-range subject는 층화되어 있지 않다. test에 In-range subject가 없는 분할: {no_inr}."],
        dict(n_splits=10, rows=(70, 30), pairs=(630, 270)))

    # ---- Development10_LeaveOneSubjectOut ----------------------------------
    splits = [dict(split_id=f"fold_{s}", test=[s], train=[x for x in dev_subjects if x != s]) for s in dev_subjects]
    lf = json.loads((ROOT / "results" / "loso" / "config" / "loso_folds.json").read_text(encoding="utf-8"))["folds"]
    same_folds = [(f["fold_id"], f["test"], f["train"]) for f in lf] == [(sp["split_id"], sp["test"], sp["train"]) for sp in splits]
    same_pairs = all([(p["input_trial"], p["target_trial"]) for p in f["test_pairs"]] ==
                     list(zip(*[pair_table(dev.select(f["test"]))[c].tolist() for c in ("input_trial", "target_trial")]))
                     for f in lf)
    results["Development10_LeaveOneSubjectOut"] = write_set(
        "Development10_LeaveOneSubjectOut", dev_raw, dev, splits, False,
        dict(folds_equal_results_loso_folds=same_folds, test_pairs_equal_results_loso_folds=same_pairs,
             each_subject_tested_once=sorted(s for sp in splits for s in sp["test"]) == dev_subjects),
        ["results/loso/config/loso_folds.json 과 같은 fold, 같은 테스트 쌍 순서."],
        dict(n_splits=10, rows=(90, 10), pairs=(810, 90)))

    # ---- Union16_Holdout73 -------------------------------------------------
    tests = union_split_plan(uni)
    near_u, inr_u = set(near_target_subjects(uni)), set(in_range_subjects(uni))
    splits = [dict(split_id=f"split_{k:02d}", test=t, train=[s for s in uni.subjects if s not in t]) for k, t in enumerate(tests)]
    cnt = pd.Series([s for t in tests for s in t]).value_counts()
    n_near = [len(set(t) & near_u) for t in tests]
    n_inr = [len(set(t) & inr_u) for t in tests]
    results["Union16_Holdout73"] = write_set(
        "Union16_Holdout73", uni_raw, uni, splits, True,
        dict(cohort_composition_3_plus_2=all(sum(cohort(s) == "development" for s in t) == 3 and len(t) == 5 for t in tests),
             near_target_1_or_2_per_split=all(n in (1, 2) for n in n_near), near_target_total_16=sum(n_near) == 16,
             in_range_1_to_3_per_split=all(1 <= n <= 3 for n in n_inr), in_range_total_18_to_20=18 <= sum(n_inr) <= 20,
             development_each_tested_3=all(cnt.get(s, 0) == 3 for s in dev_subjects),
             external_tested_3_or_4_two_at_4=sorted(int(cnt.get(s, 0)) for s in ext_subjects) == [3, 3, 3, 3, 4, 4],
             splits_all_different=len({tuple(t) for t in tests}) == len(tests)),
        ["test 5명 = Development cohort 3명 + External cohort 2명. 고정된 첫 분할 없음, 난수 시드 0.",
         f"유효 테스트 조합 {EXPECTED_VALID_UNION_SETS}개를 (Development 3명 조합 사전순) x (External 2명 조합 사전순)으로 나열한 뒤 "
         "numpy default_rng(0).permutation 순서로 깊이 우선 탐색.",
         "기존 코드의 'test에 Near-target subject 정확히 1명' 규칙은 쓰지 않음."],
        dict(n_splits=10, rows=(110, 50), pairs=(990, 450)))

    # ---- Union16_LeaveOneSubjectOut ----------------------------------------
    splits = [dict(split_id=f"fold_{s}", test=[s], train=[x for x in uni.subjects if x != s]) for s in uni.subjects]
    results["Union16_LeaveOneSubjectOut"] = write_set(
        "Union16_LeaveOneSubjectOut", uni_raw, uni, splits, False,
        dict(each_subject_tested_once=sorted(s for sp in splits for s in sp["test"]) == uni.subjects),
        [], dict(n_splits=16, rows=(150, 10), pairs=(1350, 90)))

    # ---- ExternalTest6 -----------------------------------------------------
    results["ExternalTest6"] = write_set(
        "ExternalTest6", uni_raw, uni, [dict(split_id="external_test", train=dev_subjects, test=ext_subjects)], False,
        dict(train_is_development_cohort=True, test_is_external_cohort=ext_subjects == [f"S10{k}" for k in range(1, 7)]),
        ["학습 = Development cohort 10명 전체, 평가 = External cohort 6명. 최종 평가 1회."],
        dict(n_splits=1, rows=(100, 60), pairs=(900, 540)))

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for name, r in results.items():
        m = r["manifest"]
        print(f"\n== {name}: 분할 {m['n_splits']}개, 행 {m['rows_per_split']}, 쌍 {m['pairs_per_split']}, "
              f"검사 통과 {m['all_checks_passed']}")
        print("   검사:", m["checks"])
        if "test_pct_near_target" in r["plan"]:
            print(r["plan"].drop(columns=["train"]).to_string(index=False))
            print("   피험자별 테스트 횟수:", m["test_count_per_subject"])


if __name__ == "__main__":
    main()
