"""세 모델(평균 기반, 선형, BiGRU)이 공유하는 데이터 불러오기와 분할.

비교가 공정하려면 모든 모델이 이 파일의 함수만 써서 데이터를 받아야 한다.
  - load_spectra()     : data/processed/spectra.csv -> 흡광도 A = -log10(R/100), HbI = A570
  - holdout_split()    : 피험자 단위 7:3 분할 (rep = 0..N_SPLITS-1, 서로 다른 분할 10개)
  - repeated_splits()  : 위 분할 10개를 차례로 반환
  - make_pairs()       : 같은 피험자 안의 측정 쌍 i -> j (평가 정답 = 실제 측정 j)

분할 단위는 항상 '피험자'다. 같은 사람의 측정이 학습과 테스트에 나뉘어 들어가지 않는다.
"""
from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import (FIRST_SPLIT_TEST, N_SPLITS, N_TEST, N_TEST_NEAR, NEAR_TOL_MAIN, SPLIT_SEED,
                           TARGET_HBI)

ROOT = Path(__file__).resolve().parents[1]
SPECTRA_CSV = ROOT / "data" / "processed" / "spectra.csv"

WL = np.arange(400, 701, 10)                       # 31개 파장
HBI_INDEX = int(np.flatnonzero(WL == 570)[0])      # HbI = A570
FREE = WL != 570                                   # 570 nm는 목표값으로 고정되므로 평가에서 제외
NEAR_TOL = NEAR_TOL_MAIN

# ---------------------------------------------------------------------------
# 7:3 제약 무작위 분할 10회 (2026-11-12 결정, 모델 결과를 보기 전에 고정; 설정은 common/config.py)
#   '0.6 근처 피험자' = |HbI - 0.6| <= 0.02 인 측정이 하나라도 있는 피험자 (현재 S003, S004, S008)
#   각 분할 규칙: 0.6 근처 피험자 1명 -> test, 2명 -> train
#                 나머지 7명 중 2명 -> test, 5명 -> train
#   -> train 7명 / test 3명, 양쪽 모두 0.6 근처 실측이 반드시 포함된다.
#   첫 번째 분할 = 기존 분할 (S002, S008, S009). 10개 분할은 서로 모두 다르고,
#   피험자별 test 참여 횟수가 최대한 균형을 이룬다 (split_plan 참고). 분할은 모델 성능을 보고 바꾸지 않는다.
# 분할은 train / test 두 개뿐이다 (별도 검증 세트 없음).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spectra:
    a: np.ndarray            # [N, 31] 흡광도
    subject: np.ndarray      # [N] 피험자 ID
    trial: np.ndarray        # [N] 측정 번호

    @property
    def hbi(self) -> np.ndarray:
        return self.a[:, HBI_INDEX]

    @property
    def subjects(self) -> list[str]:
        return sorted(set(self.subject.tolist()))

    def select(self, subjects) -> "Spectra":
        m = np.isin(self.subject, list(subjects))
        return Spectra(self.a[m], self.subject[m], self.trial[m])

    def __len__(self):
        return len(self.a)


def load_spectra(path: Path = SPECTRA_CSV) -> Spectra:
    d = pd.read_csv(path)
    r = d[[f"{w}nm" for w in WL]].to_numpy(float) / 100.0
    if not ((r > 0) & (r <= 1)).all():
        raise ValueError("반사율이 (0, 100%] 범위를 벗어남")
    return Spectra(-np.log10(r), d["subject"].to_numpy(), d["trial"].to_numpy(int))


def _check_disjoint(*groups):
    seen = set()
    for g in groups:
        g = set(g)
        if seen & g:
            raise AssertionError(f"피험자 중복: {sorted(seen & g)}")
        seen |= g


def near_target_subjects(data: Spectra, tol: float = NEAR_TOL) -> list[str]:
    """|HbI - 0.6| <= tol 인 측정이 하나라도 있는 피험자."""
    return sorted({s for s, h in zip(data.subject, data.hbi) if abs(h - TARGET_HBI) <= tol})


def _valid_test_sets(data: Spectra) -> list[tuple[str, ...]]:
    """제약을 만족하는 모든 테스트 조합: 0.6 근처 피험자 1명 + 나머지 피험자 2명."""
    if N_TEST_NEAR != 1:
        raise NotImplementedError("N_TEST_NEAR = 1 기준으로 구현됨")
    near = near_target_subjects(data)
    rest = [s for s in data.subjects if s not in near]
    if len(near) < 2:
        raise ValueError(f"0.6 근처 피험자가 부족해 학습/테스트 양쪽에 둘 수 없음: {near}")
    return [tuple(sorted((n, *r))) for n in near for r in itertools.combinations(rest, N_TEST - 1)]


@functools.lru_cache(maxsize=None)
def _split_plan_cached(subjects: tuple, near: tuple, valid: tuple, n_splits: int, seed: int) -> tuple:
    first = tuple(sorted(FIRST_SPLIT_TEST))
    if first not in valid:
        raise ValueError(f"첫 번째 분할 {first}가 제약(0.6 근처 피험자 1명)을 만족하지 않음")
    # 균형 목표: 테스트 자리 n_splits*N_TEST개를 최대한 고르게.
    #   0.6 근처 피험자는 분할마다 정확히 1명 -> 자리 n_splits개를 len(near)명이 나눔
    #   나머지 피험자는 자리 n_splits*(N_TEST-1)개를 나눔
    def caps(group, slots):
        q, r = divmod(slots, len(group))
        return q, r                      # r명은 q+1회, 나머지는 q회
    rest = tuple(s for s in subjects if s not in near)
    nq, nr = caps(near, n_splits)
    rq, rr = caps(rest, n_splits * (N_TEST - 1))

    rng = np.random.default_rng(seed)
    cands = [valid[i] for i in rng.permutation(len(valid)) if valid[i] != first]
    count = {s: 0 for s in subjects}
    for s in first:
        count[s] += 1
    plan = [first]

    def ok():
        over_n = sum(count[s] > nq for s in near)
        over_r = sum(count[s] > rq for s in rest)
        return (all(count[s] <= nq + 1 for s in near) and over_n <= nr and
                all(count[s] <= rq + 1 for s in rest) and over_r <= rr)

    def dfs(start):
        if len(plan) == n_splits:
            return True
        for k in range(start, len(cands)):
            c = cands[k]
            for s in c:
                count[s] += 1
            if ok():
                plan.append(c)
                if dfs(k + 1):
                    return True
                plan.pop()
            for s in c:
                count[s] -= 1
        return False

    if not dfs(0):
        raise RuntimeError("균형 조건을 만족하는 분할 조합을 찾지 못함")
    return tuple(plan)


def split_plan(data: Spectra, n_splits: int = N_SPLITS, seed: int = SPLIT_SEED) -> list[list[str]]:
    """서로 다른 테스트 피험자 조합 n_splits개 (모델 학습 전에 고정).

    - 첫 번째 분할 = 기존 분할 FIRST_SPLIT_TEST (S002, S008, S009)
    - 모든 분할이 제약을 만족: 0.6 근처 피험자 1명 test, 나머지 0.6 근처 피험자는 train
    - 피험자별 test 참여 횟수가 최대한 균형: 0.6 근처 피험자 4/3/3회, 나머지 7명 3회 x 6명 + 2회 x 1명
      (분할마다 0.6 근처 피험자가 정확히 1명이므로 모두 3회로 맞추는 것은 불가능)
    - 조건을 만족하는 조합 중 하나를 SPLIT_SEED 순서의 탐색으로 결정 (재현 가능)
    """
    plan = _split_plan_cached(tuple(data.subjects), tuple(near_target_subjects(data)),
                              tuple(_valid_test_sets(data)), n_splits, seed)
    assert len(set(plan)) == n_splits
    return [list(p) for p in plan]


def holdout_split(data: Spectra, rep: int = 0) -> tuple[Spectra, Spectra]:
    """rep번째 7:3 분할: 학습 7명 / 테스트 3명."""
    test_subjects = split_plan(data)[rep]
    train_subjects = [s for s in data.subjects if s not in test_subjects]
    _check_disjoint(train_subjects, test_subjects)
    near = set(near_target_subjects(data))
    assert len(near & set(test_subjects)) == N_TEST_NEAR and len(near & set(train_subjects)) >= 1
    return data.select(train_subjects), data.select(test_subjects)


def repeated_splits(data: Spectra):
    """분할 10개를 차례로 반환: (rep, 학습, 테스트). 각 rep 안에서 모든 모델이 같은 분할을 쓴다."""
    for rep in range(N_SPLITS):
        yield (rep, *holdout_split(data, rep))


def make_pairs(data: Spectra):
    """같은 피험자 안의 모든 순서쌍 i != j (피험자당 10 x 9 = 90쌍).

    입력 = 측정 i, 목표 HbI = 측정 j의 실제 HbI, 정답 = 측정 j의 실제 스펙트럼.
    weight는 피험자마다 총합이 1/피험자수가 되도록 정규화 (측정 수가 달라도 피험자 기여 동일).
    """
    X, Y, W, I, J = [], [], [], [], []
    subs = data.subjects
    for s in subs:
        ix = np.flatnonzero(data.subject == s)
        n = len(ix)
        for i in ix:
            for j in ix:
                if i != j:
                    X.append(data.a[i]); Y.append(data.a[j])
                    W.append(1.0 / (len(subs) * n * (n - 1))); I.append(i); J.append(j)
    return np.asarray(X), np.asarray(Y), np.asarray(W), np.asarray(I), np.asarray(J)


if __name__ == "__main__":
    data = load_spectra()
    print(f"전체: {len(data.subjects)}명, {len(data)}개 측정, 파장 {len(WL)}개")
    print(f"0.6 근처 피험자 (|HbI-0.6|<={NEAR_TOL}): {near_target_subjects(data)}")
    print(f"\n7:3 분할 {N_SPLITS}회 (seed {SPLIT_SEED})")
    for rep, tr, te in repeated_splits(data):
        n_tr = int((abs(tr.hbi - TARGET_HBI) <= NEAR_TOL).sum())
        n_te = int((abs(te.hbi - TARGET_HBI) <= NEAR_TOL).sum())
        print(f"  rep {rep}: test {te.subjects}  | 0.6 근처 실측 train {n_tr}개 / test {n_te}개")
