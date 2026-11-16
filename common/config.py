"""실험 설정 (2026-11-12 확정, 모델 결과를 보기 전에 고정).

모든 모델과 실험 스크립트는 이 값만 참조한다. 테스트 결과를 보고 바꾸지 않는다.
"""

TARGET_HBI = 0.6

# HbI 근접 범위: |HbI - 0.6| <= tol
NEAR_TOL_MAIN = 0.02        # 주 분석 (분할 제약의 '0.6 근처 피험자' 정의에도 사용)
NEAR_TOL_SUPP = 0.03        # 보조 분석

# 7:3 분할 반복
N_SPLITS = 10               # 서로 다른 분할 10개
FIRST_SPLIT_TEST = ("S002", "S008", "S009")   # 첫 번째 분할 = 기존 분할
SPLIT_SEED = 0              # 나머지 9개 분할 선택 난수 시드
N_TEST = 3                  # 테스트 피험자 수 (학습 7명)
N_TEST_NEAR = 1             # 테스트에 넣을 '0.6 근처 피험자' 수 (나머지 2명은 학습)

# B1: HbI-Binned Mean Correction
B1_BIN_WIDTH = 0.05         # 초기 기준값
B1_BIN_START = 0.40         # 고정 bin 경계: 0.40, 0.45, ..., 0.75 (전체 데이터 HbI 0.439-0.714를 포함)
B1_BIN_STOP = 0.75          # 모든 분할·모델 실행에서 동일하게 사용 (데이터에 따라 바꾸지 않음)

# BiGRU (Model_Ablation/ablation.py 의 CFG와 같은 값; epoch은 검증 없이 고정)
BIGRU = dict(hidden=8, lr=3e-3, slope_l2=1e-3, clip=1.0, epochs=150)
BIGRU_SEEDS = (0, 1, 2)     # 학습 초기화 시드
