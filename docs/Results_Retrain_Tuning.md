# 재학습·하이퍼파라미터 튜닝·구간별 분석 결과 (Phase 1–4, Development cohort만 사용)

이 문서는 `scripts/make_docs.py --results`가 결과 파일에서 생성한다. External cohort(S101–S106)는 전처리와 분할 파일 생성 외에는 어떤 모델의 학습, 하이퍼파라미터 선택, 평가에도 쓰이지 않았다. 판정 문구는 넣지 않았다.

단위: absorbance RMSE, Peak RMSE, Peak bias, 표준편차 지표, 구간 bias는 ×10⁻³ absorbance, reflectance RMSE는 %p. absorbance RMSE와 reflectance RMSE는 570 nm 제외 30파장. BiGRU의 피험자 값은 seed 0, 1, 2 지표의 평균. p 값은 다중 비교 보정이 없는 기술적 참고값 (정확 Wilcoxon signed-rank, 부호검정, 양측).

모델 명칭: HbI-binned mean = 코드의 B1, Global Linear = B2, PLS_v2 = `pls_v2_k10`, PLS_v1 = `pls_v1_k10`, Ridge = `ridge`, BiGRU = 튜닝한 BiGRU (`results/bigru_tuned`), BiGRU (fixed setting) = hidden 8, slope_l2 1e-3, 150 epoch.

## 1. Phase 1: 고정 설정 재현

General reconstruction. 단위: absorbance RMSE ×10⁻³ / reflectance RMSE %p / Peak RMSE ×10⁻³. `scripts/pls_compare.py`의 `results/compare/*/comparison_table.csv` 값이며, 2026-11-23 참고 수치와 표기 자릿수까지 모두 일치했다 (BiGRU (fixed setting)의 LeaveOneSubjectOut 값은 참고 수치가 없던 신규 값).

| 모델 | Development10_Holdout73 | Development10_LeaveOneSubjectOut |
|---|---|---|
| HbI-binned mean | 17.632 / 1.230 / 2.422 | 17.414 / 1.221 / 2.528 |
| Global Linear | 12.049 / 0.913 / 0.687 | 12.126 / 0.915 / 0.716 |
| PLS_v1 (성분 수 상한 5) | 11.935 / 0.948 / 0.740 | 11.258 / 0.873 / 0.748 |
| PLS_v1 (성분 수 상한 10) | 11.910 / 0.946 / 0.736 | 11.367 / 0.893 / 0.741 |
| PLS_v2 (성분 수 상한 5) | 11.971 / 0.928 / 0.693 | 11.542 / 0.927 / 0.703 |
| PLS_v2 (성분 수 상한 10) | 11.850 / 0.929 / 0.687 | 11.057 / 0.875 / 0.684 |
| Ridge | 11.694 / 0.923 / 0.687 | 11.136 / 0.885 / 0.687 |
| BiGRU (fixed setting) | 11.322 / 0.855 / 1.041 | 11.280 / 0.847 / 1.015 |

Raw(보정 없음)는 정답이 다른 측정이므로 General reconstruction 표에 넣지 않고 Fixed-target repeatability에서만 보고한다.

선택된 하이퍼파라미터 (분할 순서대로)

| 모델 | 분할 세트 | 선택값 | 상한(격자 끝) 도달 횟수 |
|---|---|---|---|
| PLS_v2 (상한 10) | LeaveOneSubjectOut | 7, 3, 6, 5, 6, 6, 6, 6, 8, 5 | 0 |
| PLS_v2 (상한 10) | Holdout73 | 1, 6, 2, 5, 7, 5, 8, 2, 6, 5 | 0 |
| PLS_v1 (상한 10) | LeaveOneSubjectOut | 6, 6, 5, 6, 6, 6, 5, 5, 4, 5 | 0 |
| PLS_v1 (상한 10) | Holdout73 | 4, 5, 6, 5, 4, 5, 5, 6, 5, 5 | 0 |
| PLS_v2 (상한 5) | LeaveOneSubjectOut | 5, 3, 5, 5, 5, 5, 5, 5, 4, 5 | 8 |
| PLS_v2 (상한 5) | Holdout73 | 1, 4, 2, 5, 1, 5, 5, 2, 5, 5 | 5 |
| PLS_v1 (상한 5) | LeaveOneSubjectOut | 5, 5, 5, 5, 5, 4, 5, 5, 4, 5 | 8 |
| PLS_v1 (상한 5) | Holdout73 | 4, 5, 5, 5, 4, 5, 5, 5, 5, 5 | 8 |
| Ridge alpha | LeaveOneSubjectOut | 1, 100, 1, 10, 1, 1, 1, 0.001, 1, 10 | 0 |
| Ridge alpha | Holdout73 | 1e+10, 1, 100, 10, 1, 1e+10, 0.1, 10, 1, 10 | 2 |

Fixed-target repeatability (Development10_Holdout73, ×10⁻³): Raw Peak 표준편차 2.148, spectral 표준편차 22.61; HbI-binned mean Peak 표준편차 1.713, spectral 표준편차 11.83; Global Linear Peak 표준편차 0.486, spectral 표준편차 7.27; BiGRU (fixed setting) Peak 표준편차 0.734, spectral 표준편차 7.05.

## 2. Phase 2: 데이터와 분할

| 분할 세트 | 분할 수 | 행 (train / test) | 쌍 (train / test) | 검사 |
|---|---|---|---|---|
| Development10_Holdout73 | 10 | 70 / 30 | 630 / 270 | 통과 |
| Development10_LeaveOneSubjectOut | 10 | 90 / 10 | 810 / 90 | 통과 |
| ExternalTest6 | 1 | 100 / 60 | 900 / 540 | 통과 |
| Union16_Holdout73 | 10 | 110 / 50 | 990 / 450 | 통과 |
| Union16_LeaveOneSubjectOut | 16 | 150 / 10 | 1350 / 90 | 통과 |

**Development10_Holdout73** 분할별 Near-target subject, In-range subject (명 수와 비율 %)

| 분할 | test | test Near-target | train Near-target | test In-range | train In-range |
|---|---|---|---|---|---|
| split_00 | S002 S008 S009 | 1 (33.3) | 2 (28.6) | 1 (33.3) | 3 (42.9) |
| split_01 | S003 S006 S009 | 1 (33.3) | 2 (28.6) | 1 (33.3) | 3 (42.9) |
| split_02 | S002 S004 S009 | 1 (33.3) | 2 (28.6) | 2 (66.7) | 2 (28.6) |
| split_03 | S002 S003 S007 | 1 (33.3) | 2 (28.6) | 0 (0.0) | 4 (57.1) |
| split_04 | S001 S005 S008 | 1 (33.3) | 2 (28.6) | 1 (33.3) | 3 (42.9) |
| split_05 | S005 S006 S008 | 1 (33.3) | 2 (28.6) | 0 (0.0) | 4 (57.1) |
| split_06 | S001 S003 S006 | 1 (33.3) | 2 (28.6) | 1 (33.3) | 3 (42.9) |
| split_07 | S001 S004 S007 | 1 (33.3) | 2 (28.6) | 2 (66.7) | 2 (28.6) |
| split_08 | S003 S007 S010 | 1 (33.3) | 2 (28.6) | 1 (33.3) | 3 (42.9) |
| split_09 | S004 S005 S010 | 1 (33.3) | 2 (28.6) | 2 (66.7) | 2 (28.6) |

피험자별 테스트 횟수: S001 3, S002 3, S003 4, S004 3, S005 3, S006 3, S007 3, S008 3, S009 3, S010 2

- 기존 코드(common/data_loader.split_plan)의 분할 그대로.
- In-range subject는 층화되어 있지 않다. test에 In-range subject가 없는 분할: ['split_03', 'split_05'].

**Union16_Holdout73** 분할별 Near-target subject, In-range subject (명 수와 비율 %)

| 분할 | test | test Near-target | train Near-target | test In-range | train In-range |
|---|---|---|---|---|---|
| split_00 | S002 S004 S009 S103 S106 | 1 (20.0) | 4 (36.4) | 3 (60.0) | 3 (27.3) |
| split_01 | S001 S005 S007 S101 S106 | 1 (20.0) | 4 (36.4) | 3 (60.0) | 3 (27.3) |
| split_02 | S002 S005 S006 S101 S105 | 2 (40.0) | 3 (27.3) | 1 (20.0) | 5 (45.5) |
| split_03 | S001 S004 S010 S104 S105 | 2 (40.0) | 3 (27.3) | 3 (60.0) | 3 (27.3) |
| split_04 | S003 S008 S010 S102 S104 | 2 (40.0) | 3 (27.3) | 1 (20.0) | 5 (45.5) |
| split_05 | S002 S003 S010 S102 S105 | 2 (40.0) | 3 (27.3) | 1 (20.0) | 5 (45.5) |
| split_06 | S004 S007 S008 S103 S104 | 2 (40.0) | 3 (27.3) | 1 (20.0) | 5 (45.5) |
| split_07 | S003 S006 S009 S101 S106 | 2 (40.0) | 3 (27.3) | 3 (60.0) | 3 (27.3) |
| split_08 | S005 S007 S009 S101 S102 | 1 (20.0) | 4 (36.4) | 2 (40.0) | 4 (36.4) |
| split_09 | S001 S006 S008 S102 S103 | 1 (20.0) | 4 (36.4) | 1 (20.0) | 5 (45.5) |

피험자별 테스트 횟수: S001 3, S002 3, S003 3, S004 3, S005 3, S006 3, S007 3, S008 3, S009 3, S010 3, S101 4, S102 4, S103 3, S104 3, S105 3, S106 3

- test 5명 = Development cohort 3명 + External cohort 2명. 고정된 첫 분할 없음, 난수 시드 0.
- 유효 테스트 조합 1167개를 (Development 3명 조합 사전순) x (External 2명 조합 사전순)으로 나열한 뒤 numpy default_rng(0).permutation 순서로 깊이 우선 탐색.
- 기존 코드의 'test에 Near-target subject 정확히 1명' 규칙은 쓰지 않음.

## 3. Phase 3: 재학습과 하이퍼파라미터 튜닝 (Development cohort만 사용)

BiGRU 격자: hidden {4, 8, 16} × slope_l2 {0, 1e-3, 1e-2, 1e-1} × epoch {0, 25, 50, 100, 150, 200, 300} = 84 후보. 선택 기준 = 학습 피험자 안 Leave-One-Subject-Out 내부 검증 쌍 전체의 absorbance RMSE (570 nm 제외), 선택용 seed 0. 검증 RMSE 단위 ×10⁻³.

**Development10_LeaveOneSubjectOut**

| 분할 | test | hidden | slope_l2 | epoch | 검증 RMSE | Global Linear 검증 RMSE | 격자 경계 | Peak-constrained selection (hidden / slope_l2 / epoch) |
|---|---|---|---|---|---|---|---|---|
| fold_S001 | S001 | 8 | 0 | 100 | 11.367 | 12.559 | slope_l2_min | epoch 0 (Global Linear와 동일) / – / 0 (통과 후보 12/84) |
| fold_S002 | S002 | 8 | 0 | 200 | 10.950 | 12.711 | slope_l2_min | epoch 0 (Global Linear와 동일) / – / 0 (통과 후보 12/84) |
| fold_S003 | S003 | 8 | 0 | 200 | 11.442 | 13.027 | slope_l2_min | epoch 0 (Global Linear와 동일) / – / 0 (통과 후보 12/84) |
| fold_S004 | S004 | 8 | 0.001 | 200 | 10.960 | 12.063 | 없음 | 16 / 0.01 / 25 (통과 후보 13/84) |
| fold_S005 | S005 | 8 | 0 | 300 | 11.239 | 12.659 | slope_l2_min, epoch_max | epoch 0 (Global Linear와 동일) / – / 0 (통과 후보 12/84) |
| fold_S006 | S006 | 8 | 0.001 | 200 | 11.275 | 12.808 | 없음 | epoch 0 (Global Linear와 동일) / – / 0 (통과 후보 12/84) |
| fold_S007 | S007 | 16 | 0.001 | 300 | 11.399 | 12.472 | hidden_max, epoch_max | epoch 0 (Global Linear와 동일) / – / 0 (통과 후보 12/84) |
| fold_S008 | S008 | 8 | 0 | 100 | 11.353 | 12.296 | slope_l2_min | epoch 0 (Global Linear와 동일) / – / 0 (통과 후보 12/84) |
| fold_S009 | S009 | 8 | 0 | 300 | 11.007 | 12.005 | slope_l2_min, epoch_max | epoch 0 (Global Linear와 동일) / – / 0 (통과 후보 12/84) |
| fold_S010 | S010 | 8 | 0.01 | 300 | 11.823 | 12.921 | epoch_max | epoch 0 (Global Linear와 동일) / – / 0 (통과 후보 12/84) |

격자 경계 도달 8/10회 (hidden 최소 0, hidden 최대 1, slope_l2 최소(0) 6, slope_l2 최대 0, epoch 최대(300) 4), epoch 0 선택 0/10회. 모든 분할에서 epoch 0 후보 12개의 검증 값이 서로 같고 Global Linear와 같음을 확인했다.

**Development10_Holdout73**

| 분할 | test | hidden | slope_l2 | epoch | 검증 RMSE | Global Linear 검증 RMSE | 격자 경계 | Peak-constrained selection (hidden / slope_l2 / epoch) |
|---|---|---|---|---|---|---|---|---|
| split_00 | S002 S008 S009 | 4 | 0.001 | 150 | 10.572 | 11.423 | hidden_min | epoch 0 (Global Linear와 동일) / – / 0 (통과 후보 12/84) |
| split_01 | S003 S006 S009 | 8 | 0.001 | 100 | 11.860 | 12.946 | 없음 | epoch 0 (Global Linear와 동일) / – / 0 (통과 후보 12/84) |
| split_02 | S002 S004 S009 | 16 | 0.001 | 300 | 10.640 | 11.830 | hidden_max, epoch_max | epoch 0 (Global Linear와 동일) / – / 0 (통과 후보 12/84) |
| split_03 | S002 S003 S007 | 8 | 0.001 | 300 | 11.431 | 13.343 | epoch_max | epoch 0 (Global Linear와 동일) / – / 0 (통과 후보 12/84) |
| split_04 | S001 S005 S008 | 8 | 0 | 100 | 11.582 | 12.717 | slope_l2_min | epoch 0 (Global Linear와 동일) / – / 0 (통과 후보 12/84) |
| split_05 | S005 S006 S008 | 8 | 0.01 | 300 | 12.094 | 12.835 | epoch_max | epoch 0 (Global Linear와 동일) / – / 0 (통과 후보 12/84) |
| split_06 | S001 S003 S006 | 8 | 0 | 100 | 12.303 | 13.618 | slope_l2_min | epoch 0 (Global Linear와 동일) / – / 0 (통과 후보 12/84) |
| split_07 | S001 S004 S007 | 8 | 0.01 | 300 | 11.166 | 11.884 | epoch_max | epoch 0 (Global Linear와 동일) / – / 0 (통과 후보 12/84) |
| split_08 | S003 S007 S010 | 16 | 0.001 | 300 | 12.389 | 13.704 | hidden_max, epoch_max | epoch 0 (Global Linear와 동일) / – / 0 (통과 후보 12/84) |
| split_09 | S004 S005 S010 | 8 | 0.01 | 300 | 11.602 | 12.607 | epoch_max | 16 / 0.01 / 25 (통과 후보 15/84) |

격자 경계 도달 9/10회 (hidden 최소 1, hidden 최대 2, slope_l2 최소(0) 2, slope_l2 최대 0, epoch 최대(300) 6), epoch 0 선택 0/10회. 모든 분할에서 epoch 0 후보 12개의 검증 값이 서로 같고 Global Linear와 같음을 확인했다.

**Development10_AllSubjects**

| 분할 | test | hidden | slope_l2 | epoch | 검증 RMSE | Global Linear 검증 RMSE | 격자 경계 | Peak-constrained selection (hidden / slope_l2 / epoch) |
|---|---|---|---|---|---|---|---|---|
| all_subjects | – | 8 | 0 | 200 | 11.123 | 12.505 | slope_l2_min | epoch 0 (Global Linear와 동일) / – / 0 (통과 후보 12/84) |

격자 경계 도달 1/1회 (hidden 최소 0, hidden 최대 0, slope_l2 최소(0) 1, slope_l2 최대 0, epoch 최대(300) 0), epoch 0 선택 0/1회. 모든 분할에서 epoch 0 후보 12개의 검증 값이 서로 같고 Global Linear와 같음을 확인했다.

**PLS 계열과 Ridge** — `scripts/pls_compare.py`의 `pls_v2_k10`, `pls_v1_k10`, `ridge` 결과(Phase 1에서 생성)를 그대로 재사용했다 (선택값은 1장의 표). ExternalTest6용 Development cohort 10명 전체 선택: PLS_v2 성분 수 6, PLS_v1 성분 수 6, Ridge alpha 1 (기대 6, 6과 일치: True).

BiGRU 튜닝 총 실행 시간: 30.9분 (10개 프로세스 병렬, 프로세스당 CPU 1 thread; 300 epoch 학습 2,040회). 튜닝 루프의 epoch별 학습 이력과 최종 가중치는 `BiGRUModel.fit`과 완전 일치했고, 병렬 실행 결과는 단일 실행과 완전 일치했다.

## 4. Phase 4: 평가와 분석

### 4.1 주 비교

### Development10_LeaveOneSubjectOut (주 프로토콜)

집계: 피험자 10명의 평균 ± 표준편차 (ddof=1).

쌍체 비교: 분석 단위 = 피험자. 차이 = 앞 모델 - 뒤 모델, 개선 = 차이 < 0. peak_bias와 bias는 |값| 감소를 개선으로 판정. p 값은 다중 비교 보정이 없는 기술적 참고값 (정확 Wilcoxon signed-rank, 부호검정, 양측).

**General reconstruction** (Peak bias는 부호 있는 값, 예측 − 정답)

| 모델 | absorbance RMSE | reflectance RMSE | Peak RMSE | Peak bias |
|---|---|---|---|---|
| HbI-binned mean | 17.414 ± 3.010 | 1.221 ± 0.222 | 2.528 ± 0.727 | -0.000 ± 0.000 |
| Global Linear | 12.126 ± 3.222 | 0.915 ± 0.245 | 0.716 ± 0.203 | -0.000 ± 0.000 |
| PLS_v2 | 11.057 ± 2.862 | 0.875 ± 0.248 | 0.684 ± 0.201 | 0.006 ± 0.036 |
| BiGRU | 11.301 ± 2.759 | 0.832 ± 0.185 | 0.949 ± 0.522 | 0.078 ± 0.128 |

주의: General reconstruction은 피험자 안의 모든 순서쌍을 쓰므로, 보정이 i → j와 j → i에서 정확히 반대 부호인 HbI-binned mean과 Global Linear는 Peak bias가 구조적으로 0이다 (오차가 쌍끼리 상쇄). 따라서 이 두 모델을 뒤 모델로 둔 Peak bias 비교의 k/n과 p 값은 정보가 없고, 앞 모델의 부호 있는 bias 값 자체만 참고한다.

사전 지정 비교 (General reconstruction)

| 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|
| PLS_v2 minus Global Linear | absorbance RMSE | 10 | -1.068 | 2.021 | 0.639 | 8/10 | 0.064 | 0.109 |
| BiGRU minus Global Linear | absorbance RMSE | 10 | -0.824 | 1.320 | 0.417 | 8/10 | 0.064 | 0.109 |
| PLS_v2 minus HbI-binned mean | absorbance RMSE | 10 | -6.356 | 3.549 | 1.122 | 10/10 | 0.002 | 0.002 |
| BiGRU minus HbI-binned mean | absorbance RMSE | 10 | -6.113 | 3.436 | 1.086 | 10/10 | 0.002 | 0.002 |
| BiGRU minus PLS_v2 | absorbance RMSE | 10 | 0.244 | 1.842 | 0.582 | 6/10 | 1.000 | 0.754 |
| BiGRU minus BiGRU (fixed setting) | absorbance RMSE | 10 | 0.021 | 0.512 | 0.162 | 6/10 | 1.000 | 0.754 |
| PLS_v2 minus Global Linear | reflectance RMSE | 10 | -0.039 | 0.152 | 0.048 | 8/10 | 0.131 | 0.109 |
| BiGRU minus Global Linear | reflectance RMSE | 10 | -0.083 | 0.156 | 0.049 | 8/10 | 0.160 | 0.109 |
| PLS_v2 minus HbI-binned mean | reflectance RMSE | 10 | -0.346 | 0.275 | 0.087 | 9/10 | 0.004 | 0.021 |
| BiGRU minus HbI-binned mean | reflectance RMSE | 10 | -0.389 | 0.245 | 0.077 | 10/10 | 0.002 | 0.002 |
| BiGRU minus PLS_v2 | reflectance RMSE | 10 | -0.043 | 0.163 | 0.051 | 8/10 | 0.232 | 0.109 |
| BiGRU minus BiGRU (fixed setting) | reflectance RMSE | 10 | -0.014 | 0.030 | 0.009 | 8/10 | 0.193 | 0.109 |
| PLS_v2 minus Global Linear | Peak RMSE | 10 | -0.032 | 0.125 | 0.040 | 4/10 | 1.000 | 0.754 |
| BiGRU minus Global Linear | Peak RMSE | 10 | 0.233 | 0.409 | 0.129 | 2/10 | 0.105 | 0.109 |
| PLS_v2 minus HbI-binned mean | Peak RMSE | 10 | -1.843 | 0.796 | 0.252 | 10/10 | 0.002 | 0.002 |
| BiGRU minus HbI-binned mean | Peak RMSE | 10 | -1.578 | 0.944 | 0.298 | 9/10 | 0.004 | 0.021 |
| BiGRU minus PLS_v2 | Peak RMSE | 10 | 0.265 | 0.499 | 0.158 | 5/10 | 0.375 | 1.000 |
| BiGRU minus BiGRU (fixed setting) | Peak RMSE | 10 | -0.066 | 0.189 | 0.060 | 6/10 | 0.492 | 0.754 |
| PLS_v2 minus Global Linear | Peak bias | 10 | 0.006 | 0.036 | 0.012 | 0/10 | 0.002 | 0.002 |
| BiGRU minus Global Linear | Peak bias | 10 | 0.078 | 0.128 | 0.040 | 0/10 | 0.002 | 0.002 |
| PLS_v2 minus HbI-binned mean | Peak bias | 10 | 0.006 | 0.036 | 0.012 | 0/10 | 0.002 | 0.002 |
| BiGRU minus HbI-binned mean | Peak bias | 10 | 0.078 | 0.128 | 0.040 | 0/10 | 0.002 | 0.002 |
| BiGRU minus PLS_v2 | Peak bias | 10 | 0.072 | 0.122 | 0.039 | 1/10 | 0.004 | 0.021 |
| BiGRU minus BiGRU (fixed setting) | Peak bias | 10 | -0.037 | 0.159 | 0.050 | 4/10 | 0.625 | 0.754 |

보조 표: Supplementary 모델 (General reconstruction)

| 모델 | absorbance RMSE | reflectance RMSE | Peak RMSE | Peak bias |
|---|---|---|---|---|
| PLS_v1 | 11.367 ± 2.603 | 0.893 ± 0.225 | 0.741 ± 0.219 | 0.009 ± 0.218 |
| Ridge | 11.136 ± 2.894 | 0.885 ± 0.247 | 0.687 ± 0.207 | -0.006 ± 0.054 |
| BiGRU (fixed setting) | 11.280 ± 2.705 | 0.846 ± 0.187 | 1.015 ± 0.613 | 0.115 ± 0.223 |
| BiGRU (Peak-constrained selection) | 12.042 ± 3.114 | 0.907 ± 0.231 | 0.716 ± 0.203 | -0.002 ± 0.007 |
| PLS_v2 (max 5 components) | 11.542 ± 3.179 | 0.927 ± 0.311 | 0.703 ± 0.188 | -0.001 ± 0.029 |
| PLS_v1 (max 5 components) | 11.257 ± 2.279 | 0.873 ± 0.208 | 0.748 ± 0.224 | -0.009 ± 0.171 |

| 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|
| PLS_v1 minus Global Linear | absorbance RMSE | 10 | -0.759 | 2.257 | 0.714 | 8/10 | 0.432 | 0.109 |
| Ridge minus Global Linear | absorbance RMSE | 10 | -0.990 | 1.863 | 0.589 | 8/10 | 0.084 | 0.109 |
| BiGRU (fixed setting) minus Global Linear | absorbance RMSE | 10 | -0.846 | 1.091 | 0.345 | 8/10 | 0.064 | 0.109 |
| BiGRU (Peak-constrained selection) minus Global Linear | absorbance RMSE | 10 | -0.083 | 0.263 | 0.083 | 1/10 | 1.000 | 1.000 |
| PLS_v2 (max 5 components) minus Global Linear | absorbance RMSE | 10 | -0.584 | 0.985 | 0.312 | 8/10 | 0.049 | 0.109 |
| PLS_v1 (max 5 components) minus Global Linear | absorbance RMSE | 10 | -0.868 | 1.971 | 0.623 | 8/10 | 0.432 | 0.109 |
| BiGRU (Peak-constrained selection) minus BiGRU | absorbance RMSE | 10 | 0.741 | 1.133 | 0.358 | 2/10 | 0.064 | 0.109 |
| PLS_v1 minus Global Linear | reflectance RMSE | 10 | -0.022 | 0.137 | 0.043 | 7/10 | 0.770 | 0.344 |
| Ridge minus Global Linear | reflectance RMSE | 10 | -0.030 | 0.119 | 0.038 | 7/10 | 0.232 | 0.344 |
| BiGRU (fixed setting) minus Global Linear | reflectance RMSE | 10 | -0.068 | 0.143 | 0.045 | 8/10 | 0.131 | 0.109 |
| BiGRU (Peak-constrained selection) minus Global Linear | reflectance RMSE | 10 | -0.007 | 0.023 | 0.007 | 1/10 | 1.000 | 1.000 |
| PLS_v2 (max 5 components) minus Global Linear | reflectance RMSE | 10 | 0.012 | 0.211 | 0.067 | 7/10 | 0.432 | 0.344 |
| PLS_v1 (max 5 components) minus Global Linear | reflectance RMSE | 10 | -0.042 | 0.091 | 0.029 | 7/10 | 0.193 | 0.344 |
| BiGRU (Peak-constrained selection) minus BiGRU | reflectance RMSE | 10 | 0.075 | 0.148 | 0.047 | 2/10 | 0.193 | 0.109 |
| PLS_v1 minus Global Linear | Peak RMSE | 10 | 0.024 | 0.050 | 0.016 | 4/10 | 0.275 | 0.754 |
| Ridge minus Global Linear | Peak RMSE | 10 | -0.029 | 0.122 | 0.039 | 4/10 | 0.695 | 0.754 |
| BiGRU (fixed setting) minus Global Linear | Peak RMSE | 10 | 0.298 | 0.505 | 0.160 | 2/10 | 0.064 | 0.109 |
| BiGRU (Peak-constrained selection) minus Global Linear | Peak RMSE | 10 | -0.000 | 0.001 | 0.000 | 1/10 | 1.000 | 1.000 |
| PLS_v2 (max 5 components) minus Global Linear | Peak RMSE | 10 | -0.013 | 0.068 | 0.021 | 4/10 | 1.000 | 0.754 |
| PLS_v1 (max 5 components) minus Global Linear | Peak RMSE | 10 | 0.031 | 0.044 | 0.014 | 2/10 | 0.027 | 0.109 |
| BiGRU (Peak-constrained selection) minus BiGRU | Peak RMSE | 10 | -0.233 | 0.409 | 0.129 | 8/10 | 0.105 | 0.109 |
| PLS_v1 minus Global Linear | Peak bias | 10 | 0.009 | 0.218 | 0.069 | 0/10 | 0.002 | 0.002 |
| Ridge minus Global Linear | Peak bias | 10 | -0.006 | 0.054 | 0.017 | 0/10 | 0.002 | 0.002 |
| BiGRU (fixed setting) minus Global Linear | Peak bias | 10 | 0.115 | 0.223 | 0.071 | 0/10 | 0.002 | 0.002 |
| BiGRU (Peak-constrained selection) minus Global Linear | Peak bias | 10 | -0.002 | 0.007 | 0.002 | 0/10 | 1.000 | 1.000 |
| PLS_v2 (max 5 components) minus Global Linear | Peak bias | 10 | -0.001 | 0.029 | 0.009 | 0/10 | 0.002 | 0.002 |
| PLS_v1 (max 5 components) minus Global Linear | Peak bias | 10 | -0.009 | 0.171 | 0.054 | 0/10 | 0.002 | 0.002 |
| BiGRU (Peak-constrained selection) minus BiGRU | Peak bias | 10 | -0.081 | 0.131 | 0.042 | 10/10 | 0.002 | 0.002 |

**Near-target accuracy, 허용 폭 0.02** (대상 피험자 S003 S004 S008; 기술 통계만)

| 모델 | absorbance RMSE | Peak RMSE |
|---|---|---|
| HbI-binned mean | 18.308 ± 1.258 (n=3) | 1.555 ± 0.830 (n=3) |
| Global Linear | 11.279 ± 3.549 (n=3) | 0.673 ± 0.147 (n=3) |
| PLS_v2 | 10.076 ± 2.014 (n=3) | 0.762 ± 0.224 (n=3) |
| BiGRU | 9.740 ± 2.560 (n=3) | 1.001 ± 0.584 (n=3) |
| PLS_v1 | 8.937 ± 2.347 (n=3) | 0.772 ± 0.254 (n=3) |
| Ridge | 10.136 ± 1.834 (n=3) | 0.784 ± 0.256 (n=3) |
| BiGRU (fixed setting) | 10.006 ± 2.802 (n=3) | 0.924 ± 0.438 (n=3) |
| BiGRU (Peak-constrained selection) | 11.126 ± 3.538 (n=3) | 0.684 ± 0.160 (n=3) |
| PLS_v2 (max 5 components) | 9.999 ± 2.055 (n=3) | 0.758 ± 0.226 (n=3) |
| PLS_v1 (max 5 components) | 8.907 ± 2.314 (n=3) | 0.792 ± 0.286 (n=3) |

**Near-target accuracy, 허용 폭 0.03** (대상 피험자 S001 S002 S003 S004 S005 S008 S009 S010; 예상 8명과 일치: True)

| 모델 | absorbance RMSE | Peak RMSE |
|---|---|---|
| HbI-binned mean | 16.677 ± 1.551 (n=8) | 2.050 ± 0.674 (n=8) |
| Global Linear | 12.514 ± 3.159 (n=8) | 0.749 ± 0.190 (n=8) |
| PLS_v2 | 11.350 ± 2.416 (n=8) | 0.711 ± 0.190 (n=8) |
| BiGRU | 11.929 ± 2.577 (n=8) | 0.958 ± 0.502 (n=8) |
| PLS_v1 | 11.867 ± 4.316 (n=8) | 0.815 ± 0.245 (n=8) |
| Ridge | 11.160 ± 2.276 (n=8) | 0.730 ± 0.187 (n=8) |
| BiGRU (fixed setting) | 11.895 ± 2.626 (n=8) | 0.921 ± 0.455 (n=8) |
| BiGRU (Peak-constrained selection) | 12.452 ± 3.173 (n=8) | 0.750 ± 0.190 (n=8) |
| PLS_v2 (max 5 components) | 12.153 ± 2.899 (n=8) | 0.747 ± 0.155 (n=8) |
| PLS_v1 (max 5 components) | 11.713 ± 3.668 (n=8) | 0.806 ± 0.213 (n=8) |

| 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|
| PLS_v2 minus Global Linear | absorbance RMSE | 8 | -1.164 | 3.447 | 1.219 | 5/8 | 0.742 | 0.727 |
| BiGRU minus Global Linear | absorbance RMSE | 8 | -0.585 | 2.011 | 0.711 | 4/8 | 0.641 | 1.000 |
| PLS_v2 minus HbI-binned mean | absorbance RMSE | 8 | -5.327 | 2.292 | 0.810 | 8/8 | 0.008 | 0.008 |
| BiGRU minus HbI-binned mean | absorbance RMSE | 8 | -4.748 | 1.768 | 0.625 | 8/8 | 0.008 | 0.008 |
| BiGRU minus PLS_v2 | absorbance RMSE | 8 | 0.579 | 2.077 | 0.734 | 4/8 | 0.641 | 1.000 |
| BiGRU minus BiGRU (fixed setting) | absorbance RMSE | 8 | 0.033 | 0.430 | 0.152 | 4/8 | 0.945 | 1.000 |
| PLS_v2 minus Global Linear | Peak RMSE | 8 | -0.038 | 0.225 | 0.080 | 2/8 | 0.461 | 0.289 |
| BiGRU minus Global Linear | Peak RMSE | 8 | 0.209 | 0.407 | 0.144 | 1/8 | 0.109 | 0.070 |
| PLS_v2 minus HbI-binned mean | Peak RMSE | 8 | -1.338 | 0.690 | 0.244 | 8/8 | 0.008 | 0.008 |
| BiGRU minus HbI-binned mean | Peak RMSE | 8 | -1.092 | 0.988 | 0.349 | 7/8 | 0.023 | 0.070 |
| BiGRU minus PLS_v2 | Peak RMSE | 8 | 0.246 | 0.504 | 0.178 | 2/8 | 0.148 | 0.289 |
| BiGRU minus BiGRU (fixed setting) | Peak RMSE | 8 | 0.037 | 0.139 | 0.049 | 4/8 | 0.742 | 1.000 |

**Fixed-target repeatability** (테스트 측정 전부를 HbI 0.6으로 복원했을 때의 개인 내 표준편차; n = 집계 단위 수)

전체

| 모델 | Peak 표준편차 | spectral 표준편차 |
|---|---|---|
| Raw | 2.202 ± 0.837 (n=10) | 23.117 ± 10.614 (n=10) |
| HbI-binned mean | 1.787 ± 0.514 (n=10) | 11.720 ± 2.070 (n=10) |
| Global Linear | 0.507 ± 0.143 (n=10) | 7.331 ± 1.815 (n=10) |
| PLS_v2 | 0.481 ± 0.148 (n=10) | 6.712 ± 1.566 (n=10) |
| BiGRU | 0.686 ± 0.328 (n=10) | 6.825 ± 1.453 (n=10) |
| PLS_v1 | 0.502 ± 0.165 (n=10) | 4.114 ± 1.589 (n=10) |
| Ridge | 0.492 ± 0.156 (n=10) | 6.593 ± 1.503 (n=10) |
| BiGRU (fixed setting) | 0.711 ± 0.388 (n=10) | 6.957 ± 1.437 (n=10) |
| BiGRU (Peak-constrained selection) | 0.506 ± 0.143 (n=10) | 7.294 ± 1.752 (n=10) |
| PLS_v2 (max 5 components) | 0.496 ± 0.136 (n=10) | 7.263 ± 2.005 (n=10) |
| PLS_v1 (max 5 components) | 0.531 ± 0.162 (n=10) | 4.340 ± 1.372 (n=10) |

In-range subject

| 모델 | Peak 표준편차 | spectral 표준편차 |
|---|---|---|
| Raw | 2.979 ± 0.298 (n=4) | 32.874 ± 7.730 (n=4) |
| HbI-binned mean | 1.750 ± 0.673 (n=4) | 11.041 ± 1.855 (n=4) |
| Global Linear | 0.593 ± 0.156 (n=4) | 8.042 ± 1.957 (n=4) |
| PLS_v2 | 0.534 ± 0.201 (n=4) | 7.290 ± 1.591 (n=4) |
| BiGRU | 0.974 ± 0.315 (n=4) | 7.684 ± 1.369 (n=4) |
| PLS_v1 | 0.617 ± 0.164 (n=4) | 5.311 ± 1.697 (n=4) |
| Ridge | 0.533 ± 0.200 (n=4) | 7.049 ± 1.568 (n=4) |
| BiGRU (fixed setting) | 1.103 ± 0.271 (n=4) | 7.754 ± 1.271 (n=4) |
| BiGRU (Peak-constrained selection) | 0.592 ± 0.155 (n=4) | 7.947 ± 1.825 (n=4) |
| PLS_v2 (max 5 components) | 0.571 ± 0.160 (n=4) | 8.035 ± 2.296 (n=4) |
| PLS_v1 (max 5 components) | 0.638 ± 0.166 (n=4) | 5.067 ± 1.884 (n=4) |

그 밖의 피험자

| 모델 | Peak 표준편차 | spectral 표준편차 |
|---|---|---|
| Raw | 1.684 ± 0.636 (n=6) | 16.612 ± 6.322 (n=6) |
| HbI-binned mean | 1.812 ± 0.450 (n=6) | 12.173 ± 2.244 (n=6) |
| Global Linear | 0.449 ± 0.113 (n=6) | 6.858 ± 1.719 (n=6) |
| PLS_v2 | 0.446 ± 0.106 (n=6) | 6.326 ± 1.566 (n=6) |
| BiGRU | 0.495 ± 0.157 (n=6) | 6.252 ± 1.300 (n=6) |
| PLS_v1 | 0.425 ± 0.124 (n=6) | 3.316 ± 0.954 (n=6) |
| Ridge | 0.466 ± 0.134 (n=6) | 6.289 ± 1.521 (n=6) |
| BiGRU (fixed setting) | 0.450 ± 0.147 (n=6) | 6.425 ± 1.377 (n=6) |
| BiGRU (Peak-constrained selection) | 0.449 ± 0.113 (n=6) | 6.858 ± 1.719 (n=6) |
| PLS_v2 (max 5 components) | 0.446 ± 0.101 (n=6) | 6.749 ± 1.811 (n=6) |
| PLS_v1 (max 5 components) | 0.459 ± 0.123 (n=6) | 3.855 ± 0.743 (n=6) |

사전 지정 비교 (Fixed-target repeatability, 전체 피험자)

| 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|
| PLS_v2 minus Global Linear | Peak 표준편차 | 10 | -0.026 | 0.091 | 0.029 | 5/10 | 0.922 | 1.000 |
| BiGRU minus Global Linear | Peak 표준편차 | 10 | 0.180 | 0.261 | 0.083 | 2/10 | 0.049 | 0.109 |
| PLS_v2 minus HbI-binned mean | Peak 표준편차 | 10 | -1.307 | 0.570 | 0.180 | 10/10 | 0.002 | 0.002 |
| BiGRU minus HbI-binned mean | Peak 표준편차 | 10 | -1.101 | 0.672 | 0.212 | 9/10 | 0.004 | 0.021 |
| BiGRU minus PLS_v2 | Peak 표준편차 | 10 | 0.205 | 0.293 | 0.093 | 3/10 | 0.049 | 0.344 |
| BiGRU minus BiGRU (fixed setting) | Peak 표준편차 | 10 | -0.025 | 0.177 | 0.056 | 5/10 | 1.000 | 1.000 |
| PLS_v2 minus Global Linear | spectral 표준편차 | 10 | -0.620 | 1.123 | 0.355 | 7/10 | 0.160 | 0.344 |
| BiGRU minus Global Linear | spectral 표준편차 | 10 | -0.507 | 0.902 | 0.285 | 8/10 | 0.049 | 0.109 |
| PLS_v2 minus HbI-binned mean | spectral 표준편차 | 10 | -5.009 | 2.335 | 0.738 | 10/10 | 0.002 | 0.002 |
| BiGRU minus HbI-binned mean | spectral 표준편차 | 10 | -4.895 | 2.080 | 0.658 | 10/10 | 0.002 | 0.002 |
| BiGRU minus PLS_v2 | spectral 표준편차 | 10 | 0.113 | 1.213 | 0.384 | 4/10 | 0.846 | 0.754 |
| BiGRU minus BiGRU (fixed setting) | spectral 표준편차 | 10 | -0.132 | 0.260 | 0.082 | 7/10 | 0.160 | 0.344 |

**HbI 변화 크기 구간별 absorbance RMSE** (|ΔHbI| 구간, 방향 전체; n = 그 구간에 쌍이 있는 집계 단위 수. 해당 구간에 쌍이 없는 피험자는 그 구간의 n에서 빠진다)

방향: 전체

| 모델 | 0.000-0.025 | 0.025-0.050 | 0.050-0.100 | 0.100-0.300 |
|---|---|---|---|---|
| HbI-binned mean | 13.912 ± 3.686 (n=10) | 17.695 ± 4.994 (n=10) | 17.674 ± 3.518 (n=9) | 18.177 ± 5.280 (n=8) |
| Global Linear | 8.712 ± 1.495 (n=10) | 9.547 ± 2.663 (n=10) | 13.016 ± 2.918 (n=9) | 17.421 ± 7.829 (n=8) |
| PLS_v2 | 8.511 ± 1.500 (n=10) | 8.704 ± 2.471 (n=10) | 11.439 ± 3.527 (n=9) | 15.149 ± 7.538 (n=8) |
| BiGRU | 8.596 ± 1.649 (n=10) | 9.059 ± 2.337 (n=10) | 12.112 ± 3.256 (n=9) | 15.564 ± 5.613 (n=8) |
| PLS_v1 | 9.280 ± 1.552 (n=10) | 10.144 ± 1.983 (n=10) | 11.878 ± 2.435 (n=9) | 14.847 ± 6.428 (n=8) |
| Ridge | 8.552 ± 1.453 (n=10) | 8.794 ± 2.632 (n=10) | 11.558 ± 3.592 (n=9) | 15.446 ± 7.410 (n=8) |
| BiGRU (fixed setting) | 8.585 ± 1.580 (n=10) | 8.890 ± 2.285 (n=10) | 12.036 ± 2.885 (n=9) | 15.154 ± 6.259 (n=8) |
| BiGRU (Peak-constrained selection) | 8.713 ± 1.495 (n=10) | 9.535 ± 2.667 (n=10) | 12.944 ± 2.920 (n=9) | 17.160 ± 7.250 (n=8) |

방향: 증가 (ΔHbI > 0)

| 모델 | 0.000-0.025 | 0.025-0.050 | 0.050-0.100 | 0.100-0.300 |
|---|---|---|---|---|
| HbI-binned mean | 14.004 ± 3.696 (n=10) | 17.695 ± 4.994 (n=10) | 17.674 ± 3.518 (n=9) | 18.177 ± 5.280 (n=8) |
| Global Linear | 8.744 ± 1.538 (n=10) | 9.547 ± 2.663 (n=10) | 13.016 ± 2.918 (n=9) | 17.421 ± 7.829 (n=8) |
| PLS_v2 | 8.570 ± 1.551 (n=10) | 8.826 ± 2.538 (n=10) | 11.741 ± 3.703 (n=9) | 17.091 ± 9.622 (n=8) |
| BiGRU | 8.634 ± 1.698 (n=10) | 8.914 ± 2.289 (n=10) | 11.735 ± 3.096 (n=9) | 15.256 ± 6.506 (n=8) |
| PLS_v1 | 8.986 ± 2.133 (n=10) | 9.710 ± 3.481 (n=10) | 12.367 ± 4.195 (n=9) | 15.319 ± 10.027 (n=8) |
| Ridge | 8.596 ± 1.500 (n=10) | 8.835 ± 2.619 (n=10) | 11.636 ± 3.658 (n=9) | 16.844 ± 9.139 (n=8) |
| BiGRU (fixed setting) | 8.627 ± 1.644 (n=10) | 8.817 ± 2.231 (n=10) | 11.900 ± 2.946 (n=9) | 14.687 ± 6.974 (n=8) |
| BiGRU (Peak-constrained selection) | 8.746 ± 1.538 (n=10) | 9.539 ± 2.665 (n=10) | 12.974 ± 2.917 (n=9) | 17.323 ± 7.609 (n=8) |

방향: 감소 (ΔHbI < 0)

| 모델 | 0.000-0.025 | 0.025-0.050 | 0.050-0.100 | 0.100-0.300 |
|---|---|---|---|---|
| HbI-binned mean | 14.004 ± 3.696 (n=10) | 17.695 ± 4.994 (n=10) | 17.674 ± 3.518 (n=9) | 18.177 ± 5.280 (n=8) |
| Global Linear | 8.744 ± 1.538 (n=10) | 9.547 ± 2.663 (n=10) | 13.016 ± 2.918 (n=9) | 17.421 ± 7.829 (n=8) |
| PLS_v2 | 8.508 ± 1.531 (n=10) | 8.575 ± 2.424 (n=10) | 11.098 ± 3.454 (n=9) | 12.646 ± 5.392 (n=8) |
| BiGRU | 8.620 ± 1.687 (n=10) | 9.189 ± 2.432 (n=10) | 12.420 ± 3.592 (n=9) | 15.549 ± 5.268 (n=8) |
| PLS_v1 | 9.574 ± 1.455 (n=10) | 10.261 ± 1.625 (n=10) | 10.797 ± 2.919 (n=9) | 13.127 ± 4.561 (n=8) |
| Ridge | 8.565 ± 1.491 (n=10) | 8.750 ± 2.656 (n=10) | 11.439 ± 3.670 (n=9) | 13.454 ± 6.361 (n=8) |
| BiGRU (fixed setting) | 8.604 ± 1.612 (n=10) | 8.955 ± 2.362 (n=10) | 12.129 ± 2.971 (n=9) | 15.386 ± 5.924 (n=8) |
| BiGRU (Peak-constrained selection) | 8.744 ± 1.538 (n=10) | 9.530 ± 2.669 (n=10) | 12.914 ± 2.925 (n=9) | 16.989 ± 6.887 (n=8) |

Peak RMSE, 방향 전체

| 모델 | 0.000-0.025 | 0.025-0.050 | 0.050-0.100 | 0.100-0.300 |
|---|---|---|---|---|
| HbI-binned mean | 1.654 ± 0.760 (n=10) | 2.832 ± 0.688 (n=10) | 2.568 ± 1.286 (n=9) | 3.052 ± 1.032 (n=8) |
| Global Linear | 0.672 ± 0.240 (n=10) | 0.643 ± 0.263 (n=10) | 0.689 ± 0.209 (n=9) | 0.773 ± 0.334 (n=8) |
| PLS_v2 | 0.677 ± 0.243 (n=10) | 0.635 ± 0.238 (n=10) | 0.698 ± 0.207 (n=9) | 0.725 ± 0.234 (n=8) |
| BiGRU | 0.673 ± 0.239 (n=10) | 0.745 ± 0.407 (n=10) | 0.858 ± 0.435 (n=9) | 1.398 ± 1.000 (n=8) |
| PLS_v1 | 0.700 ± 0.241 (n=10) | 0.673 ± 0.273 (n=10) | 0.720 ± 0.224 (n=9) | 0.803 ± 0.339 (n=8) |
| Ridge | 0.678 ± 0.245 (n=10) | 0.631 ± 0.241 (n=10) | 0.700 ± 0.232 (n=9) | 0.739 ± 0.225 (n=8) |
| BiGRU (fixed setting) | 0.670 ± 0.244 (n=10) | 0.736 ± 0.376 (n=10) | 0.872 ± 0.423 (n=9) | 1.492 ± 1.137 (n=8) |
| BiGRU (Peak-constrained selection) | 0.672 ± 0.241 (n=10) | 0.642 ± 0.261 (n=10) | 0.689 ± 0.209 (n=9) | 0.771 ± 0.333 (n=8) |

구간별 쌍체 차이 (방향 전체; 증가·감소 방향은 `comparisons_delta_hbi.csv`)

| 구간 | 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|---|
| dh|0.000-0.025|all | PLS_v2 minus Global Linear | absorbance RMSE | 10 | -0.201 | 0.228 | 0.072 | 9/10 | 0.020 | 0.021 |
| dh|0.000-0.025|all | BiGRU minus Global Linear | absorbance RMSE | 10 | -0.116 | 0.274 | 0.087 | 6/10 | 0.275 | 0.754 |
| dh|0.000-0.025|all | BiGRU minus PLS_v2 | absorbance RMSE | 10 | 0.085 | 0.229 | 0.072 | 3/10 | 0.275 | 0.344 |
| dh|0.000-0.025|all | PLS_v2 minus Global Linear | Peak RMSE | 10 | 0.005 | 0.008 | 0.002 | 2/10 | 0.037 | 0.109 |
| dh|0.000-0.025|all | BiGRU minus Global Linear | Peak RMSE | 10 | 0.001 | 0.013 | 0.004 | 3/10 | 0.492 | 0.344 |
| dh|0.000-0.025|all | BiGRU minus PLS_v2 | Peak RMSE | 10 | -0.004 | 0.017 | 0.005 | 5/10 | 0.625 | 1.000 |
| dh|0.025-0.050|all | PLS_v2 minus Global Linear | absorbance RMSE | 10 | -0.842 | 0.729 | 0.230 | 8/10 | 0.010 | 0.109 |
| dh|0.025-0.050|all | BiGRU minus Global Linear | absorbance RMSE | 10 | -0.488 | 0.658 | 0.208 | 8/10 | 0.037 | 0.109 |
| dh|0.025-0.050|all | BiGRU minus PLS_v2 | absorbance RMSE | 10 | 0.355 | 0.691 | 0.219 | 4/10 | 0.275 | 0.754 |
| dh|0.025-0.050|all | PLS_v2 minus Global Linear | Peak RMSE | 10 | -0.008 | 0.055 | 0.017 | 5/10 | 0.922 | 1.000 |
| dh|0.025-0.050|all | BiGRU minus Global Linear | Peak RMSE | 10 | 0.102 | 0.234 | 0.074 | 4/10 | 0.322 | 0.754 |
| dh|0.025-0.050|all | BiGRU minus PLS_v2 | Peak RMSE | 10 | 0.110 | 0.282 | 0.089 | 5/10 | 0.492 | 1.000 |
| dh|0.050-0.100|all | PLS_v2 minus Global Linear | absorbance RMSE | 9 | -1.577 | 2.544 | 0.848 | 7/9 | 0.027 | 0.180 |
| dh|0.050-0.100|all | BiGRU minus Global Linear | absorbance RMSE | 9 | -0.904 | 1.288 | 0.429 | 7/9 | 0.055 | 0.180 |
| dh|0.050-0.100|all | BiGRU minus PLS_v2 | absorbance RMSE | 9 | 0.673 | 2.982 | 0.994 | 5/9 | 0.910 | 1.000 |
| dh|0.050-0.100|all | PLS_v2 minus Global Linear | Peak RMSE | 9 | 0.009 | 0.060 | 0.020 | 3/9 | 0.359 | 0.508 |
| dh|0.050-0.100|all | BiGRU minus Global Linear | Peak RMSE | 9 | 0.169 | 0.358 | 0.119 | 3/9 | 0.164 | 0.508 |
| dh|0.050-0.100|all | BiGRU minus PLS_v2 | Peak RMSE | 9 | 0.160 | 0.363 | 0.121 | 3/9 | 0.301 | 0.508 |
| dh|0.100-0.300|all | PLS_v2 minus Global Linear | absorbance RMSE | 8 | -2.271 | 3.808 | 1.346 | 6/8 | 0.195 | 0.289 |
| dh|0.100-0.300|all | BiGRU minus Global Linear | absorbance RMSE | 8 | -1.857 | 3.896 | 1.378 | 5/8 | 0.195 | 0.727 |
| dh|0.100-0.300|all | BiGRU minus PLS_v2 | absorbance RMSE | 8 | 0.414 | 3.816 | 1.349 | 3/8 | 0.547 | 0.727 |
| dh|0.100-0.300|all | PLS_v2 minus Global Linear | Peak RMSE | 8 | -0.048 | 0.321 | 0.113 | 4/8 | 0.945 | 1.000 |
| dh|0.100-0.300|all | BiGRU minus Global Linear | Peak RMSE | 8 | 0.625 | 0.927 | 0.328 | 2/8 | 0.109 | 0.289 |
| dh|0.100-0.300|all | BiGRU minus PLS_v2 | Peak RMSE | 8 | 0.673 | 1.104 | 0.390 | 3/8 | 0.195 | 0.727 |

### Development10_Holdout73 (보조 프로토콜)

집계: 피험자 -> 분할 안 평균 -> 분할 10개의 평균 ± 표준편차. 분할끼리 피험자가 겹치므로 분할 간 표준편차를 독립 표본의 오차로 해석하지 않는다.

쌍체 비교: 분석 단위 = 피험자 (피험자별로 분할 평균을 낸 뒤 n = 10; 주 표의 평균과 쌍체 표의 평균이 다를 수 있음). 차이 = 앞 모델 - 뒤 모델, 개선 = 차이 < 0. peak_bias와 bias는 |값| 감소를 개선으로 판정. p 값은 다중 비교 보정이 없는 기술적 참고값 (정확 Wilcoxon signed-rank, 부호검정, 양측).

**General reconstruction** (Peak bias는 부호 있는 값, 예측 − 정답)

| 모델 | absorbance RMSE | reflectance RMSE | Peak RMSE | Peak bias |
|---|---|---|---|---|
| HbI-binned mean | 17.632 ± 1.979 | 1.230 ± 0.138 | 2.422 ± 0.625 | -0.000 ± 0.000 |
| Global Linear | 12.049 ± 1.841 | 0.913 ± 0.136 | 0.687 ± 0.109 | -0.000 ± 0.000 |
| PLS_v2 | 11.850 ± 2.518 | 0.929 ± 0.173 | 0.687 ± 0.104 | -0.003 ± 0.040 |
| BiGRU | 11.572 ± 1.346 | 0.877 ± 0.097 | 0.950 ± 0.407 | 0.073 ± 0.075 |

주의: General reconstruction은 피험자 안의 모든 순서쌍을 쓰므로, 보정이 i → j와 j → i에서 정확히 반대 부호인 HbI-binned mean과 Global Linear는 Peak bias가 구조적으로 0이다 (오차가 쌍끼리 상쇄). 따라서 이 두 모델을 뒤 모델로 둔 Peak bias 비교의 k/n과 p 값은 정보가 없고, 앞 모델의 부호 있는 bias 값 자체만 참고한다.

사전 지정 비교 (General reconstruction)

| 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|
| PLS_v2 minus Global Linear | absorbance RMSE | 10 | -0.185 | 1.395 | 0.441 | 7/10 | 0.492 | 0.344 |
| BiGRU minus Global Linear | absorbance RMSE | 10 | -0.467 | 0.987 | 0.312 | 7/10 | 0.193 | 0.344 |
| PLS_v2 minus HbI-binned mean | absorbance RMSE | 10 | -5.587 | 4.426 | 1.399 | 9/10 | 0.004 | 0.021 |
| BiGRU minus HbI-binned mean | absorbance RMSE | 10 | -5.869 | 3.814 | 1.206 | 10/10 | 0.002 | 0.002 |
| BiGRU minus PLS_v2 | absorbance RMSE | 10 | -0.282 | 1.727 | 0.546 | 5/10 | 0.922 | 1.000 |
| BiGRU minus BiGRU (fixed setting) | absorbance RMSE | 10 | 0.257 | 0.246 | 0.078 | 0/10 | 0.002 | 0.002 |
| PLS_v2 minus Global Linear | reflectance RMSE | 10 | 0.020 | 0.165 | 0.052 | 7/10 | 0.770 | 0.344 |
| BiGRU minus Global Linear | reflectance RMSE | 10 | -0.034 | 0.100 | 0.032 | 8/10 | 0.193 | 0.109 |
| PLS_v2 minus HbI-binned mean | reflectance RMSE | 10 | -0.285 | 0.354 | 0.112 | 9/10 | 0.027 | 0.021 |
| BiGRU minus HbI-binned mean | reflectance RMSE | 10 | -0.339 | 0.274 | 0.087 | 10/10 | 0.002 | 0.002 |
| BiGRU minus PLS_v2 | reflectance RMSE | 10 | -0.054 | 0.163 | 0.052 | 7/10 | 0.375 | 0.344 |
| BiGRU minus BiGRU (fixed setting) | reflectance RMSE | 10 | 0.024 | 0.025 | 0.008 | 3/10 | 0.037 | 0.344 |
| PLS_v2 minus Global Linear | Peak RMSE | 10 | 0.000 | 0.047 | 0.015 | 3/10 | 0.557 | 0.344 |
| BiGRU minus Global Linear | Peak RMSE | 10 | 0.286 | 0.346 | 0.109 | 1/10 | 0.010 | 0.021 |
| PLS_v2 minus HbI-binned mean | Peak RMSE | 10 | -1.727 | 0.649 | 0.205 | 10/10 | 0.002 | 0.002 |
| BiGRU minus HbI-binned mean | Peak RMSE | 10 | -1.442 | 0.899 | 0.284 | 9/10 | 0.004 | 0.021 |
| BiGRU minus PLS_v2 | Peak RMSE | 10 | 0.285 | 0.375 | 0.119 | 3/10 | 0.027 | 0.344 |
| BiGRU minus BiGRU (fixed setting) | Peak RMSE | 10 | -0.100 | 0.142 | 0.045 | 6/10 | 0.131 | 0.754 |
| PLS_v2 minus Global Linear | Peak bias | 10 | -0.003 | 0.020 | 0.006 | 0/10 | 0.002 | 0.002 |
| BiGRU minus Global Linear | Peak bias | 10 | 0.075 | 0.154 | 0.049 | 0/10 | 0.002 | 0.002 |
| PLS_v2 minus HbI-binned mean | Peak bias | 10 | -0.003 | 0.020 | 0.006 | 0/10 | 0.002 | 0.002 |
| BiGRU minus HbI-binned mean | Peak bias | 10 | 0.075 | 0.154 | 0.049 | 0/10 | 0.002 | 0.002 |
| BiGRU minus PLS_v2 | Peak bias | 10 | 0.078 | 0.153 | 0.048 | 2/10 | 0.010 | 0.109 |
| BiGRU minus BiGRU (fixed setting) | Peak bias | 10 | -0.028 | 0.049 | 0.016 | 8/10 | 0.037 | 0.109 |

보조 표: Supplementary 모델 (General reconstruction)

| 모델 | absorbance RMSE | reflectance RMSE | Peak RMSE | Peak bias |
|---|---|---|---|---|
| PLS_v1 | 11.910 ± 1.456 | 0.946 ± 0.153 | 0.736 ± 0.157 | -0.034 ± 0.251 |
| Ridge | 11.694 ± 1.995 | 0.923 ± 0.152 | 0.687 ± 0.113 | 0.008 ± 0.022 |
| BiGRU (fixed setting) | 11.322 ± 1.436 | 0.855 ± 0.096 | 1.041 ± 0.591 | 0.099 ± 0.127 |
| BiGRU (Peak-constrained selection) | 11.996 ± 1.816 | 0.908 ± 0.131 | 0.694 ± 0.125 | -0.001 ± 0.004 |
| PLS_v2 (max 5 components) | 11.971 ± 2.352 | 0.928 ± 0.167 | 0.693 ± 0.099 | -0.018 ± 0.041 |
| PLS_v1 (max 5 components) | 11.935 ± 1.476 | 0.948 ± 0.154 | 0.740 ± 0.158 | -0.038 ± 0.249 |

| 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|
| PLS_v1 minus Global Linear | absorbance RMSE | 10 | -0.225 | 2.100 | 0.664 | 7/10 | 0.625 | 0.344 |
| Ridge minus Global Linear | absorbance RMSE | 10 | -0.337 | 0.942 | 0.298 | 7/10 | 0.275 | 0.344 |
| BiGRU (fixed setting) minus Global Linear | absorbance RMSE | 10 | -0.723 | 1.091 | 0.345 | 8/10 | 0.064 | 0.109 |
| BiGRU (Peak-constrained selection) minus Global Linear | absorbance RMSE | 10 | -0.060 | 0.118 | 0.037 | 4/10 | 0.125 | 0.125 |
| PLS_v2 (max 5 components) minus Global Linear | absorbance RMSE | 10 | -0.064 | 0.891 | 0.282 | 6/10 | 0.770 | 0.754 |
| PLS_v1 (max 5 components) minus Global Linear | absorbance RMSE | 10 | -0.200 | 2.038 | 0.644 | 7/10 | 0.625 | 0.344 |
| BiGRU (Peak-constrained selection) minus BiGRU | absorbance RMSE | 10 | 0.406 | 0.925 | 0.293 | 3/10 | 0.232 | 0.344 |
| PLS_v1 minus Global Linear | reflectance RMSE | 10 | 0.026 | 0.101 | 0.032 | 4/10 | 0.557 | 0.754 |
| Ridge minus Global Linear | reflectance RMSE | 10 | 0.014 | 0.104 | 0.033 | 7/10 | 0.922 | 0.344 |
| BiGRU (fixed setting) minus Global Linear | reflectance RMSE | 10 | -0.059 | 0.107 | 0.034 | 8/10 | 0.084 | 0.109 |
| BiGRU (Peak-constrained selection) minus Global Linear | reflectance RMSE | 10 | -0.006 | 0.011 | 0.003 | 3/10 | 0.250 | 0.250 |
| PLS_v2 (max 5 components) minus Global Linear | reflectance RMSE | 10 | 0.018 | 0.120 | 0.038 | 6/10 | 0.846 | 0.754 |
| PLS_v1 (max 5 components) minus Global Linear | reflectance RMSE | 10 | 0.028 | 0.094 | 0.030 | 4/10 | 0.432 | 0.754 |
| BiGRU (Peak-constrained selection) minus BiGRU | reflectance RMSE | 10 | 0.029 | 0.096 | 0.030 | 3/10 | 0.322 | 0.344 |
| PLS_v1 minus Global Linear | Peak RMSE | 10 | 0.053 | 0.060 | 0.019 | 1/10 | 0.010 | 0.021 |
| Ridge minus Global Linear | Peak RMSE | 10 | 0.000 | 0.054 | 0.017 | 3/10 | 0.432 | 0.344 |
| BiGRU (fixed setting) minus Global Linear | Peak RMSE | 10 | 0.385 | 0.478 | 0.151 | 1/10 | 0.020 | 0.021 |
| BiGRU (Peak-constrained selection) minus Global Linear | Peak RMSE | 10 | 0.009 | 0.018 | 0.006 | 0/10 | 0.250 | 0.250 |
| PLS_v2 (max 5 components) minus Global Linear | Peak RMSE | 10 | 0.006 | 0.035 | 0.011 | 2/10 | 0.322 | 0.109 |
| PLS_v1 (max 5 components) minus Global Linear | Peak RMSE | 10 | 0.057 | 0.075 | 0.024 | 2/10 | 0.014 | 0.109 |
| BiGRU (Peak-constrained selection) minus BiGRU | Peak RMSE | 10 | -0.277 | 0.335 | 0.106 | 9/10 | 0.010 | 0.021 |
| PLS_v1 minus Global Linear | Peak bias | 10 | -0.048 | 0.253 | 0.080 | 0/10 | 0.002 | 0.002 |
| Ridge minus Global Linear | Peak bias | 10 | 0.008 | 0.026 | 0.008 | 0/10 | 0.002 | 0.002 |
| BiGRU (fixed setting) minus Global Linear | Peak bias | 10 | 0.103 | 0.187 | 0.059 | 0/10 | 0.002 | 0.002 |
| BiGRU (Peak-constrained selection) minus Global Linear | Peak bias | 10 | -0.001 | 0.003 | 0.001 | 0/10 | 0.125 | 0.125 |
| PLS_v2 (max 5 components) minus Global Linear | Peak bias | 10 | -0.018 | 0.025 | 0.008 | 0/10 | 0.002 | 0.002 |
| PLS_v1 (max 5 components) minus Global Linear | Peak bias | 10 | -0.053 | 0.261 | 0.082 | 0/10 | 0.002 | 0.002 |
| BiGRU (Peak-constrained selection) minus BiGRU | Peak bias | 10 | -0.076 | 0.155 | 0.049 | 10/10 | 0.002 | 0.002 |

**Near-target accuracy, 허용 폭 0.02** (대상 피험자 S003 S004 S008; 기술 통계만)

| 모델 | absorbance RMSE | Peak RMSE |
|---|---|---|
| HbI-binned mean | 18.416 ± 1.215 (n=10) | 1.729 ± 0.639 (n=10) |
| Global Linear | 11.178 ± 3.452 (n=10) | 0.653 ± 0.134 (n=10) |
| PLS_v2 | 11.316 ± 3.245 (n=10) | 0.712 ± 0.167 (n=10) |
| BiGRU | 10.146 ± 2.532 (n=10) | 0.928 ± 0.636 (n=10) |
| PLS_v1 | 8.900 ± 2.963 (n=10) | 0.804 ± 0.323 (n=10) |
| Ridge | 11.307 ± 3.408 (n=10) | 0.698 ± 0.184 (n=10) |
| BiGRU (fixed setting) | 10.127 ± 2.900 (n=10) | 1.045 ± 0.915 (n=10) |
| BiGRU (Peak-constrained selection) | 11.121 ± 3.436 (n=10) | 0.656 ± 0.137 (n=10) |
| PLS_v2 (max 5 components) | 11.790 ± 3.418 (n=10) | 0.694 ± 0.170 (n=10) |
| PLS_v1 (max 5 components) | 8.987 ± 2.968 (n=10) | 0.824 ± 0.342 (n=10) |

**Near-target accuracy, 허용 폭 0.03** (대상 피험자 S001 S002 S003 S004 S005 S008 S009 S010; 예상 8명과 일치: True)

| 모델 | absorbance RMSE | Peak RMSE |
|---|---|---|
| HbI-binned mean | 16.776 ± 1.432 (n=10) | 1.991 ± 0.480 (n=10) |
| Global Linear | 12.068 ± 1.622 (n=10) | 0.702 ± 0.081 (n=10) |
| PLS_v2 | 11.799 ± 3.131 (n=10) | 0.698 ± 0.118 (n=10) |
| BiGRU | 11.540 ± 1.048 (n=10) | 0.849 ± 0.337 (n=10) |
| PLS_v1 | 12.042 ± 2.282 (n=10) | 0.787 ± 0.155 (n=10) |
| Ridge | 11.798 ± 2.553 (n=10) | 0.700 ± 0.121 (n=10) |
| BiGRU (fixed setting) | 11.572 ± 1.060 (n=10) | 0.936 ± 0.478 (n=10) |
| BiGRU (Peak-constrained selection) | 12.055 ± 1.619 (n=10) | 0.705 ± 0.086 (n=10) |
| PLS_v2 (max 5 components) | 12.020 ± 2.688 (n=10) | 0.711 ± 0.081 (n=10) |
| PLS_v1 (max 5 components) | 12.057 ± 2.285 (n=10) | 0.792 ± 0.155 (n=10) |

| 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|
| PLS_v2 minus Global Linear | absorbance RMSE | 8 | 0.058 | 2.384 | 0.843 | 4/8 | 1.000 | 1.000 |
| BiGRU minus Global Linear | absorbance RMSE | 8 | -0.604 | 2.402 | 0.849 | 4/8 | 0.844 | 1.000 |
| PLS_v2 minus HbI-binned mean | absorbance RMSE | 8 | -4.319 | 1.987 | 0.703 | 8/8 | 0.008 | 0.008 |
| BiGRU minus HbI-binned mean | absorbance RMSE | 8 | -4.981 | 2.158 | 0.763 | 8/8 | 0.008 | 0.008 |
| BiGRU minus PLS_v2 | absorbance RMSE | 8 | -0.662 | 2.800 | 0.990 | 5/8 | 0.461 | 0.727 |
| BiGRU minus BiGRU (fixed setting) | absorbance RMSE | 8 | -0.072 | 0.199 | 0.070 | 5/8 | 0.461 | 0.727 |
| PLS_v2 minus Global Linear | Peak RMSE | 8 | 0.002 | 0.100 | 0.035 | 1/8 | 0.195 | 0.070 |
| BiGRU minus Global Linear | Peak RMSE | 8 | 0.185 | 0.318 | 0.113 | 1/8 | 0.109 | 0.070 |
| PLS_v2 minus HbI-binned mean | Peak RMSE | 8 | -1.243 | 0.592 | 0.209 | 8/8 | 0.008 | 0.008 |
| BiGRU minus HbI-binned mean | Peak RMSE | 8 | -1.060 | 0.882 | 0.312 | 7/8 | 0.016 | 0.070 |
| BiGRU minus PLS_v2 | Peak RMSE | 8 | 0.183 | 0.345 | 0.122 | 3/8 | 0.250 | 0.727 |
| BiGRU minus BiGRU (fixed setting) | Peak RMSE | 8 | -0.102 | 0.129 | 0.045 | 6/8 | 0.055 | 0.289 |

**Fixed-target repeatability** (테스트 측정 전부를 HbI 0.6으로 복원했을 때의 개인 내 표준편차; n = 집계 단위 수)

전체

| 모델 | Peak 표준편차 | spectral 표준편차 |
|---|---|---|
| Raw | 2.148 ± 0.443 (n=10) | 22.613 ± 5.106 (n=10) |
| HbI-binned mean | 1.713 ± 0.442 (n=10) | 11.830 ± 1.364 (n=10) |
| Global Linear | 0.486 ± 0.077 (n=10) | 7.273 ± 1.036 (n=10) |
| PLS_v2 | 0.490 ± 0.078 (n=10) | 7.191 ± 1.380 (n=10) |
| BiGRU | 0.673 ± 0.296 (n=10) | 6.992 ± 0.811 (n=10) |
| PLS_v1 | 0.495 ± 0.080 (n=10) | 4.231 ± 0.838 (n=10) |
| Ridge | 0.490 ± 0.084 (n=10) | 7.100 ± 1.212 (n=10) |
| BiGRU (fixed setting) | 0.734 ± 0.397 (n=10) | 7.049 ± 0.912 (n=10) |
| BiGRU (Peak-constrained selection) | 0.491 ± 0.088 (n=10) | 7.257 ± 1.022 (n=10) |
| PLS_v2 (max 5 components) | 0.493 ± 0.073 (n=10) | 7.316 ± 1.076 (n=10) |
| PLS_v1 (max 5 components) | 0.500 ± 0.081 (n=10) | 4.250 ± 0.826 (n=10) |

In-range subject

| 모델 | Peak 표준편차 | spectral 표준편차 |
|---|---|---|
| Raw | 3.038 ± 0.216 (n=8) | 34.188 ± 6.656 (n=8) |
| HbI-binned mean | 1.576 ± 0.560 (n=8) | 11.546 ± 1.597 (n=8) |
| Global Linear | 0.559 ± 0.134 (n=8) | 8.085 ± 1.469 (n=8) |
| PLS_v2 | 0.529 ± 0.153 (n=8) | 8.364 ± 1.737 (n=8) |
| BiGRU | 0.937 ± 0.456 (n=8) | 7.948 ± 0.686 (n=8) |
| PLS_v1 | 0.577 ± 0.134 (n=8) | 5.201 ± 1.466 (n=8) |
| Ridge | 0.531 ± 0.150 (n=8) | 7.907 ± 1.442 (n=8) |
| BiGRU (fixed setting) | 1.078 ± 0.559 (n=8) | 7.909 ± 0.950 (n=8) |
| BiGRU (Peak-constrained selection) | 0.568 ± 0.147 (n=8) | 8.048 ± 1.454 (n=8) |
| PLS_v2 (max 5 components) | 0.544 ± 0.136 (n=8) | 8.440 ± 1.766 (n=8) |
| PLS_v1 (max 5 components) | 0.585 ± 0.133 (n=8) | 5.173 ± 1.468 (n=8) |

그 밖의 피험자

| 모델 | Peak 표준편차 | spectral 표준편차 |
|---|---|---|
| Raw | 1.694 ± 0.489 (n=10) | 16.607 ± 4.143 (n=10) |
| HbI-binned mean | 1.734 ± 0.455 (n=10) | 11.676 ± 1.687 (n=10) |
| Global Linear | 0.429 ± 0.085 (n=10) | 6.631 ± 1.122 (n=10) |
| PLS_v2 | 0.459 ± 0.073 (n=10) | 6.358 ± 1.198 (n=10) |
| BiGRU | 0.464 ± 0.099 (n=10) | 6.390 ± 1.095 (n=10) |
| PLS_v1 | 0.432 ± 0.100 (n=10) | 3.605 ± 0.930 (n=10) |
| Ridge | 0.456 ± 0.082 (n=10) | 6.579 ± 1.131 (n=10) |
| BiGRU (fixed setting) | 0.488 ± 0.156 (n=10) | 6.525 ± 1.037 (n=10) |
| BiGRU (Peak-constrained selection) | 0.430 ± 0.086 (n=10) | 6.642 ± 1.125 (n=10) |
| PLS_v2 (max 5 components) | 0.458 ± 0.074 (n=10) | 6.516 ± 1.061 (n=10) |
| PLS_v1 (max 5 components) | 0.433 ± 0.100 (n=10) | 3.707 ± 0.906 (n=10) |

In-range subject 행은 test에 In-range subject가 있는 분할만 평균했다 (분할 3과 분할 5에는 없음; n이 그 분할 수).

사전 지정 비교 (Fixed-target repeatability, 전체 피험자)

| 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|
| PLS_v2 minus Global Linear | Peak 표준편차 | 10 | 0.004 | 0.046 | 0.015 | 3/10 | 0.625 | 0.344 |
| BiGRU minus Global Linear | Peak 표준편차 | 10 | 0.204 | 0.257 | 0.081 | 2/10 | 0.049 | 0.109 |
| PLS_v2 minus HbI-binned mean | Peak 표준편차 | 10 | -1.217 | 0.452 | 0.143 | 10/10 | 0.002 | 0.002 |
| BiGRU minus HbI-binned mean | Peak 표준편차 | 10 | -1.017 | 0.636 | 0.201 | 9/10 | 0.004 | 0.021 |
| BiGRU minus PLS_v2 | Peak 표준편차 | 10 | 0.199 | 0.272 | 0.086 | 2/10 | 0.037 | 0.109 |
| BiGRU minus BiGRU (fixed setting) | Peak 표준편차 | 10 | -0.065 | 0.093 | 0.029 | 7/10 | 0.084 | 0.344 |
| PLS_v2 minus Global Linear | spectral 표준편차 | 10 | -0.073 | 0.898 | 0.284 | 4/10 | 0.922 | 0.754 |
| BiGRU minus Global Linear | spectral 표준편차 | 10 | -0.276 | 0.822 | 0.260 | 5/10 | 0.492 | 1.000 |
| PLS_v2 minus HbI-binned mean | spectral 표준편차 | 10 | -4.527 | 2.752 | 0.870 | 10/10 | 0.002 | 0.002 |
| BiGRU minus HbI-binned mean | spectral 표준편차 | 10 | -4.729 | 2.275 | 0.720 | 10/10 | 0.002 | 0.002 |
| BiGRU minus PLS_v2 | spectral 표준편차 | 10 | -0.202 | 1.165 | 0.368 | 7/10 | 0.557 | 0.344 |
| BiGRU minus BiGRU (fixed setting) | spectral 표준편차 | 10 | -0.067 | 0.135 | 0.043 | 7/10 | 0.193 | 0.344 |

**HbI 변화 크기 구간별 absorbance RMSE** (|ΔHbI| 구간, 방향 전체; n = 그 구간에 쌍이 있는 집계 단위 수. 해당 구간에 쌍이 없는 피험자는 그 구간의 n에서 빠진다)

방향: 전체

| 모델 | 0.000-0.025 | 0.025-0.050 | 0.050-0.100 | 0.100-0.300 |
|---|---|---|---|---|
| HbI-binned mean | 14.061 ± 2.350 (n=10) | 17.887 ± 3.388 (n=10) | 18.310 ± 2.060 (n=10) | 18.771 ± 3.091 (n=10) |
| Global Linear | 8.624 ± 0.752 (n=10) | 9.420 ± 1.413 (n=10) | 12.928 ± 1.536 (n=10) | 17.405 ± 3.572 (n=10) |
| PLS_v2 | 8.510 ± 0.775 (n=10) | 8.941 ± 1.054 (n=10) | 12.036 ± 2.513 (n=10) | 17.285 ± 5.015 (n=10) |
| BiGRU | 8.546 ± 0.815 (n=10) | 9.023 ± 1.521 (n=10) | 12.310 ± 1.470 (n=10) | 16.641 ± 2.509 (n=10) |
| PLS_v1 | 9.832 ± 1.140 (n=10) | 10.608 ± 1.435 (n=10) | 12.201 ± 1.263 (n=10) | 14.927 ± 3.534 (n=10) |
| Ridge | 8.539 ± 0.788 (n=10) | 8.988 ± 1.378 (n=10) | 12.135 ± 2.358 (n=10) | 16.948 ± 3.784 (n=10) |
| BiGRU (fixed setting) | 8.527 ± 0.841 (n=10) | 8.915 ± 1.469 (n=10) | 12.089 ± 1.659 (n=10) | 15.873 ± 3.104 (n=10) |
| BiGRU (Peak-constrained selection) | 8.622 ± 0.751 (n=10) | 9.405 ± 1.396 (n=10) | 12.890 ± 1.555 (n=10) | 17.274 ± 3.522 (n=10) |

방향: 증가 (ΔHbI > 0)

| 모델 | 0.000-0.025 | 0.025-0.050 | 0.050-0.100 | 0.100-0.300 |
|---|---|---|---|---|
| HbI-binned mean | 14.150 ± 2.365 (n=10) | 17.887 ± 3.388 (n=10) | 18.310 ± 2.060 (n=10) | 18.771 ± 3.091 (n=10) |
| Global Linear | 8.660 ± 0.793 (n=10) | 9.420 ± 1.413 (n=10) | 12.928 ± 1.536 (n=10) | 17.405 ± 3.572 (n=10) |
| PLS_v2 | 8.568 ± 0.798 (n=10) | 9.022 ± 1.128 (n=10) | 12.299 ± 2.636 (n=10) | 18.747 ± 6.256 (n=10) |
| BiGRU | 8.571 ± 0.914 (n=10) | 8.904 ± 1.604 (n=10) | 12.163 ± 1.693 (n=10) | 15.733 ± 2.089 (n=10) |
| PLS_v1 | 9.533 ± 1.474 (n=10) | 9.788 ± 2.147 (n=10) | 12.467 ± 2.297 (n=10) | 15.251 ± 5.285 (n=10) |
| Ridge | 8.601 ± 0.833 (n=10) | 9.088 ± 1.382 (n=10) | 12.427 ± 2.599 (n=10) | 18.847 ± 5.211 (n=10) |
| BiGRU (fixed setting) | 8.570 ± 0.941 (n=10) | 8.863 ± 1.511 (n=10) | 11.992 ± 1.849 (n=10) | 15.324 ± 3.235 (n=10) |
| BiGRU (Peak-constrained selection) | 8.659 ± 0.792 (n=10) | 9.410 ± 1.402 (n=10) | 12.911 ± 1.543 (n=10) | 17.371 ± 3.555 (n=10) |

방향: 감소 (ΔHbI < 0)

| 모델 | 0.000-0.025 | 0.025-0.050 | 0.050-0.100 | 0.100-0.300 |
|---|---|---|---|---|
| HbI-binned mean | 14.150 ± 2.365 (n=10) | 17.887 ± 3.388 (n=10) | 18.310 ± 2.060 (n=10) | 18.771 ± 3.091 (n=10) |
| Global Linear | 8.660 ± 0.793 (n=10) | 9.420 ± 1.413 (n=10) | 12.928 ± 1.536 (n=10) | 17.405 ± 3.572 (n=10) |
| PLS_v2 | 8.521 ± 0.820 (n=10) | 8.853 ± 0.999 (n=10) | 11.737 ± 2.477 (n=10) | 15.209 ± 4.773 (n=10) |
| BiGRU | 8.591 ± 0.790 (n=10) | 9.128 ± 1.472 (n=10) | 12.383 ± 1.413 (n=10) | 17.100 ± 3.795 (n=10) |
| PLS_v1 | 10.140 ± 0.986 (n=10) | 11.016 ± 1.472 (n=10) | 11.265 ± 2.178 (n=10) | 13.383 ± 3.626 (n=10) |
| Ridge | 8.548 ± 0.820 (n=10) | 8.884 ± 1.385 (n=10) | 11.805 ± 2.215 (n=10) | 14.398 ± 3.316 (n=10) |
| BiGRU (fixed setting) | 8.556 ± 0.811 (n=10) | 8.956 ± 1.449 (n=10) | 12.132 ± 1.590 (n=10) | 16.133 ± 3.415 (n=10) |
| BiGRU (Peak-constrained selection) | 8.657 ± 0.792 (n=10) | 9.399 ± 1.390 (n=10) | 12.868 ± 1.570 (n=10) | 17.170 ± 3.517 (n=10) |

Peak RMSE, 방향 전체

| 모델 | 0.000-0.025 | 0.025-0.050 | 0.050-0.100 | 0.100-0.300 |
|---|---|---|---|---|
| HbI-binned mean | 1.622 ± 0.605 (n=10) | 2.681 ± 0.836 (n=10) | 2.517 ± 0.820 (n=10) | 2.950 ± 1.148 (n=10) |
| Global Linear | 0.654 ± 0.138 (n=10) | 0.623 ± 0.128 (n=10) | 0.644 ± 0.114 (n=10) | 0.730 ± 0.223 (n=10) |
| PLS_v2 | 0.658 ± 0.137 (n=10) | 0.625 ± 0.109 (n=10) | 0.667 ± 0.102 (n=10) | 0.721 ± 0.153 (n=10) |
| BiGRU | 0.662 ± 0.150 (n=10) | 0.730 ± 0.264 (n=10) | 0.883 ± 0.418 (n=10) | 1.492 ± 0.855 (n=10) |
| PLS_v1 | 0.697 ± 0.183 (n=10) | 0.676 ± 0.170 (n=10) | 0.704 ± 0.162 (n=10) | 0.802 ± 0.244 (n=10) |
| Ridge | 0.657 ± 0.138 (n=10) | 0.624 ± 0.117 (n=10) | 0.664 ± 0.116 (n=10) | 0.709 ± 0.183 (n=10) |
| BiGRU (fixed setting) | 0.665 ± 0.170 (n=10) | 0.763 ± 0.374 (n=10) | 0.961 ± 0.641 (n=10) | 1.772 ± 1.341 (n=10) |
| BiGRU (Peak-constrained selection) | 0.654 ± 0.138 (n=10) | 0.626 ± 0.134 (n=10) | 0.652 ± 0.128 (n=10) | 0.753 ± 0.236 (n=10) |

구간별 쌍체 차이 (방향 전체; 증가·감소 방향은 `comparisons_delta_hbi.csv`)

| 구간 | 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|---|
| dh|0.000-0.025|all | PLS_v2 minus Global Linear | absorbance RMSE | 10 | -0.113 | 0.196 | 0.062 | 7/10 | 0.160 | 0.344 |
| dh|0.000-0.025|all | BiGRU minus Global Linear | absorbance RMSE | 10 | -0.085 | 0.150 | 0.048 | 6/10 | 0.160 | 0.754 |
| dh|0.000-0.025|all | BiGRU minus PLS_v2 | absorbance RMSE | 10 | 0.029 | 0.151 | 0.048 | 5/10 | 0.695 | 1.000 |
| dh|0.000-0.025|all | PLS_v2 minus Global Linear | Peak RMSE | 10 | 0.004 | 0.004 | 0.001 | 1/10 | 0.006 | 0.021 |
| dh|0.000-0.025|all | BiGRU minus Global Linear | Peak RMSE | 10 | 0.010 | 0.019 | 0.006 | 2/10 | 0.064 | 0.109 |
| dh|0.000-0.025|all | BiGRU minus PLS_v2 | Peak RMSE | 10 | 0.006 | 0.021 | 0.007 | 6/10 | 0.846 | 0.754 |
| dh|0.025-0.050|all | PLS_v2 minus Global Linear | absorbance RMSE | 10 | -0.483 | 0.703 | 0.222 | 7/10 | 0.084 | 0.344 |
| dh|0.025-0.050|all | BiGRU minus Global Linear | absorbance RMSE | 10 | -0.411 | 0.502 | 0.159 | 8/10 | 0.027 | 0.109 |
| dh|0.025-0.050|all | BiGRU minus PLS_v2 | absorbance RMSE | 10 | 0.073 | 0.556 | 0.176 | 6/10 | 0.846 | 0.754 |
| dh|0.025-0.050|all | PLS_v2 minus Global Linear | Peak RMSE | 10 | 0.001 | 0.033 | 0.010 | 3/10 | 0.557 | 0.344 |
| dh|0.025-0.050|all | BiGRU minus Global Linear | Peak RMSE | 10 | 0.113 | 0.197 | 0.062 | 1/10 | 0.014 | 0.021 |
| dh|0.025-0.050|all | BiGRU minus PLS_v2 | Peak RMSE | 10 | 0.112 | 0.223 | 0.071 | 3/10 | 0.084 | 0.344 |
| dh|0.050-0.100|all | PLS_v2 minus Global Linear | absorbance RMSE | 9 | -0.633 | 1.436 | 0.479 | 6/9 | 0.250 | 0.508 |
| dh|0.050-0.100|all | BiGRU minus Global Linear | absorbance RMSE | 9 | -0.511 | 1.153 | 0.384 | 6/9 | 0.359 | 0.508 |
| dh|0.050-0.100|all | BiGRU minus PLS_v2 | absorbance RMSE | 9 | 0.121 | 1.835 | 0.612 | 4/9 | 0.820 | 1.000 |
| dh|0.050-0.100|all | PLS_v2 minus Global Linear | Peak RMSE | 9 | 0.022 | 0.042 | 0.014 | 2/9 | 0.164 | 0.180 |
| dh|0.050-0.100|all | BiGRU minus Global Linear | Peak RMSE | 9 | 0.283 | 0.328 | 0.109 | 1/9 | 0.008 | 0.039 |
| dh|0.050-0.100|all | BiGRU minus PLS_v2 | Peak RMSE | 9 | 0.261 | 0.341 | 0.114 | 2/9 | 0.020 | 0.180 |
| dh|0.100-0.300|all | PLS_v2 minus Global Linear | absorbance RMSE | 8 | 0.429 | 3.142 | 1.111 | 4/8 | 0.945 | 1.000 |
| dh|0.100-0.300|all | BiGRU minus Global Linear | absorbance RMSE | 8 | -1.108 | 2.701 | 0.955 | 6/8 | 0.383 | 0.289 |
| dh|0.100-0.300|all | BiGRU minus PLS_v2 | absorbance RMSE | 8 | -1.537 | 4.322 | 1.528 | 5/8 | 0.383 | 0.727 |
| dh|0.100-0.300|all | PLS_v2 minus Global Linear | Peak RMSE | 8 | 0.029 | 0.161 | 0.057 | 3/8 | 0.641 | 0.727 |
| dh|0.100-0.300|all | BiGRU minus Global Linear | Peak RMSE | 8 | 0.783 | 0.712 | 0.252 | 1/8 | 0.023 | 0.070 |
| dh|0.100-0.300|all | BiGRU minus PLS_v2 | Peak RMSE | 8 | 0.754 | 0.786 | 0.278 | 1/8 | 0.023 | 0.070 |

### 4.2 파장 구간별 복원 오차

구간: Soret = 400–440 nm (5파장); Band 450-460 = 450–460 nm (2파장); Carotenoid = 470–510 nm (5파장); Q-band = 520–590 nm (7파장); Red = 600–700 nm (11파장). Q-band는 570 nm 제외. 5개 구간의 파장 수 합 = 30. 5개 구간의 오차 제곱합으로 다시 계산한 전체 absorbance RMSE가 `common/metrics.py`의 값과 같음을 모든 피험자·모델·seed에서 확인했다 (허용 1e-12).

### Development10_LeaveOneSubjectOut

구간 absorbance RMSE ×10⁻³

| 모델 | Soret (5) | Band 450-460 (2) | Carotenoid (5) | Q-band (7) | Red (11) |
|---|---|---|---|---|---|
| HbI-binned mean | 23.073 ± 6.599 | 18.094 ± 6.037 | 15.411 ± 5.906 | 18.681 ± 3.542 | 12.867 ± 2.776 |
| Global Linear | 19.288 ± 7.994 | 10.558 ± 4.679 | 7.480 ± 2.885 | 4.217 ± 0.925 | 12.200 ± 4.460 |
| PLS_v2 | 16.723 ± 6.449 | 7.951 ± 2.277 | 6.480 ± 1.716 | 4.114 ± 0.894 | 12.236 ± 4.199 |
| BiGRU | 18.250 ± 7.315 | 10.198 ± 3.983 | 7.202 ± 3.143 | 4.797 ± 1.435 | 10.942 ± 3.062 |
| PLS_v1 | 17.353 ± 5.669 | 9.571 ± 3.676 | 7.088 ± 2.532 | 4.078 ± 0.818 | 12.241 ± 3.921 |
| Ridge | 16.537 ± 7.204 | 7.840 ± 2.337 | 6.585 ± 1.774 | 4.311 ± 0.933 | 12.359 ± 4.090 |
| BiGRU (fixed setting) | 17.846 ± 6.642 | 9.957 ± 3.825 | 7.597 ± 3.429 | 4.571 ± 1.274 | 11.286 ± 3.107 |

구간 bias (예측 − 정답) ×10⁻³

| 모델 | Soret (5) | Band 450-460 (2) | Carotenoid (5) | Q-band (7) | Red (11) |
|---|---|---|---|---|---|
| HbI-binned mean | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.000 ± 0.000 |
| Global Linear | -0.000 ± 0.000 | -0.000 ± 0.000 | -0.000 ± 0.000 | -0.000 ± 0.000 | -0.000 ± 0.000 |
| PLS_v2 | 1.675 ± 2.051 | 0.044 ± 0.714 | -0.593 ± 0.456 | -0.352 ± 0.269 | -1.307 ± 1.134 |
| BiGRU | 0.329 ± 1.570 | 0.171 ± 1.347 | 0.225 ± 0.775 | -0.011 ± 0.577 | 0.031 ± 1.037 |
| PLS_v1 | 1.315 ± 10.567 | 0.253 ± 4.391 | -0.233 ± 2.398 | -0.406 ± 1.271 | -0.410 ± 7.223 |
| Ridge | 1.669 ± 1.889 | -0.012 ± 0.654 | -0.658 ± 0.455 | -0.361 ± 0.258 | -1.433 ± 1.078 |
| BiGRU (fixed setting) | 0.502 ± 2.321 | 0.100 ± 1.892 | 0.228 ± 1.099 | -0.067 ± 0.859 | -0.113 ± 0.806 |

570 nm: h_in → h* 고정, 오차 0 (모든 모델).

구간별 쌍체 차이

| 구간 | 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|---|
| band|Soret | PLS_v2 minus Global Linear | absorbance RMSE | 10 | -2.566 | 4.689 | 1.483 | 6/10 | 0.193 | 0.754 |
| band|Soret | BiGRU minus Global Linear | absorbance RMSE | 10 | -1.039 | 5.113 | 1.617 | 6/10 | 0.492 | 0.754 |
| band|Soret | BiGRU minus PLS_v2 | absorbance RMSE | 10 | 1.527 | 2.769 | 0.876 | 4/10 | 0.193 | 0.754 |
| band|Soret | PLS_v2 minus Global Linear | bias | 10 | 1.675 | 2.051 | 0.649 | 0/10 | 0.002 | 0.002 |
| band|Soret | BiGRU minus Global Linear | bias | 10 | 0.329 | 1.570 | 0.496 | 0/10 | 0.002 | 0.002 |
| band|Soret | BiGRU minus PLS_v2 | bias | 10 | -1.346 | 2.344 | 0.741 | 6/10 | 0.322 | 0.754 |
| band|Band 450-460 | PLS_v2 minus Global Linear | absorbance RMSE | 10 | -2.607 | 4.607 | 1.457 | 7/10 | 0.049 | 0.344 |
| band|Band 450-460 | BiGRU minus Global Linear | absorbance RMSE | 10 | -0.360 | 1.482 | 0.468 | 5/10 | 0.557 | 1.000 |
| band|Band 450-460 | BiGRU minus PLS_v2 | absorbance RMSE | 10 | 2.247 | 3.922 | 1.240 | 1/10 | 0.006 | 0.021 |
| band|Band 450-460 | PLS_v2 minus Global Linear | bias | 10 | 0.044 | 0.714 | 0.226 | 0/10 | 0.002 | 0.002 |
| band|Band 450-460 | BiGRU minus Global Linear | bias | 10 | 0.171 | 1.347 | 0.426 | 0/10 | 0.002 | 0.002 |
| band|Band 450-460 | BiGRU minus PLS_v2 | bias | 10 | 0.127 | 1.367 | 0.432 | 3/10 | 0.131 | 0.344 |
| band|Carotenoid | PLS_v2 minus Global Linear | absorbance RMSE | 10 | -0.999 | 2.744 | 0.868 | 7/10 | 0.275 | 0.344 |
| band|Carotenoid | BiGRU minus Global Linear | absorbance RMSE | 10 | -0.278 | 0.823 | 0.260 | 6/10 | 0.625 | 0.754 |
| band|Carotenoid | BiGRU minus PLS_v2 | absorbance RMSE | 10 | 0.722 | 2.869 | 0.907 | 5/10 | 1.000 | 1.000 |
| band|Carotenoid | PLS_v2 minus Global Linear | bias | 10 | -0.593 | 0.456 | 0.144 | 0/10 | 0.002 | 0.002 |
| band|Carotenoid | BiGRU minus Global Linear | bias | 10 | 0.225 | 0.775 | 0.245 | 0/10 | 0.002 | 0.002 |
| band|Carotenoid | BiGRU minus PLS_v2 | bias | 10 | 0.818 | 0.817 | 0.258 | 6/10 | 0.846 | 0.754 |
| band|Q-band | PLS_v2 minus Global Linear | absorbance RMSE | 10 | -0.103 | 0.763 | 0.241 | 6/10 | 0.557 | 0.754 |
| band|Q-band | BiGRU minus Global Linear | absorbance RMSE | 10 | 0.580 | 1.287 | 0.407 | 4/10 | 0.322 | 0.754 |
| band|Q-band | BiGRU minus PLS_v2 | absorbance RMSE | 10 | 0.684 | 0.995 | 0.315 | 2/10 | 0.037 | 0.109 |
| band|Q-band | PLS_v2 minus Global Linear | bias | 10 | -0.352 | 0.269 | 0.085 | 0/10 | 0.002 | 0.002 |
| band|Q-band | BiGRU minus Global Linear | bias | 10 | -0.011 | 0.577 | 0.182 | 0/10 | 0.002 | 0.002 |
| band|Q-band | BiGRU minus PLS_v2 | bias | 10 | 0.341 | 0.569 | 0.180 | 4/10 | 0.492 | 0.754 |
| band|Red | PLS_v2 minus Global Linear | absorbance RMSE | 10 | 0.036 | 3.057 | 0.967 | 7/10 | 0.492 | 0.344 |
| band|Red | BiGRU minus Global Linear | absorbance RMSE | 10 | -1.258 | 3.126 | 0.988 | 8/10 | 0.193 | 0.109 |
| band|Red | BiGRU minus PLS_v2 | absorbance RMSE | 10 | -1.293 | 2.494 | 0.789 | 8/10 | 0.105 | 0.109 |
| band|Red | PLS_v2 minus Global Linear | bias | 10 | -1.307 | 1.134 | 0.359 | 0/10 | 0.002 | 0.002 |
| band|Red | BiGRU minus Global Linear | bias | 10 | 0.031 | 1.037 | 0.328 | 0/10 | 0.002 | 0.002 |
| band|Red | BiGRU minus PLS_v2 | bias | 10 | 1.338 | 1.259 | 0.398 | 9/10 | 0.027 | 0.021 |

### Development10_Holdout73

구간 absorbance RMSE ×10⁻³

| 모델 | Soret (5) | Band 450-460 (2) | Carotenoid (5) | Q-band (7) | Red (11) |
|---|---|---|---|---|---|
| HbI-binned mean | 23.191 ± 3.404 | 18.496 ± 4.156 | 15.725 ± 4.643 | 19.117 ± 2.004 | 12.754 ± 1.503 |
| Global Linear | 19.016 ± 4.210 | 10.476 ± 2.536 | 7.337 ± 1.387 | 4.184 ± 0.457 | 12.190 ± 2.404 |
| PLS_v2 | 17.797 ± 5.186 | 8.813 ± 2.823 | 6.751 ± 1.395 | 4.266 ± 0.695 | 12.819 ± 2.796 |
| BiGRU | 18.343 ± 3.721 | 10.248 ± 1.984 | 7.304 ± 1.536 | 4.480 ± 0.442 | 11.757 ± 1.642 |
| PLS_v1 | 17.754 ± 2.959 | 9.975 ± 2.105 | 7.365 ± 1.471 | 4.300 ± 0.516 | 12.956 ± 2.686 |
| Ridge | 17.265 ± 4.594 | 8.784 ± 2.813 | 6.829 ± 1.404 | 4.291 ± 0.477 | 12.767 ± 2.560 |
| BiGRU (fixed setting) | 17.795 ± 3.775 | 10.130 ± 2.125 | 7.582 ± 1.808 | 4.533 ± 0.710 | 11.409 ± 1.595 |

구간 bias (예측 − 정답) ×10⁻³

| 모델 | Soret (5) | Band 450-460 (2) | Carotenoid (5) | Q-band (7) | Red (11) |
|---|---|---|---|---|---|
| HbI-binned mean | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Global Linear | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| PLS_v2 | 0.994 ± 1.466 | -0.335 ± 0.846 | -0.717 ± 0.572 | -0.343 ± 0.204 | -1.175 ± 0.999 |
| BiGRU | 0.833 ± 2.083 | 0.585 ± 1.797 | 0.408 ± 1.087 | -0.133 ± 0.430 | -0.171 ± 0.925 |
| PLS_v1 | 0.859 ± 8.024 | -0.074 ± 3.951 | -0.406 ± 2.983 | -0.149 ± 1.062 | -0.540 ± 6.973 |
| Ridge | 1.191 ± 0.969 | -0.002 ± 0.254 | -0.485 ± 0.413 | -0.286 ± 0.226 | -1.185 ± 0.970 |
| BiGRU (fixed setting) | 0.670 ± 1.874 | 0.309 ± 1.536 | 0.331 ± 0.781 | -0.005 ± 0.453 | -0.205 ± 0.621 |

570 nm: h_in → h* 고정, 오차 0 (모든 모델).

구간별 쌍체 차이

| 구간 | 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|---|
| band|Soret | PLS_v2 minus Global Linear | absorbance RMSE | 10 | -1.277 | 1.762 | 0.557 | 7/10 | 0.084 | 0.344 |
| band|Soret | BiGRU minus Global Linear | absorbance RMSE | 10 | -0.717 | 3.646 | 1.153 | 6/10 | 0.322 | 0.754 |
| band|Soret | BiGRU minus PLS_v2 | absorbance RMSE | 10 | 0.560 | 3.256 | 1.030 | 5/10 | 1.000 | 1.000 |
| band|Soret | PLS_v2 minus Global Linear | bias | 10 | 1.044 | 1.206 | 0.382 | 0/10 | 0.002 | 0.002 |
| band|Soret | BiGRU minus Global Linear | bias | 10 | 0.742 | 2.941 | 0.930 | 0/10 | 0.002 | 0.002 |
| band|Soret | BiGRU minus PLS_v2 | bias | 10 | -0.302 | 2.328 | 0.736 | 5/10 | 0.695 | 1.000 |
| band|Band 450-460 | PLS_v2 minus Global Linear | absorbance RMSE | 10 | -1.773 | 2.010 | 0.636 | 6/10 | 0.084 | 0.754 |
| band|Band 450-460 | BiGRU minus Global Linear | absorbance RMSE | 10 | -0.222 | 1.353 | 0.428 | 6/10 | 0.557 | 0.754 |
| band|Band 450-460 | BiGRU minus PLS_v2 | absorbance RMSE | 10 | 1.552 | 1.941 | 0.614 | 3/10 | 0.064 | 0.344 |
| band|Band 450-460 | PLS_v2 minus Global Linear | bias | 10 | -0.332 | 0.362 | 0.114 | 0/10 | 0.002 | 0.002 |
| band|Band 450-460 | BiGRU minus Global Linear | bias | 10 | 0.492 | 2.602 | 0.823 | 0/10 | 0.002 | 0.002 |
| band|Band 450-460 | BiGRU minus PLS_v2 | bias | 10 | 0.824 | 2.532 | 0.801 | 3/10 | 0.160 | 0.344 |
| band|Carotenoid | PLS_v2 minus Global Linear | absorbance RMSE | 10 | -0.589 | 1.759 | 0.556 | 7/10 | 0.232 | 0.344 |
| band|Carotenoid | BiGRU minus Global Linear | absorbance RMSE | 10 | 0.020 | 0.639 | 0.202 | 7/10 | 0.557 | 0.344 |
| band|Carotenoid | BiGRU minus PLS_v2 | absorbance RMSE | 10 | 0.609 | 1.666 | 0.527 | 4/10 | 0.375 | 0.754 |
| band|Carotenoid | PLS_v2 minus Global Linear | bias | 10 | -0.729 | 0.502 | 0.159 | 0/10 | 0.002 | 0.002 |
| band|Carotenoid | BiGRU minus Global Linear | bias | 10 | 0.356 | 1.556 | 0.492 | 0/10 | 0.002 | 0.002 |
| band|Carotenoid | BiGRU minus PLS_v2 | bias | 10 | 1.085 | 1.669 | 0.528 | 6/10 | 1.000 | 0.754 |
| band|Q-band | PLS_v2 minus Global Linear | absorbance RMSE | 10 | 0.109 | 0.815 | 0.258 | 6/10 | 0.846 | 0.754 |
| band|Q-band | BiGRU minus Global Linear | absorbance RMSE | 10 | 0.345 | 0.809 | 0.256 | 6/10 | 0.695 | 0.754 |
| band|Q-band | BiGRU minus PLS_v2 | absorbance RMSE | 10 | 0.236 | 0.807 | 0.255 | 4/10 | 0.557 | 0.754 |
| band|Q-band | PLS_v2 minus Global Linear | bias | 10 | -0.350 | 0.250 | 0.079 | 0/10 | 0.002 | 0.002 |
| band|Q-band | BiGRU minus Global Linear | bias | 10 | -0.125 | 0.525 | 0.166 | 0/10 | 0.002 | 0.002 |
| band|Q-band | BiGRU minus PLS_v2 | bias | 10 | 0.225 | 0.393 | 0.124 | 6/10 | 0.375 | 0.754 |
| band|Red | PLS_v2 minus Global Linear | absorbance RMSE | 10 | 0.684 | 2.915 | 0.922 | 6/10 | 0.922 | 0.754 |
| band|Red | BiGRU minus Global Linear | absorbance RMSE | 10 | -0.420 | 2.012 | 0.636 | 8/10 | 0.322 | 0.109 |
| band|Red | BiGRU minus PLS_v2 | absorbance RMSE | 10 | -1.104 | 2.686 | 0.849 | 7/10 | 0.275 | 0.344 |
| band|Red | PLS_v2 minus Global Linear | bias | 10 | -1.199 | 0.986 | 0.312 | 0/10 | 0.002 | 0.002 |
| band|Red | BiGRU minus Global Linear | bias | 10 | -0.150 | 0.725 | 0.229 | 0/10 | 0.002 | 0.002 |
| band|Red | BiGRU minus PLS_v2 | bias | 10 | 1.049 | 0.874 | 0.276 | 10/10 | 0.002 | 0.002 |

### 4.3 구간 단위 SHAP (Development10_LeaveOneSubjectOut)

모델 × 입력 그룹의 평균 |SHAP| ×10⁻³ (스펙트럼 변화량 30파장 평균 / Peak 변화량). 900쌍, BiGRU는 seed 3개 각각 계산 후 평균.

| 모델 | h* (target) | h_in (570 nm) | Soret | Band 450-460 | Carotenoid | Q-band | Red |
|---|---|---|---|---|---|---|---|
| HbI-binned mean | 21.180 / 1.811 | 21.304 / 1.823 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| Global Linear | 21.407 / 2.092 | 21.537 / 2.104 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| PLS_v2 | 20.659 / 2.095 | 88.761 / 3.515 | 1.705 / 0.048 | 1.069 / 0.020 | 6.538 / 0.061 | 81.350 / 3.347 | 0.923 / 0.025 |
| BiGRU | 21.637 / 2.065 | 22.830 / 2.083 | 0.827 / 0.038 | 0.411 / 0.104 | 1.054 / 0.234 | 1.278 / 0.194 | 0.904 / 0.049 |
| PLS_v1 | 21.774 / 2.079 | 4.859 / 0.605 | 8.185 / 0.183 | 1.596 / 0.088 | 5.896 / 0.202 | 13.355 / 1.681 | 6.520 / 0.251 |
| Ridge | 20.694 / 2.100 | 82.229 / 3.500 | 2.131 / 0.162 | 3.330 / 0.529 | 9.398 / 0.451 | 74.083 / 3.377 | 1.367 / 0.037 |

입력 그룹 × 출력 구간의 평균 |SHAP| ×10⁻³ (Proposal 모델과 Global Linear; 전체 모델은 `shap_input_group_by_output_band.csv`)

Global Linear

| 입력 그룹 | 출력 Soret | 출력 Band 450-460 | 출력 Carotenoid | 출력 Q-band | 출력 Red |
|---|---|---|---|---|---|
| h* (target) | 19.988 | 24.581 | 24.017 | 39.571 | 8.729 |
| h_in (570 nm) | 20.130 | 24.738 | 24.164 | 39.809 | 8.772 |
| Soret | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Band 450-460 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Carotenoid | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Q-band | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Red | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

PLS_v2

| 입력 그룹 | 출력 Soret | 출력 Band 450-460 | 출력 Carotenoid | 출력 Q-band | 출력 Red |
|---|---|---|---|---|---|
| h* (target) | 21.205 | 24.398 | 23.095 | 39.094 | 6.893 |
| h_in (570 nm) | 122.750 | 83.783 | 72.901 | 49.394 | 106.478 |
| Soret | 2.722 | 1.242 | 0.452 | 0.482 | 2.675 |
| Band 450-460 | 0.484 | 0.508 | 0.844 | 0.427 | 1.948 |
| Carotenoid | 3.080 | 1.367 | 3.397 | 2.260 | 13.200 |
| Q-band | 119.677 | 75.796 | 64.858 | 22.916 | 109.621 |
| Red | 1.118 | 0.614 | 0.539 | 0.341 | 1.436 |

BiGRU

| 입력 그룹 | 출력 Soret | 출력 Band 450-460 | 출력 Carotenoid | 출력 Q-band | 출력 Red |
|---|---|---|---|---|---|
| h* (target) | 20.668 | 25.018 | 24.240 | 39.698 | 8.787 |
| h_in (570 nm) | 22.947 | 26.988 | 25.813 | 40.814 | 9.221 |
| Soret | 1.634 | 0.825 | 0.807 | 0.677 | 0.564 |
| Band 450-460 | 1.113 | 0.873 | 0.235 | 0.220 | 0.210 |
| Carotenoid | 2.513 | 2.424 | 0.968 | 0.475 | 0.549 |
| Q-band | 1.773 | 1.789 | 1.995 | 1.072 | 0.765 |
| Red | 1.146 | 0.958 | 0.784 | 0.884 | 0.852 |

h_in 그룹 기여와 스펙트럼 그룹(5개 합) 기여의 Pearson 상관 (쌍 × 출력 파장 전체 / Peak)

| 모델 | 스펙트럼 변화량 | Peak 변화량 |
|---|---|---|
| HbI-binned mean | 0.038 | -0.019 |
| Global Linear | -0.016 | 0.036 |
| PLS_v2 | -0.980 | -0.913 |
| BiGRU | -0.325 | 0.216 |
| PLS_v1 | 0.807 | 0.973 |
| Ridge | -0.976 | -0.875 |

해석 주의: PLS_v2와 Ridge는 입력이 Δh x (A_in - A_in(570)) 형태라 h_in 그룹과 스펙트럼 그룹의 기여가 반대 부호로 상쇄될 수 있다. 그룹별 크기만으로 중요도를 단정하지 않는다. 그룹을 서로 다른 측정에서 섞으면 실제로 없는 스펙트럼이 생길 수 있다 (interventional SHAP의 한계).

검사: 입력 그룹 7개가 특징 32개를 한 번씩 덮음, 조합 128개 전부 계산, 효율성 최대 오차 1.18e-15, 기존 6개 그룹을 넣었을 때 `shap_analysis.shapley`와 일치 True (최대 차이 0.0e+00), Ridge 재학습 계수 = 저장 계수 True.

### 4.4 잔차 기울기의 파장 4구간 집계 (Development10_LeaveOneSubjectOut)

정의: r_model(λ) = (A*_model(λ) - A*_GlobalLinear(λ)) / Δh, Δh = 0인 쌍 제외. 피험자별 평균 후 구간 집계. BiGRU는 seed 평균. 개인 기준: d_s(λ) = b_s(λ) - b_GlobalLinear(λ); b_s는 평가에만 사용. 단위: absorbance / HbI (무차원 기울기).

| 구간 | 모델 | n | 구간 평균 잔차 기울기 (평균 ± 표준편차) | 구간 크기(RMS) 평균 | 개인 기준과의 Pearson 상관 | 부호 일치 | Global Linear 기울기 구간 평균 |
|---|---|---|---|---|---|---|---|
| Soret | PLS_v2 | 10 | 0.0227 ± 0.1483 | 0.1064 | 0.722 | 7/10 | 0.4440 |
| Band 450-460 | PLS_v2 | 10 | -0.0091 ± 0.1030 | 0.0731 | 0.804 | 9/10 | 0.5459 |
| Q-band | PLS_v2 | 10 | -0.0122 ± 0.0102 | 0.0235 | 0.150 | 8/10 | 0.8761 |
| Red | PLS_v2 | 10 | -0.0474 ± 0.0609 | 0.0629 | 0.311 | 8/10 | 0.1930 |
| Peak direction | PLS_v2 | 10 | -0.0001 ± 0.0032 | – | 0.176 | 6/10 | -0.0462 |
| Soret | BiGRU | 10 | -0.0124 ± 0.1040 | 0.0871 | 0.255 | 6/10 | 0.4440 |
| Band 450-460 | BiGRU | 10 | -0.0122 ± 0.0490 | 0.0381 | 0.402 | 7/10 | 0.5459 |
| Q-band | BiGRU | 10 | -0.0322 ± 0.0369 | 0.0414 | 0.670 | 8/10 | 0.8761 |
| Red | BiGRU | 10 | -0.0523 ± 0.0698 | 0.0708 | 0.635 | 8/10 | 0.1930 |
| Peak direction | BiGRU | 10 | 0.0017 ± 0.0080 | – | -0.056 | 4/10 | -0.0462 |
| Soret | Ridge | 10 | 0.0053 ± 0.1363 | 0.0910 | 0.629 | 7/10 | 0.4440 |
| Band 450-460 | Ridge | 10 | -0.0135 ± 0.0952 | 0.0668 | 0.803 | 9/10 | 0.5459 |
| Q-band | Ridge | 10 | -0.0088 ± 0.0102 | 0.0204 | 0.004 | 6/10 | 0.8761 |
| Red | Ridge | 10 | -0.0348 ± 0.0551 | 0.0543 | 0.308 | 6/10 | 0.1930 |
| Peak direction | Ridge | 10 | -0.0002 ± 0.0033 | – | 0.295 | 8/10 | -0.0462 |
| Soret | Personal reference | 10 | -0.0262 ± 0.1838 | 0.1619 | – | – | 0.4440 |
| Band 450-460 | Personal reference | 10 | -0.0384 ± 0.1133 | 0.1024 | – | – | 0.5459 |
| Q-band | Personal reference | 10 | -0.0157 ± 0.0358 | 0.0482 | – | – | 0.8761 |
| Red | Personal reference | 10 | -0.0665 ± 0.1797 | 0.1516 | – | – | 0.1930 |
| Peak direction | Personal reference | 10 | -0.0026 ± 0.0072 | – | – | – | -0.0462 |

'Personal reference' 행은 개인 기준 잔차 기울기 d_s 자체의 구간 평균이다. 'Peak direction' = r(490) − (r(470) + r(510))/2 는 4구간에 포함하지 않은 별도 항목이다.

제외한 모델: Global Linear — 정의상 잔차 기울기 0; HbI-binned mean — 보정이 Δh에 비례하지 않아(bin 평균 차이) 기울기 형태로 정의되지 않음; PLS_v1 — 보정이 Δh에 비례하지 않아(절편과 A_in, h_in, h*의 선형식) 기울기 형태로 정의되지 않음.

검사: Δh = 0 제외 순서쌍 4개, BiGRU 잔차 기울기와 `BiGRUModel.residual`의 최대 차이 1.8e-13 (기준 1e-5), Δh를 바꿨을 때 PLS_v2 최대 차이 1.0e-13, Ridge 1.0e-13.

### 4.5 HbI 변화 크기 구간별 오차

4.1의 분할 세트별 표에 포함했다.

### 4.6 Peak-constrained selection (BiGRU, 보조 분석)

선택값은 3장의 표, 지표는 4.1의 보조 표 'BiGRU (Peak-constrained selection)' 행. 주 결과를 대체하지 않는다.

### 4.7 S006 exclusion sensitivity (보조 분석)

S006이 test가 아닌 9개 fold에서 학습 피험자 8명(S006 제외)으로 다시 학습. 하이퍼파라미터는 해당 fold의 주 결과 선택값으로 고정.

| 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|
| HbI-binned mean [S006 excluded] minus HbI-binned mean | absorbance RMSE | 9 | 0.888 | 0.893 | 0.298 | 1/9 | 0.020 | 0.039 |
| Global Linear [S006 excluded] minus Global Linear | absorbance RMSE | 9 | 0.004 | 0.044 | 0.015 | 4/9 | 0.820 | 1.000 |
| PLS_v2 [S006 excluded] minus PLS_v2 | absorbance RMSE | 9 | 0.005 | 0.093 | 0.031 | 3/9 | 0.734 | 0.508 |
| BiGRU [S006 excluded] minus BiGRU | absorbance RMSE | 9 | -0.037 | 0.097 | 0.032 | 6/9 | 0.301 | 0.508 |
| HbI-binned mean [S006 excluded] minus HbI-binned mean | Peak RMSE | 9 | 0.133 | 0.254 | 0.085 | 3/9 | 0.203 | 0.508 |
| Global Linear [S006 excluded] minus Global Linear | Peak RMSE | 9 | -0.000 | 0.005 | 0.002 | 4/9 | 0.910 | 1.000 |
| PLS_v2 [S006 excluded] minus PLS_v2 | Peak RMSE | 9 | 0.002 | 0.007 | 0.002 | 4/9 | 0.570 | 1.000 |
| BiGRU [S006 excluded] minus BiGRU | Peak RMSE | 9 | 0.020 | 0.048 | 0.016 | 2/9 | 0.203 | 0.180 |

## 5. 검사 통과 여부

| 검사 | 결과 |
|---|---|
| Phase 0: 명세와 코드 대조 17항목, 단위 테스트 | 모두 일치. 기존 테스트 44개 통과 (실패 0). 새 테스트 12개 포함 56개 통과 |
| Phase 1: 참고 수치와 일치 (마지막 자리 ±1 이내) | 모든 모델·두 분할 세트에서 차이 0. 선택 성분 수와 Fixed-target repeatability 값도 일치 |
| Phase 2: External cohort 전처리 기대값 | HbI 범위 6명, Near-target subject 5명, In-range subject 6명, 완전 중복 0, S102 Peak_RS 회귀 (−331.304, 274.815, 76.033) 모두 일치 |
| Phase 2: 분할 세트 5개의 검사 | 모두 통과 (train/test 피험자 교집합 없음, 쌍은 같은 피험자 안, 행·쌍 수, 기존 `data/splits`·`loso_folds.json`과 동일, Union16 규칙) |
| Phase 3: 튜닝 루프와 `BiGRUModel.fit` 일치 | epoch별 학습 이력과 최종 가중치 완전 일치 (3개 설정) |
| Phase 3: 병렬 실행과 단일 실행 일치 | 완전 일치 |
| Phase 3: epoch 0 후보 = Global Linear | 모든 분할·안쪽 fold에서 제곱오차 합이 완전 일치 |
| Phase 3: 재학습 검사 | 570 nm = 목표 HbI, 유한 값, 기울기 = 같은 분할의 Global Linear 기울기 모두 통과. epoch 0 선택 분할의 예측 = Global Linear 예측 (완전 일치) |
| Phase 4.1: 테스트 쌍·정답·목표 HbI = 기존 저장 예측 | 일치. 불러온 모델의 예측 = 저장 예측 (최대 차이 1e-9 이내) |
| Phase 4.1: Near-target accuracy 허용 폭 0.03 대상 피험자 | 예상 8명과 일치 |
| Phase 4.2: 5개 구간의 파장 수 합 30, 구간 제곱합으로 다시 계산한 전체 RMSE | 통과 (허용 1e-12) |
| Phase 4.3: SHAP 그룹이 특징 32개를 한 번씩 덮음, 기존 6개 그룹에서 기존 함수와 같은 값, 효율성 | 통과 (4.3 본문) |
| Phase 4.4: BiGRU 잔차 기울기 = `BiGRUModel.residual` (1e-5 이내), PLS_v2·Ridge는 Δh와 무관 | 통과 (4.4 본문). Δh = 0 제외 순서쌍 4개 (명세의 예상과 같음) |

## 6. 명세와 다르게 처리했거나 명세에 없던 항목

- **공용 모듈 추가.** 최종 산출물 목록에 없는 `scripts/final_utils.py`(구간 정의, 분할 읽기, 쌍체 차이), `scripts/final_eval.py`(모델 불러오기·학습, 피험자별 지표, 집계), `scripts/make_docs.py`(결과 파일에서 문서 생성)를 추가했다. 지표와 집계 코드를 스크립트마다 복사하지 않기 위한 것이며, 지표 계산은 `common/metrics.py`의 함수만 쓴다.
- **파장 구간별 복원 오차의 계산 위치.** `scripts/band_analysis.py`가 `final_eval.split_records`를 호출해 계산한다 (주 비교와 같은 함수). 잔차 기울기 집계는 `band_analysis.py` 안에 있다.
- **Peak-constrained selection의 선택값.** 추가 학습 없이 `selection.json`의 기록으로 고른다는 명세대로, 튜닝 시점에 `selection.json`에 함께 적어 두고 `tune_bigru.py --peak-constrained`가 같은 값을 다시 계산해 일치를 확인한 뒤 재학습한다.
- **epoch 0이 선택된 분할의 모델 파일.** 학습 없이 `BiGRUModel(epochs=0)`을 저장한다 (예측은 Global Linear와 완전 일치). 뒤의 SHAP·잔차 기울기 코드가 같은 경로를 쓰게 하기 위함이다.
- **쌍체 비교 함수.** `loso.paired`는 부호 규칙이 달라 쓰지 않고 `wilcoxon_exact`, `sign_test`만 재사용했다. 차이 = 앞 모델 − 뒤 모델, 개선 = 차이 < 0, 표준오차 = 표준편차 / √n.
- **Raw.** General reconstruction에는 넣지 않았다 (보정 없음이면 정답이 다른 측정이라 비교 의미가 다름). Fixed-target repeatability에서만 보고한다. 기존 스크립트(`evaluate.py`, `loso.py`)와 같은 처리다.
- **Ridge의 격자 끝.** Development10_Holdout73의 분할 0과 5에서 Ridge가 alpha 1e10(6회 확장 후에도 격자 끝)을 골랐다. 기존 구현의 규칙 그대로이며 바꾸지 않았다.
- **`requirements.txt`.** openpyxl 한 줄을 추가했다 (사용자 승인, 규칙 4의 예외, 별도 커밋).
- **ExternalTest6·Union16 스크립트.** `scripts/external_test.py`와 `scripts/run_union16.py`는 동결 전에 작성해 Phase 4 커밋에 넣었다. `--approved` 없이는 실행되지 않으며 Phase 5 시점까지 실행하지 않았다.
