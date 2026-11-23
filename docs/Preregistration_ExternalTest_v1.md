# ExternalTest6 동결 문서 (Preregistration v1)

이 문서는 External cohort(S101–S106)를 어떤 모델에도 넣기 전에 작성했다. 아래 설정, 지표, 비교, 구간 정의는 ExternalTest6 결과를 본 뒤 바꾸지 않는다. 기계가 읽는 값은 `docs/Preregistration_ExternalTest_v1.json`에 있고 `scripts/external_test.py`와 `scripts/run_union16.py`는 그 파일만 읽는다.

## 1. 동결한 설정 (Development10_AllSubjects에서 선택)

| 모델 | 설정 | 선택 방법 |
|---|---|---|
| HbI-binned mean | bin 폭 0.05, 경계 0.40–0.75 | 고정 (튜닝 없음) |
| Global Linear | – | 튜닝 없음 |
| PLS_v2 | 성분 수 6 | Development cohort 10명 안 Leave-One-Subject-Out, 상한 10 |
| BiGRU | hidden 8, slope_l2 0, epoch 200, lr 3e-3, clip 1.0, seed 0/1/2 | Development cohort 10명 안 Leave-One-Subject-Out, 84 후보 (검증 absorbance RMSE 11.123×10⁻³, Global Linear 12.505×10⁻³) |
| PLS_v1 | 성분 수 6 | 같은 내부 절차, 상한 10 |
| Ridge | alpha 1 | 같은 내부 절차 (격자 1e-4…1e4, 끝이면 확장) |
| BiGRU (fixed setting) | hidden 8, slope_l2 1e-3, epoch 150 | `common/config.py` 고정값 |
| BiGRU (Peak-constrained selection) | epoch 0 (Global Linear와 동일), lr 3e-3, clip 1.0, seed 0/1/2 | 내부 검증 Peak RMSE ≤ epoch 0의 값인 후보 12/84개 중 검증 absorbance RMSE 최소 |

## 2. 코드, 분할, 환경

- 코드 커밋 (Phase 4 커밋): `a617ae6edab89b4889a753f4899281daf2aba8b8` (브랜치 `experiment/retrain-and-tuning-2026-11`)
- `data/splits_final/*/manifest.json`의 SHA-256

| 분할 세트 | manifest.json SHA-256 |
|---|---|
| Development10_Holdout73 | `9e662cc9b117319bafa85ebd83ecc45d757bd83950e4ba101af4e8e5921defa5` |
| Development10_LeaveOneSubjectOut | `55e3a3df3cdd671709d23a80cf069d720e7ff9d002379f9cc2702861bf82b836` |
| ExternalTest6 | `0e6a6bc19d838daae0cc48e90d9c0ebc974f095457dd6e1905a6d5911bd6bd3e` |
| Union16_Holdout73 | `bbf4d77a8bd9533eda90ddd9ff3b6684ede74506217b88a1a6f0661f6d9c40c7` |
| Union16_LeaveOneSubjectOut | `c73b1a7f99ed5a878df6b14a45afee0ce76515d4b4353137979864d86c7de5b1` |

- 환경: python 3.11.18, torch 2.11.0+cpu, numpy 2.4.6, pandas 3.1.1, matplotlib 3.11.2, openpyxl 3.1.5, platform Windows-10-10.0.26200-SP0. 전체 목록은 `docs/Environment.md`.

## 3. ExternalTest6에서 계산할 것 (Phase 6 정의 그대로)

- 학습: Development cohort 10명 전체. 하이퍼파라미터는 1장의 값으로 고정. BiGRU seed 0, 1, 2 (피험자 값 = seed별 지표의 평균).
- 평가: S101–S106, `data/splits_final/ExternalTest6`의 테스트 쌍 (피험자당 90쌍, 합계 540쌍). 1회만 실행하고, 평가 후 설정을 바꿔 다시 실행하지 않는다.
- 주 표와 사전 지정 비교: Raw, HbI-binned mean, Global Linear, PLS_v2, BiGRU. 보조 표: PLS_v1, Ridge, BiGRU (fixed setting), BiGRU (Peak-constrained selection).
- 실험과 지표
  - General reconstruction: absorbance RMSE (주 지표, 570 nm 제외 30파장), reflectance RMSE, Peak RMSE (제약 지표), Peak bias.
  - Fixed-target repeatability: 개인 내 Peak 표준편차, spectral 표준편차. In-range subject(S101, S106)와 그 밖의 피험자를 분리해 보고.
  - Near-target accuracy: 0.6 ± 0.02 실측이 S101과 S105에 1개씩뿐이므로 기술 통계로만 보고.
- 사전 지정 비교 (피험자 6명 쌍체 차이; 차이 = 앞 모델 − 뒤 모델, 개선 = 차이 < 0, Peak bias는 |bias| 감소; 평균, 표준편차, 표준오차, k/n, 정확 Wilcoxon p, 부호검정 p): PLS_v2 minus Global Linear, BiGRU minus Global Linear, PLS_v2 minus HbI-binned mean, BiGRU minus HbI-binned mean, BiGRU minus PLS_v2, BiGRU minus BiGRU (fixed setting). p 값은 다중 비교 보정이 없는 기술적 참고값이며 n = 6에서 부호검정 양측 p의 최솟값은 0.031이다. 판정 문구는 넣지 않는다.
- 구간별 분석 (Development cohort 분석과 같은 정의, 같은 함수)
  - 파장 구간별 복원 오차: Soret = 400, 410, 420, 430, 440 nm; Band 450-460 = 450, 460 nm; Carotenoid = 470, 480, 490, 500, 510 nm; Q-band = 520, 530, 540, 550, 560, 580, 590 nm; Red = 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700 nm. 570 nm는 h_in → h* 고정, 오차 0.
  - 잔차 기울기 4구간 (Soret, Band 450-460, Q-band, Red): r_model(λ) = (A*_model(λ) − A*_GlobalLinear(λ)) / Δh, Δh = 0인 쌍 제외, 대상 PLS_v2, BiGRU, Ridge. 개인 기준 d_s = b_s − b_GlobalLinear. Peak 방향 r(490) − (r(470) + r(510))/2 는 별도 항목.
  - HbI 변화 크기 구간: |ΔHbI| 0–0.025, 0.025–0.05, 0.05–0.10, 0.10–0.30 × 방향(증가, 감소, 전체).
- 추가 기준값: 피험자별 10회 측정의 Peak 대 HbI 선형회귀로 얻은 HbI 0.6에서의 값과 그 표준오차. 모델로 0.6 복원한 10개 스펙트럼의 Peak 평균과의 차이를 Peak_RS 단위로 보고 (Peak_RS = 8387.2 × Peak + 83.495, Hwang et al. 2023, Sensors 23(17):7654 식 (2)). 이 기준값은 추정값이며 독립 정답이 아니다. In-range subject가 아닌 4명은 외삽임을 표시한다.

## 4. Union16 분할 (Phase 7, 보조)

- `Union16_LeaveOneSubjectOut`, `Union16_Holdout73`에서 모든 모델을 1장의 고정값으로 다시 학습한다 (PLS는 성분 수를 명시 전달, 다시 고르지 않음).
- 규칙 3의 의도된 예외: 고정값은 Development cohort 10명 전체로 고른 값이므로 Development cohort 피험자가 test인 분할에서는 그 피험자가 하이퍼파라미터 선택에 쓰였다. External cohort 피험자가 test인 분할에는 이 문제가 없다. Union16은 보조 결과다.
- 학습 피험자 수 효과의 주 비교는 같은 고정값끼리 한다: External cohort 6명은 10명 학습(ExternalTest6) 대 15명 학습, Development cohort 10명은 9명 학습(같은 고정값으로 다시 학습) 대 15명 학습. fold별 튜닝값으로 학습한 기존 9명 학습 결과는 참고 행으로 둔다. 이 결과는 ExternalTest6과 독립이 아니므로 데이터 증가 효과의 추정으로만 표기한다.

## 5. Phase 3과 Phase 4 결과 요약 (Development10_LeaveOneSubjectOut, 피험자 10명 평균 ± 표준편차)

단위: absorbance RMSE, Peak RMSE, Peak bias ×10⁻³, reflectance RMSE %p. 전체 표는 `docs/Results_Retrain_Tuning.md`.

| 모델 | absorbance RMSE | reflectance RMSE | Peak RMSE | Peak bias |
|---|---|---|---|---|
| HbI-binned mean | 17.414 ± 3.010 | 1.221 ± 0.222 | 2.528 ± 0.727 | -0.000 ± 0.000 |
| Global Linear | 12.126 ± 3.222 | 0.915 ± 0.245 | 0.716 ± 0.203 | -0.000 ± 0.000 |
| PLS_v2 | 11.057 ± 2.862 | 0.875 ± 0.248 | 0.684 ± 0.201 | 0.006 ± 0.036 |
| BiGRU | 11.301 ± 2.759 | 0.832 ± 0.185 | 0.949 ± 0.522 | 0.078 ± 0.128 |
| PLS_v1 | 11.367 ± 2.603 | 0.893 ± 0.225 | 0.741 ± 0.219 | 0.009 ± 0.218 |
| Ridge | 11.136 ± 2.894 | 0.885 ± 0.247 | 0.687 ± 0.207 | -0.006 ± 0.054 |
| BiGRU (fixed setting) | 11.280 ± 2.705 | 0.846 ± 0.187 | 1.015 ± 0.613 | 0.115 ± 0.223 |
| BiGRU (Peak-constrained selection) | 12.042 ± 3.114 | 0.907 ± 0.231 | 0.716 ± 0.203 | -0.002 ± 0.007 |

| 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|
| PLS_v2 minus Global Linear | absorbance RMSE | 10 | -1.068 | 2.021 | 0.639 | 8/10 | 0.064 | 0.109 |
| BiGRU minus Global Linear | absorbance RMSE | 10 | -0.824 | 1.320 | 0.417 | 8/10 | 0.064 | 0.109 |
| PLS_v2 minus HbI-binned mean | absorbance RMSE | 10 | -6.356 | 3.549 | 1.122 | 10/10 | 0.002 | 0.002 |
| BiGRU minus HbI-binned mean | absorbance RMSE | 10 | -6.113 | 3.436 | 1.086 | 10/10 | 0.002 | 0.002 |
| BiGRU minus PLS_v2 | absorbance RMSE | 10 | 0.244 | 1.842 | 0.582 | 6/10 | 1.000 | 0.754 |
| BiGRU minus BiGRU (fixed setting) | absorbance RMSE | 10 | 0.021 | 0.512 | 0.162 | 6/10 | 1.000 | 0.754 |
| PLS_v2 minus Global Linear | Peak RMSE | 10 | -0.032 | 0.125 | 0.040 | 4/10 | 1.000 | 0.754 |
| BiGRU minus Global Linear | Peak RMSE | 10 | 0.233 | 0.409 | 0.129 | 2/10 | 0.105 | 0.109 |
| PLS_v2 minus HbI-binned mean | Peak RMSE | 10 | -1.843 | 0.796 | 0.252 | 10/10 | 0.002 | 0.002 |
| BiGRU minus HbI-binned mean | Peak RMSE | 10 | -1.578 | 0.944 | 0.298 | 9/10 | 0.004 | 0.021 |
| BiGRU minus PLS_v2 | Peak RMSE | 10 | 0.265 | 0.499 | 0.158 | 5/10 | 0.375 | 1.000 |
| BiGRU minus BiGRU (fixed setting) | Peak RMSE | 10 | -0.066 | 0.189 | 0.060 | 6/10 | 0.492 | 0.754 |

BiGRU 선택값 요약 (전체는 `docs/Results_Retrain_Tuning.md` 3장): Development10_AllSubjects = hidden 8, slope_l2 0, epoch 200.

## 6. 명세 보충 사항 (2026-11-23 사용자 답변, 그대로 기록)

사전 보고 확인했습니다. 아래대로 진행하세요. 이 답변은 프롬프트 명세의 일부로 취급하고, `docs/Preregistration_ExternalTest_v1.md`에 "명세 보충 사항"으로 그대로 기록하세요.

### 1. 환경

- conda 환경 `skin_spectra`를 새로 만든다. Python 3.11.
- torch는 2.11.0 CPU 빌드로 설치한다: `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu`. 모든 실행이 CPU(규칙 8)이므로 CUDA 빌드는 쓰지 않는다. 그 인덱스에 해당 버전이 없으면 다른 버전으로 바꾸지 말고 멈추고 보고한다.
- 나머지 패키지는 `requirements.txt` 그대로 설치한다.
- openpyxl은 환경에 설치하고, `requirements.txt`에도 설치된 버전으로 한 줄 추가한다. 별도 커밋으로 하고 사유(`common/prepare_data.py`가 xlsx를 읽을 때 필요)를 커밋 메시지에 적는다. 규칙 4의 예외로 허용한다.
- 실제 설치된 버전 전체를 `docs/`에 기록한다.
- 기대값 일치 기준을 다음으로 바꾼다. 참고 수치는 Linux, numpy 2.5.3, pandas 3.0.5에서 계산한 값이다.
  - 일치 기준: 표에 적힌 자릿수의 마지막 자리에서 ±1 이내.
  - HbI-binned mean 또는 Global Linear가 기준을 벗어나면 멈추고 보고한다 (닫힌 식이라 환경 차이로 설명되지 않음).
  - PLS 계열 또는 Ridge가 기준을 벗어나면 선택된 성분 수(또는 alpha)가 기대값과 같은지 먼저 확인하고, 차이를 보고한 뒤 계속 진행한다.
  - BiGRU는 차이를 보고만 하고 계속 진행한다.

### 2. 쌍체 차이의 부호와 개선 정의

기본안대로 한다.

- 차이 = 앞 모델 − 뒤 모델. 개선 = 차이 < 0. `wilcoxon_exact`와 `sign_test`만 재사용하고 k/n은 새로 계산한다.
- Peak bias는 |bias| 감소를 개선으로 정의하고, 표에는 모델별 부호 있는 bias도 함께 적는다.
- Fixed-target repeatability의 표준편차 지표도 낮을수록 개선이다.

### 3. 질문 1의 판정 기준

기본안대로 한다. 스크립트와 결과 문서에는 판정 문구를 넣지 않고 평균 차이, 표준편차, 표준오차, k/n, p 값만 적는다. 해석은 사용자가 작성한다.

### 4. Holdout73의 쌍체 비교 단위

기본안대로 한다. 피험자별로 분할 평균을 낸 뒤 n = 10 쌍체 비교를 하고, 주 표의 평균(분할 평균의 평균)과 쌍체 표의 평균이 다를 수 있다는 주석을 단다.

### 5. Union16의 하이퍼파라미터와 규칙 3

기본안대로 한다.

- 규칙 3의 의도된 예외로 명시한다. Union16은 보조 결과이며, Development cohort 피험자가 test인 분할에서는 그 피험자가 하이퍼파라미터 선택에 쓰였다는 점을 표 주석과 동결 문서에 적는다. External cohort 피험자가 test인 분할에는 이 문제가 없다는 점도 함께 적는다.
- 학습 피험자 수 효과의 주 비교는 같은 고정값끼리 한다. 9명 학습 쪽을 같은 고정값으로 다시 학습한 행을 추가하고, fold별 튜닝값으로 학습한 기존 행은 참고 행으로 둔다.

### 6. Phase 6과 Phase 7의 모델 범위

기본안대로 전부 포함하되 표를 나눈다.

- 주 표와 사전 지정 비교: Raw, HbI-binned mean, Global Linear, PLS_v2, BiGRU.
- 보조 표: PLS_v1, Ridge, BiGRU (fixed setting), BiGRU (Peak-constrained selection).
- BiGRU (Peak-constrained selection)의 `Development10_AllSubjects` 선택값도 동결 문서에 기록한다.

### 7. BiGRU 튜닝의 세부 정의

세 가지 모두 기본안대로 한다.

- 내부 검증 Peak RMSE는 검증 쌍 전체로 계산한다 (absorbance RMSE와 같은 방식, 안쪽 fold 전체를 합쳐서).
- 검증 예측은 `BiGRUModel.predict`와 같은 float64 조립을 쓴다.
- 동률 규칙은 값이 정확히 같을 때만 적용한다. history 일치 검사는 완전 일치로 한다. 완전 일치가 실패하면 허용오차를 임의로 넣지 말고 최대 절대 차이와 함께 멈추고 보고한다.
- epoch 0이 선택된 분할은 표에 hidden과 slope_l2 대신 "epoch 0 (Global Linear와 동일)"로 적는다.

### 8. Near-target accuracy와 Fixed-target repeatability의 빈칸

기본안대로 한다.

- 허용 폭 0.02: Leave-One-Subject-Out에서 대상이 3명뿐이므로 기술 통계만 보고한다.
- 허용 폭 0.03: Development cohort의 대상 피험자는 8명으로 예상한다 (S001, S002, S003, S004, S005, S008, S009, S010). 쌍체 차이(평균, 표준오차, k/n)를 보조로 보고한다. 대상 피험자가 이와 다르면 보고한다.
- Holdout73의 분할 3과 분할 5: In-range subject의 반복성은 해당 피험자가 있는 분할만 평균하고 분할 수를 표기한다.

### 9. 스크립트 배치

기본안대로 한다.

- Development cohort 10명 전체의 PLS와 Ridge 선택 → `scripts/final_comparison.py`
- Peak-constrained selection의 재학습 → `scripts/tune_bigru.py`의 옵션
- S006 exclusion sensitivity → `scripts/sensitivity_exclude_S006.py`
- 새 테스트 → `tests/test_final.py` (기존 테스트 파일은 수정하지 않음)

### 사소한 가정

여섯 가지 모두 승인한다. 보충 사항은 다음과 같다.

- Peak_RS 식의 근거: Hwang et al. 2023, Sensors 23(17):7654의 식 (2). 결과 문서에 이 출처를 적는다.
- 검사 항목 추가 (`common/prepare_external.py`): S102(원 파일 S002)의 측정 10행으로 Peak_RS 대 HbI 선형회귀를 하면 기울기 −331.304, 절편 274.815, HbI 0.6에서의 값 76.033이 나와야 한다. 이 값은 원 파일 하단의 분석 행(Slope, Bias, Results)과 같다.

승인된 여섯 가지 가정 (사전 보고의 문구)

- Union16 분할의 후보 순서: 조합을 사전순으로 나열한 뒤 seed 0 순열을 적용하고 `manifest.json`에 기록한다. "Near-target 합계 16" 조건 때문에 S101과 S105 중 정확히 한 명만 4회 테스트가 된다.
- Ridge의 SHAP: `B2RidgeModel`에 `load`가 없어 기존 스크립트처럼 다시 학습하고 저장 계수와 대조한다.
- epoch 0이 선택된 fold: `BiGRUModel(epochs=0)`으로 저장해 SHAP과 잔차 기울기 코드가 같은 경로를 타게 한다.
- SHAP의 "h_in과 스펙트럼 그룹 기여의 상관": 스펙트럼 5개 그룹의 합과 h_in 사이의 상관으로 계산한다.
- Peak_RS 식(8387.2 × Peak + 83.495): 명세 값을 그대로 쓴다. 외삽이 되는 In-range 밖 4명도 계산하되 외삽임을 표시한다.
- 동결 문서의 커밋 해시: 문서 자신이 들어가는 커밋이 아니라 Phase 4 커밋을 적는다.

### 다음 단계

환경을 만든 뒤 Phase 0의 17번(단위 테스트)을 실행하고 Phase 0 정식 보고를 한다. 불일치가 없으면 Phase 1로 진행한다.

## 7. v1.1 보충 (External cohort 결과를 보기 전에 결정, 2026-11-23)

코드와 `docs/Preregistration_ExternalTest_v1.json`의 설정값은 바꾸지 않았다. 이 절은 해석 규칙만 추가한다.

- 뒤 모델이 Global Linear 또는 HbI-binned mean인 쌍체 비교에서 Peak bias와 구간 bias 행은 해석하지 않는다. 테스트 쌍이 피험자 안의 모든 순서쌍이고 두 모델의 보정은 i → j와 j → i에서 정확히 반대 부호이므로, 두 모델의 bias가 구조적으로 0이기 때문이다. 표는 그대로 생성하고 주의 문구를 유지한다.
- BiGRU (Peak-constrained selection)는 `Development10_AllSubjects`에서 epoch 0이 선택되었으므로 ExternalTest6에서 Global Linear와 같은 값이 나온다.
