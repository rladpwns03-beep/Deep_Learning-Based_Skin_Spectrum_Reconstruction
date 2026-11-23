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
