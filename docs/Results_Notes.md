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
