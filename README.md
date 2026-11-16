# Deep_Learning-Based_Skin_Spectrum_Reconstruction
메카트로닉스 팀프로젝트: 딥러닝 기반 스킨 스펙트럼 복원

## 모델 (`models/`)
| 모델 | 파일 | 보정식 |
|---|---|---|
| B1: HbI-Binned Mean Correction | `hb_index.py` | A* = A + Mean(목표 bin) − Mean(입력 bin). bin 폭 0.05, 학습 데이터 평균 |
| B2: Global Linear Correction | `linear.py` | A* = A + (h* − h)·b(λ). b는 학습 피험자 내 중심화 후 모은 기울기 |
| BiGRU | `bigru.py` | A* = A + (h* − h)·(b(λ) + r_BiGRU(λ)). b(λ)는 B2 기울기 |

세 모델 모두 570 nm 값을 목표 HbI로 고정하고, 입력 HbI와 목표 HbI가 같으면 원본을 그대로 반환한다.

### B1의 한계
- **Bin 경계 불연속성**: 보정량이 bin 단위의 계단 함수다. 같은 bin 안에서는 HbI가 달라도 보정량이 0이다(570 nm만 바뀜). 반대로 경계를 조금만 넘어도 보정량이 bin 평균의 차이만큼 갑자기 바뀐다.
- **피험자별 피부색 차이에 의한 편향**: bin 평균은 서로 다른 피험자의 스펙트럼을 그대로 평균한다. 그래서 bin마다 피험자 구성이 다르면 평균 차이에 HbI 효과와 피부색 차이가 섞인다. 측정 수와 피험자 수가 적은 높은 HbI bin에서 특히 크게 작용할 수 있다.
- 빈 bin은 가장 가까운 유효 bin의 평균으로 대체한다. 실행할 때마다 bin별 측정 수, 고유 피험자 수, fallback 사용 횟수를 기록한다.

## 폴더 구조
```
common/          데이터 읽기·분할 생성·지표 (data_loader, make_splits, metrics, prepare_data, plot_data)
models/          B1(hb_index), B2(linear), BiGRU(bigru), PLS_v1(pls), PLS_v2(b2_pls), Ridge(b2_ridge)
scripts/         실험 실행 스크립트 (아래 표)
tests/           단위 테스트 (python -m unittest discover -s tests)
Model_Ablation/  초기 모델 비교 실험 (evaluate·tests가 Peak 정의 비교용으로 ablation.py를 import)
data/            raw → processed/spectra.csv, splits/ (7:3 분할 계획)
results/         holdout(7:3 분할 10회), loso, pls_v1, pls_v2, ridge, *_k10, compare, shap
```

## 실행 순서 (프로젝트 루트에서)
| 단계 | 명령 | 결과 |
|---|---|---|
| 1. 학습 (7:3 분할) | `python scripts/train_models.py` | `results/holdout/models` |
| 2. 평가 (7:3 분할) | `python scripts/evaluate.py` | `results/holdout/{metrics,predictions,figures}` |
| 3. LOSO | `python scripts/loso.py --out-version loso` | `results/loso` |
| 4. PLS·Ridge 비교 | `python -X utf8 scripts/pls_compare.py` | `results/<모델>/<분할>`, `results/compare` |
| 5. SHAP | `python -X utf8 scripts/shap_analysis.py` | `results/shap/loso` |
| 보조 | `scripts/alpha_sweep.py`, `scripts/pair_strategy.py`, `scripts/plot_peak.py` | `results/holdout/` 하위 |
| 이전 방식 | `scripts/pls_loso.py` (현재는 4단계 `pls_v1_k10`으로 대체) | `results/pls_v1/legacy_loso` |

모든 스크립트는 결과가 이미 있으면 덮어쓰지 않고 멈춘다.
