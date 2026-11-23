# ExternalTest6과 Union16 결과 (Phase 6–7)

이 문서는 `scripts/make_docs.py --external`이 결과 파일에서 생성한다. 판정 문구는 넣지 않았다. 단위: absorbance RMSE, Peak RMSE, Peak bias, 표준편차 지표, 구간 bias는 ×10⁻³ absorbance, reflectance RMSE는 %p. p 값은 다중 비교 보정이 없는 기술적 참고값 (정확 Wilcoxon signed-rank, 부호검정, 양측).

## 1. ExternalTest6 (학습 = Development cohort 10명 전체, 평가 = External cohort S101–S106, 1회 실행)

- 동결 문서: `docs\Preregistration_ExternalTest_v1.json` (SHA-256 `16bdf79e78b53cba112886f60e0554f5cd267e0856dfd1f7f72d174556066027`), 태그 `preregistration-externaltest-v1.1`. 실행 시점의 코드 커밋 `d005825540cf197ff37234eb75130564b720dad9`.
- 고정 설정: PLS_v2 성분 수 6, PLS_v1 성분 수 6, Ridge alpha 1, BiGRU hidden 8, slope_l2 0, epoch 200 (seed 0, 1, 2). 실행 후 설정을 바꿔 다시 실행하지 않았다.
- 쌍체 비교: 피험자 6명 쌍체 차이. 차이 = 앞 모델 - 뒤 모델, 개선 = 차이 < 0. p 값은 다중 비교 보정이 없는 기술적 참고값 (정확 Wilcoxon signed-rank, 부호검정, 양측). n = 6에서 부호검정 양측 p의 최솟값은 0.031.

주의 (동결 문서 v1.1 보충): 뒤 모델이 Global Linear 또는 HbI-binned mean인 쌍체 비교에서 Peak bias와 구간 bias 행은 해석하지 않는다 (두 모델의 bias가 구조적으로 0). BiGRU (Peak-constrained selection)는 epoch 0이 선택되어 Global Linear와 같은 값이다.

### 1.1 General reconstruction (피험자당 90쌍, 피험자 6명 평균 ± 표준편차)

| 모델 | absorbance RMSE | reflectance RMSE | Peak RMSE | Peak bias |
|---|---|---|---|---|
| HbI-binned mean | 18.718 ± 3.043 | 1.412 ± 0.282 | 2.651 ± 0.969 | 0.000 ± 0.000 |
| Global Linear | 12.910 ± 4.401 | 1.107 ± 0.405 | 0.893 ± 0.421 | 0.000 ± 0.000 |
| PLS_v2 | 12.009 ± 4.082 | 0.994 ± 0.405 | 0.921 ± 0.401 | -0.007 ± 0.044 |
| BiGRU | 11.862 ± 4.130 | 0.937 ± 0.345 | 1.087 ± 0.536 | 0.101 ± 0.086 |

사전 지정 비교

| 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|
| PLS_v2 minus Global Linear | absorbance RMSE | 6 | -0.901 | 1.597 | 0.652 | 4/6 | 0.219 | 0.688 |
| BiGRU minus Global Linear | absorbance RMSE | 6 | -1.047 | 1.577 | 0.644 | 4/6 | 0.219 | 0.688 |
| PLS_v2 minus HbI-binned mean | absorbance RMSE | 6 | -6.710 | 2.599 | 1.061 | 6/6 | 0.031 | 0.031 |
| BiGRU minus HbI-binned mean | absorbance RMSE | 6 | -6.856 | 1.559 | 0.637 | 6/6 | 0.031 | 0.031 |
| BiGRU minus PLS_v2 | absorbance RMSE | 6 | -0.146 | 2.783 | 1.136 | 2/6 | 0.562 | 0.688 |
| BiGRU minus BiGRU (fixed setting) | absorbance RMSE | 6 | -0.179 | 0.740 | 0.302 | 3/6 | 0.688 | 1.000 |
| PLS_v2 minus Global Linear | reflectance RMSE | 6 | -0.114 | 0.193 | 0.079 | 4/6 | 0.312 | 0.688 |
| BiGRU minus Global Linear | reflectance RMSE | 6 | -0.170 | 0.201 | 0.082 | 4/6 | 0.156 | 0.688 |
| PLS_v2 minus HbI-binned mean | reflectance RMSE | 6 | -0.418 | 0.247 | 0.101 | 6/6 | 0.031 | 0.031 |
| BiGRU minus HbI-binned mean | reflectance RMSE | 6 | -0.475 | 0.177 | 0.072 | 6/6 | 0.031 | 0.031 |
| BiGRU minus PLS_v2 | reflectance RMSE | 6 | -0.057 | 0.284 | 0.116 | 1/6 | 0.438 | 0.219 |
| BiGRU minus BiGRU (fixed setting) | reflectance RMSE | 6 | -0.043 | 0.079 | 0.032 | 4/6 | 0.562 | 0.688 |
| PLS_v2 minus Global Linear | Peak RMSE | 6 | 0.029 | 0.066 | 0.027 | 3/6 | 0.562 | 1.000 |
| BiGRU minus Global Linear | Peak RMSE | 6 | 0.195 | 0.224 | 0.091 | 1/6 | 0.156 | 0.219 |
| PLS_v2 minus HbI-binned mean | Peak RMSE | 6 | -1.730 | 0.646 | 0.264 | 6/6 | 0.031 | 0.031 |
| BiGRU minus HbI-binned mean | Peak RMSE | 6 | -1.564 | 0.631 | 0.258 | 6/6 | 0.031 | 0.031 |
| BiGRU minus PLS_v2 | Peak RMSE | 6 | 0.166 | 0.186 | 0.076 | 1/6 | 0.156 | 0.219 |
| BiGRU minus BiGRU (fixed setting) | Peak RMSE | 6 | 0.081 | 0.083 | 0.034 | 1/6 | 0.062 | 0.219 |
| PLS_v2 minus Global Linear | Peak bias | 6 | -0.007 | 0.044 | 0.018 | 0/6 | 0.031 | 0.031 |
| BiGRU minus Global Linear | Peak bias | 6 | 0.101 | 0.086 | 0.035 | 0/6 | 0.031 | 0.031 |
| PLS_v2 minus HbI-binned mean | Peak bias | 6 | -0.007 | 0.044 | 0.018 | 0/6 | 0.031 | 0.031 |
| BiGRU minus HbI-binned mean | Peak bias | 6 | 0.101 | 0.086 | 0.035 | 0/6 | 0.031 | 0.031 |
| BiGRU minus PLS_v2 | Peak bias | 6 | 0.108 | 0.097 | 0.040 | 0/6 | 0.031 | 0.031 |
| BiGRU minus BiGRU (fixed setting) | Peak bias | 6 | 0.040 | 0.029 | 0.012 | 1/6 | 0.312 | 0.219 |

보조 표

| 모델 | absorbance RMSE | reflectance RMSE | Peak RMSE | Peak bias |
|---|---|---|---|---|
| PLS_v1 | 12.792 ± 5.062 | 1.067 ± 0.483 | 0.883 ± 0.406 | 0.022 ± 0.177 |
| Ridge | 12.075 ± 4.041 | 1.022 ± 0.425 | 0.851 ± 0.315 | 0.006 ± 0.038 |
| BiGRU (fixed setting) | 12.041 ± 4.218 | 0.980 ± 0.376 | 1.006 ± 0.479 | 0.061 ± 0.106 |
| BiGRU (Peak-constrained selection) | 12.910 ± 4.401 | 1.107 ± 0.405 | 0.893 ± 0.421 | 0.000 ± 0.000 |

| 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|
| PLS_v1 minus Global Linear | absorbance RMSE | 6 | -0.117 | 3.086 | 1.260 | 3/6 | 0.844 | 1.000 |
| Ridge minus Global Linear | absorbance RMSE | 6 | -0.835 | 1.497 | 0.611 | 4/6 | 0.219 | 0.688 |
| BiGRU (fixed setting) minus Global Linear | absorbance RMSE | 6 | -0.869 | 0.906 | 0.370 | 6/6 | 0.031 | 0.031 |
| BiGRU (Peak-constrained selection) minus Global Linear | absorbance RMSE | 6 | 0.000 | 0.000 | 0.000 | 0/6 | 1.000 | 1.000 |
| BiGRU (Peak-constrained selection) minus BiGRU | absorbance RMSE | 6 | 1.047 | 1.577 | 0.644 | 2/6 | 0.219 | 0.688 |
| PLS_v1 minus Global Linear | reflectance RMSE | 6 | -0.040 | 0.333 | 0.136 | 4/6 | 0.562 | 0.688 |
| Ridge minus Global Linear | reflectance RMSE | 6 | -0.085 | 0.185 | 0.076 | 3/6 | 0.438 | 1.000 |
| BiGRU (fixed setting) minus Global Linear | reflectance RMSE | 6 | -0.127 | 0.135 | 0.055 | 5/6 | 0.094 | 0.219 |
| BiGRU (Peak-constrained selection) minus Global Linear | reflectance RMSE | 6 | 0.000 | 0.000 | 0.000 | 0/6 | – | – |
| BiGRU (Peak-constrained selection) minus BiGRU | reflectance RMSE | 6 | 0.170 | 0.201 | 0.082 | 2/6 | 0.156 | 0.688 |
| PLS_v1 minus Global Linear | Peak RMSE | 6 | -0.010 | 0.039 | 0.016 | 4/6 | 0.562 | 0.688 |
| Ridge minus Global Linear | Peak RMSE | 6 | -0.041 | 0.114 | 0.047 | 3/6 | 0.688 | 1.000 |
| BiGRU (fixed setting) minus Global Linear | Peak RMSE | 6 | 0.114 | 0.182 | 0.074 | 3/6 | 0.438 | 1.000 |
| BiGRU (Peak-constrained selection) minus Global Linear | Peak RMSE | 6 | -0.000 | 0.000 | 0.000 | 2/6 | 0.500 | 0.500 |
| BiGRU (Peak-constrained selection) minus BiGRU | Peak RMSE | 6 | -0.195 | 0.224 | 0.091 | 5/6 | 0.156 | 0.219 |
| PLS_v1 minus Global Linear | Peak bias | 6 | 0.022 | 0.177 | 0.072 | 0/6 | 0.031 | 0.031 |
| Ridge minus Global Linear | Peak bias | 6 | 0.006 | 0.038 | 0.016 | 0/6 | 0.031 | 0.031 |
| BiGRU (fixed setting) minus Global Linear | Peak bias | 6 | 0.061 | 0.106 | 0.043 | 0/6 | 0.031 | 0.031 |
| BiGRU (Peak-constrained selection) minus Global Linear | Peak bias | 6 | -0.000 | 0.000 | 0.000 | 1/6 | 1.000 | 1.000 |
| BiGRU (Peak-constrained selection) minus BiGRU | Peak bias | 6 | -0.101 | 0.086 | 0.035 | 6/6 | 0.031 | 0.031 |

### 1.2 Fixed-target repeatability (In-range subject = S101, S106)

전체 (6명)

| 모델 | Peak 표준편차 | spectral 표준편차 |
|---|---|---|
| Raw | 2.301 ± 0.515 (n=6) | 18.318 ± 4.698 (n=6) |
| HbI-binned mean | 1.875 ± 0.685 (n=6) | 12.776 ± 2.305 (n=6) |
| Global Linear | 0.631 ± 0.298 (n=6) | 8.239 ± 2.882 (n=6) |
| PLS_v2 | 0.665 ± 0.256 (n=6) | 7.410 ± 1.995 (n=6) |
| BiGRU | 0.798 ± 0.421 (n=6) | 8.020 ± 3.070 (n=6) |
| PLS_v1 | 0.594 ± 0.272 (n=6) | 4.196 ± 1.668 (n=6) |
| Ridge | 0.626 ± 0.194 (n=6) | 7.439 ± 2.028 (n=6) |
| BiGRU (fixed setting) | 0.685 ± 0.310 (n=6) | 8.329 ± 3.198 (n=6) |
| BiGRU (Peak-constrained selection) | 0.631 ± 0.298 (n=6) | 8.239 ± 2.882 (n=6) |

In-range subject (S101, S106)

| 모델 | Peak 표준편차 | spectral 표준편차 |
|---|---|---|
| Raw | 2.586 ± 0.350 (n=2) | 23.488 ± 2.642 (n=2) |
| HbI-binned mean | 1.351 ± 0.014 (n=2) | 13.804 ± 1.797 (n=2) |
| Global Linear | 0.552 ± 0.049 (n=2) | 8.340 ± 2.063 (n=2) |
| PLS_v2 | 0.594 ± 0.043 (n=2) | 8.148 ± 2.177 (n=2) |
| BiGRU | 0.697 ± 0.039 (n=2) | 7.896 ± 2.517 (n=2) |
| PLS_v1 | 0.509 ± 0.037 (n=2) | 3.559 ± 0.233 (n=2) |
| Ridge | 0.580 ± 0.014 (n=2) | 7.700 ± 2.321 (n=2) |
| BiGRU (fixed setting) | 0.645 ± 0.066 (n=2) | 8.106 ± 2.366 (n=2) |
| BiGRU (Peak-constrained selection) | 0.552 ± 0.049 (n=2) | 8.340 ± 2.063 (n=2) |

그 밖의 피험자 (4명)

| 모델 | Peak 표준편차 | spectral 표준편차 |
|---|---|---|
| Raw | 2.158 ± 0.565 (n=4) | 15.734 ± 2.781 (n=4) |
| HbI-binned mean | 2.136 ± 0.713 (n=4) | 12.262 ± 2.593 (n=4) |
| Global Linear | 0.671 ± 0.375 (n=4) | 8.188 ± 3.523 (n=4) |
| PLS_v2 | 0.701 ± 0.322 (n=4) | 7.041 ± 2.124 (n=4) |
| BiGRU | 0.849 ± 0.534 (n=4) | 8.082 ± 3.685 (n=4) |
| PLS_v1 | 0.637 ± 0.340 (n=4) | 4.515 ± 2.053 (n=4) |
| Ridge | 0.649 ± 0.246 (n=4) | 7.309 ± 2.235 (n=4) |
| BiGRU (fixed setting) | 0.706 ± 0.396 (n=4) | 8.440 ± 3.890 (n=4) |
| BiGRU (Peak-constrained selection) | 0.671 ± 0.375 (n=4) | 8.188 ± 3.523 (n=4) |

사전 지정 비교 (전체 6명)

| 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|
| PLS_v2 minus Global Linear | Peak 표준편차 | 6 | 0.034 | 0.108 | 0.044 | 4/6 | 1.000 | 0.688 |
| BiGRU minus Global Linear | Peak 표준편차 | 6 | 0.167 | 0.136 | 0.056 | 0/6 | 0.031 | 0.031 |
| PLS_v2 minus HbI-binned mean | Peak 표준편차 | 6 | -1.209 | 0.482 | 0.197 | 6/6 | 0.031 | 0.031 |
| BiGRU minus HbI-binned mean | Peak 표준편차 | 6 | -1.076 | 0.368 | 0.150 | 6/6 | 0.031 | 0.031 |
| BiGRU minus PLS_v2 | Peak 표준편차 | 6 | 0.133 | 0.183 | 0.075 | 1/6 | 0.094 | 0.219 |
| BiGRU minus BiGRU (fixed setting) | Peak 표준편차 | 6 | 0.113 | 0.119 | 0.048 | 0/6 | 0.031 | 0.031 |
| PLS_v2 minus Global Linear | spectral 표준편차 | 6 | -0.829 | 1.646 | 0.672 | 4/6 | 0.219 | 0.688 |
| BiGRU minus Global Linear | spectral 표준편차 | 6 | -0.219 | 0.912 | 0.372 | 4/6 | 0.688 | 0.688 |
| PLS_v2 minus HbI-binned mean | spectral 표준편차 | 6 | -5.366 | 0.564 | 0.230 | 6/6 | 0.031 | 0.031 |
| BiGRU minus HbI-binned mean | spectral 표준편차 | 6 | -4.756 | 1.638 | 0.669 | 6/6 | 0.031 | 0.031 |
| BiGRU minus PLS_v2 | spectral 표준편차 | 6 | 0.610 | 1.908 | 0.779 | 4/6 | 1.000 | 0.688 |
| BiGRU minus BiGRU (fixed setting) | spectral 표준편차 | 6 | -0.309 | 0.313 | 0.128 | 5/6 | 0.062 | 0.219 |

### 1.3 Near-target accuracy, 허용 폭 0.02 (기술 통계만)

0.6 ± 0.02 실측이 S101과 S105에 1개씩뿐이므로 기술 통계로만 보고. n = 대상 피험자 수.

| 모델 | absorbance RMSE | Peak RMSE |
|---|---|---|
| HbI-binned mean | 19.709 ± 12.030 (n=2) | 1.454 ± 0.037 (n=2) |
| Global Linear | 9.952 ± 4.090 (n=2) | 0.842 ± 0.523 (n=2) |
| PLS_v2 | 9.432 ± 3.987 (n=2) | 0.851 ± 0.516 (n=2) |
| BiGRU | 8.839 ± 4.596 (n=2) | 0.932 ± 0.678 (n=2) |
| PLS_v1 | 8.656 ± 0.841 (n=2) | 0.822 ± 0.391 (n=2) |
| Ridge | 9.461 ± 3.076 (n=2) | 0.871 ± 0.510 (n=2) |
| BiGRU (fixed setting) | 9.375 ± 4.439 (n=2) | 0.876 ± 0.609 (n=2) |
| BiGRU (Peak-constrained selection) | 9.952 ± 4.090 (n=2) | 0.842 ± 0.523 (n=2) |

### 1.4 파장 구간별 복원 오차

구간 absorbance RMSE ×10⁻³

| 모델 | Soret (5) | Band 450-460 (2) | Carotenoid (5) | Q-band (7) | Red (11) |
|---|---|---|---|---|---|
| HbI-binned mean | 22.835 ± 5.828 | 21.268 ± 4.233 | 19.166 ± 2.883 | 16.498 ± 3.731 | 16.188 ± 5.902 |
| Global Linear | 15.345 ± 6.521 | 11.740 ± 5.505 | 10.641 ± 4.694 | 4.898 ± 1.630 | 15.413 ± 6.168 |
| PLS_v2 | 15.956 ± 5.418 | 11.454 ± 4.075 | 9.990 ± 3.439 | 4.549 ± 1.460 | 13.445 ± 6.296 |
| BiGRU | 16.371 ± 6.454 | 11.879 ± 6.027 | 10.299 ± 5.356 | 5.223 ± 1.663 | 12.190 ± 5.637 |
| PLS_v1 | 16.602 ± 6.828 | 11.672 ± 5.407 | 10.204 ± 4.662 | 5.433 ± 2.848 | 14.556 ± 7.090 |
| Ridge | 15.392 ± 4.871 | 11.002 ± 3.838 | 9.614 ± 3.092 | 4.643 ± 1.467 | 14.001 ± 6.690 |
| BiGRU (fixed setting) | 15.716 ± 6.027 | 11.740 ± 5.693 | 10.407 ± 5.098 | 4.926 ± 1.766 | 13.171 ± 5.978 |

구간 bias (예측 − 정답) ×10⁻³

| 모델 | Soret (5) | Band 450-460 (2) | Carotenoid (5) | Q-band (7) | Red (11) |
|---|---|---|---|---|---|
| HbI-binned mean | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.000 ± 0.000 | 0.000 ± 0.000 |
| Global Linear | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.000 ± 0.000 | -0.000 ± 0.000 | -0.000 ± 0.000 |
| PLS_v2 | 0.723 ± 1.336 | -0.463 ± 0.811 | -0.960 ± 0.762 | -0.441 ± 0.285 | -1.980 ± 1.077 |
| BiGRU | 0.233 ± 1.410 | 0.101 ± 1.091 | 0.483 ± 0.753 | 0.350 ± 0.666 | -0.010 ± 0.620 |
| PLS_v1 | 1.269 ± 13.343 | 0.022 ± 6.675 | -1.148 ± 4.709 | -1.303 ± 4.034 | -3.755 ± 9.271 |
| Ridge | 1.331 ± 0.986 | -0.254 ± 0.564 | -0.943 ± 0.655 | -0.495 ± 0.280 | -2.215 ± 1.089 |
| BiGRU (fixed setting) | 0.486 ± 1.171 | 0.168 ± 0.810 | 0.455 ± 0.498 | 0.350 ± 0.548 | 0.237 ± 0.376 |

570 nm: h_in → h* 고정, 오차 0 (모든 모델).

| 구간 | 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|---|
| band|Soret | PLS_v2 minus Global Linear | absorbance RMSE | 6 | 0.611 | 2.166 | 0.884 | 2/6 | 0.688 | 0.688 |
| band|Soret | BiGRU minus Global Linear | absorbance RMSE | 6 | 1.027 | 3.667 | 1.497 | 2/6 | 0.438 | 0.688 |
| band|Soret | BiGRU minus PLS_v2 | absorbance RMSE | 6 | 0.415 | 4.670 | 1.906 | 5/6 | 0.438 | 0.219 |
| band|Soret | PLS_v2 minus Global Linear | bias | 6 | 0.723 | 1.336 | 0.545 | 0/6 | 0.031 | 0.031 |
| band|Soret | BiGRU minus Global Linear | bias | 6 | 0.233 | 1.410 | 0.576 | 0/6 | 0.031 | 0.031 |
| band|Soret | BiGRU minus PLS_v2 | bias | 6 | -0.491 | 2.260 | 0.923 | 3/6 | 0.844 | 1.000 |
| band|Band 450-460 | PLS_v2 minus Global Linear | absorbance RMSE | 6 | -0.286 | 2.207 | 0.901 | 3/6 | 0.688 | 1.000 |
| band|Band 450-460 | BiGRU minus Global Linear | absorbance RMSE | 6 | 0.139 | 1.422 | 0.581 | 3/6 | 1.000 | 1.000 |
| band|Band 450-460 | BiGRU minus PLS_v2 | absorbance RMSE | 6 | 0.425 | 3.126 | 1.276 | 4/6 | 1.000 | 0.688 |
| band|Band 450-460 | PLS_v2 minus Global Linear | bias | 6 | -0.463 | 0.811 | 0.331 | 0/6 | 0.031 | 0.031 |
| band|Band 450-460 | BiGRU minus Global Linear | bias | 6 | 0.101 | 1.091 | 0.445 | 0/6 | 0.031 | 0.031 |
| band|Band 450-460 | BiGRU minus PLS_v2 | bias | 6 | 0.564 | 1.788 | 0.730 | 4/6 | 1.000 | 0.688 |
| band|Carotenoid | PLS_v2 minus Global Linear | absorbance RMSE | 6 | -0.651 | 2.391 | 0.976 | 4/6 | 0.562 | 0.688 |
| band|Carotenoid | BiGRU minus Global Linear | absorbance RMSE | 6 | -0.342 | 0.848 | 0.346 | 5/6 | 0.438 | 0.219 |
| band|Carotenoid | BiGRU minus PLS_v2 | absorbance RMSE | 6 | 0.309 | 2.697 | 1.101 | 3/6 | 0.844 | 1.000 |
| band|Carotenoid | PLS_v2 minus Global Linear | bias | 6 | -0.960 | 0.762 | 0.311 | 0/6 | 0.031 | 0.031 |
| band|Carotenoid | BiGRU minus Global Linear | bias | 6 | 0.483 | 0.753 | 0.307 | 0/6 | 0.031 | 0.031 |
| band|Carotenoid | BiGRU minus PLS_v2 | bias | 6 | 1.443 | 1.482 | 0.605 | 5/6 | 0.438 | 0.219 |
| band|Q-band | PLS_v2 minus Global Linear | absorbance RMSE | 6 | -0.349 | 0.617 | 0.252 | 4/6 | 0.219 | 0.688 |
| band|Q-band | BiGRU minus Global Linear | absorbance RMSE | 6 | 0.325 | 1.237 | 0.505 | 1/6 | 0.312 | 0.219 |
| band|Q-band | BiGRU minus PLS_v2 | absorbance RMSE | 6 | 0.674 | 1.405 | 0.574 | 1/6 | 0.219 | 0.219 |
| band|Q-band | PLS_v2 minus Global Linear | bias | 6 | -0.441 | 0.285 | 0.116 | 0/6 | 0.031 | 0.031 |
| band|Q-band | BiGRU minus Global Linear | bias | 6 | 0.350 | 0.666 | 0.272 | 0/6 | 0.031 | 0.031 |
| band|Q-band | BiGRU minus PLS_v2 | bias | 6 | 0.791 | 0.812 | 0.331 | 2/6 | 0.312 | 0.688 |
| band|Red | PLS_v2 minus Global Linear | absorbance RMSE | 6 | -1.968 | 3.097 | 1.264 | 4/6 | 0.219 | 0.688 |
| band|Red | BiGRU minus Global Linear | absorbance RMSE | 6 | -3.223 | 3.702 | 1.511 | 4/6 | 0.156 | 0.688 |
| band|Red | BiGRU minus PLS_v2 | absorbance RMSE | 6 | -1.255 | 4.486 | 1.831 | 2/6 | 1.000 | 0.688 |
| band|Red | PLS_v2 minus Global Linear | bias | 6 | -1.980 | 1.077 | 0.439 | 0/6 | 0.031 | 0.031 |
| band|Red | BiGRU minus Global Linear | bias | 6 | -0.010 | 0.620 | 0.253 | 0/6 | 0.031 | 0.031 |
| band|Red | BiGRU minus PLS_v2 | bias | 6 | 1.970 | 1.307 | 0.534 | 6/6 | 0.031 | 0.031 |

### 1.5 잔차 기울기의 파장 4구간 집계

| 구간 | 모델 | n | 구간 평균 잔차 기울기 (평균 ± 표준편차) | 구간 크기(RMS) 평균 | 개인 기준과의 Pearson 상관 | 부호 일치 | Global Linear 기울기 구간 평균 |
|---|---|---|---|---|---|---|---|
| Soret | PLS_v2 | 6 | 0.0073 ± 0.0719 | 0.0625 | -0.037 | 2/6 | 0.4448 |
| Band 450-460 | PLS_v2 | 6 | -0.0224 ± 0.0407 | 0.0398 | 0.150 | 3/6 | 0.5467 |
| Q-band | PLS_v2 | 6 | -0.0210 ± 0.0183 | 0.0303 | -0.163 | 6/6 | 0.8763 |
| Red | PLS_v2 | 6 | -0.1034 ± 0.1158 | 0.1164 | -0.073 | 5/6 | 0.1935 |
| Peak direction | PLS_v2 | 6 | -0.0009 ± 0.0013 | – | -0.584 | 3/6 | -0.0462 |
| Soret | BiGRU | 6 | 0.0423 ± 0.0727 | 0.0749 | 0.319 | 2/6 | 0.4448 |
| Band 450-460 | BiGRU | 6 | 0.0047 ± 0.0292 | 0.0253 | -0.149 | 4/6 | 0.5467 |
| Q-band | BiGRU | 6 | -0.0552 ± 0.0398 | 0.0629 | -0.103 | 5/6 | 0.8763 |
| Red | BiGRU | 6 | -0.1144 ± 0.0663 | 0.1161 | 0.204 | 5/6 | 0.1935 |
| Peak direction | BiGRU | 6 | -0.0007 ± 0.0050 | – | -0.957 | 1/6 | -0.0462 |
| Soret | Ridge | 6 | 0.0028 ± 0.0593 | 0.0462 | 0.142 | 2/6 | 0.4448 |
| Band 450-460 | Ridge | 6 | -0.0255 ± 0.0357 | 0.0389 | 0.355 | 4/6 | 0.5467 |
| Q-band | Ridge | 6 | -0.0193 ± 0.0180 | 0.0293 | -0.128 | 6/6 | 0.8763 |
| Red | Ridge | 6 | -0.0869 ± 0.1144 | 0.1028 | -0.120 | 5/6 | 0.1935 |
| Peak direction | Ridge | 6 | -0.0015 ± 0.0021 | – | 0.682 | 4/6 | -0.0462 |
| Soret | Personal reference | 6 | -0.1203 ± 0.1206 | 0.1296 | – | – | 0.4448 |
| Band 450-460 | Personal reference | 6 | -0.1321 ± 0.1266 | 0.1461 | – | – | 0.5467 |
| Q-band | Personal reference | 6 | -0.0497 ± 0.0279 | 0.0699 | – | – | 0.8763 |
| Red | Personal reference | 6 | -0.2183 ± 0.1176 | 0.2206 | – | – | 0.1935 |
| Peak direction | Personal reference | 6 | -0.0076 ± 0.0133 | – | – | – | -0.0462 |

검사: Δh = 0 제외 순서쌍 4개, BiGRU 잔차 기울기와 `BiGRUModel.residual`의 최대 차이 6.7e-13.

### 1.6 HbI 변화 크기 구간별 오차 (n = 그 구간에 쌍이 있는 피험자 수)

absorbance RMSE, 방향: 전체

| 모델 | 0.000-0.025 | 0.025-0.050 | 0.050-0.100 | 0.100-0.300 |
|---|---|---|---|---|
| HbI-binned mean | 12.357 ± 4.503 (n=6) | 18.420 ± 4.326 (n=6) | 22.003 ± 6.462 (n=6) | 22.808 ± 5.347 (n=5) |
| Global Linear | 6.489 ± 2.004 (n=6) | 9.868 ± 4.960 (n=6) | 16.613 ± 6.542 (n=6) | 18.878 ± 5.378 (n=5) |
| PLS_v2 | 6.512 ± 2.103 (n=6) | 10.880 ± 3.919 (n=6) | 14.846 ± 6.310 (n=6) | 17.923 ± 3.962 (n=5) |
| BiGRU | 6.346 ± 1.938 (n=6) | 9.691 ± 4.000 (n=6) | 15.046 ± 6.217 (n=6) | 16.625 ± 5.678 (n=5) |
| PLS_v1 | 10.562 ± 4.514 (n=6) | 10.834 ± 3.687 (n=6) | 14.665 ± 6.239 (n=6) | 15.872 ± 5.223 (n=5) |
| Ridge | 6.454 ± 2.055 (n=6) | 10.736 ± 4.050 (n=6) | 15.176 ± 6.130 (n=6) | 17.731 ± 4.476 (n=5) |
| BiGRU (fixed setting) | 6.350 ± 1.970 (n=6) | 9.552 ± 4.468 (n=6) | 15.400 ± 6.331 (n=6) | 17.056 ± 5.558 (n=5) |
| BiGRU (Peak-constrained selection) | 6.489 ± 2.004 (n=6) | 9.868 ± 4.960 (n=6) | 16.613 ± 6.542 (n=6) | 18.878 ± 5.378 (n=5) |

absorbance RMSE, 방향: 증가

| 모델 | 0.000-0.025 | 0.025-0.050 | 0.050-0.100 | 0.100-0.300 |
|---|---|---|---|---|
| HbI-binned mean | 12.413 ± 4.443 (n=6) | 18.420 ± 4.326 (n=6) | 22.003 ± 6.462 (n=6) | 22.808 ± 5.347 (n=5) |
| Global Linear | 6.519 ± 1.996 (n=6) | 9.868 ± 4.960 (n=6) | 16.613 ± 6.542 (n=6) | 18.878 ± 5.378 (n=5) |
| PLS_v2 | 6.516 ± 2.155 (n=6) | 10.893 ± 3.742 (n=6) | 14.221 ± 5.639 (n=6) | 17.713 ± 1.560 (n=5) |
| BiGRU | 6.402 ± 1.964 (n=6) | 9.793 ± 4.234 (n=6) | 15.353 ± 6.558 (n=6) | 16.585 ± 5.427 (n=5) |
| PLS_v1 | 10.256 ± 4.698 (n=6) | 10.464 ± 3.274 (n=6) | 12.952 ± 2.244 (n=6) | 13.570 ± 2.160 (n=5) |
| Ridge | 6.454 ± 2.092 (n=6) | 10.697 ± 3.860 (n=6) | 14.561 ± 5.423 (n=6) | 17.122 ± 2.212 (n=5) |
| BiGRU (fixed setting) | 6.387 ± 1.993 (n=6) | 9.600 ± 4.691 (n=6) | 15.775 ± 6.585 (n=6) | 17.559 ± 5.546 (n=5) |
| BiGRU (Peak-constrained selection) | 6.519 ± 1.996 (n=6) | 9.868 ± 4.960 (n=6) | 16.613 ± 6.542 (n=6) | 18.878 ± 5.378 (n=5) |

absorbance RMSE, 방향: 감소

| 모델 | 0.000-0.025 | 0.025-0.050 | 0.050-0.100 | 0.100-0.300 |
|---|---|---|---|---|
| HbI-binned mean | 12.413 ± 4.443 (n=6) | 18.420 ± 4.326 (n=6) | 22.003 ± 6.462 (n=6) | 22.808 ± 5.347 (n=5) |
| Global Linear | 6.519 ± 1.996 (n=6) | 9.868 ± 4.960 (n=6) | 16.613 ± 6.542 (n=6) | 18.878 ± 5.378 (n=5) |
| PLS_v2 | 6.578 ± 2.074 (n=6) | 10.858 ± 4.116 (n=6) | 15.415 ± 7.000 (n=6) | 17.704 ± 6.931 (n=5) |
| BiGRU | 6.352 ± 1.898 (n=6) | 9.578 ± 3.772 (n=6) | 14.705 ± 5.880 (n=6) | 16.521 ± 6.254 (n=5) |
| PLS_v1 | 10.846 ± 4.555 (n=6) | 10.602 ± 5.647 (n=6) | 15.165 ± 10.569 (n=6) | 16.880 ± 9.661 (n=5) |
| Ridge | 6.520 ± 2.034 (n=6) | 10.767 ± 4.252 (n=6) | 15.740 ± 6.842 (n=6) | 17.865 ± 7.470 (n=5) |
| BiGRU (fixed setting) | 6.371 ± 1.932 (n=6) | 9.497 ± 4.253 (n=6) | 15.008 ± 6.077 (n=6) | 16.457 ± 5.770 (n=5) |
| BiGRU (Peak-constrained selection) | 6.519 ± 1.996 (n=6) | 9.868 ± 4.960 (n=6) | 16.613 ± 6.542 (n=6) | 18.878 ± 5.378 (n=5) |

Peak RMSE, 방향 전체

| 모델 | 0.000-0.025 | 0.025-0.050 | 0.050-0.100 | 0.100-0.300 |
|---|---|---|---|---|
| HbI-binned mean | 1.536 ± 0.380 (n=6) | 2.600 ± 0.810 (n=6) | 3.210 ± 1.945 (n=6) | 2.970 ± 1.471 (n=5) |
| Global Linear | 0.611 ± 0.086 (n=6) | 0.602 ± 0.285 (n=6) | 1.060 ± 0.661 (n=6) | 1.374 ± 0.830 (n=5) |
| PLS_v2 | 0.618 ± 0.087 (n=6) | 0.618 ± 0.280 (n=6) | 1.082 ± 0.643 (n=6) | 1.455 ± 0.758 (n=5) |
| BiGRU | 0.632 ± 0.097 (n=6) | 0.686 ± 0.328 (n=6) | 1.224 ± 0.827 (n=6) | 1.955 ± 0.955 (n=5) |
| PLS_v1 | 0.640 ± 0.089 (n=6) | 0.606 ± 0.265 (n=6) | 1.031 ± 0.639 (n=6) | 1.334 ± 0.795 (n=5) |
| Ridge | 0.607 ± 0.084 (n=6) | 0.575 ± 0.229 (n=6) | 0.999 ± 0.530 (n=6) | 1.263 ± 0.572 (n=5) |
| BiGRU (fixed setting) | 0.616 ± 0.090 (n=6) | 0.641 ± 0.308 (n=6) | 1.135 ± 0.743 (n=6) | 1.739 ± 0.872 (n=5) |
| BiGRU (Peak-constrained selection) | 0.611 ± 0.086 (n=6) | 0.602 ± 0.285 (n=6) | 1.060 ± 0.661 (n=6) | 1.374 ± 0.830 (n=5) |

구간별 쌍체 차이 (방향 전체)

| 구간 | 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|---|
| dh|0.000-0.025|all | PLS_v2 minus Global Linear | absorbance RMSE | 6 | 0.022 | 0.572 | 0.233 | 2/6 | 0.688 | 0.688 |
| dh|0.000-0.025|all | BiGRU minus Global Linear | absorbance RMSE | 6 | -0.143 | 0.243 | 0.099 | 4/6 | 0.312 | 0.688 |
| dh|0.000-0.025|all | BiGRU minus PLS_v2 | absorbance RMSE | 6 | -0.165 | 0.499 | 0.204 | 4/6 | 0.688 | 0.688 |
| dh|0.000-0.025|all | PLS_v2 minus Global Linear | Peak RMSE | 6 | 0.006 | 0.016 | 0.007 | 2/6 | 0.438 | 0.688 |
| dh|0.000-0.025|all | BiGRU minus Global Linear | Peak RMSE | 6 | 0.020 | 0.032 | 0.013 | 3/6 | 0.438 | 1.000 |
| dh|0.000-0.025|all | BiGRU minus PLS_v2 | Peak RMSE | 6 | 0.014 | 0.030 | 0.012 | 2/6 | 0.312 | 0.688 |
| dh|0.025-0.050|all | PLS_v2 minus Global Linear | absorbance RMSE | 6 | 1.012 | 1.891 | 0.772 | 2/6 | 0.438 | 0.688 |
| dh|0.025-0.050|all | BiGRU minus Global Linear | absorbance RMSE | 6 | -0.176 | 1.291 | 0.527 | 2/6 | 1.000 | 0.688 |
| dh|0.025-0.050|all | BiGRU minus PLS_v2 | absorbance RMSE | 6 | -1.188 | 1.374 | 0.561 | 5/6 | 0.156 | 0.219 |
| dh|0.025-0.050|all | PLS_v2 minus Global Linear | Peak RMSE | 6 | 0.016 | 0.030 | 0.012 | 2/6 | 0.312 | 0.688 |
| dh|0.025-0.050|all | BiGRU minus Global Linear | Peak RMSE | 6 | 0.084 | 0.115 | 0.047 | 2/6 | 0.219 | 0.688 |
| dh|0.025-0.050|all | BiGRU minus PLS_v2 | Peak RMSE | 6 | 0.068 | 0.113 | 0.046 | 1/6 | 0.219 | 0.219 |
| dh|0.050-0.100|all | PLS_v2 minus Global Linear | absorbance RMSE | 6 | -1.767 | 2.918 | 1.191 | 4/6 | 0.219 | 0.688 |
| dh|0.050-0.100|all | BiGRU minus Global Linear | absorbance RMSE | 6 | -1.567 | 2.360 | 0.964 | 4/6 | 0.312 | 0.688 |
| dh|0.050-0.100|all | BiGRU minus PLS_v2 | absorbance RMSE | 6 | 0.200 | 4.519 | 1.845 | 2/6 | 0.688 | 0.688 |
| dh|0.050-0.100|all | PLS_v2 minus Global Linear | Peak RMSE | 6 | 0.022 | 0.059 | 0.024 | 3/6 | 0.562 | 1.000 |
| dh|0.050-0.100|all | BiGRU minus Global Linear | Peak RMSE | 6 | 0.165 | 0.231 | 0.094 | 1/6 | 0.156 | 0.219 |
| dh|0.050-0.100|all | BiGRU minus PLS_v2 | Peak RMSE | 6 | 0.142 | 0.226 | 0.092 | 1/6 | 0.219 | 0.219 |
| dh|0.100-0.300|all | PLS_v2 minus Global Linear | absorbance RMSE | 5 | -0.955 | 2.785 | 1.245 | 3/5 | 0.625 | 1.000 |
| dh|0.100-0.300|all | BiGRU minus Global Linear | absorbance RMSE | 5 | -2.252 | 3.471 | 1.552 | 4/5 | 0.188 | 0.375 |
| dh|0.100-0.300|all | BiGRU minus PLS_v2 | absorbance RMSE | 5 | -1.297 | 5.396 | 2.413 | 3/5 | 1.000 | 1.000 |
| dh|0.100-0.300|all | PLS_v2 minus Global Linear | Peak RMSE | 5 | 0.081 | 0.214 | 0.096 | 3/5 | 0.812 | 1.000 |
| dh|0.100-0.300|all | BiGRU minus Global Linear | Peak RMSE | 5 | 0.581 | 0.538 | 0.241 | 0/5 | 0.062 | 0.062 |
| dh|0.100-0.300|all | BiGRU minus PLS_v2 | Peak RMSE | 5 | 0.500 | 0.352 | 0.157 | 0/5 | 0.062 | 0.062 |

### 1.7 추가 기준값: 피험자 내 Peak 대 HbI 선형회귀의 HbI 0.6 값 (Peak_RS 단위)

Peak_RS = 8387.2 x Peak + 83.495 (Hwang et al. 2023, Sensors 23(17):7654 식 (2)). 기준값은 피험자 내 선형회귀의 추정값이며 독립 정답이 아니다. In-range subject가 아니면 외삽.

| 피험자 | 구분 | 기준값 ± 표준오차 | HbI-binned mean − 기준값 | Global Linear − 기준값 | PLS_v2 − 기준값 | BiGRU − 기준값 | PLS_v1 − 기준값 | Ridge − 기준값 | BiGRU (fixed setting) − 기준값 |
|---|---|---|---|---|---|---|---|---|---|
| S101 | In-range | 140.74 ± 1.38 | -10.47 | +2.79 | +2.93 | +4.18 | +2.19 | +3.04 | +3.57 |
| S102 | 외삽 | 76.03 ± 2.66 | -17.65 | -5.73 | -7.66 | -9.18 | -4.91 | -5.91 | -8.01 |
| S103 | 외삽 | 122.66 ± 3.44 | +5.34 | +5.58 | +5.21 | +6.01 | +5.67 | +4.10 | +4.49 |
| S104 | 외삽 | 142.20 ± 3.59 | +30.60 | +28.39 | +27.96 | +35.42 | +26.45 | +23.63 | +30.71 |
| S105 | 외삽 | 100.79 ± 2.25 | -10.30 | +4.91 | +5.39 | +4.33 | +6.29 | +5.39 | +4.41 |
| S106 | In-range | 79.16 ± 2.22 | -13.50 | -2.08 | -3.72 | -2.52 | +0.10 | -3.28 | -1.42 |

## 2. Union16 분할 (보조, 고정 설정)

- 규칙 3의 의도된 예외: 고정 하이퍼파라미터는 Development cohort 10명 전체로 고른 값이므로 Development cohort 피험자가 test인 분할에서는 그 피험자가 선택에 쓰였다. External cohort 피험자가 test인 분할에는 이 문제가 없다.
- ExternalTest6과 독립이 아니므로 일반화 성능 추정치가 아니라 데이터 증가 효과의 추정으로만 표기한다.
- 하이퍼파라미터는 동결 문서의 값으로 고정했다 (다시 고르지 않음).

### Union16_LeaveOneSubjectOut

집계: 피험자 16명의 평균 ± 표준편차.

| 모델 | absorbance RMSE | reflectance RMSE | Peak RMSE | Peak bias |
|---|---|---|---|---|
| HbI-binned mean | 15.698 ± 3.005 | 1.143 ± 0.207 | 1.768 ± 0.978 | -0.000 ± 0.000 |
| Global Linear | 11.934 ± 4.025 | 0.940 ± 0.321 | 0.787 ± 0.298 | -0.000 ± 0.000 |
| PLS_v2 | 11.275 ± 3.012 | 0.909 ± 0.287 | 0.876 ± 0.314 | -0.049 ± 0.039 |
| BiGRU | 11.361 ± 3.405 | 0.874 ± 0.252 | 1.026 ± 0.508 | 0.107 ± 0.208 |
| PLS_v1 | 11.391 ± 3.831 | 0.906 ± 0.329 | 0.826 ± 0.285 | -0.014 ± 0.275 |
| Ridge | 11.062 ± 3.022 | 0.892 ± 0.279 | 0.751 ± 0.248 | -0.004 ± 0.054 |
| BiGRU (fixed setting) | 11.165 ± 3.458 | 0.858 ± 0.270 | 0.934 ± 0.419 | 0.064 ± 0.148 |
| BiGRU (Peak-constrained selection) | 11.934 ± 4.025 | 0.940 ± 0.321 | 0.787 ± 0.298 | -0.000 ± 0.000 |

| 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|
| PLS_v2 minus Global Linear | absorbance RMSE | 16 | -0.659 | 2.204 | 0.551 | 9/16 | 0.562 | 0.804 |
| BiGRU minus Global Linear | absorbance RMSE | 16 | -0.574 | 1.259 | 0.315 | 10/16 | 0.193 | 0.454 |
| PLS_v2 minus HbI-binned mean | absorbance RMSE | 16 | -4.423 | 3.045 | 0.761 | 14/16 | 0.000 | 0.004 |
| BiGRU minus HbI-binned mean | absorbance RMSE | 16 | -4.337 | 2.105 | 0.526 | 16/16 | 0.000 | 0.000 |
| BiGRU minus PLS_v2 | absorbance RMSE | 16 | 0.086 | 2.104 | 0.526 | 6/16 | 0.597 | 0.454 |
| BiGRU minus BiGRU (fixed setting) | absorbance RMSE | 16 | 0.196 | 0.485 | 0.121 | 5/16 | 0.130 | 0.210 |
| PLS_v2 minus Global Linear | reflectance RMSE | 16 | -0.032 | 0.148 | 0.037 | 8/16 | 0.528 | 1.000 |
| BiGRU minus Global Linear | reflectance RMSE | 16 | -0.066 | 0.135 | 0.034 | 12/16 | 0.093 | 0.077 |
| PLS_v2 minus HbI-binned mean | reflectance RMSE | 16 | -0.234 | 0.253 | 0.063 | 13/16 | 0.002 | 0.021 |
| BiGRU minus HbI-binned mean | reflectance RMSE | 16 | -0.269 | 0.171 | 0.043 | 16/16 | 0.000 | 0.000 |
| BiGRU minus PLS_v2 | reflectance RMSE | 16 | -0.035 | 0.184 | 0.046 | 7/16 | 0.821 | 0.804 |
| BiGRU minus BiGRU (fixed setting) | reflectance RMSE | 16 | 0.016 | 0.058 | 0.015 | 6/16 | 0.298 | 0.454 |
| PLS_v2 minus Global Linear | Peak RMSE | 16 | 0.089 | 0.215 | 0.054 | 5/16 | 0.065 | 0.210 |
| BiGRU minus Global Linear | Peak RMSE | 16 | 0.239 | 0.348 | 0.087 | 5/16 | 0.018 | 0.210 |
| PLS_v2 minus HbI-binned mean | Peak RMSE | 16 | -0.892 | 0.954 | 0.238 | 15/16 | 0.000 | 0.001 |
| BiGRU minus HbI-binned mean | Peak RMSE | 16 | -0.742 | 0.837 | 0.209 | 14/16 | 0.001 | 0.004 |
| BiGRU minus PLS_v2 | Peak RMSE | 16 | 0.150 | 0.443 | 0.111 | 5/16 | 0.159 | 0.210 |
| BiGRU minus BiGRU (fixed setting) | Peak RMSE | 16 | 0.092 | 0.137 | 0.034 | 4/16 | 0.016 | 0.077 |

| 모델 | Peak 표준편차 | spectral 표준편차 |
|---|---|---|
| Raw | 2.239 ± 0.715 (n=16) | 21.317 ± 8.983 (n=16) |
| HbI-binned mean | 1.250 ± 0.692 (n=16) | 10.582 ± 2.006 (n=16) |
| Global Linear | 0.556 ± 0.211 (n=16) | 7.395 ± 2.457 (n=16) |
| PLS_v2 | 0.622 ± 0.212 (n=16) | 6.807 ± 1.682 (n=16) |
| BiGRU | 0.758 ± 0.354 (n=16) | 6.887 ± 2.350 (n=16) |
| PLS_v1 | 0.550 ± 0.197 (n=16) | 4.193 ± 1.714 (n=16) |
| Ridge | 0.566 ± 0.192 (n=16) | 6.875 ± 1.752 (n=16) |
| BiGRU (fixed setting) | 0.661 ± 0.291 (n=16) | 6.906 ± 2.345 (n=16) |
| BiGRU (Peak-constrained selection) | 0.556 ± 0.211 (n=16) | 7.395 ± 2.457 (n=16) |

### Union16_Holdout73

집계: 피험자 → 분할 안 평균 → 분할 10개의 평균 ± 표준편차; 쌍체 비교는 피험자별 분할 평균 뒤 n = 16.

| 모델 | absorbance RMSE | reflectance RMSE | Peak RMSE | Peak bias |
|---|---|---|---|---|
| HbI-binned mean | 15.985 ± 1.178 | 1.154 ± 0.085 | 1.750 ± 0.385 | -0.000 ± 0.000 |
| Global Linear | 11.895 ± 1.831 | 0.929 ± 0.163 | 0.790 ± 0.129 | -0.000 ± 0.000 |
| PLS_v2 | 11.842 ± 1.795 | 0.967 ± 0.167 | 0.830 ± 0.128 | -0.032 ± 0.025 |
| BiGRU | 11.669 ± 1.385 | 0.887 ± 0.122 | 1.111 ± 0.382 | 0.126 ± 0.183 |
| PLS_v1 | 11.721 ± 1.724 | 0.934 ± 0.148 | 0.824 ± 0.125 | -0.016 ± 0.152 |
| Ridge | 11.637 ± 1.451 | 0.957 ± 0.144 | 0.759 ± 0.120 | -0.011 ± 0.032 |
| BiGRU (fixed setting) | 11.337 ± 1.476 | 0.863 ± 0.126 | 0.984 ± 0.279 | 0.080 ± 0.144 |
| BiGRU (Peak-constrained selection) | 11.895 ± 1.831 | 0.929 ± 0.163 | 0.790 ± 0.129 | -0.000 ± 0.000 |

| 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|
| PLS_v2 minus Global Linear | absorbance RMSE | 16 | -0.065 | 2.284 | 0.571 | 5/16 | 0.464 | 0.210 |
| BiGRU minus Global Linear | absorbance RMSE | 16 | -0.284 | 1.414 | 0.354 | 8/16 | 0.464 | 1.000 |
| PLS_v2 minus HbI-binned mean | absorbance RMSE | 16 | -4.016 | 3.318 | 0.830 | 13/16 | 0.001 | 0.021 |
| BiGRU minus HbI-binned mean | absorbance RMSE | 16 | -4.235 | 2.251 | 0.563 | 16/16 | 0.000 | 0.000 |
| BiGRU minus PLS_v2 | absorbance RMSE | 16 | -0.218 | 2.052 | 0.513 | 8/16 | 0.860 | 1.000 |
| BiGRU minus BiGRU (fixed setting) | absorbance RMSE | 16 | 0.316 | 0.430 | 0.107 | 5/16 | 0.011 | 0.210 |
| PLS_v2 minus Global Linear | reflectance RMSE | 16 | 0.037 | 0.169 | 0.042 | 6/16 | 0.375 | 0.454 |
| BiGRU minus Global Linear | reflectance RMSE | 16 | -0.048 | 0.159 | 0.040 | 12/16 | 0.175 | 0.077 |
| PLS_v2 minus HbI-binned mean | reflectance RMSE | 16 | -0.176 | 0.293 | 0.073 | 12/16 | 0.051 | 0.077 |
| BiGRU minus HbI-binned mean | reflectance RMSE | 16 | -0.261 | 0.175 | 0.044 | 16/16 | 0.000 | 0.000 |
| BiGRU minus PLS_v2 | reflectance RMSE | 16 | -0.085 | 0.208 | 0.052 | 9/16 | 0.252 | 0.804 |
| BiGRU minus BiGRU (fixed setting) | reflectance RMSE | 16 | 0.022 | 0.046 | 0.011 | 5/16 | 0.083 | 0.210 |
| PLS_v2 minus Global Linear | Peak RMSE | 16 | 0.041 | 0.206 | 0.051 | 7/16 | 0.433 | 0.804 |
| BiGRU minus Global Linear | Peak RMSE | 16 | 0.332 | 0.475 | 0.119 | 1/16 | 0.000 | 0.001 |
| PLS_v2 minus HbI-binned mean | Peak RMSE | 16 | -0.928 | 0.599 | 0.150 | 16/16 | 0.000 | 0.000 |
| BiGRU minus HbI-binned mean | Peak RMSE | 16 | -0.636 | 0.634 | 0.158 | 13/16 | 0.001 | 0.021 |
| BiGRU minus PLS_v2 | Peak RMSE | 16 | 0.292 | 0.581 | 0.145 | 2/16 | 0.013 | 0.004 |
| BiGRU minus BiGRU (fixed setting) | Peak RMSE | 16 | 0.130 | 0.157 | 0.039 | 3/16 | 0.001 | 0.021 |

| 모델 | Peak 표준편차 | spectral 표준편차 |
|---|---|---|
| Raw | 2.242 ± 0.251 (n=10) | 21.260 ± 3.291 (n=10) |
| HbI-binned mean | 1.238 ± 0.273 (n=10) | 10.750 ± 0.827 (n=10) |
| Global Linear | 0.558 ± 0.091 (n=10) | 7.365 ± 1.098 (n=10) |
| PLS_v2 | 0.597 ± 0.068 (n=10) | 7.386 ± 1.061 (n=10) |
| BiGRU | 0.790 ± 0.246 (n=10) | 7.099 ± 1.112 (n=10) |
| PLS_v1 | 0.548 ± 0.090 (n=10) | 4.257 ± 0.827 (n=10) |
| Ridge | 0.564 ± 0.074 (n=10) | 7.250 ± 0.872 (n=10) |
| BiGRU (fixed setting) | 0.675 ± 0.187 (n=10) | 6.975 ± 0.802 (n=10) |
| BiGRU (Peak-constrained selection) | 0.558 ± 0.091 (n=10) | 7.365 ± 1.098 (n=10) |

### Development10_LeaveOneSubjectOut_FixedHyperparameters

집계: Development cohort 10명, 9명 학습, 동결 문서의 고정값 (학습 피험자 수 비교용).

| 모델 | absorbance RMSE | reflectance RMSE | Peak RMSE | Peak bias |
|---|---|---|---|---|
| HbI-binned mean | 17.414 ± 3.010 | 1.221 ± 0.222 | 2.528 ± 0.727 | -0.000 ± 0.000 |
| Global Linear | 12.126 ± 3.222 | 0.915 ± 0.245 | 0.716 ± 0.203 | -0.000 ± 0.000 |
| PLS_v2 | 10.427 ± 2.737 | 0.829 ± 0.197 | 0.685 ± 0.212 | 0.006 ± 0.034 |
| BiGRU | 11.141 ± 2.572 | 0.828 ± 0.178 | 1.059 ± 0.706 | 0.175 ± 0.228 |
| PLS_v1 | 11.011 ± 2.585 | 0.853 ± 0.229 | 0.750 ± 0.198 | -0.004 ± 0.229 |
| Ridge | 10.540 ± 2.698 | 0.839 ± 0.207 | 0.666 ± 0.203 | 0.007 ± 0.038 |
| BiGRU (fixed setting) | 11.280 ± 2.705 | 0.846 ± 0.187 | 1.015 ± 0.613 | 0.115 ± 0.223 |
| BiGRU (Peak-constrained selection) | 12.126 ± 3.222 | 0.915 ± 0.245 | 0.716 ± 0.203 | -0.000 ± 0.000 |

| 비교 | 지표 | n | 차이 평균 | 차이 표준편차 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|
| PLS_v2 minus Global Linear | absorbance RMSE | 10 | -1.699 | 2.249 | 0.711 | 8/10 | 0.037 | 0.109 |
| BiGRU minus Global Linear | absorbance RMSE | 10 | -0.984 | 1.478 | 0.468 | 7/10 | 0.084 | 0.344 |
| PLS_v2 minus HbI-binned mean | absorbance RMSE | 10 | -6.987 | 3.319 | 1.050 | 10/10 | 0.002 | 0.002 |
| BiGRU minus HbI-binned mean | absorbance RMSE | 10 | -6.273 | 3.565 | 1.127 | 10/10 | 0.002 | 0.002 |
| BiGRU minus PLS_v2 | absorbance RMSE | 10 | 0.714 | 2.143 | 0.678 | 6/10 | 0.695 | 0.754 |
| BiGRU minus BiGRU (fixed setting) | absorbance RMSE | 10 | -0.139 | 0.473 | 0.150 | 7/10 | 0.557 | 0.344 |
| PLS_v2 minus Global Linear | reflectance RMSE | 10 | -0.086 | 0.161 | 0.051 | 9/10 | 0.084 | 0.021 |
| BiGRU minus Global Linear | reflectance RMSE | 10 | -0.087 | 0.181 | 0.057 | 8/10 | 0.193 | 0.109 |
| PLS_v2 minus HbI-binned mean | reflectance RMSE | 10 | -0.392 | 0.230 | 0.073 | 10/10 | 0.002 | 0.002 |
| BiGRU minus HbI-binned mean | reflectance RMSE | 10 | -0.394 | 0.262 | 0.083 | 10/10 | 0.002 | 0.002 |
| BiGRU minus PLS_v2 | reflectance RMSE | 10 | -0.001 | 0.160 | 0.051 | 6/10 | 0.695 | 0.754 |
| BiGRU minus BiGRU (fixed setting) | reflectance RMSE | 10 | -0.019 | 0.045 | 0.014 | 8/10 | 0.193 | 0.109 |
| PLS_v2 minus Global Linear | Peak RMSE | 10 | -0.031 | 0.122 | 0.038 | 3/10 | 0.625 | 0.344 |
| BiGRU minus Global Linear | Peak RMSE | 10 | 0.342 | 0.579 | 0.183 | 3/10 | 0.105 | 0.344 |
| PLS_v2 minus HbI-binned mean | Peak RMSE | 10 | -1.842 | 0.807 | 0.255 | 10/10 | 0.002 | 0.002 |
| BiGRU minus HbI-binned mean | Peak RMSE | 10 | -1.469 | 1.038 | 0.328 | 9/10 | 0.004 | 0.021 |
| BiGRU minus PLS_v2 | Peak RMSE | 10 | 0.373 | 0.654 | 0.207 | 4/10 | 0.131 | 0.754 |
| BiGRU minus BiGRU (fixed setting) | Peak RMSE | 10 | 0.044 | 0.180 | 0.057 | 5/10 | 0.695 | 1.000 |

| 모델 | Peak 표준편차 | spectral 표준편차 |
|---|---|---|
| Raw | 2.202 ± 0.837 (n=10) | 23.117 ± 10.614 (n=10) |
| HbI-binned mean | 1.787 ± 0.514 (n=10) | 11.720 ± 2.070 (n=10) |
| Global Linear | 0.507 ± 0.143 (n=10) | 7.331 ± 1.815 (n=10) |
| PLS_v2 | 0.484 ± 0.156 (n=10) | 6.238 ± 1.725 (n=10) |
| BiGRU | 0.746 ± 0.457 (n=10) | 6.881 ± 1.443 (n=10) |
| PLS_v1 | 0.510 ± 0.138 (n=10) | 3.854 ± 1.474 (n=10) |
| Ridge | 0.475 ± 0.151 (n=10) | 6.183 ± 1.537 (n=10) |
| BiGRU (fixed setting) | 0.711 ± 0.388 (n=10) | 6.957 ± 1.437 (n=10) |
| BiGRU (Peak-constrained selection) | 0.507 ± 0.143 (n=10) | 7.331 ± 1.815 (n=10) |

### 학습 피험자 수 효과 (General reconstruction, 피험자 쌍체 차이; 차이 = 15명 학습 − 더 적은 학습, 개선 = 차이 < 0)

| 대상 | 역할 | 비교 | 모델 | 지표 | n | 차이 평균 | 표준오차 | 개선 k/n | Wilcoxon p | 부호검정 p |
|---|---|---|---|---|---|---|---|---|---|---|
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | HbI-binned mean | absorbance RMSE | 6 | -2.392 | 0.472 | 6/6 | 0.031 | 0.031 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | HbI-binned mean | reflectance RMSE | 6 | -0.193 | 0.038 | 6/6 | 0.031 | 0.031 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | HbI-binned mean | Peak RMSE | 6 | -1.057 | 0.047 | 6/6 | 0.031 | 0.031 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | Global Linear | absorbance RMSE | 6 | -1.527 | 0.154 | 6/6 | 0.031 | 0.031 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | Global Linear | reflectance RMSE | 6 | -0.174 | 0.021 | 6/6 | 0.031 | 0.031 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | Global Linear | Peak RMSE | 6 | 0.000 | 0.028 | 4/6 | 1.000 | 0.688 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | PLS_v2 | absorbance RMSE | 6 | -0.712 | 0.252 | 5/6 | 0.062 | 0.219 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | PLS_v2 | reflectance RMSE | 6 | -0.078 | 0.023 | 5/6 | 0.062 | 0.219 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | PLS_v2 | Peak RMSE | 6 | 0.017 | 0.050 | 3/6 | 0.844 | 1.000 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | BiGRU | absorbance RMSE | 6 | -0.823 | 0.205 | 6/6 | 0.031 | 0.031 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | BiGRU | reflectance RMSE | 6 | -0.059 | 0.025 | 5/6 | 0.094 | 0.219 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | BiGRU | Peak RMSE | 6 | -0.089 | 0.029 | 5/6 | 0.062 | 0.219 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | PLS_v1 | absorbance RMSE | 6 | -1.049 | 0.166 | 6/6 | 0.031 | 0.031 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | PLS_v1 | reflectance RMSE | 6 | -0.133 | 0.026 | 6/6 | 0.031 | 0.031 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | PLS_v1 | Peak RMSE | 6 | 0.025 | 0.024 | 3/6 | 0.688 | 1.000 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | Ridge | absorbance RMSE | 6 | -1.162 | 0.497 | 5/6 | 0.062 | 0.219 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | Ridge | reflectance RMSE | 6 | -0.147 | 0.053 | 6/6 | 0.031 | 0.031 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | Ridge | Peak RMSE | 6 | -0.037 | 0.014 | 5/6 | 0.094 | 0.219 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | BiGRU (fixed setting) | absorbance RMSE | 6 | -1.181 | 0.223 | 6/6 | 0.031 | 0.031 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | BiGRU (fixed setting) | reflectance RMSE | 6 | -0.109 | 0.025 | 5/6 | 0.062 | 0.219 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | BiGRU (fixed setting) | Peak RMSE | 6 | -0.029 | 0.017 | 3/6 | 0.438 | 1.000 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | BiGRU (Peak-constrained selection) | absorbance RMSE | 6 | -1.527 | 0.154 | 6/6 | 0.031 | 0.031 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | BiGRU (Peak-constrained selection) | reflectance RMSE | 6 | -0.174 | 0.021 | 6/6 | 0.031 | 0.031 |
| External cohort 6명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 10 train (ExternalTest6) | BiGRU (Peak-constrained selection) | Peak RMSE | 6 | 0.000 | 0.028 | 4/6 | 1.000 | 0.688 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | HbI-binned mean | absorbance RMSE | 10 | -2.093 | 1.178 | 6/10 | 0.322 | 0.754 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | HbI-binned mean | reflectance RMSE | 10 | -0.124 | 0.070 | 6/10 | 0.375 | 0.754 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | HbI-binned mean | Peak RMSE | 10 | -0.655 | 0.254 | 7/10 | 0.049 | 0.344 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | Global Linear | absorbance RMSE | 10 | 0.139 | 0.514 | 6/10 | 1.000 | 0.754 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | Global Linear | reflectance RMSE | 10 | 0.030 | 0.067 | 6/10 | 0.770 | 0.754 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | Global Linear | Peak RMSE | 10 | 0.007 | 0.022 | 5/10 | 0.922 | 1.000 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | PLS_v2 | absorbance RMSE | 10 | 0.835 | 0.590 | 5/10 | 0.492 | 1.000 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | PLS_v2 | reflectance RMSE | 10 | 0.076 | 0.062 | 6/10 | 0.846 | 0.754 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | PLS_v2 | Peak RMSE | 10 | 0.153 | 0.077 | 3/10 | 0.064 | 0.344 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | BiGRU | absorbance RMSE | 10 | 0.412 | 0.493 | 4/10 | 0.375 | 0.754 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | BiGRU | reflectance RMSE | 10 | 0.044 | 0.031 | 4/10 | 0.375 | 0.754 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | BiGRU | Peak RMSE | 10 | -0.017 | 0.087 | 7/10 | 0.492 | 0.344 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | PLS_v1 | absorbance RMSE | 10 | 0.168 | 0.516 | 4/10 | 0.695 | 0.754 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | PLS_v1 | reflectance RMSE | 10 | 0.037 | 0.043 | 5/10 | 0.492 | 1.000 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | PLS_v1 | Peak RMSE | 10 | 0.026 | 0.020 | 5/10 | 0.492 | 1.000 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | Ridge | absorbance RMSE | 10 | 0.612 | 0.627 | 6/10 | 0.770 | 0.754 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | Ridge | reflectance RMSE | 10 | 0.063 | 0.064 | 6/10 | 0.770 | 0.754 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | Ridge | Peak RMSE | 10 | 0.046 | 0.025 | 4/10 | 0.232 | 0.754 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | BiGRU (fixed setting) | absorbance RMSE | 10 | 0.067 | 0.430 | 4/10 | 0.770 | 0.754 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | BiGRU (fixed setting) | reflectance RMSE | 10 | 0.004 | 0.035 | 7/10 | 0.922 | 0.344 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | BiGRU (fixed setting) | Peak RMSE | 10 | -0.107 | 0.080 | 8/10 | 0.160 | 0.109 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | BiGRU (Peak-constrained selection) | absorbance RMSE | 10 | 0.139 | 0.514 | 6/10 | 1.000 | 0.754 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | BiGRU (Peak-constrained selection) | reflectance RMSE | 10 | 0.030 | 0.067 | 6/10 | 0.770 | 0.754 |
| Development cohort 10명 | 주 비교 (같은 고정값) | 15 train (Union16) minus 9 train (same fixed hyperparameters) | BiGRU (Peak-constrained selection) | Peak RMSE | 10 | 0.007 | 0.022 | 5/10 | 0.922 | 1.000 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | HbI-binned mean | absorbance RMSE | 10 | -2.093 | 1.178 | 6/10 | 0.322 | 0.754 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | HbI-binned mean | reflectance RMSE | 10 | -0.124 | 0.070 | 6/10 | 0.375 | 0.754 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | HbI-binned mean | Peak RMSE | 10 | -0.655 | 0.254 | 7/10 | 0.049 | 0.344 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | Global Linear | absorbance RMSE | 10 | 0.139 | 0.514 | 6/10 | 1.000 | 0.754 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | Global Linear | reflectance RMSE | 10 | 0.030 | 0.067 | 6/10 | 0.770 | 0.754 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | Global Linear | Peak RMSE | 10 | 0.007 | 0.022 | 5/10 | 0.922 | 1.000 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | PLS_v2 | absorbance RMSE | 10 | 0.205 | 0.619 | 7/10 | 0.557 | 0.344 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | PLS_v2 | reflectance RMSE | 10 | 0.029 | 0.059 | 7/10 | 0.625 | 0.344 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | PLS_v2 | Peak RMSE | 10 | 0.154 | 0.071 | 3/10 | 0.064 | 0.344 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | BiGRU | absorbance RMSE | 10 | 0.252 | 0.517 | 5/10 | 0.846 | 1.000 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | BiGRU | reflectance RMSE | 10 | 0.040 | 0.035 | 4/10 | 0.492 | 0.754 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | BiGRU | Peak RMSE | 10 | 0.093 | 0.053 | 4/10 | 0.193 | 0.754 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | PLS_v1 | absorbance RMSE | 10 | -0.187 | 0.451 | 5/10 | 0.625 | 1.000 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | PLS_v1 | reflectance RMSE | 10 | -0.004 | 0.046 | 6/10 | 0.922 | 0.754 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | PLS_v1 | Peak RMSE | 10 | 0.036 | 0.021 | 4/10 | 0.160 | 0.754 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | Ridge | absorbance RMSE | 10 | 0.016 | 0.780 | 7/10 | 0.625 | 0.344 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | Ridge | reflectance RMSE | 10 | 0.018 | 0.072 | 7/10 | 0.625 | 0.344 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | Ridge | Peak RMSE | 10 | 0.026 | 0.020 | 4/10 | 0.275 | 0.754 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | BiGRU (fixed setting) | absorbance RMSE | 10 | 0.067 | 0.430 | 4/10 | 0.770 | 0.754 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | BiGRU (fixed setting) | reflectance RMSE | 10 | 0.004 | 0.035 | 7/10 | 0.922 | 0.344 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | BiGRU (fixed setting) | Peak RMSE | 10 | -0.107 | 0.080 | 8/10 | 0.160 | 0.109 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | BiGRU (Peak-constrained selection) | absorbance RMSE | 10 | 0.223 | 0.547 | 6/10 | 0.922 | 0.754 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | BiGRU (Peak-constrained selection) | reflectance RMSE | 10 | 0.037 | 0.069 | 6/10 | 0.695 | 0.754 |
| Development cohort 10명 | 참고 행 (설정 차이가 섞임) | 15 train (Union16) minus 9 train (fold-wise tuned) | BiGRU (Peak-constrained selection) | Peak RMSE | 10 | 0.007 | 0.022 | 5/10 | 0.922 | 1.000 |
