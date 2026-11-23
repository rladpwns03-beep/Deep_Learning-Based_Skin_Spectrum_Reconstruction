"""결과 파일에서 문서를 만든다 (숫자를 손으로 옮기지 않기 위함). 판정 문구는 넣지 않는다.

실행 (code 폴더, conda 환경 skin_spectra):
    python -X utf8 scripts/make_docs.py --results                       # docs/Results_Retrain_Tuning.md (Phase 1-4)
    python -X utf8 scripts/make_docs.py --preregistration --commit HASH # docs/Preregistration_ExternalTest_v1.md / .json (없을 때만)
    python -X utf8 scripts/make_docs.py --external                      # docs/Results_ExternalTest_Union16.md (Phase 6-7 뒤)

단위: absorbance RMSE, Peak RMSE, Peak bias, 표준편차 지표, 구간 bias = x10^-3 absorbance / reflectance RMSE = %p.
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))
from final_utils import BANDS, MAIN_MODELS, P_NOTE, RES, SLOPE_BANDS, SPLITS, jdump, jload   # noqa: E402

DOCS = ROOT / "docs"
SCALE = dict(a_rmse=1e3, peak_rmse=1e3, peak_bias=1e3, bias=1e3, peak_sd=1e3, spectral_sd=1e3, r_rmse_pp=1.0)
LABEL = dict(a_rmse="absorbance RMSE", r_rmse_pp="reflectance RMSE", peak_rmse="Peak RMSE", peak_bias="Peak bias",
             bias="bias", peak_sd="Peak 표준편차", spectral_sd="spectral 표준편차")
SUPP = ["PLS_v1", "Ridge", "BiGRU (fixed setting)", "BiGRU (Peak-constrained selection)",
        "PLS_v2 (max 5 components)", "PLS_v1 (max 5 components)"]
OLD_NAME = {"B1": "HbI-binned mean", "B2": "Global Linear", "BiGRU": "BiGRU (fixed setting)", "PLS_v1": "PLS_v1 (성분 수 상한 5)",
            "PLS_v2": "PLS_v2 (성분 수 상한 5)", "Ridge": "Ridge", "PLS_v1_k10": "PLS_v1 (성분 수 상한 10)",
            "PLS_v2_k10": "PLS_v2 (성분 수 상한 10)"}
SETS = {"Development10_LeaveOneSubjectOut": "주 프로토콜", "Development10_Holdout73": "보조 프로토콜"}
DH = ["0.000-0.025", "0.025-0.050", "0.050-0.100", "0.100-0.300"]


def table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def num(v, nd=3):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def pm(mean, sd, metric, nd=3):
    s = SCALE.get(metric, 1.0)
    if mean is None or not np.isfinite(mean):
        return "–"
    return f"{mean * s:.{nd}f} ± {sd * s:.{nd}f}" if sd is not None and np.isfinite(sd) else f"{mean * s:.{nd}f}"


def cell(summ, section, metric, model, with_n=False):
    r = summ[(summ.section == section) & (summ.metric == metric) & (summ.model == model)]
    if r.empty:
        return "–"
    r = r.iloc[0]
    return pm(r["mean"], r["sd"], metric) + (f" (n={int(r['n'])})" if with_n else "")


def wide(summ, section, metrics, models, with_n=False) -> str:
    rows = [[m] + [cell(summ, section, k, m, with_n) for k in metrics] for m in models
            if not summ[(summ.section == section) & (summ.model == m)].empty]
    return table(["모델"] + [LABEL[k] for k in metrics], rows)


def cmp_table(cmp, sections, metrics=None, show_section=False) -> str:
    c = cmp[cmp.section.isin(sections)]
    if metrics:
        c = c[c.metric.isin(metrics)]
    rows = []
    for r in c.itertuples():
        s = SCALE.get(r.metric, 1.0)
        rows.append(([r.section] if show_section else []) + [
            r.comparison, LABEL.get(r.metric, r.metric), int(r.n), num(r.diff_mean * s), num(r.diff_sd * s), num(r.diff_se * s),
            f"{int(r.n_improved)}/{int(r.n)}", num(r.wilcoxon_p), num(r.sign_p)])
    return table((["구간"] if show_section else []) + ["비교", "지표", "n", "차이 평균", "차이 표준편차", "표준오차", "개선 k/n",
                                                      "Wilcoxon p", "부호검정 p"], rows)


def read(p):
    return pd.read_csv(p, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
def phase1() -> str:
    out = ["## 1. Phase 1: 고정 설정 재현\n",
           "General reconstruction. 단위: absorbance RMSE ×10⁻³ / reflectance RMSE %p / Peak RMSE ×10⁻³. "
           "`scripts/pls_compare.py`의 `results/compare/*/comparison_table.csv` 값이며, 2026-11-23 참고 수치와 표기 자릿수까지 모두 일치했다 "
           "(BiGRU (fixed setting)의 LeaveOneSubjectOut 값은 참고 수치가 없던 신규 값).\n"]
    t = {sc: read(RES / "compare" / sc / "comparison_table.csv") for sc in ("holdout", "loso")}
    rows = []
    for k in ["B1", "B2", "PLS_v1", "PLS_v1_k10", "PLS_v2", "PLS_v2_k10", "Ridge", "BiGRU"]:
        r = [OLD_NAME[k]]
        for sc in ("holdout", "loso"):
            x = t[sc][(t[sc].section == "general") & (t[sc].model == k)].iloc[0]
            r.append(f"{x.a_rmse:.3f} / {x.r_rmse_pp:.3f} / {x.peak_rmse:.3f}")
        rows.append(r)
    out.append(table(["모델", "Development10_Holdout73", "Development10_LeaveOneSubjectOut"], rows))
    out.append("Raw(보정 없음)는 정답이 다른 측정이므로 General reconstruction 표에 넣지 않고 Fixed-target repeatability에서만 보고한다.\n")
    out.append("선택된 하이퍼파라미터 (분할 순서대로)\n")
    rows = []
    for k, lab in (("pls_v2_k10", "PLS_v2 (상한 10)"), ("pls_v1_k10", "PLS_v1 (상한 10)"), ("pls_v2", "PLS_v2 (상한 5)"),
                   ("pls_v1", "PLS_v1 (상한 5)"), ("ridge", "Ridge alpha")):
        for sc, sn in (("loso", "LeaveOneSubjectOut"), ("holdout", "Holdout73")):
            s = jload(RES / k / sc / "selection.json")["splits"]
            ch = [x["chosen"] for x in s]
            top = max(s[0]["grid"]) if k != "ridge" else None
            hit = sum(c == top for c in ch) if top else sum(bool(x.get("at_boundary")) for x in s)
            rows.append([lab, sn, ", ".join(f"{c:g}" for c in ch), hit])
    out.append(table(["모델", "분할 세트", "선택값", "상한(격자 끝) 도달 횟수"], rows))
    t06 = jload(RES / "holdout/metrics/target06_reconstruction.json")["fixed_target_consistency"]["summary"]
    out.append("Fixed-target repeatability (Development10_Holdout73, ×10⁻³): " + "; ".join(
        f"{OLD_NAME.get(m, m)} Peak 표준편차 {t06['peak_sd'][m]['mean'] * 1e3:.3f}, spectral 표준편차 {t06['spectral_sd'][m]['mean'] * 1e3:.2f}"
        for m in ("Raw", "B1", "B2", "BiGRU")) + ".\n")
    return "\n".join(out)


def phase2() -> str:
    out = ["## 2. Phase 2: 데이터와 분할\n"]
    rows = []
    for d in sorted(SPLITS.iterdir()):
        m = jload(d / "manifest.json")
        rows.append([d.name, m["n_splits"], f"{m['rows_per_split']['train']} / {m['rows_per_split']['test']}",
                     f"{m['pairs_per_split']['train']} / {m['pairs_per_split']['test']}", "통과" if m["all_checks_passed"] else "실패"])
    out.append(table(["분할 세트", "분할 수", "행 (train / test)", "쌍 (train / test)", "검사"], rows))
    for name in ("Development10_Holdout73", "Union16_Holdout73"):
        p, m = read(SPLITS / name / "split_plan.csv"), jload(SPLITS / name / "manifest.json")
        out.append(f"**{name}** 분할별 Near-target subject, In-range subject (명 수와 비율 %)\n")
        out.append(table(["분할", "test", "test Near-target", "train Near-target", "test In-range", "train In-range"],
                         [[r.split_id, r.test, f"{r.test_n_near_target} ({r.test_pct_near_target})",
                           f"{r.train_n_near_target} ({r.train_pct_near_target})", f"{r.test_n_in_range} ({r.test_pct_in_range})",
                           f"{r.train_n_in_range} ({r.train_pct_in_range})"] for r in p.itertuples()]))
        out.append("피험자별 테스트 횟수: " + ", ".join(f"{s} {v}" for s, v in m["test_count_per_subject"].items()) + "\n")
        out.append("\n".join(f"- {n}" for n in m["notes"]) + "\n")
    return "\n".join(out)


def chosen_text(c):
    if c["epoch"] == 0:
        return ["epoch 0 (Global Linear와 동일)", "–", "0"]
    return [c["hidden"], f"{c['slope_l2']:g}", c["epoch"]]


def boundary_text(c):
    if c["epoch"] == 0:
        return "–"
    b = [k for k, v in c["at_grid_boundary"].items() if v]
    return ", ".join(b) if b else "없음"


def phase3() -> str:
    out = ["## 3. Phase 3: 재학습과 하이퍼파라미터 튜닝 (Development cohort만 사용)\n",
           "BiGRU 격자: hidden {4, 8, 16} × slope_l2 {0, 1e-3, 1e-2, 1e-1} × epoch {0, 25, 50, 100, 150, 200, 300} = 84 후보. "
           "선택 기준 = 학습 피험자 안 Leave-One-Subject-Out 내부 검증 쌍 전체의 absorbance RMSE (570 nm 제외), 선택용 seed 0. "
           "검증 RMSE 단위 ×10⁻³.\n"]
    total = 0.0
    for unit in ("Development10_LeaveOneSubjectOut", "Development10_Holdout73", "Development10_AllSubjects"):
        s = jload(RES / "bigru_tuned" / unit / "selection.json")
        total += s["tuning_seconds"]
        rows = []
        for r in s["splits"]:
            c, pc = r["chosen"], r["peak_constrained_chosen"]
            rows.append([r["split_id"], " ".join(r["test"]) or "–", *chosen_text(c), num(c["val_a_rmse"] * 1e3),
                         num(r["global_linear_validation"]["val_a_rmse"] * 1e3), boundary_text(c),
                         " / ".join(str(x) for x in chosen_text(pc)) + f" (통과 후보 {pc['n_candidates_passing']}/84)"])
        out.append(f"**{unit}**\n")
        out.append(table(["분할", "test", "hidden", "slope_l2", "epoch", "검증 RMSE", "Global Linear 검증 RMSE", "격자 경계",
                          "Peak-constrained selection (hidden / slope_l2 / epoch)"], rows))
        ch = [r["chosen"] for r in s["splits"]]
        nb = sum(any(c["at_grid_boundary"].values()) and c["epoch"] != 0 for c in ch)
        out.append(f"격자 경계 도달 {nb}/{len(ch)}회 (hidden 최소 {sum(c['at_grid_boundary']['hidden_min'] and c['epoch'] != 0 for c in ch)}, "
                   f"hidden 최대 {sum(c['at_grid_boundary']['hidden_max'] and c['epoch'] != 0 for c in ch)}, "
                   f"slope_l2 최소(0) {sum(c['at_grid_boundary']['slope_l2_min'] and c['epoch'] != 0 for c in ch)}, "
                   f"slope_l2 최대 {sum(c['at_grid_boundary']['slope_l2_max'] and c['epoch'] != 0 for c in ch)}, "
                   f"epoch 최대(300) {sum(c['at_grid_boundary']['epoch_max'] for c in ch)}), "
                   f"epoch 0 선택 {sum(c['epoch'] == 0 for c in ch)}/{len(ch)}회. "
                   "모든 분할에서 epoch 0 후보 12개의 검증 값이 서로 같고 Global Linear와 같음을 확인했다.\n")
    a = jload(RES / "final_comparison" / "Development10_AllSubjects_selection.json")
    out.append("**PLS 계열과 Ridge** — `scripts/pls_compare.py`의 `pls_v2_k10`, `pls_v1_k10`, `ridge` 결과(Phase 1에서 생성)를 그대로 재사용했다 "
               "(선택값은 1장의 표). ExternalTest6용 Development cohort 10명 전체 선택: "
               f"PLS_v2 성분 수 {a['PLS_v2']['n_components']}, PLS_v1 성분 수 {a['PLS_v1']['n_components']}, Ridge alpha {a['Ridge']['alpha']:g} "
               f"(기대 6, 6과 일치: {a['as_expected']}).\n")
    out.append(f"BiGRU 튜닝 총 실행 시간: {total / 60:.1f}분 (10개 프로세스 병렬, 프로세스당 CPU 1 thread; 300 epoch 학습 2,040회). "
               "튜닝 루프의 epoch별 학습 이력과 최종 가중치는 `BiGRUModel.fit`과 완전 일치했고, 병렬 실행 결과는 단일 실행과 완전 일치했다.\n")
    return "\n".join(out)


def phase4_main(split_set, role) -> str:
    d = RES / "final_comparison" / split_set
    summ, cm, cs = read(d / "summary.csv"), read(d / "comparisons_main.csv"), read(d / "comparisons_supplementary.csv")
    cd, meta = read(d / "comparisons_delta_hbi.csv"), jload(d / "metrics.json")
    gm = ["a_rmse", "r_rmse_pp", "peak_rmse", "peak_bias"]
    o = [f"### {split_set} ({role})\n", f"집계: {meta['aggregation']}\n\n쌍체 비교: {meta['paired_comparison']}\n",
         "**General reconstruction** (Peak bias는 부호 있는 값, 예측 − 정답)\n", wide(summ, "general", gm, MAIN_MODELS),
         "주의: General reconstruction은 피험자 안의 모든 순서쌍을 쓰므로, 보정이 i → j와 j → i에서 정확히 반대 부호인 HbI-binned mean과 "
         "Global Linear는 Peak bias가 구조적으로 0이다 (오차가 쌍끼리 상쇄). 따라서 이 두 모델을 뒤 모델로 둔 Peak bias 비교의 k/n과 p 값은 "
         "정보가 없고, 앞 모델의 부호 있는 bias 값 자체만 참고한다.\n",
         "사전 지정 비교 (General reconstruction)\n", cmp_table(cm, ["general"]),
         "보조 표: Supplementary 모델 (General reconstruction)\n", wide(summ, "general", gm, SUPP), cmp_table(cs, ["general"]),
         f"**Near-target accuracy, 허용 폭 0.02** (대상 피험자 {' '.join(meta['near_target']['tol_0p02_subjects'])}; 기술 통계만)\n",
         wide(summ, "near_0.02", ["a_rmse", "peak_rmse"], MAIN_MODELS + SUPP, True),
         f"**Near-target accuracy, 허용 폭 0.03** (대상 피험자 {' '.join(meta['near_target']['tol_0p03_subjects'])}; "
         f"예상 8명과 일치: {meta['near_target']['tol_0p03_as_expected']})\n",
         wide(summ, "near_0.03", ["a_rmse", "peak_rmse"], MAIN_MODELS + SUPP, True), cmp_table(cm, ["near_0.03"]),
         "**Fixed-target repeatability** (테스트 측정 전부를 HbI 0.6으로 복원했을 때의 개인 내 표준편차; n = 집계 단위 수)\n"]
    for sec, lab in (("repeatability", "전체"), ("repeatability_in_range", "In-range subject"), ("repeatability_other", "그 밖의 피험자")):
        o += [f"{lab}\n", wide(summ, sec, ["peak_sd", "spectral_sd"], ["Raw"] + MAIN_MODELS + SUPP, True)]
    if split_set.endswith("Holdout73"):
        o.append("In-range subject 행은 test에 In-range subject가 있는 분할만 평균했다 (분할 3과 분할 5에는 없음; n이 그 분할 수).\n")
    o += ["사전 지정 비교 (Fixed-target repeatability, 전체 피험자)\n", cmp_table(cm, ["repeatability"]),
          "**HbI 변화 크기 구간별 absorbance RMSE** (|ΔHbI| 구간, 방향 전체; n = 그 구간에 쌍이 있는 집계 단위 수. "
          "해당 구간에 쌍이 없는 피험자는 그 구간의 n에서 빠진다)\n"]
    for direction, lab in (("all", "전체"), ("increase", "증가 (ΔHbI > 0)"), ("decrease", "감소 (ΔHbI < 0)")):
        rows = [[m] + [cell(summ, f"dh|{b}|{direction}", "a_rmse", m, True) for b in DH] for m in MAIN_MODELS + SUPP[:4]]
        o += [f"방향: {lab}\n", table(["모델"] + DH, rows)]
    rows = [[m] + [cell(summ, f"dh|{b}|all", "peak_rmse", m, True) for b in DH] for m in MAIN_MODELS + SUPP[:4]]
    o += ["Peak RMSE, 방향 전체\n", table(["모델"] + DH, rows),
          "구간별 쌍체 차이 (방향 전체; 증가·감소 방향은 `comparisons_delta_hbi.csv`)\n",
          cmp_table(cd, [f"dh|{b}|all" for b in DH], ["a_rmse", "peak_rmse"], True)]
    return "\n".join(o)


def phase4_bands(split_set) -> str:
    d = RES / "band_analysis" / split_set
    summ, cmp_ = read(d / "band_error_summary.csv"), read(d / "band_error_comparisons.csv")
    models = MAIN_MODELS + SUPP[:3]
    o = [f"### {split_set}\n"]
    for met in ("a_rmse", "bias"):
        rows = [[m] + [cell(summ, f"band|{b}", met, m) for b in BANDS] for m in models]
        o += [f"구간 {LABEL[met] if met == 'a_rmse' else 'bias (예측 − 정답)'} ×10⁻³\n",
              table(["모델"] + [f"{b} ({len(w)})" for b, w in BANDS.items()], rows)]
    o += ["570 nm: h_in → h* 고정, 오차 0 (모든 모델).\n", "구간별 쌍체 차이\n",
          cmp_table(cmp_, [f"band|{b}" for b in BANDS], None, True)]
    return "\n".join(o)


def phase4_shap() -> str:
    d = RES / "shap_bands" / "Development10_LeaveOneSubjectOut"
    s, j = read(d / "shap_summary.csv"), jload(d / "shap_summary.json")
    ob, co = read(d / "shap_input_group_by_output_band.csv"), read(d / "shap_hin_vs_spectral_correlation.csv")
    groups = list(j["groups"])
    o = ["모델 × 입력 그룹의 평균 |SHAP| ×10⁻³ (스펙트럼 변화량 30파장 평균 / Peak 변화량). 900쌍, BiGRU는 seed 3개 각각 계산 후 평균.\n"]
    rows = []
    for m in j["models"]:
        x = s[s.model_base == m].set_index("group")
        rows.append([m] + [f"{x.loc[g, 'mean_abs_spectral'] * 1e3:.3f} / {x.loc[g, 'abs_peak'] * 1e3:.3f}" for g in groups])
    o.append(table(["모델"] + groups, rows))
    o.append("입력 그룹 × 출력 구간의 평균 |SHAP| ×10⁻³ (Proposal 모델과 Global Linear; 전체 모델은 `shap_input_group_by_output_band.csv`)\n")
    for m in ("Global Linear", "PLS_v2", "BiGRU"):
        x = ob[ob.model_base == m].set_index("group")
        o += [f"{m}\n", table(["입력 그룹"] + [f"출력 {b}" for b in BANDS],
                              [[g] + [f"{x.loc[g, f'out|{b}'] * 1e3:.3f}" for b in BANDS] for g in groups])]
    o.append("h_in 그룹 기여와 스펙트럼 그룹(5개 합) 기여의 Pearson 상관 (쌍 × 출력 파장 전체 / Peak)\n")
    o.append(table(["모델", "스펙트럼 변화량", "Peak 변화량"], [[r.model, num(r.spectral), num(r.peak)] for r in co.itertuples()]))
    o.append(f"해석 주의: {j['interpretation_caution']} 그룹을 서로 다른 측정에서 섞으면 실제로 없는 스펙트럼이 생길 수 있다 "
             "(interventional SHAP의 한계).\n")
    ck = j["checks"]
    o.append(f"검사: 입력 그룹 7개가 특징 32개를 한 번씩 덮음, 조합 {j['n_coalitions']}개 전부 계산, 효율성 최대 오차 {ck['max_efficiency_error']:.2e}, "
             f"기존 6개 그룹을 넣었을 때 `shap_analysis.shapley`와 일치 {ck['old_groups_match_shap_analysis'][0]['old_groups_match_shap_analysis']} "
             f"(최대 차이 {ck['old_groups_match_shap_analysis'][0]['max_abs_diff']:.1e}), Ridge 재학습 계수 = 저장 계수 {ck['ridge_refit_matches_saved']}.\n")
    return "\n".join(o)


def slope_table(path) -> str:
    s = read(path)
    rows = []
    for r in s.itertuples():
        rows.append([r.item, r.model, int(r.n_subjects), f"{r.mean:.4f} ± {r.sd:.4f}", num(r.rms_magnitude_mean, 4),
                     num(r.pearson_with_personal_reference), "–" if pd.isna(r.n_sign_agree) else f"{int(r.n_sign_agree)}/{int(r.n_subjects)}",
                     num(r.global_linear_slope_mean, 4)])
    return table(["구간", "모델", "n", "구간 평균 잔차 기울기 (평균 ± 표준편차)", "구간 크기(RMS) 평균", "개인 기준과의 Pearson 상관",
                  "부호 일치", "Global Linear 기울기 구간 평균"], rows)


def phase4_slopes() -> str:
    d = RES / "band_analysis" / "Development10_LeaveOneSubjectOut"
    c = jload(d / "residual_slope_config.json")
    o = [f"정의: {c['definition']} 개인 기준: {c['personal_reference']}. 단위: absorbance / HbI (무차원 기울기).\n",
         slope_table(d / "residual_slope_summary.csv"),
         "'Personal reference' 행은 개인 기준 잔차 기울기 d_s 자체의 구간 평균이다. 'Peak direction' = r(490) − (r(470) + r(510))/2 는 4구간에 포함하지 않은 별도 항목이다.\n",
         "제외한 모델: " + "; ".join(f"{k} — {v}" for k, v in c["excluded_models"].items()) + ".\n",
         f"검사: Δh = 0 제외 순서쌍 {c['n_pairs_excluded_dh0']}개, BiGRU 잔차 기울기와 `BiGRUModel.residual`의 최대 차이 "
         f"{c['checks']['bigru_vs_residual_max_abs']:.1e} (기준 1e-5), Δh를 바꿨을 때 PLS_v2 최대 차이 "
         f"{c['checks']['PLS_v2_dh_invariance_max_abs']:.1e}, Ridge {c['checks']['Ridge_dh_invariance_max_abs']:.1e}.\n"]
    return "\n".join(o)


def phase4_s006() -> str:
    d = RES / "sensitivity_exclude_S006"
    return "\n".join(["S006이 test가 아닌 9개 fold에서 학습 피험자 8명(S006 제외)으로 다시 학습. 하이퍼파라미터는 해당 fold의 주 결과 선택값으로 고정.\n",
                      cmp_table(read(d / "comparisons.csv"), ["general"])])


def results_doc():
    parts = ["# 재학습·하이퍼파라미터 튜닝·구간별 분석 결과 (Phase 1–4, Development cohort만 사용)\n",
             "이 문서는 `scripts/make_docs.py --results`가 결과 파일에서 생성한다. External cohort(S101–S106)는 전처리와 분할 파일 생성 외에는 "
             "어떤 모델의 학습, 하이퍼파라미터 선택, 평가에도 쓰이지 않았다. 판정 문구는 넣지 않았다.\n",
             "단위: absorbance RMSE, Peak RMSE, Peak bias, 표준편차 지표, 구간 bias는 ×10⁻³ absorbance, reflectance RMSE는 %p. "
             "absorbance RMSE와 reflectance RMSE는 570 nm 제외 30파장. BiGRU의 피험자 값은 seed 0, 1, 2 지표의 평균. " + P_NOTE + "\n",
             "모델 명칭: HbI-binned mean = 코드의 B1, Global Linear = B2, PLS_v2 = `pls_v2_k10`, PLS_v1 = `pls_v1_k10`, Ridge = `ridge`, "
             "BiGRU = 튜닝한 BiGRU (`results/bigru_tuned`), BiGRU (fixed setting) = hidden 8, slope_l2 1e-3, 150 epoch.\n",
             phase1(), phase2(), phase3(), "## 4. Phase 4: 평가와 분석\n", "### 4.1 주 비교\n"]
    parts += [phase4_main(s, r) for s, r in SETS.items()]
    parts += ["### 4.2 파장 구간별 복원 오차\n", "구간: " + "; ".join(f"{b} = {w[0]}–{w[-1]} nm ({len(w)}파장)" for b, w in BANDS.items()) +
              ". Q-band는 570 nm 제외. 5개 구간의 파장 수 합 = 30. 5개 구간의 오차 제곱합으로 다시 계산한 전체 absorbance RMSE가 "
              "`common/metrics.py`의 값과 같음을 모든 피험자·모델·seed에서 확인했다 (허용 1e-12).\n"]
    parts += [phase4_bands(s) for s in SETS]
    parts += ["### 4.3 구간 단위 SHAP (Development10_LeaveOneSubjectOut)\n", phase4_shap(),
              "### 4.4 잔차 기울기의 파장 4구간 집계 (Development10_LeaveOneSubjectOut)\n", phase4_slopes(),
              "### 4.5 HbI 변화 크기 구간별 오차\n", "4.1의 분할 세트별 표에 포함했다.\n",
              "### 4.6 Peak-constrained selection (BiGRU, 보조 분석)\n",
              "선택값은 3장의 표, 지표는 4.1의 보조 표 'BiGRU (Peak-constrained selection)' 행. 주 결과를 대체하지 않는다.\n",
              "### 4.7 S006 exclusion sensitivity (보조 분석)\n", phase4_s006()]
    extra = DOCS / "Results_Notes.md"
    if extra.exists():
        parts.append(extra.read_text(encoding="utf-8"))
    (DOCS / "Results_Retrain_Tuning.md").write_text("\n".join(parts), encoding="utf-8")
    print("작성:", DOCS / "Results_Retrain_Tuning.md")


# ---------------------------------------------------------------------------
def preregistration(commit):
    jpath, mpath = DOCS / "Preregistration_ExternalTest_v1.json", DOCS / "Preregistration_ExternalTest_v1.md"
    if jpath.exists() or mpath.exists():
        sys.exit("[중단] 동결 문서가 이미 있음. 덮어쓰지 않는다.")
    import torch
    import matplotlib
    import openpyxl
    sel = jload(RES / "bigru_tuned" / "Development10_AllSubjects" / "selection.json")["splits"][0]
    a = jload(RES / "final_comparison" / "Development10_AllSubjects_selection.json")
    keep = lambda c: {k: c[k] for k in ("hidden", "slope_l2", "epoch")}                     # noqa: E731
    hp = dict(pls_v2_components=a["PLS_v2"]["n_components"], pls_v1_components=a["PLS_v1"]["n_components"],
              ridge_alpha=a["Ridge"]["alpha"], bigru=keep(sel["chosen"]), bigru_peak_constrained=keep(sel["peak_constrained_chosen"]),
              bigru_fixed_setting=dict(hidden=8, slope_l2=1e-3, epoch=150), bigru_common=dict(lr=3e-3, clip=1.0, seeds=[0, 1, 2]),
              hbi_binned_mean=dict(bin_width=0.05, edges="0.40-0.75"))
    manifests = {d.name: hashlib.sha256((d / "manifest.json").read_bytes()).hexdigest() for d in sorted(SPLITS.iterdir())}
    env = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
               matplotlib=matplotlib.__version__, openpyxl=openpyxl.__version__, platform=platform.platform())
    jdump(jpath, dict(version="v1", frozen_at_code_commit=commit, hyperparameters=hp, split_manifest_sha256=manifests, environment=env,
                      validation=dict(bigru=dict(val_a_rmse=sel["chosen"]["val_a_rmse"], val_peak_rmse=sel["chosen"]["val_peak_rmse"],
                                                 global_linear=sel["global_linear_validation"]))))
    c, pc = sel["chosen"], sel["peak_constrained_chosen"]
    lo = RES / "final_comparison" / "Development10_LeaveOneSubjectOut"
    summ, cm = read(lo / "summary.csv"), read(lo / "comparisons_main.csv")
    bs = lambda x: "epoch 0 (Global Linear와 동일)" if x["epoch"] == 0 else f"hidden {x['hidden']}, slope_l2 {x['slope_l2']:g}, epoch {x['epoch']}"   # noqa: E731
    md = [
        "# ExternalTest6 동결 문서 (Preregistration v1)\n",
        "이 문서는 External cohort(S101–S106)를 어떤 모델에도 넣기 전에 작성했다. 아래 설정, 지표, 비교, 구간 정의는 ExternalTest6 결과를 본 뒤 바꾸지 않는다. "
        "기계가 읽는 값은 `docs/Preregistration_ExternalTest_v1.json`에 있고 `scripts/external_test.py`와 `scripts/run_union16.py`는 그 파일만 읽는다.\n",
        "## 1. 동결한 설정 (Development10_AllSubjects에서 선택)\n",
        table(["모델", "설정", "선택 방법"], [
            ["HbI-binned mean", "bin 폭 0.05, 경계 0.40–0.75", "고정 (튜닝 없음)"],
            ["Global Linear", "–", "튜닝 없음"],
            ["PLS_v2", f"성분 수 {hp['pls_v2_components']}", "Development cohort 10명 안 Leave-One-Subject-Out, 상한 10"],
            ["BiGRU", bs(c) + ", lr 3e-3, clip 1.0, seed 0/1/2",
             f"Development cohort 10명 안 Leave-One-Subject-Out, 84 후보 (검증 absorbance RMSE {c['val_a_rmse'] * 1e3:.3f}×10⁻³, "
             f"Global Linear {sel['global_linear_validation']['val_a_rmse'] * 1e3:.3f}×10⁻³)"],
            ["PLS_v1", f"성분 수 {hp['pls_v1_components']}", "같은 내부 절차, 상한 10"],
            ["Ridge", f"alpha {hp['ridge_alpha']:g}", "같은 내부 절차 (격자 1e-4…1e4, 끝이면 확장)"],
            ["BiGRU (fixed setting)", "hidden 8, slope_l2 1e-3, epoch 150", "`common/config.py` 고정값"],
            ["BiGRU (Peak-constrained selection)", bs(pc) + ", lr 3e-3, clip 1.0, seed 0/1/2",
             f"내부 검증 Peak RMSE ≤ epoch 0의 값인 후보 {pc['n_candidates_passing']}/84개 중 검증 absorbance RMSE 최소"]]),
        "## 2. 코드, 분할, 환경\n",
        f"- 코드 커밋 (Phase 4 커밋): `{commit}` (브랜치 `experiment/retrain-and-tuning-2026-11`)\n"
        "- `data/splits_final/*/manifest.json`의 SHA-256\n",
        table(["분할 세트", "manifest.json SHA-256"], [[k, f"`{v}`"] for k, v in manifests.items()]),
        "- 환경: " + ", ".join(f"{k} {v}" for k, v in env.items()) + ". 전체 목록은 `docs/Environment.md`.\n",
        "## 3. ExternalTest6에서 계산할 것 (Phase 6 정의 그대로)\n",
        "- 학습: Development cohort 10명 전체. 하이퍼파라미터는 1장의 값으로 고정. BiGRU seed 0, 1, 2 (피험자 값 = seed별 지표의 평균).\n"
        "- 평가: S101–S106, `data/splits_final/ExternalTest6`의 테스트 쌍 (피험자당 90쌍, 합계 540쌍). 1회만 실행하고, 평가 후 설정을 바꿔 다시 실행하지 않는다.\n"
        "- 주 표와 사전 지정 비교: Raw, HbI-binned mean, Global Linear, PLS_v2, BiGRU. 보조 표: PLS_v1, Ridge, BiGRU (fixed setting), "
        "BiGRU (Peak-constrained selection).\n"
        "- 실험과 지표\n"
        "  - General reconstruction: absorbance RMSE (주 지표, 570 nm 제외 30파장), reflectance RMSE, Peak RMSE (제약 지표), Peak bias.\n"
        "  - Fixed-target repeatability: 개인 내 Peak 표준편차, spectral 표준편차. In-range subject(S101, S106)와 그 밖의 피험자를 분리해 보고.\n"
        "  - Near-target accuracy: 0.6 ± 0.02 실측이 S101과 S105에 1개씩뿐이므로 기술 통계로만 보고.\n"
        "- 사전 지정 비교 (피험자 6명 쌍체 차이; 차이 = 앞 모델 − 뒤 모델, 개선 = 차이 < 0, Peak bias는 |bias| 감소; 평균, 표준편차, 표준오차, k/n, "
        "정확 Wilcoxon p, 부호검정 p): PLS_v2 minus Global Linear, BiGRU minus Global Linear, PLS_v2 minus HbI-binned mean, "
        "BiGRU minus HbI-binned mean, BiGRU minus PLS_v2, BiGRU minus BiGRU (fixed setting). "
        "p 값은 다중 비교 보정이 없는 기술적 참고값이며 n = 6에서 부호검정 양측 p의 최솟값은 0.031이다. 판정 문구는 넣지 않는다.\n"
        "- 구간별 분석 (Development cohort 분석과 같은 정의, 같은 함수)\n"
        "  - 파장 구간별 복원 오차: " + "; ".join(f"{b} = {', '.join(map(str, w))} nm" for b, w in BANDS.items()) + ". 570 nm는 h_in → h* 고정, 오차 0.\n"
        "  - 잔차 기울기 4구간 (" + ", ".join(SLOPE_BANDS) + "): r_model(λ) = (A*_model(λ) − A*_GlobalLinear(λ)) / Δh, Δh = 0인 쌍 제외, "
        "대상 PLS_v2, BiGRU, Ridge. 개인 기준 d_s = b_s − b_GlobalLinear. Peak 방향 r(490) − (r(470) + r(510))/2 는 별도 항목.\n"
        "  - HbI 변화 크기 구간: |ΔHbI| 0–0.025, 0.025–0.05, 0.05–0.10, 0.10–0.30 × 방향(증가, 감소, 전체).\n"
        "- 추가 기준값: 피험자별 10회 측정의 Peak 대 HbI 선형회귀로 얻은 HbI 0.6에서의 값과 그 표준오차. 모델로 0.6 복원한 10개 스펙트럼의 Peak 평균과의 "
        "차이를 Peak_RS 단위로 보고 (Peak_RS = 8387.2 × Peak + 83.495, Hwang et al. 2023, Sensors 23(17):7654 식 (2)). "
        "이 기준값은 추정값이며 독립 정답이 아니다. In-range subject가 아닌 4명은 외삽임을 표시한다.\n",
        "## 4. Union16 분할 (Phase 7, 보조)\n",
        "- `Union16_LeaveOneSubjectOut`, `Union16_Holdout73`에서 모든 모델을 1장의 고정값으로 다시 학습한다 (PLS는 성분 수를 명시 전달, 다시 고르지 않음).\n"
        "- 규칙 3의 의도된 예외: 고정값은 Development cohort 10명 전체로 고른 값이므로 Development cohort 피험자가 test인 분할에서는 그 피험자가 "
        "하이퍼파라미터 선택에 쓰였다. External cohort 피험자가 test인 분할에는 이 문제가 없다. Union16은 보조 결과다.\n"
        "- 학습 피험자 수 효과의 주 비교는 같은 고정값끼리 한다: External cohort 6명은 10명 학습(ExternalTest6) 대 15명 학습, Development cohort 10명은 "
        "9명 학습(같은 고정값으로 다시 학습) 대 15명 학습. fold별 튜닝값으로 학습한 기존 9명 학습 결과는 참고 행으로 둔다. "
        "이 결과는 ExternalTest6과 독립이 아니므로 데이터 증가 효과의 추정으로만 표기한다.\n",
        "## 5. Phase 3과 Phase 4 결과 요약 (Development10_LeaveOneSubjectOut, 피험자 10명 평균 ± 표준편차)\n",
        "단위: absorbance RMSE, Peak RMSE, Peak bias ×10⁻³, reflectance RMSE %p. 전체 표는 `docs/Results_Retrain_Tuning.md`.\n",
        wide(summ, "general", ["a_rmse", "r_rmse_pp", "peak_rmse", "peak_bias"], MAIN_MODELS + SUPP[:4]),
        cmp_table(cm, ["general"], ["a_rmse", "peak_rmse"]),
        "BiGRU 선택값 요약 (전체는 `docs/Results_Retrain_Tuning.md` 3장): Development10_AllSubjects = " + bs(c) + ".\n",
        "## 6. 명세 보충 사항 (2026-11-23 사용자 답변, 그대로 기록)\n",
        (DOCS / "Spec_Supplement_2026-11-23.md").read_text(encoding="utf-8")]
    mpath.write_text("\n".join(md), encoding="utf-8")
    print("작성:", mpath, jpath)


def external_doc():
    d, u = RES / "external_test", RES / "union16"
    cfg = jload(d / "config.json")
    summ, cm, cs = read(d / "summary.csv"), read(d / "comparisons_main.csv"), read(d / "comparisons_supplementary.csv")
    cb, cd, pr = read(d / "comparisons_band.csv"), read(d / "comparisons_delta_hbi.csv"), read(d / "peak_rs_reference.csv")
    gm = ["a_rmse", "r_rmse_pp", "peak_rmse", "peak_bias"]
    supp, hp = SUPP[:4], cfg["hyperparameters"]
    bias_note = ("주의 (동결 문서 v1.1 보충): 뒤 모델이 Global Linear 또는 HbI-binned mean인 쌍체 비교에서 Peak bias와 구간 bias 행은 해석하지 않는다 "
                 "(두 모델의 bias가 구조적으로 0). BiGRU (Peak-constrained selection)는 epoch 0이 선택되어 Global Linear와 같은 값이다.\n")
    o = ["# ExternalTest6과 Union16 결과 (Phase 6–7)\n",
         "이 문서는 `scripts/make_docs.py --external`이 결과 파일에서 생성한다. 판정 문구는 넣지 않았다. "
         "단위: absorbance RMSE, Peak RMSE, Peak bias, 표준편차 지표, 구간 bias는 ×10⁻³ absorbance, reflectance RMSE는 %p. " + P_NOTE + "\n",
         "## 1. ExternalTest6 (학습 = Development cohort 10명 전체, 평가 = External cohort S101–S106, 1회 실행)\n",
         f"- 동결 문서: `{cfg['preregistration']['file']}` (SHA-256 `{cfg['preregistration']['sha256']}`), 태그 `preregistration-externaltest-v1.1`. "
         f"실행 시점의 코드 커밋 `{cfg['code_commit']}`.\n"
         f"- 고정 설정: PLS_v2 성분 수 {hp['pls_v2_components']}, PLS_v1 성분 수 {hp['pls_v1_components']}, Ridge alpha {hp['ridge_alpha']:g}, "
         f"BiGRU hidden {hp['bigru']['hidden']}, slope_l2 {hp['bigru']['slope_l2']:g}, epoch {hp['bigru']['epoch']} (seed 0, 1, 2). 실행 후 설정을 바꿔 다시 실행하지 않았다.\n"
         f"- 쌍체 비교: {cfg['paired_comparison']}\n", bias_note,
         "### 1.1 General reconstruction (피험자당 90쌍, 피험자 6명 평균 ± 표준편차)\n", wide(summ, "general", gm, MAIN_MODELS),
         "사전 지정 비교\n", cmp_table(cm, ["general"]), "보조 표\n", wide(summ, "general", gm, supp), cmp_table(cs, ["general"]),
         "### 1.2 Fixed-target repeatability (In-range subject = S101, S106)\n"]
    for sec, lab in (("repeatability", "전체 (6명)"), ("repeatability_in_range", "In-range subject (S101, S106)"), ("repeatability_other", "그 밖의 피험자 (4명)")):
        o += [f"{lab}\n", wide(summ, sec, ["peak_sd", "spectral_sd"], ["Raw"] + MAIN_MODELS + supp, True)]
    o += ["사전 지정 비교 (전체 6명)\n", cmp_table(cm, ["repeatability"]),
          f"### 1.3 Near-target accuracy, 허용 폭 0.02 (기술 통계만)\n", f"{cfg['near_target']}. n = 대상 피험자 수.\n",
          wide(summ, "near_0.02", ["a_rmse", "peak_rmse"], MAIN_MODELS + supp, True),
          "### 1.4 파장 구간별 복원 오차\n"]
    for met in ("a_rmse", "bias"):
        rows = [[m] + [cell(summ, f"band|{b}", met, m) for b in BANDS] for m in MAIN_MODELS + supp[:3]]
        o += [f"구간 {'absorbance RMSE' if met == 'a_rmse' else 'bias (예측 − 정답)'} ×10⁻³\n",
              table(["모델"] + [f"{b} ({len(w)})" for b, w in BANDS.items()], rows)]
    o += ["570 nm: h_in → h* 고정, 오차 0 (모든 모델).\n", cmp_table(cb, [f"band|{b}" for b in BANDS], None, True),
          "### 1.5 잔차 기울기의 파장 4구간 집계\n", slope_table(d / "residual_slope_summary.csv"),
          f"검사: Δh = 0 제외 순서쌍 {cfg['residual_slope_checks']['n_excluded_dh0']}개, BiGRU 잔차 기울기와 `BiGRUModel.residual`의 최대 차이 "
          f"{cfg['residual_slope_checks']['bigru_vs_residual_max_abs']:.1e}.\n",
          "### 1.6 HbI 변화 크기 구간별 오차 (n = 그 구간에 쌍이 있는 피험자 수)\n"]
    for direction, lab in (("all", "전체"), ("increase", "증가"), ("decrease", "감소")):
        rows = [[m] + [cell(summ, f"dh|{b}|{direction}", "a_rmse", m, True) for b in DH] for m in MAIN_MODELS + supp]
        o += [f"absorbance RMSE, 방향: {lab}\n", table(["모델"] + DH, rows)]
    rows = [[m] + [cell(summ, f"dh|{b}|all", "peak_rmse", m, True) for b in DH] for m in MAIN_MODELS + supp]
    o += ["Peak RMSE, 방향 전체\n", table(["모델"] + DH, rows), "구간별 쌍체 차이 (방향 전체)\n",
          cmp_table(cd, [f"dh|{b}|all" for b in DH], ["a_rmse", "peak_rmse"], True),
          "### 1.7 추가 기준값: 피험자 내 Peak 대 HbI 선형회귀의 HbI 0.6 값 (Peak_RS 단위)\n",
          f"{cfg['peak_rs_reference']}\n"]
    models = MAIN_MODELS + supp[:3]
    rows = []
    for s, g in pr.groupby("subject"):
        g = g.set_index("model")
        r0 = g.iloc[0]
        rows.append([s, "외삽" if r0.extrapolated else "In-range", f"{r0.reference_peak_rs:.2f} ± {r0.reference_se_peak_rs:.2f}"] +
                    [f"{g.loc[m, 'model_minus_reference_peak_rs']:+.2f}" for m in models])
    o.append(table(["피험자", "구분", "기준값 ± 표준오차"] + [f"{m} − 기준값" for m in models], rows))
    o.append("## 2. Union16 분할 (보조, 고정 설정)\n")
    note = jload(u / "training_size_effect_note.json")
    o.append(f"- {note['rule3_exception']}\n- {note['note']}\n- 하이퍼파라미터는 동결 문서의 값으로 고정했다 (다시 고르지 않음).\n")
    for name, lab in (("Union16_LeaveOneSubjectOut", "피험자 16명의 평균 ± 표준편차"),
                      ("Union16_Holdout73", "피험자 → 분할 안 평균 → 분할 10개의 평균 ± 표준편차; 쌍체 비교는 피험자별 분할 평균 뒤 n = 16"),
                      ("Development10_LeaveOneSubjectOut_FixedHyperparameters", "Development cohort 10명, 9명 학습, 동결 문서의 고정값 (학습 피험자 수 비교용)")):
        s2, c2 = read(u / name / "summary.csv"), read(u / name / "comparisons_main.csv")
        o += [f"### {name}\n", f"집계: {lab}.\n", wide(s2, "general", gm, MAIN_MODELS + supp),
              cmp_table(c2, ["general"], ["a_rmse", "r_rmse_pp", "peak_rmse"]),
              wide(s2, "repeatability", ["peak_sd", "spectral_sd"], ["Raw"] + MAIN_MODELS + supp, True)]
    e = read(u / "training_size_effect.csv")
    o.append("### 학습 피험자 수 효과 (General reconstruction, 피험자 쌍체 차이; 차이 = 15명 학습 − 더 적은 학습, 개선 = 차이 < 0)\n")
    rows = [[r.cohort, r.role, r.comparison, r.model, LABEL[r.metric], int(r.n), num(r.diff_mean * SCALE[r.metric]), num(r.diff_se * SCALE[r.metric]),
             f"{int(r.n_improved)}/{int(r.n)}", num(r.wilcoxon_p), num(r.sign_p)] for r in e.itertuples()]
    o.append(table(["대상", "역할", "비교", "모델", "지표", "n", "차이 평균", "표준오차", "개선 k/n", "Wilcoxon p", "부호검정 p"], rows))
    (DOCS / "Results_ExternalTest_Union16.md").write_text("\n".join(o), encoding="utf-8")
    print("작성:", DOCS / "Results_ExternalTest_Union16.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", action="store_true")
    ap.add_argument("--results", action="store_true")
    ap.add_argument("--preregistration", action="store_true")
    ap.add_argument("--commit", default=None)
    a = ap.parse_args()
    DOCS.mkdir(exist_ok=True)
    if a.results:
        results_doc()
    if a.external:
        external_doc()
    if a.preregistration:
        if not a.commit:
            ap.error("--commit 이 필요함")
        preregistration(a.commit)


if __name__ == "__main__":
    main()
