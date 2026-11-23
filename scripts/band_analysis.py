"""구간별 분석 (Development cohort만 사용): (1) 파장 구간별 복원 오차, (2) 잔차 기울기의 파장 4구간 집계.

실행 (code 폴더, conda 환경 skin_spectra):
    python -X utf8 scripts/band_analysis.py          # results/band_analysis/ (이미 결과가 있으면 중단)

파장 구간 (고정)
  Soret 400-440 (5) | Band 450-460 (2) | Carotenoid 470-510 (5) | Q-band 520-560, 580, 590 (7, 570 제외) | Red 600-700 (11)
  570 nm: h_in -> h* 고정, 오차 0 (모든 모델). 5개 구간의 파장 수 합 = 30.

(1) 파장 구간별 복원 오차 (두 분할 세트, 모델 전부)
  피험자별·모델별·구간별 absorbance RMSE (그 피험자의 모든 테스트 쌍 x 구간 파장의 오차 제곱 평균의 제곱근)와 bias (예측 - 정답).
  검사: 5개 구간의 오차 제곱합으로 다시 계산한 전체 absorbance RMSE = common/metrics.group_metrics 의 a_rmse.

(2) 잔차 기울기 (Development10_LeaveOneSubjectOut)
  r_model(λ) = (A*_model(λ) - A*_GlobalLinear(λ)) / Δh,  Δh = 목표 HbI - 입력 HbI.  Δh = 0인 쌍은 제외.
  대상: PLS_v2, BiGRU (튜닝, seed 평균), Ridge. Global Linear는 정의상 0.
  제외: HbI-binned mean과 PLS_v1은 보정이 Δh에 비례하지 않아 기울기 형태로 정의되지 않는다.
  집계 구간: Soret, Band 450-460, Q-band (570 제외), Red. 570 nm와 Carotenoid 구간은 이 집계에서 제외하고,
  Peak 방향 잔차 기울기 r(490) - (r(470) + r(510))/2 를 별도로 보고한다.
  개인 기준 잔차 기울기 d_s(λ) = b_s(λ) - b_GlobalLinear(λ): b_s = 테스트 피험자 s의 10회 측정으로 구한 피험자 내 최소제곱 기울기
  (평가에만 사용, 학습과 선택에는 쓰지 않음), b_GlobalLinear = 그 fold의 학습 피험자로 구한 기울기.
  검사: BiGRU는 위 값이 BiGRUModel.residual(a, target) 과 1e-5 이내, PLS_v2와 Ridge는 Δh의 크기와 부호가 달라도 같은 값.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))
from common.data_loader import HBI_INDEX, WL                                   # noqa: E402
from common.metrics import peak                                                # noqa: E402
from final_utils import (BAND_COMPARISONS, BAND_INDEX, BANDS, DEV_CSV, P_NOTE, RES, SLOPE_BANDS, jdump, load_splits,   # noqa: E402
                         load_spectra, test_pairs)
from final_eval import comparison_table, load_dev_models, split_records, summary_table   # noqa: E402
from models.bigru import BiGRUModel                                            # noqa: E402
from models.linear import fit_slope                                            # noqa: E402

OUT = RES / "band_analysis"
SPLIT_SETS = ["Development10_LeaveOneSubjectOut", "Development10_Holdout73"]
SLOPE_SET = "Development10_LeaveOneSubjectOut"
SLOPE_MODELS = ["PLS_v2", "BiGRU", "Ridge"]
EXCLUDED = {"Global Linear": "정의상 잔차 기울기 0",
            "HbI-binned mean": "보정이 Δh에 비례하지 않아(bin 평균 차이) 기울기 형태로 정의되지 않음",
            "PLS_v1": "보정이 Δh에 비례하지 않아(절편과 A_in, h_in, h*의 선형식) 기울기 형태로 정의되지 않음"}


def residual_slope_rows(models: dict, gl, te, X, T, h_in, subj, split_id="") -> tuple:
    """테스트 피험자별 평균 잔차 기울기와 개인 기준. 반환: (행 목록, 검사 dict)."""
    dh = T - h_in
    valid = dh != 0
    PGL = gl.predict(X, T)
    r_by_model, checks = {}, dict(n_pairs=int(len(dh)), n_excluded_dh0=int((~valid).sum()))
    for name in SLOPE_MODELS:
        rs = []
        for m in models[name]:
            r = (m.predict(X, T) - PGL)[valid] / dh[valid, None]
            if isinstance(m, BiGRUModel):
                d = float(np.abs(r - m.residual(X, T)[valid]).max())
                checks["bigru_vs_residual_max_abs"] = max(checks.get("bigru_vs_residual_max_abs", 0.0), d)
            else:                                           # 같은 입력, 다른 Δh (크기·부호) -> 같은 기울기여야 함
                for shift in (0.05, -0.1):
                    T2 = h_in + shift
                    r2 = (m.predict(X, T2) - gl.predict(X, T2)) / shift
                    key = f"{name}_dh_invariance_max_abs"
                    checks[key] = max(checks.get(key, 0.0), float(np.abs(r2[valid] - r).max()))
            rs.append(r)
        r_by_model[name] = np.mean(rs, axis=0)              # seed 평균
    sv = subj[valid]
    rows = []
    for s in te.subjects:
        d_s = fit_slope(te.select([s])) - gl.slope          # 개인 기준 (570 nm: 1 - 1 = 0)
        base = dict(split_id=split_id, subject=s, n_pairs_used=int((sv == s).sum()))
        for name, r in {**r_by_model, "Personal reference": None}.items():
            v = d_s if r is None else r[sv == s].mean(0)
            row = dict(base, model=name, peak_direction=float(peak(v)))
            for b in SLOPE_BANDS:
                row[f"mean|{b}"] = float(v[BAND_INDEX[b]].mean())
                row[f"rms|{b}"] = float(np.sqrt(np.mean(v[BAND_INDEX[b]] ** 2)))
            row.update({f"wl{int(w)}": float(x) for w, x in zip(WL, v)})
            rows.append(row)
        row = dict(base, model="Global Linear slope", peak_direction=float(peak(gl.slope)))
        for b in SLOPE_BANDS:
            row[f"mean|{b}"] = float(gl.slope[BAND_INDEX[b]].mean())
            row[f"rms|{b}"] = float(np.sqrt(np.mean(gl.slope[BAND_INDEX[b]] ** 2)))
        row.update({f"wl{int(w)}": float(x) for w, x in zip(WL, gl.slope)})
        rows.append(row)
    return rows, checks


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else None


def residual_slope_summary(df: pd.DataFrame) -> pd.DataFrame:
    """구간별·모델별 요약 (피험자 기준). Peak 방향은 별도 항목 (4구간에 포함하지 않음)."""
    ref = df[df.model == "Personal reference"].set_index("subject")
    gls = df[df.model == "Global Linear slope"].set_index("subject")
    rows = []
    for name in SLOPE_MODELS + ["Personal reference"]:
        d = df[df.model == name].set_index("subject").loc[ref.index]
        for item in SLOPE_BANDS + ["Peak direction"]:
            col = "peak_direction" if item == "Peak direction" else f"mean|{item}"
            v, r = d[col].to_numpy(), ref[col].to_numpy()
            rows.append(dict(
                item=item, model=name, n_subjects=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else None,
                rms_magnitude_mean=None if item == "Peak direction" else float(d[f"rms|{item}"].mean()),
                pearson_with_personal_reference=None if name == "Personal reference" else _corr(v, r),
                n_sign_agree=None if name == "Personal reference" else int((np.sign(v) == np.sign(r)).sum()),
                global_linear_slope_mean=float(gls[col].mean())))
    return pd.DataFrame(rows)


def run_band_errors(split_set, data):
    out = OUT / split_set
    frames = []
    for sp in load_splits(split_set):
        sid = sp["split_id"]
        ii, jj, subj = test_pairs(split_set, sid, data)
        recs = split_records(load_dev_models(split_set, sid), data.select(sp["test"]), data.a[ii], data.a[jj], data.hbi[jj],
                             data.hbi[ii], subj)
        f = pd.DataFrame([r for r in recs if r["section"].startswith("band|") or r["section"] == "general"])
        f.insert(0, "split_id", sid)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df = df[df.section.str.startswith("band|") | ((df.section == "general") & (df.metric == "a_rmse"))]
    by_split = split_set.endswith("Holdout73")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "band_error_records.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    summary_table(df, by_split).to_csv(out / "band_error_summary.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    comparison_table(df, BAND_COMPARISONS, ["band|"]).to_csv(out / "band_error_comparisons.csv", index=False,
                                                             encoding="utf-8-sig", float_format="%.10g")
    jdump(out / "band_error_config.json", dict(
        split_set=split_set, bands=BANDS, n_wavelengths={b: len(v) for b, v in BANDS.items()},
        n_wavelengths_total=sum(len(v) for v in BANDS.values()), nm570="570 nm: h_in -> h* 고정, 오차 0 (모든 모델)",
        definitions=dict(a_rmse="피험자의 모든 테스트 쌍 x 구간 파장의 오차 제곱 평균의 제곱근", bias="오차 평균 (예측 - 정답)"),
        aggregation="피험자 -> 분할 안 평균 -> 분할 평균 ± 표준편차" if by_split else "피험자 10명의 평균 ± 표준편차",
        paired_comparison="분석 단위 = 피험자. 차이 = 앞 모델 - 뒤 모델, 개선 = 차이 < 0 (bias는 |bias| 감소). " + P_NOTE,
        checks=dict(band_sse_recombines_to_total_rmse=True, n_wavelengths_sum_is_30=True)))
    print(f"[구간 오차] 저장: {out}")


def run_residual_slopes(data):
    out = OUT / SLOPE_SET
    rows, checks = [], []
    for sp in load_splits(SLOPE_SET):
        sid = sp["split_id"]
        models = load_dev_models(SLOPE_SET, sid)
        ii, jj, subj = test_pairs(SLOPE_SET, sid, data)
        r, ck = residual_slope_rows(models, models["Global Linear"][0], data.select(sp["test"]), data.a[ii], data.hbi[jj],
                                    data.hbi[ii], subj, sid)
        rows += r
        checks.append(dict(split_id=sid, **ck))
    ck = pd.DataFrame(checks)
    worst = dict(bigru_vs_residual_max_abs=float(ck.bigru_vs_residual_max_abs.max()),
                 **{c: float(ck[c].max()) for c in ck.columns if c.endswith("dh_invariance_max_abs")})
    if worst["bigru_vs_residual_max_abs"] > 1e-5 or any(v > 1e-9 for k, v in worst.items() if "invariance" in k):
        raise RuntimeError(f"잔차 기울기 검사 실패: {worst}")
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "residual_slope_per_subject.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    residual_slope_summary(df).to_csv(out / "residual_slope_summary.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    jdump(out / "residual_slope_config.json", dict(
        split_set=SLOPE_SET, models=SLOPE_MODELS, excluded_models=EXCLUDED, bands=SLOPE_BANDS,
        excluded_from_band_aggregation="570 nm (h_in, h*)와 Carotenoid 구간 (470-510 nm). Peak 방향은 별도 항목.",
        definition="r_model(λ) = (A*_model(λ) - A*_GlobalLinear(λ)) / Δh, Δh = 0인 쌍 제외. 피험자별 평균 후 구간 집계. BiGRU는 seed 평균.",
        personal_reference="d_s(λ) = b_s(λ) - b_GlobalLinear(λ); b_s는 평가에만 사용",
        n_pairs_excluded_dh0=int(ck.n_excluded_dh0.sum()), checks=worst, per_split_checks=checks))
    print(f"[잔차 기울기] 저장: {out} | Δh=0 제외 쌍 {int(ck.n_excluded_dh0.sum())} | 검사 {worst}")


def main():
    if OUT.exists() and any(OUT.rglob("*.csv")):
        sys.exit(f"[중단] {OUT} 에 이미 결과가 있음. 덮어쓰지 않음.")
    torch.set_num_threads(1)
    data = load_spectra(DEV_CSV)
    for s in SPLIT_SETS:
        run_band_errors(s, data)
    run_residual_slopes(data)


if __name__ == "__main__":
    main()
