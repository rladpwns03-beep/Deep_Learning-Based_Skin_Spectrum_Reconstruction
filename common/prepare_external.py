"""External cohort 전처리: data/raw_external -> data/processed (spectra_external.csv, spectra_union16.csv).

실행:  python -X utf8 common/prepare_external.py   (code 폴더에서)

  - 입력: data/raw_external/S001_오른손_엄지.csv ... S006_오른손_엄지.csv (cp949, 읽기 전용)
  - 측정 행만 읽는다: 첫 열이 정수이고 'Data Name'에 "(YYYY-MM-DD" 형식의 측정 시각이 있는 행.
    (S002 파일 아래쪽의 분석 행은 읽지 않는다.) 파일마다 정확히 10행이 아니면 오류.
  - 피험자 ID: 파일의 S001-S006 -> S101-S106 (Development cohort의 S001-S010과 다른 사람).
  - prepare_data.load_subject 와 같은 검증 (패딩 구간 복사 여부, 결측, 반사율 범위).
  - data/processed/spectra.csv 는 읽기만 한다.

검사 (하나라도 다르면 오류로 멈춤)
  - 피험자별 HbI 범위, 16명 기준 Near-target subject / In-range subject, 160개 스펙트럼 완전 중복 없음
  - S102(원 파일 S002) 측정 10행의 Peak_RS 대 HbI 선형회귀 = 원 파일 하단 분석 행의 값
    (Peak_RS = 8387.2 x Peak + 83.495, Hwang et al. 2023, Sensors 23(17):7654 식 (2))
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import NEAR_TOL_MAIN, TARGET_HBI                              # noqa: E402
from common.prepare_data import PAD_HIGH, PAD_LOW, WL_COLS, parse_time           # noqa: E402

RAW = ROOT / "data" / "raw_external"
OUT = ROOT / "data" / "processed"
N_ROWS = 10
PEAK_RS_SCALE, PEAK_RS_OFFSET = 8387.2, 83.495

EXPECTED_HBI = {"S101": (0.4973, 0.6411), "S102": (0.4297, 0.5607), "S103": (0.4511, 0.5547),
                "S104": (0.4604, 0.5690), "S105": (0.4916, 0.5825), "S106": (0.4604, 0.6499)}
EXPECTED_NEAR = ["S003", "S004", "S008", "S101", "S105"]
EXPECTED_IN_RANGE = ["S001", "S004", "S009", "S010", "S101", "S106"]
EXPECTED_S102_REGRESSION = dict(slope=-331.304, intercept=274.815, at_target=76.033)


def subject_id(path: Path) -> str:
    m = re.fullmatch(r"S00([1-6])_오른손_엄지", path.stem)
    if m is None:
        raise ValueError(f"예상하지 않은 파일 이름: {path.name}")
    return f"S10{m.group(1)}"


def load_external(path: Path) -> pd.DataFrame:
    f = pd.read_csv(path, encoding="cp949", dtype=str)
    first = f.iloc[:, 0].fillna("").str.strip()
    is_meas = first.str.fullmatch(r"\d+") & f["Data Name"].fillna("").str.contains(r"\(\d{4}-\d{2}-\d{2}")
    f = f[is_meas].reset_index(drop=True)
    if len(f) != N_ROWS:
        raise ValueError(f"{path.name}: 측정 행이 {len(f)}개 (기대 {N_ROWS})")
    R = f[WL_COLS + PAD_LOW + PAD_HIGH].astype(float)
    assert (R[PAD_LOW].to_numpy() == R[["400nm"]].to_numpy()).all(), f"{path.name}: 360~390nm가 400nm와 다름"
    assert (R[PAD_HIGH].to_numpy() == R[["700nm"]].to_numpy()).all(), f"{path.name}: 710~740nm가 700nm와 다름"
    assert R[WL_COLS].notna().all().all(), f"{path.name}: 결측값 있음"
    assert ((R[WL_COLS] > 0) & (R[WL_COLS] < 100)).all().all(), f"{path.name}: 반사율 범위(0~100%) 밖 값"
    trial = first[is_meas].astype(int).reset_index(drop=True)
    assert trial.tolist() == list(range(1, N_ROWS + 1)), f"{path.name}: 측정 번호가 1..{N_ROWS}가 아님"
    return pd.DataFrame({
        "subject": subject_id(path),
        "trial": trial,
        "measured_at": f["Data Name"].map(parse_time),
        "L": f["L*(D65)"].astype(float),
        "a": f["a*(D65)"].astype(float),
        "b": f["b*(D65)"].astype(float),
    }).join(R[WL_COLS])


def absorbance(df: pd.DataFrame) -> np.ndarray:
    return -np.log10(df[WL_COLS].to_numpy(float) / 100.0)


def peak_rs_regression(df: pd.DataFrame) -> dict:
    """측정 10행의 Peak_RS 대 HbI 최소제곱 직선과 HbI 0.6에서의 값."""
    a = absorbance(df)
    col = {c: i for i, c in enumerate(WL_COLS)}
    hbi = a[:, col["570nm"]]
    pk = a[:, col["490nm"]] - (a[:, col["470nm"]] + a[:, col["510nm"]]) / 2
    slope, intercept = np.polyfit(hbi, PEAK_RS_SCALE * pk + PEAK_RS_OFFSET, 1)
    return dict(slope=float(slope), intercept=float(intercept), at_target=float(slope * TARGET_HBI + intercept))


def main():
    files = sorted(RAW.glob("S*.csv"))
    if len(files) != 6:
        raise RuntimeError(f"{RAW}: S*.csv가 {len(files)}개 (기대 6)")
    ext = pd.concat([load_external(p) for p in files], ignore_index=True)
    dev = pd.read_csv(OUT / "spectra.csv")
    assert list(dev.columns) == list(ext.columns), "spectra.csv와 열 구성이 다름"
    assert not set(dev.subject) & set(ext.subject)
    union = pd.concat([dev, ext], ignore_index=True)
    assert not union.duplicated(["subject", "trial"]).any()

    problems = []
    hbi = absorbance(union)[:, WL_COLS.index("570nm")]
    rng = {s: (hbi[(union.subject == s).to_numpy()].min(), hbi[(union.subject == s).to_numpy()].max())
           for s in sorted(union.subject.unique())}
    for s, (lo, hi) in EXPECTED_HBI.items():
        if round(rng[s][0], 4) != lo or round(rng[s][1], 4) != hi:
            problems.append(f"{s}: HbI 범위 {rng[s][0]:.4f}-{rng[s][1]:.4f} (기대 {lo}-{hi})")
    near = sorted({s for s, h in zip(union.subject, hbi) if abs(h - TARGET_HBI) <= NEAR_TOL_MAIN})
    in_range = sorted(s for s, (lo, hi) in rng.items() if lo <= TARGET_HBI <= hi)
    if near != EXPECTED_NEAR:
        problems.append(f"Near-target subject {near} (기대 {EXPECTED_NEAR})")
    if in_range != EXPECTED_IN_RANGE:
        problems.append(f"In-range subject {in_range} (기대 {EXPECTED_IN_RANGE})")
    n_dup = int(union.duplicated(WL_COLS).sum())
    if n_dup:
        problems.append(f"완전 중복 스펙트럼 {n_dup}개")
    reg = peak_rs_regression(ext[ext.subject == "S102"])
    for k, v in EXPECTED_S102_REGRESSION.items():
        if round(reg[k], 3) != v:
            problems.append(f"S102 Peak_RS 회귀 {k} = {reg[k]:.4f} (기대 {v})")
    if problems:
        raise RuntimeError("기대값과 다름 (파일을 쓰지 않음):\n  " + "\n  ".join(problems))

    ext.to_csv(OUT / "spectra_external.csv", index=False, encoding="utf-8")
    union.to_csv(OUT / "spectra_union16.csv", index=False, encoding="utf-8")
    print(f"spectra_external.csv: {ext.subject.nunique()}명, {len(ext)}행")
    print(f"spectra_union16.csv: {union.subject.nunique()}명, {len(union)}행")
    for s in EXPECTED_HBI:
        print(f"  {s}: HbI {rng[s][0]:.4f} - {rng[s][1]:.4f}")
    print(f"Near-target subject ({len(near)}/16): {near}")
    print(f"In-range subject ({len(in_range)}/16): {in_range}")
    print(f"완전 중복 스펙트럼: {n_dup}")
    print(f"S102 Peak_RS 회귀: slope {reg['slope']:.3f}, intercept {reg['intercept']:.3f}, HbI 0.6에서 {reg['at_target']:.3f}")


if __name__ == "__main__":
    main()
