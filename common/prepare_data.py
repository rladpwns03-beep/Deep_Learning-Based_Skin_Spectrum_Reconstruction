"""data/raw -> data/processed 전처리.

실행:  python common/prepare_data.py   (code 폴더에서)

원본(data/raw)은 읽기만 하고 절대 수정하지 않는다.
생성 파일:
  data/processed/spectra.csv       피험자 x 측정 1회 = 1행, 400~700nm 반사율(%) 31개 파장
  data/processed/galaxy_watch.csv  Galaxy Watch 8 측정값 (피험자 x trial)

처리 내용:
  1. cp949 인코딩 CSV -> UTF-8
  2. 측정기 유효 구간 400~700nm만 남김
     (360~390nm는 400nm 값, 710~740nm는 700nm 값이 그대로 복사돼 있어 실측값이 아님)
  3. 비어 있는 열('------': Target No., dE*ab, Judgement, dL*, da*, db*) 제거
  4. 'Data Name'의 측정 시각(예: "1 (2026-09-30 오후 2:39:51)")을 ISO 형식으로 변환
값 자체(반사율)는 변환하지 않는다. 흡광도(-log10(R/100)) 변환은 data_loader에서 한다.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

WL = np.arange(400, 701, 10)              # 유효 파장 31개
WL_COLS = [f"{w}nm" for w in WL]
PAD_LOW = ["360nm", "370nm", "380nm", "390nm"]
PAD_HIGH = ["710nm", "720nm", "730nm", "740nm"]


def parse_time(name: str) -> str:
    m = re.search(r"\((\d{4}-\d{2}-\d{2})\s*(오전|오후)\s*(\d{1,2}):(\d{2}):(\d{2})\)", name)
    if m is None:
        raise ValueError(f"측정 시각을 읽을 수 없음: {name!r}")
    day, ampm, hh, mm, ss = m.groups()
    hh = int(hh) % 12 + (12 if ampm == "오후" else 0)
    return f"{day} {hh:02d}:{mm}:{ss}"


def load_subject(path: Path) -> pd.DataFrame:
    f = pd.read_csv(path, encoding="cp949")
    R = f[WL_COLS + PAD_LOW + PAD_HIGH].astype(float)

    # 검증: 패딩 구간이 정말 400/700nm 복사값인지 확인 (아니면 유효 구간 가정이 틀린 것)
    assert (R[PAD_LOW].to_numpy() == R[["400nm"]].to_numpy()).all(), f"{path.name}: 360~390nm가 400nm와 다름"
    assert (R[PAD_HIGH].to_numpy() == R[["700nm"]].to_numpy()).all(), f"{path.name}: 710~740nm가 700nm와 다름"
    assert R[WL_COLS].notna().all().all(), f"{path.name}: 결측값 있음"
    assert ((R[WL_COLS] > 0) & (R[WL_COLS] < 100)).all().all(), f"{path.name}: 반사율 범위(0~100%) 밖 값"

    return pd.DataFrame({
        "subject": path.stem,
        "trial": f.iloc[:, 0].astype(int),
        "measured_at": f["Data Name"].map(parse_time),
        "L": f["L*(D65)"].astype(float),
        "a": f["a*(D65)"].astype(float),
        "b": f["b*(D65)"].astype(float),
    }).join(R[WL_COLS])


def load_watch(path: Path) -> pd.DataFrame:
    """엑셀 상단의 Galaxy Watch8 표(행: trial1~3, 열: S001~S010)만 추출.
    아래쪽 CM-700d 흡광도 블록은 행 라벨이 없어 피험자 대응을 단정할 수 없으므로 가져오지 않는다."""
    x = pd.read_excel(path, header=None)
    head = x.iloc[1]
    cols = {j: v for j, v in head.items() if isinstance(v, str) and v.startswith("S")}
    rows = []
    for i in range(len(x)):
        label = x.iat[i, 1]
        if isinstance(label, str) and label.startswith("trial"):
            for j, sid in cols.items():
                rows.append(dict(subject=sid, trial=int(label[5:]), value=float(x.iat[i, j])))
    return pd.DataFrame(rows).sort_values(["subject", "trial"], ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW.glob("S*.csv"))
    assert files, f"{RAW}에 S*.csv가 없음"

    spec = pd.concat([load_subject(p) for p in files], ignore_index=True)
    assert not spec.duplicated(["subject", "trial"]).any()
    spec.to_csv(OUT / "spectra.csv", index=False, encoding="utf-8")
    print(f"spectra.csv: {spec.subject.nunique()}명, {len(spec)}행, 파장 {len(WL_COLS)}개 ({WL[0]}~{WL[-1]}nm)")
    print(spec.groupby("subject").size().to_string())

    xlsx = sorted(RAW.glob("Galaxy Watch*.xlsx"))
    if xlsx:
        w = load_watch(xlsx[0])
        w.to_csv(OUT / "galaxy_watch.csv", index=False, encoding="utf-8")
        print(f"galaxy_watch.csv: {w.subject.nunique()}명, {len(w)}행")


if __name__ == "__main__":
    main()
