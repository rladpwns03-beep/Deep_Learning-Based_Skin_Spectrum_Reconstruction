"""S006 exclusion sensitivity (보조 분석, Development cohort만 사용).

실행 (code 폴더, conda 환경 skin_spectra):
    python -X utf8 scripts/sensitivity_exclude_S006.py     # results/sensitivity_exclude_S006/ (이미 있으면 중단)

배경: S006은 HbI 범위가 0.033으로 가장 좁다 (0.485-0.518).
  - Development10_LeaveOneSubjectOut 에서 S006이 test가 아닌 9개 fold에 대해, 학습 피험자에서 S006을 뺀 8명으로 다시 학습.
  - 하이퍼파라미터는 해당 fold의 주 결과에서 선택된 값으로 고정 (BiGRU의 hidden, slope_l2, epoch와 PLS_v2의 성분 수). 다시 고르지 않는다.
  - 모델: HbI-binned mean, Global Linear, PLS_v2, BiGRU.
  - 보고: 모델별 "S006 제외 학습 minus S006 포함 학습"의 General reconstruction absorbance RMSE와 Peak RMSE 쌍체 차이 (피험자 9명).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))
from final_utils import DEV_CSV, P_NOTE, RES, jdump, jload, load_splits, load_spectra, test_pairs   # noqa: E402
from final_eval import comparison_table, fit_models, load_dev_models, split_records, summary_table   # noqa: E402

SPLIT_SET = "Development10_LeaveOneSubjectOut"
EXCLUDE = "S006"
MODELS = ["HbI-binned mean", "Global Linear", "PLS_v2", "BiGRU"]
OUT = RES / "sensitivity_exclude_S006"
TAG = " [S006 excluded]"


def main():
    if OUT.exists() and any(OUT.iterdir()):
        sys.exit(f"[중단] {OUT} 에 이미 결과가 있음. 덮어쓰지 않음.")
    torch.set_num_threads(1)
    data = load_spectra(DEV_CSV)
    bigru = {r["split_id"]: r["chosen"] for r in jload(RES / "bigru_tuned" / SPLIT_SET / "selection.json")["splits"]}
    pls = {r["split"]: r["chosen"] for r in jload(RES / "pls_v2_k10" / "loso" / "selection.json")["splits"]}
    frames, used = [], []
    for sp in load_splits(SPLIT_SET):
        sid = sp["split_id"]
        if EXCLUDE in sp["test"]:
            continue
        train8 = [s for s in sp["train"] if s != EXCLUDE]
        assert len(train8) == 8 and not set(train8) & set(sp["test"])
        hp = dict(pls_v2_components=pls[sid], bigru=bigru[sid])
        inc = {m: v for m, v in load_dev_models(SPLIT_SET, sid, with_peak_constrained=False).items() if m in MODELS}
        assert inc["PLS_v2"][0].n_components == pls[sid] and inc["BiGRU"][0].epochs == bigru[sid]["epoch"]
        exc = {m + TAG: v for m, v in fit_models(data.select(train8), hp, MODELS).items()}
        ii, jj, subj = test_pairs(SPLIT_SET, sid, data)
        recs = split_records({**inc, **exc}, data.select(sp["test"]), data.a[ii], data.a[jj], data.hbi[jj], data.hbi[ii], subj)
        f = pd.DataFrame([r for r in recs if r["section"] == "general" and r["metric"] in ("a_rmse", "peak_rmse")])
        f.insert(0, "split_id", sid)
        frames.append(f)
        used.append(dict(split_id=sid, test=sp["test"], train=train8, pls_v2_components=pls[sid],
                         bigru={k: bigru[sid][k] for k in ("hidden", "slope_l2", "epoch")}))
        print(f"[{sid}] 8명 학습 완료", flush=True)
    df = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "records.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    summary_table(df, False).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    cmp_ = comparison_table(df, [(m + TAG, m) for m in MODELS])
    cmp_.to_csv(OUT / "comparisons.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    jdump(OUT / "config.json", dict(
        split_set=SPLIT_SET, excluded_training_subject=EXCLUDE, n_folds=len(used), folds=used, models=MODELS,
        hyperparameters="해당 fold의 주 결과에서 선택된 값으로 고정 (다시 고르지 않음)",
        comparison="'S006 제외 학습 minus S006 포함 학습', 피험자 9명 쌍체 차이. 개선 = 차이 < 0. " + P_NOTE))
    print(cmp_[["metric", "comparison", "n", "diff_mean", "diff_se", "n_improved"]].to_string(index=False))


if __name__ == "__main__":
    main()
