"""잔차/변화량 회귀 모델(PLS_v1, PLS_v2, Ridge) 평가 + 기존 B1/B2/BiGRU와 같은 테스트 피험자·같은 쌍·같은 지표로 비교.

실행 (code 폴더, conda 환경 skin_spectra):
    python -X utf8 scripts/pls_compare.py                                   # 모든 모델, 두 분할
    python -X utf8 scripts/pls_compare.py --scheme loso --models pls_v2 ridge

모델 (--models)
  pls_v1      예전 PLS (models/pls.py): X=[A_in(31), h_in, h*] 표준화, Y=A_target-A_in 표준화, A*(570)=h*.  성분 1..5
  pls_v2      B2 + PLS 잔차 보정 (models/b2_pls.py): u=Δh(A_in-A_in(570)) 30밴드, e=A_target-A_B2 30밴드,
              u·e 열별 표준화, A* = A_B2 + f(u)-f(0).                                                 성분 1..5
  ridge       B2 + Ridge 잔차 보정 (models/b2_ridge.py): pls_v2와 같은 u, e, 같은 u 표준화, 다중출력 Ridge,
              A* = A_B2 + f(u)-f(0).  alpha {1e-4..1e4}, 끝이면 내부 검증에서 10배씩 확장
  pls_v1_k10, pls_v2_k10   민감도 분석: 성분 1..10

저장 (이미 있으면 그 분할은 건너뜀; results/holdout·loso는 읽기만 함)
  results/<model>/<scheme>/   models/<split>.json, selection.json, predictions.json
  results/compare/<scheme>/   comparison_table.csv, per_subject_delta.csv, metrics.json, config.json

설계 (결과를 보기 전에 고정)
  - 분할: holdout = 7:3 분할 10개 (data/splits/split_plan.csv), loso = LOSO 10 fold (results/loso/config/loso_folds.json).
  - 검증: 기존 실험에 검증 분할이 없으므로 학습 피험자 안 피험자 단위 LOSO. 기준 = 570 nm 제외 흡광도 RMSE
    (내부 검증 쌍 전체 pooled; 피험자마다 쌍 수가 같아 피험자 평균 MSE와 순위 동일). 테스트 피험자는 최종 평가에만 사용.
  - 테스트 쌍: 기존 예측 파일에 저장된 쌍 그대로 (테스트 피험자 안 모든 i != j).
    목표 HbI = 측정 j의 실측 A(570), 정답 = 측정 j의 실측 스펙트럼 (보간값 아님).
  - B1/B2/BiGRU: 저장된 예측에 같은 지표 함수를 다시 적용 (재학습 없음). BiGRU = seed 0/1/2 지표 평균.
  - 지표: common/metrics.py group_metrics (a_rmse·r_rmse_pp = 570 제외 30파장) + 570 포함 31파장 흡광도 RMSE.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # 프로젝트 루트 (scripts/ 의 상위)
sys.path.insert(0, str(ROOT))
from common import config as C                                          # noqa: E402
from common.data_loader import HBI_INDEX, WL, holdout_split, load_spectra   # noqa: E402
from common.metrics import group_metrics                                 # noqa: E402
from models.b2_pls import B2PLSModel                                     # noqa: E402
from models.b2_ridge import ALPHAS, B2RidgeModel                         # noqa: E402
from models.linear import apply_slope                                    # noqa: E402
from models.pls import PLSModel                                          # noqa: E402

RES = ROOT / "results"
TOLS = {"main": C.NEAR_TOL_MAIN, "supplementary": C.NEAR_TOL_SUPP}
METRICS = ["a_rmse_31", "a_rmse", "r_rmse_pp", "peak_rmse"]
BASE_MODELS = ["B1", "B2", "BiGRU"]
REGISTRY = {   # 폴더/키 -> (표 이름, 생성자, 설명)
    "pls_v1": ("PLS_v1", lambda: PLSModel(max_components=5), "예전 PLS: X=[A_in,h_in,h*] -> A_target-A_in, 성분 1..5"),
    "pls_v2": ("PLS_v2", lambda: B2PLSModel(max_components=5), "B2 + PLS 잔차, 성분 1..5"),
    "ridge": ("Ridge", lambda: B2RidgeModel(), f"B2 + Ridge 잔차, alpha {ALPHAS} (+끝이면 확장)"),
    "pls_v1_k10": ("PLS_v1_k10", lambda: PLSModel(max_components=10), "민감도: 예전 PLS 성분 1..10"),
    "pls_v2_k10": ("PLS_v2_k10", lambda: B2PLSModel(max_components=10), "민감도: B2 + PLS 성분 1..10"),
}
SCHEMES = {
    "holdout": dict(pred=RES / "holdout/predictions/predictions.json", key="split_id",
                    desc="7:3 분할 10개 (results/holdout, 분할마다 0.6 근처 피험자 1명 테스트)"),
    "loso": dict(pred=RES / "loso/predictions/loso_predictions.json", key="fold_id",
                 desc="Leave-One-Subject-Out 10 fold (results/loso)"),
}


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jdump(p, obj, indent=1):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")


def metrics(P, Y):
    g = group_metrics(P, Y)
    return dict(a_rmse_31=float(np.sqrt(np.mean((P - Y) ** 2))), a_rmse=g["a_rmse"], r_rmse_pp=g["r_rmse_pp"],
                peak_rmse=g["peak_rmse"], n_pairs=g["n_pairs"])


def train_subjects(scheme, data, sid, test):
    if scheme == "loso":
        f = {f["fold_id"]: f for f in jload(RES / "loso/config/loso_folds.json")["folds"]}[sid]
        assert f["test"] == test, (sid, f["test"], test)
        return f["train"]
    tr, te = holdout_split(data, int(sid.split("_")[1]))
    assert te.subjects == test, (sid, te.subjects, test)
    return tr.subjects


def selection(m):
    """선택된 하이퍼파라미터와 내부 검증 곡선 (RMSE, 570 제외)."""
    if isinstance(m, B2RidgeModel):
        return dict(param="alpha", chosen=m.alpha, grid=m.cv["alphas"], rmse=m.cv["rmse"], extended=m.cv["extended"],
                    at_boundary=m.cv["at_boundary_after_extension"])
    if isinstance(m, B2PLSModel):
        return dict(param="n_components", chosen=m.n_components, grid=m.cv["components"], rmse=m.cv["rmse"])
    return dict(param="n_components", chosen=m.n_components, grid=m.cv["components"],
                rmse=np.sqrt(m.cv["mean_mse"]).tolist())


def run(scheme, data, keys, run_id):
    cfg = SCHEMES[scheme]
    cmp_out = RES / "compare" / scheme
    outs = {k: RES / k / scheme for k in keys}
    busy = [p for p in [cmp_out, *outs.values()] if p.exists() and any(p.rglob("*.json"))]
    if busy:
        print(f"[건너뜀] {scheme}: 이미 결과가 있음 -> {[str(p.relative_to(ROOT)) for p in busy]}. 덮어쓰지 않음.")
        return None
    names = [REGISTRY[k][0] for k in keys]
    row = {(s, t): i for i, (s, t) in enumerate(zip(data.subject, data.trial))}
    g = pd.DataFrame(jload(cfg["pred"])["general_reconstruction"])

    recs, checks = [], []
    sel = {k: [] for k in keys}
    preds = {k: [] for k in keys}
    for sid, d in g.groupby(cfg["key"], sort=True):
        test = sorted(d.subject.unique())
        train = train_subjects(scheme, data, sid, test)
        assert not set(train) & set(test)
        tr, te = data.select(train), data.select(test)
        ii = np.array([row[(s, t)] for s, t in zip(d.subject, d.input_trial)])
        jj = np.array([row[(s, t)] for s, t in zip(d.subject, d.target_trial)])
        X, Y, T = data.a[ii], data.a[jj], data.hbi[jj]
        assert np.allclose(Y, np.array(d.actual.tolist()), atol=1e-12) and np.allclose(T, d.target_hbi, atol=1e-12)
        allp = {k: np.array([p[k] for p in d.predictions]) for k in d.predictions.iloc[0]}
        line = []
        for k in keys:
            m = REGISTRY[k][1]().fit(tr)
            m.save(outs[k] / "models" / f"{sid}.json")
            P = m.predict(X, T)
            allp[REGISTRY[k][0]] = P
            ident = m.predict(te.a, te.hbi)
            ck = dict(model=k, split=sid, hbi570_exact=bool(np.array_equal(P[:, HBI_INDEX], T)),
                      identity_max_abs_diff=float(np.abs(ident - te.a).max()),
                      identity_exact=bool(np.array_equal(ident, te.a)),
                      fixed06_hbi570_exact=bool(np.array_equal(m.predict(te.a, C.TARGET_HBI)[:, HBI_INDEX],
                                                               np.full(len(te), C.TARGET_HBI))))
            if hasattr(m, "slope"):
                ck["b2_refit_vs_stored_max_abs"] = float(np.abs(apply_slope(X, T, m.slope) - allp["B2"]).max())
            checks.append(ck)
            s_ = dict(split=sid, test=test, train=train, **selection(m)); sel[k].append(s_)
            line.append(f"{REGISTRY[k][0]}={s_['chosen']:g}")
            preds[k] += [dict(split=sid, subject=d.subject.iloc[n], input_trial=int(d.input_trial.iloc[n]),
                              target_trial=int(d.target_trial.iloc[n]), target_hbi=float(T[n]), pred=P[n].tolist())
                         for n in range(len(d))]
        print(f"[{scheme} {sid}] test {test}  " + "  ".join(line), flush=True)
        for s in test:
            sm = (d.subject == s).to_numpy()
            for k, Pk in allp.items():
                mn, seed = (k.split("_seed")[0], int(k.split("_seed")[1])) if "_seed" in k else (k, -1)
                r = dict(split=sid, subject=s, model=mn, seed=seed, general=metrics(Pk[sm], Y[sm]))
                for t, tol in TOLS.items():
                    sl = sm & (np.abs(T - C.TARGET_HBI) <= tol)
                    r[f"near_{t}"] = metrics(Pk[sl], Y[sl]) if sl.any() else None
                recs.append(r)

    order = BASE_MODELS + names

    def frame(section):
        rows = [dict(split=r["split"], subject=r["subject"], model=r["model"], seed=r["seed"], **r[section])
                for r in recs if r[section] is not None]
        return pd.DataFrame(rows).groupby(["split", "subject", "model"])[METRICS].mean().reset_index()   # BiGRU seed 평균

    result, tables, deltas = {}, [], []
    for sec in ["general", "near_main", "near_supplementary"]:
        df = frame(sec)
        per_subj = df.groupby(["subject", "model"])[METRICS].mean().unstack("model")
        if scheme == "holdout" and sec == "general":
            summary = df.groupby(["split", "model"])[METRICS].mean().groupby("model").mean()
            agg = "피험자 -> split 안 평균 -> split 10개 평균 (results/holdout과 같은 집계)"
        else:
            summary = df.groupby("model")[METRICS].mean()
            agg = "피험자(평가) 단위 평균" + (" (split별 평가)" if scheme == "holdout" else "")
        summary = summary.reindex(order)
        result[sec] = dict(aggregation=agg, n_evaluations=int(len(df) // len(order)), subjects=sorted(per_subj.index),
                           mean=summary.to_dict(orient="index"),
                           per_subject={s: {met: {mn: float(per_subj.loc[s, (met, mn)]) for mn in order} for met in METRICS}
                                        for s in per_subj.index})
        for mn in order:
            tables.append(dict(section=sec, model=mn, **{m_: summary.loc[mn, m_] * (1 if m_ == "r_rmse_pp" else 1e3)
                                                          for m_ in METRICS}))
        for s in per_subj.index:
            for mn in names:
                row_ = dict(section=sec, subject=s, model=mn)
                for m_ in METRICS[1:]:
                    b2, v = per_subj.loc[s, (m_, "B2")], per_subj.loc[s, (m_, mn)]
                    row_.update({f"{m_}_B2": b2, f"{m_}_model": v, f"{m_}_pct_vs_B2": 100 * (v - b2) / b2})
                deltas.append(row_)

    match = {}   # 저장 지표 재현 검사
    if scheme == "loso":
        sw = jload(RES / "loso/metrics/loso_model_comparison.json")["subject_wise"]
        for met in ("a_rmse", "peak_rmse"):
            match[met] = {mn: bool(np.allclose([result["general"]["per_subject"][s][met][mn] for s in sw[met][mn]],
                                               list(sw[met][mn].values()), rtol=1e-9)) for mn in BASE_MODELS}
    else:
        sw = pd.DataFrame(jload(RES / "holdout/metrics/general_reconstruction.json")["subject_wise"])
        sw = sw.groupby(["split_id", "subject", "model"])[["a_rmse", "peak_rmse", "r_rmse_pp"]].mean()
        mine = frame("general").set_index(["split", "subject", "model"])
        for met in ("a_rmse", "peak_rmse", "r_rmse_pp"):
            match[met] = {mn: bool(np.allclose(mine.xs(mn, level="model")[met].sort_index().to_numpy(),
                                               sw.xs(mn, level="model")[met].sort_index().to_numpy(), rtol=1e-9))
                          for mn in BASE_MODELS}

    for k in keys:
        jdump(outs[k] / "selection.json", dict(run_id=run_id, scheme=scheme, model=REGISTRY[k][0], description=REGISTRY[k][2],
                                               splits=sel[k], checks=[c for c in checks if c["model"] == k]))
        jdump(outs[k] / "predictions.json", dict(wavelengths_nm=WL.tolist(), model=REGISTRY[k][0],
                                                  general_reconstruction=preds[k]), indent=None)
    cmp_out.mkdir(parents=True, exist_ok=True)
    tab = pd.DataFrame(tables)
    tab.to_csv(cmp_out / "comparison_table.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    dl = pd.DataFrame(deltas)
    dl.to_csv(cmp_out / "per_subject_delta.csv", index=False, encoding="utf-8-sig", float_format="%.6g")
    jdump(cmp_out / "config.json", dict(
        run_id=run_id, scheme=scheme, split_description=cfg["desc"], test_pairs_source=str(cfg["pred"].relative_to(ROOT)),
        models={REGISTRY[k][0]: REGISTRY[k][2] for k in keys},
        scaling=("PLS_v2와 Ridge: u를 학습 쌍의 열별 평균·SD로 같은 방식 표준화. PLS_v2는 추가로 출력 e도 열별 표준화 "
                 "(NIPALS 성분 추출에 영향). Ridge는 출력 열마다 독립이라 출력 스케일과 무관하므로 출력 표준화 안 함. "
                 "PLS_v1은 X=[A_in,h_in,h*], Y=A_target-A_in 모두 열별 표준화."),
        ridge_penalty="sklearn Ridge와 같은 정의: 제곱합 + alpha·||B||², 절편 벌점 없음",
        selection="학습 피험자 안 피험자 단위 LOSO (B2 기반 모델은 안쪽 fold마다 B2 재추정), 570 제외 흡광도 RMSE 최소",
        target_spectrum="측정 j의 실측 스펙트럼 (보간 아님). 목표 HbI = 측정 j의 실측 A(570)",
        near_definition="목표 측정의 |HbI-0.6| <= tol 인 쌍 (main 0.02, supplementary 0.03)",
        comparison="B1/B2/BiGRU = 저장 예측 재사용(재학습 없음), BiGRU = seed 0/1/2 지표 평균",
        metrics_units="a_rmse_31(570 포함), a_rmse(570 제외), peak_rmse: 흡광도 / r_rmse_pp: 반사율 %p(570 제외)",
        hyperparameters_tuned_on_test=False))
    jdump(cmp_out / "metrics.json", dict(run_id=run_id, scheme=scheme, recompute_matches_stored=match, checks=checks,
                                         selection={REGISTRY[k][0]: sel[k] for k in keys}, results=result, status="completed"))
    return dict(tab=tab, checks=pd.DataFrame(checks), match=match, sel=sel, deltas=dl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=list(SCHEMES), nargs="*", default=list(SCHEMES))
    ap.add_argument("--models", choices=list(REGISTRY), nargs="*", default=list(REGISTRY))
    a = ap.parse_args()
    data = load_spectra()
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    for sc in a.scheme:
        r = run(sc, data, a.models, run_id)
        if r is None:
            continue
        ck = r["checks"]
        print(f"\n== {sc} ==")
        print("570=h* 정확:", bool(ck.hbi570_exact.all()), "| 저장 지표 재현:", r["match"])
        print("Δh=0 출력-입력 최대차:", ck.groupby("model").identity_max_abs_diff.max().to_dict())
        if "b2_refit_vs_stored_max_abs" in ck:
            print("B2 재추정 vs 저장 최대차:", ck.b2_refit_vs_stored_max_abs.max())
        print(r["tab"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
