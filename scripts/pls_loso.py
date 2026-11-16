"""예전 PLS(스펙트럼 변화 직접 예측, 옛 이름 v3) LOSO 평가 + results/loso(B1/B2/BiGRU)와 비교.

실행 (code 폴더, conda 환경 skin_spectra, results/loso 결과가 먼저 있어야 함):
    python scripts/pls_loso.py            # results/pls_v1/legacy_loso/ 에 저장 (현재는 pls_compare.py 사용) (기존 결과가 있으면 중단)

설계 (2026-11-14, 결과를 보기 전에 고정)
  - 모델: models/pls.py. X = [A_in, h_in, h*], Y = A_target - A_in, 570 nm = h*.
  - 평가: LOSO 10 fold (v2와 같은 fold, 같은 Test Pair 90개 = 같은 피험자 모든 i != j).
  - 성분 수: fold마다 Train 9명 안의 내부 LOSO로 1..10 중 MSE(570 nm 제외) 최소. Test 피험자는 선택에 쓰지 않음.
  - 지표: common/metrics.py (v1/v2와 같은 함수). 평가 단위 = 피험자.
  - 비교 기준: results/loso/predictions/loso_predictions.json 의 B1/B2/BiGRU 예측에 같은 지표 함수를 다시 적용
    (BiGRU 피험자 값 = seed별 지표 평균, loso.py와 같은 집계). v2 저장 지표와 일치하는지 검사.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # 프로젝트 루트 (scripts/ 의 상위)
sys.path.insert(0, str(ROOT))
from common import config as C                                          # noqa: E402
from common.data_loader import FREE, HBI_INDEX, WL, load_spectra        # noqa: E402
from common.metrics import consistency, group_metrics                   # noqa: E402
from models.pls import MAX_COMPONENTS, PLSModel                        # noqa: E402

V2 = ROOT / "results" / "loso"
OUT = ROOT / "results" / "pls_v1" / "legacy_loso"   # pls_compare.py --models pls_v1_k10 로 대체됨
TOLS = {"main": C.NEAR_TOL_MAIN, "supplementary": C.NEAR_TOL_SUPP}
MODELS = ["B1", "B2", "BiGRU", "PLS"]
COLORS = {"B1": "#eb6834", "B2": "#2a78d6", "BiGRU": "#1baf7a", "PLS": "#8a4fd0"}
GEN_METRICS = ["a_rmse", "r_rmse_pp", "peak_rmse", "peak_mae", "peak_bias"]


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jdump(p, obj, indent=1):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")


def style():
    plt.rcParams.update({
        "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "axes.edgecolor": "#7a7a7a", "xtick.color": "#7a7a7a", "ytick.color": "#7a7a7a",
        "xtick.labelcolor": "#444444", "ytick.labelcolor": "#444444", "axes.titlesize": 8.5,
        "legend.frameon": False, "legend.fontsize": 7})


def subject_metrics(P, Y, T):
    """피험자 하나: 일반 복원 + 0.6 근처 정확도(main/supp)."""
    r = dict(general=group_metrics(P, Y))
    for t, tol in TOLS.items():
        sel = np.abs(T - C.TARGET_HBI) <= tol
        r[f"near_{t}"] = group_metrics(P[sel], Y[sel]) if sel.any() else None
    return r


def main():
    argparse.ArgumentParser().parse_args()
    if OUT.exists() and any(OUT.rglob("*.json")):
        sys.exit(f"[중단] {OUT} 에 이미 결과가 있음. 덮어쓰지 않음.")
    if not (V2 / "predictions" / "loso_predictions.json").exists():
        sys.exit("[중단] v2 LOSO 결과가 없음. python scripts/loso.py --out-version loso 먼저 실행")
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    data = load_spectra()
    row = {(s, t): i for i, (s, t) in enumerate(zip(data.subject, data.trial))}
    folds = jload(V2 / "config" / "loso_folds.json")["folds"]
    v2pred = jload(V2 / "predictions" / "loso_predictions.json")

    jdump(OUT / "config" / "pls_config.json", dict(
        experiment_version="v3", run_id=run_id, scheme="Leave-One-Subject-Out (v2와 같은 fold)", folds="results/loso/config/loso_folds.json",
        model="PLS2 (NIPALS, numpy), X=[A_in(31), h_in, h*] 표준화, Y=A_target-A_in 표준화, A*(570)=h*",
        n_components_selection=f"fold마다 Train 9명 내부 LOSO, 1..{MAX_COMPONENTS}, 570 nm 제외 MSE 최소",
        training_pairs="P1_all_pairs (같은 피험자 모든 i != j)", target_hbi=C.TARGET_HBI, near_tolerances=TOLS,
        comparison="results/loso B1/B2/BiGRU 예측에 같은 지표 함수 재적용", hyperparameters_tuned_on_test=False))

    rows, cons_rows, fold_info, pred_g, pred_f = [], [], [], [], []
    for f in folds:
        fid, s = f["fold_id"], f["test"][0]
        tr = data.select(f["train"])
        m = PLSModel().fit(tr)
        m.save(OUT / "models" / "PLS" / f"{fid}.json")
        ii = np.array([row[(s, p["input_trial"])] for p in f["test_pairs"]])
        jj = np.array([row[(s, p["target_trial"])] for p in f["test_pairs"]])
        X, Y, T = data.a[ii], data.a[jj], data.hbi[jj]
        P = m.predict(X, T)
        assert np.array_equal(P[:, HBI_INDEX], T)
        te = data.select([s]); order = np.argsort(te.trial)
        F = m.predict(te.a[order], C.TARGET_HBI)
        fold_info.append(dict(fold_id=fid, test_subject=s, n_components=m.n_components, inner_cv_mean_mse=m.cv["mean_mse"],
                              identity_error_max_train=m.identity_error(tr.a), identity_error_max_test=m.identity_error(te.a)))
        print(f"[{fid}] PLS 성분 {m.n_components}", flush=True)
        rows.append(dict(model="PLS", seed=-1, subject=s, **subject_metrics(P, Y, T)))
        cons_rows.append(dict(model="PLS", seed=-1, subject=s, **consistency(F)))
        for n, p in enumerate(f["test_pairs"]):
            pred_g.append(dict(fold_id=fid, subject=s, input_trial=p["input_trial"], target_trial=p["target_trial"],
                               input_hbi=float(data.hbi[ii[n]]), target_hbi=float(T[n]), PLS=P[n].tolist()))
        for n, k in enumerate(order):
            pred_f.append(dict(fold_id=fid, subject=s, input_trial=int(te.trial[k]), target_hbi=C.TARGET_HBI, PLS=F[n].tolist()))

    # ---------- v2 모델: 같은 지표 함수 재적용 ----------
    g2 = pd.DataFrame(v2pred["general_reconstruction"])
    keys = sorted(g2.predictions.iloc[0].keys())
    for s, d in g2.groupby("subject"):
        Y = np.array(d.actual.tolist()); T = d.target_hbi.to_numpy()
        for k in keys:
            mn, seed = (k.split("_seed")[0], int(k.split("_seed")[1])) if "_seed" in k else (k, -1)
            P = np.array([p[k] for p in d.predictions])
            rows.append(dict(model=mn, seed=seed, subject=s, **subject_metrics(P, Y, T)))
    f2 = pd.DataFrame(v2pred["fixed_target_0p6"])
    for s, d in f2.groupby("subject"):
        d = d.sort_values("input_trial")
        for k in keys:
            mn, seed = (k.split("_seed")[0], int(k.split("_seed")[1])) if "_seed" in k else (k, -1)
            cons_rows.append(dict(model=mn, seed=seed, subject=s, **consistency(np.array([p[k] for p in d.predictions]))))

    # ---------- 피험자 표 (BiGRU = seed 평균) ----------
    def table(section, metric):
        recs = [dict(model=r["model"], seed=r["seed"], subject=r["subject"], v=r[section][metric])
                for r in rows if r[section] is not None]
        df = pd.DataFrame(recs).groupby(["model", "subject"]).v.mean().unstack("model")
        return df[[m for m in MODELS if m in df]]

    out_tab = {}
    for sec in ["general", "near_main", "near_supplementary"]:
        out_tab[sec] = {}
        for met in GEN_METRICS:
            t = table(sec, met)
            out_tab[sec][met] = dict(per_subject=t.round(8).to_dict(orient="index"),
                                     mean=t.mean().to_dict(), sd=t.std(ddof=1).to_dict(), min=t.min().to_dict(), max=t.max().to_dict(),
                                     n_subjects=int(len(t)),
                                     n_subjects_pls_better_than={m: int((t["PLS"] < t[m]).sum()) for m in t if m != "PLS"}
                                     if met != "peak_bias" else None)
    cdf = pd.DataFrame(cons_rows)
    cons = {met: cdf.groupby(["model", "subject"])[met].mean().unstack("model").round(8).to_dict(orient="index")
            for met in ("peak_sd", "spectral_sd") if met in cdf}

    # ---------- v2 저장값과 일치 검사 ----------
    chk = {}
    try:
        v2c = jload(V2 / "metrics" / "loso_model_comparison.json")["subject_wise"]
        for met in ("a_rmse", "peak_rmse"):
            mine = table("general", met)
            chk[met] = {mn: bool(np.allclose([mine.loc[s, mn] for s in v2c[met][mn]], list(v2c[met][mn].values()), rtol=1e-9))
                        for mn in ("B1", "B2", "BiGRU")}
    except Exception as e:   # noqa: BLE001
        chk = dict(error=repr(e))

    jdump(OUT / "metrics" / "v3_model_comparison.json", dict(
        experiment_version="v3", run_id=run_id, unit="absorbance (r_rmse_pp: %p)", aggregation="pair -> subject group 지표 -> 10명 평균/SD",
        pls_folds=fold_info, metrics=out_tab, consistency_fixed_0p6=cons, recompute_matches_v2=chk,
        note="B1/B2/BiGRU 값은 v2 예측에서 재계산. 0.6 근처(main)는 정답이 있는 피험자만 (n_subjects 참고). "
             "피험자 10명 비교이며 유의성 검정은 하지 않음.", status="completed"))
    jdump(OUT / "predictions" / "pls_predictions.json", dict(general_reconstruction=pred_g, fixed_target_0p6=pred_f,
                                                             wavelengths_nm=WL.tolist()), indent=None)
    make_figures(out_tab, fold_info)
    print("저장:", OUT)
    print("v2 재계산 일치:", chk)


def make_figures(tab, fold_info):
    style()
    fd = OUT / "figures"; fd.mkdir(parents=True, exist_ok=True)
    panels = [("general", "a_rmse", "일반 복원: 흡광도 RMSE"), ("general", "peak_rmse", "일반 복원: 490 nm Peak RMSE"),
              ("near_main", "a_rmse", "0.6 근처 정답: 흡광도 RMSE"), ("near_main", "peak_rmse", "0.6 근처 정답: Peak RMSE")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, (sec, met, title) in zip(axes, panels):
        ps = tab[sec][met]["per_subject"]
        subs = sorted(ps)
        jit = np.linspace(-0.15, 0.15, len(subs)) if len(subs) > 1 else np.zeros(1)
        for x, mn in enumerate(MODELS):
            v = np.array([ps[s][mn] for s in subs]) * 1e3
            ax.scatter(x + jit, v, s=14, color=COLORS[mn], alpha=0.8, lw=0)
            ax.hlines(v.mean(), x - 0.3, x + 0.3, color="#222222", lw=1.2)
        for n, s in enumerate(subs):   # 같은 피험자 연결선 (점과 같은 x 위치)
            ax.plot(np.arange(len(MODELS)) + jit[n], [ps[s][mn] * 1e3 for mn in MODELS], color="#bbbbbb", lw=0.4, zorder=0)
        ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS)
        ax.set_title(f"{title} (n={len(subs)})", loc="left", fontweight="bold"); ax.set_ylabel("x1e-3")
        ax.grid(alpha=0.25, lw=0.5, axis="y")
    fig.suptitle("LOSO 모델 비교 (v2 B1/B2/BiGRU + v3 PLS; 점: 피험자, 가로선: 평균, 회색선: 같은 피험자)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(fd / "v3_model_comparison.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    for fi in fold_info:
        c = np.array(fi["inner_cv_mean_mse"]) * 1e6
        ax.plot(range(1, len(c) + 1), c, lw=0.8, alpha=0.7, label=f"{fi['test_subject']} (k={fi['n_components']})")
    ax.set_xlabel("PLS 성분 수"); ax.set_ylabel("내부 LOSO MSE x1e-6"); ax.legend(fontsize=6, ncol=2)
    ax.set_title("fold별 성분 수 선택 (Train 9명 내부 LOSO)", loc="left", fontweight="bold"); ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(); fig.savefig(fd / "v3_pls_components.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    main()
