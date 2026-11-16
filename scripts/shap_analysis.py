"""모든 모델의 SHAP 값 (입력 그룹 단위, 정확한 Shapley 값; shap 패키지 없이 정의대로 계산).

실행 (code 폴더, conda 환경 skin_spectra):
    python -X utf8 scripts/shap_analysis.py            # results/shap/loso/ 에 저장 (이미 있으면 중단)

설계 (결과를 보기 전에 고정)
  - 분할: LOSO 10 fold (results/loso). fold마다 그 fold의 학습 모델로 테스트 피험자의 90쌍(i != j)을 설명 -> 900쌍.
  - 모델: B1, B2, BiGRU(seed 0/1/2 각각), PLS_v1, PLS_v2, Ridge.
      B1·B2·BiGRU = results/loso/models 의 저장 모델, PLS_v1·PLS_v2 = results/pls_v1|pls_v2/loso/models,
      Ridge = 같은 학습 피험자로 다시 학습 (결정적, 저장 계수와 같은지 확인).
  - 입력 = [A_in 31파장, h*] 을 6개 그룹(player)으로 묶음:
      h*, h_in(=A_in 570), A 400-460, A 470-510 (카로티노이드 피크 구간), A 520-560, A 580-700
  - 설명할 출력 (오차가 아니라 예측값):
      (1) 예측 변화량 D(λ) = A*(λ) - A_in(λ), 570 제외 30파장
      (2) Peak 변화량 dP = P(A*) - P(A_in),  P = A490 - (A470 + A510)/2
  - 값 함수 (interventional): v(S) = 평균_b f(x_S, b_~S),  b = 그 fold 학습 쌍 100개 (seed 0 무작위, 테스트 피험자 제외)
  - Shapley 값: 2^6 = 64개 조합 전부 계산 (근사 없음). 검사: base + Σφ = f(x).
  - 주의: 그룹을 서로 다른 측정에서 섞으면 실제로 없는 스펙트럼이 생길 수 있음 (interventional SHAP의 한계).
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # 프로젝트 루트 (scripts/ 의 상위)
sys.path.insert(0, str(ROOT))
from common.data_loader import FREE, HBI_INDEX, WL, load_spectra, make_pairs   # noqa: E402
from common.metrics import peak                                              # noqa: E402
from models.b2_pls import B2PLSModel                                          # noqa: E402
from models.b2_ridge import B2RidgeModel                                      # noqa: E402
from models.bigru import BiGRUModel                                           # noqa: E402
from models.hb_index import BinnedMeanModel                                   # noqa: E402
from models.linear import LinearModel                                         # noqa: E402
from models.pls import PLSModel                                               # noqa: E402

RES = ROOT / "results"
OUT = RES / "shap" / "loso"
N_BG, BG_SEED = 100, 0
H_STAR = 31                                         # 특징 벡터 z = [A_in(31), h*]
GROUPS = {
    "h*": [H_STAR],
    "h_in(570)": [HBI_INDEX],
    "A 400-460": list(np.flatnonzero((WL >= 400) & (WL <= 460))),
    "A 470-510": list(np.flatnonzero((WL >= 470) & (WL <= 510))),
    "A 520-560": list(np.flatnonzero((WL >= 520) & (WL <= 560))),
    "A 580-700": list(np.flatnonzero((WL >= 580) & (WL <= 700))),
}
GNAMES = list(GROUPS)
NG = len(GNAMES)
assert sorted(sum(GROUPS.values(), [])) == list(range(32))


def outputs(model, Z):
    """Z (n, 32) -> (n, 31): 570 제외 30파장 예측 변화량 + Peak 변화량."""
    A, T = Z[:, :31], Z[:, H_STAR]
    P = model.predict(A, T)
    return np.column_stack([(P - A)[:, FREE], peak(P) - peak(A)])


def shapley(model, X, BG):
    """정확한 그룹 Shapley 값. X (N,32), BG (B,32) -> phi (N, NG, 31), base (31,), fx (N, 31)."""
    N, B = len(X), len(BG)
    v = {}
    for S in itertools.product([0, 1], repeat=NG):
        mask = np.zeros(32, bool)
        for g, on in zip(GNAMES, S):
            if on:
                mask[GROUPS[g]] = True
        Z = np.where(mask[None, None, :], X[:, None, :], BG[None, :, :]).reshape(N * B, 32)
        v[S] = outputs(model, Z).reshape(N, B, -1).mean(1)
    phi = np.zeros((N, NG, v[(0,) * NG].shape[1]))
    for i in range(NG):
        for S in v:
            if S[i]:
                continue
            k = sum(S)
            w = math.factorial(k) * math.factorial(NG - k - 1) / math.factorial(NG)
            Si = tuple(1 if j == i else s for j, s in enumerate(S))
            phi[:, i] += w * (v[Si] - v[S])
    return phi, v[(0,) * NG].mean(0), v[(1,) * NG]


def main():
    if OUT.exists() and any(OUT.rglob("*.json")):
        sys.exit(f"[중단] {OUT} 에 이미 결과가 있음. 덮어쓰지 않음.")
    t0 = time.time()
    data = load_spectra()
    folds = json.loads((RES / "loso/config/loso_folds.json").read_text(encoding="utf-8"))["folds"]
    row = {(s, t): i for i, (s, t) in enumerate(zip(data.subject, data.trial))}
    recs, checks = [], []
    for f in folds:
        fid, s = f["fold_id"], f["test"][0]
        tr = data.select(f["train"])
        Xa, Ya, *_ = make_pairs(tr)
        rng = np.random.default_rng(BG_SEED)
        pick = rng.choice(len(Xa), N_BG, replace=False)
        BG = np.column_stack([Xa[pick], Ya[pick, HBI_INDEX]])
        ii = np.array([row[(s, p["input_trial"])] for p in f["test_pairs"]])
        jj = np.array([row[(s, p["target_trial"])] for p in f["test_pairs"]])
        X = np.column_stack([data.a[ii], data.hbi[jj]])

        ridge = B2RidgeModel().fit(tr)
        saved = json.loads((RES / f"ridge/loso/models/{fid}.json").read_text(encoding="utf-8"))
        checks.append(dict(fold=fid, ridge_refit_matches_saved=bool(np.allclose(ridge.coef, saved["coef"], rtol=1e-10, atol=1e-14))))
        models = {
            "B1": BinnedMeanModel.load(RES / f"loso/models/B1/{fid}.json"),
            "B2": LinearModel.load(RES / f"loso/models/B2/{fid}.json"),
            **{f"BiGRU_seed{k}": BiGRUModel.load(RES / f"loso/models/BiGRU/{fid}_seed_{k}.pt") for k in (0, 1, 2)},
            "PLS_v1": PLSModel.load(RES / f"pls_v1/loso/models/{fid}.json"),
            "PLS_v2": B2PLSModel.load(RES / f"pls_v2/loso/models/{fid}.json"),
            "Ridge": ridge,
        }
        for name, m in models.items():
            phi, base, fx = shapley(m, X, BG)
            err = float(np.abs(base[None] + phi.sum(1) - fx).max())
            checks.append(dict(fold=fid, model=name, efficiency_max_abs_err=err))
            for n in range(len(X)):
                for gi, g in enumerate(GNAMES):
                    recs.append(dict(fold=fid, subject=s, pair=n, model=name, group=g,
                                     mean_abs_spectral=float(np.abs(phi[n, gi, :30]).mean()),
                                     peak=float(phi[n, gi, 30]),
                                     **{f"wl{int(w)}": float(phi[n, gi, c]) for c, w in enumerate(WL[FREE])}))
        print(f"[{fid}] 완료 ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(recs)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "shap_values_per_pair.csv", index=False, encoding="utf-8-sig", float_format="%.6g")
    # 요약: 쌍별 |φ| 평균 (BiGRU는 seed별 계산 후 평균)
    df["model_base"] = df.model.str.replace(r"_seed\d", "", regex=True)
    df["abs_peak"] = df.peak.abs()
    per_seed = df.groupby(["model", "model_base", "group"])[["mean_abs_spectral", "abs_peak"]].mean().reset_index()
    summ = per_seed.groupby(["model_base", "group"])[["mean_abs_spectral", "abs_peak"]].mean()
    wl_cols = [c for c in df.columns if c.startswith("wl")]
    heat = df.groupby(["model", "model_base", "group"])[wl_cols].apply(lambda x: x.abs().mean()).groupby(["model_base", "group"]).mean()
    summ.to_csv(OUT / "shap_summary.csv", encoding="utf-8-sig", float_format="%.6g")
    heat.to_csv(OUT / "shap_by_wavelength.csv", encoding="utf-8-sig", float_format="%.6g")
    (OUT / "shap_summary.json").write_text(json.dumps(dict(
        design=__doc__, groups={g: [int(WL[i]) if i < 31 else "h*" for i in v] for g, v in GROUPS.items()},
        n_background=N_BG, background_seed=BG_SEED, n_pairs=int(df.pair.groupby(df.fold).nunique().sum()),
        checks=checks, max_efficiency_error=max(c.get("efficiency_max_abs_err", 0) for c in checks),
        bigru_seed_spread={g: per_seed[(per_seed.model_base == "BiGRU") & (per_seed.group == g)][["mean_abs_spectral", "abs_peak"]]
                           .agg(["min", "max"]).to_dict() for g in GNAMES},
        summary={m: summ.loc[m].to_dict(orient="index") for m in summ.index.get_level_values(0).unique()}),
        ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"저장: {OUT}  ({time.time() - t0:.0f}s)")


def plot():
    """results/shap/loso/shap_by_wavelength.csv -> 모델별 히트맵 (행 = 입력 그룹, 열 = 출력 파장, 값 = 평균 |SHAP|)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 8})
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#f7f9fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    h = pd.read_csv(OUT / "shap_by_wavelength.csv").set_index(["model_base", "group"])
    s = pd.read_csv(OUT / "shap_summary.csv").set_index(["model_base", "group"])
    models = ["B1", "B2", "BiGRU", "PLS_v1", "PLS_v2", "Ridge"]
    wl = [int(c[2:]) for c in h.columns]
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 13), gridspec_kw=dict(width_ratios=[30, 2.2], wspace=0.06, hspace=0.42))
    for r, m in enumerate(models):
        M = h.loc[m].reindex(GNAMES).to_numpy() * 1e3
        pk = s.loc[m].reindex(GNAMES)["abs_peak"].to_numpy()[:, None] * 1e3
        vmax = max(M.max(), 1e-9)
        ax = axes[r, 0]
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_yticks(range(NG)); ax.set_yticklabels(GNAMES)
        ax.set_xticks(range(len(wl))); ax.set_xticklabels(wl if r == len(models) - 1 else [], rotation=90, fontsize=6.5)
        if r == len(models) - 1:
            ax.set_xlabel("출력 파장 (nm, 570 제외)")
        for x in (6.5, 11.5, 16.5):           # 400-460 | 470-510 | 520-560 | 580-700 (570 제외)
            ax.axvline(x, color="white", lw=1.5)
        share = (s.loc[m].reindex(GNAMES)["mean_abs_spectral"] / s.loc[m]["mean_abs_spectral"].sum() * 100).round(1)
        ax.set_title(f"{m}   (그룹 비율 %: " + ", ".join(f"{g} {v:g}" for g, v in share.items()) + ")", loc="left",
                     fontsize=8, fontweight="bold")
        cb = fig.colorbar(im, ax=ax, pad=0.005, fraction=0.02); cb.ax.tick_params(labelsize=6); cb.set_label("x1e-3", fontsize=6)
        ax2 = axes[r, 1]
        ax2.imshow(pk, aspect="auto", cmap=cmap, vmin=0, vmax=max(pk.max(), 1e-9))
        for i, v in enumerate(pk[:, 0]):
            ax2.text(0, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                     color="white" if v > 0.6 * pk.max() else "#222222")
        ax2.set_xticks([0]); ax2.set_xticklabels(["Peak"] if r == len(models) - 1 else [], fontsize=7); ax2.set_yticks([])
        ax2.set_title("Peak", fontsize=7)
    fig.suptitle("모델별 SHAP: 입력 그룹이 예측 변화량 A*(λ)-A_in(λ)에 기여한 크기 (평균 |SHAP|, LOSO 900쌍, BiGRU = seed 3개 평균)\n"
                 "패널마다 색 척도가 다름 (모델 안에서 비교용).  PLS_v2·Ridge의 h_in과 스펙트럼 그룹은 서로 반대 부호로 크게 상쇄됨 "
                 "(상관 -0.92/-0.98) -> 개별 크기는 해석하지 말 것",
                 x=0.01, y=0.935, ha="left", fontsize=9, fontweight="bold")
    fig.savefig(OUT / "shap_heatmap.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print("그림:", OUT / "shap_heatmap.png")


if __name__ == "__main__":
    if "--figure" in sys.argv:
        plot()
    else:
        main()
        plot()
