"""BiGRU 하이퍼파라미터 튜닝 (학습 피험자 안 Leave-One-Subject-Out) + 선택된 설정으로 재학습.

실행 (code 폴더, conda 환경 skin_spectra):
    python -X utf8 scripts/tune_bigru.py --time-only                                  # 300 epoch 1회 시간 측정 + 일치 검사
    python -X utf8 scripts/tune_bigru.py --verify-parallel                            # 분할 1개: 단일 실행과 병렬 실행 결과 비교
    python -X utf8 scripts/tune_bigru.py --unit Development10_LeaveOneSubjectOut --workers 8
    python -X utf8 scripts/tune_bigru.py --unit Development10_Holdout73 --workers 8
    python -X utf8 scripts/tune_bigru.py --unit Development10_AllSubjects --workers 8  # 선택 결과만 저장 (ExternalTest6용)
    python -X utf8 scripts/tune_bigru.py --unit <...> --peak-constrained               # Peak-constrained selection 재학습

설계 (결과를 보기 전에 고정)
  - 격자: hidden {4, 8, 16} x slope_l2 {0, 1e-3, 1e-2, 1e-1} x epoch {0, 25, 50, 100, 150, 200, 300} = 84 후보.
    epoch는 한 번의 300 epoch 학습에서 해당 시점의 검증 값을 기록. 고정: lr 3e-3, clip 1.0, Adam, full batch, 선택용 seed 0.
  - 내부 검증: 학습 피험자를 1명씩 빼는 Leave-One-Subject-Out. 안쪽 fold마다 표준화 평균/표준편차와 Global Linear 기울기를
    안쪽 학습 피험자로 다시 계산 (BiGRUModel.fit 과 같은 코드 경로).
  - 선택 기준: 내부 검증 쌍 전체(안쪽 fold 전부를 합침)의 absorbance RMSE (570 nm 제외). Peak RMSE도 같은 방식으로 저장.
    검증 예측은 BiGRUModel.predict (float64 조립).
  - 동률(값이 정확히 같을 때만): 작은 hidden, 큰 slope_l2, 적은 epoch 순.
  - Peak-constrained selection: 내부 검증 Peak RMSE <= epoch 0(Global Linear)의 내부 검증 Peak RMSE 인 후보 중 absorbance RMSE 최소.
  - 바깥 분할의 테스트 피험자는 선택에 쓰지 않는다. External cohort는 이 스크립트에서 읽지 않는다.
  - 선택된 (hidden, slope_l2, epoch)으로 학습 피험자 전체에서 seed 0, 1, 2 재학습. epoch 0이면 Global Linear와 같은 예측
    (BiGRUModel(epochs=0) 으로 저장해 뒤의 분석이 같은 코드 경로를 쓰게 함).

저장: results/bigru_tuned/<unit>/  selection.json, config.json, models/, predictions.json, metrics.json, selection_parts/
      Peak-constrained: results/bigru_tuned/<unit>/peak_constrained/  (같은 구성)
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import multiprocessing as mp
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))
from common import config as C                                                   # noqa: E402
from common.data_loader import FREE, HBI_INDEX, WL, Spectra, load_spectra, make_pairs   # noqa: E402
from common.metrics import group_metrics, peak                                   # noqa: E402
from models.bigru import BiGRUModel, ResidualBiGRU, _objective, _seed_everything  # noqa: E402
from models.linear import LinearModel, apply_slope, fit_slope                    # noqa: E402
from final_utils import DEV_CSV, RES, jdump, jload, load_splits, test_pairs       # noqa: E402

GRID_HIDDEN = (4, 8, 16)
GRID_SLOPE_L2 = (0.0, 1e-3, 1e-2, 1e-1)
GRID_EPOCH = (0, 25, 50, 100, 150, 200, 300)
LR, CLIP, SELECT_SEED = 3e-3, 1.0, 0
SEEDS = tuple(C.BIGRU_SEEDS)
UNITS = ("Development10_Holdout73", "Development10_LeaveOneSubjectOut", "Development10_AllSubjects")
OUT = RES / "bigru_tuned"


def make_cfg(hidden, slope_l2):
    return dict(hidden=int(hidden), lr=LR, slope_l2=float(slope_l2), clip=CLIP)


def train_with_checkpoints(train: Spectra, hidden, slope_l2, seed, checkpoints, on_checkpoint=None):
    """BiGRUModel.fit 과 같은 학습 루프. checkpoints의 epoch마다 on_checkpoint(model) 결과를 기록한다."""
    cps = set(checkpoints)
    m = BiGRUModel(seed=seed, epochs=max(cps), cfg=make_cfg(hidden, slope_l2))
    _seed_everything(m.seed)
    m.b2 = fit_slope(train)
    m.train_subjects = train.subjects
    mean, std = train.a.mean(0), np.maximum(train.a.std(0), 1e-6)
    m.net = ResidualBiGRU(mean, std, m.b2, m.cfg["hidden"])
    X, Y, W = make_pairs(train)[:3]
    batch = tuple(torch.tensor(v, dtype=torch.float32) for v in (X, Y, W))
    opt = torch.optim.Adam(m.net.parameters(), lr=m.cfg["lr"])
    m.history, out = [], {}
    if 0 in cps and on_checkpoint is not None:
        out[0] = on_checkpoint(m)
    for ep in range(1, m.epochs + 1):                        # full batch
        m.net.train(); opt.zero_grad(set_to_none=True)
        loss, mse = _objective(m.net, batch, m.cfg["slope_l2"])
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(m.net.parameters(), m.cfg["clip"])
        opt.step()
        m.history.append((ep, loss.item(), mse.item(), grad_norm.item()))
        if ep in cps and on_checkpoint is not None:
            out[ep] = on_checkpoint(m)
    return m, out


def validation_sums(predict, va: Spectra) -> dict:
    """검증 피험자의 모든 순서쌍에 대한 제곱오차 합 (570 제외 흡광도, Peak)."""
    X, Y = make_pairs(va)[:2]
    P = predict(X, Y[:, HBI_INDEX])
    return dict(sse_a=float(np.sum((P - Y)[:, FREE] ** 2)), n_a=int(Y[:, FREE].size),
                sse_p=float(np.sum((peak(P) - peak(Y)) ** 2)), n_p=int(len(Y)))


# ---------------------------------------------------------------------------
# 병렬 작업 (바깥 분할의 학습 피험자, 검증 피험자, hidden, slope_l2) -> epoch별 검증 제곱오차 합
# ---------------------------------------------------------------------------
_DATA = None


def _init_worker():
    global _DATA
    torch.set_num_threads(1)
    _DATA = load_spectra(DEV_CSV)


def inner_task(task):
    train_subjects, val, hidden, slope_l2 = task
    tr_in = _DATA.select([s for s in train_subjects if s != val])
    va = _DATA.select([val])
    assert val not in tr_in.subjects and len(tr_in.subjects) == len(train_subjects) - 1
    t = time.time()
    _, out = train_with_checkpoints(tr_in, hidden, slope_l2, SELECT_SEED, GRID_EPOCH,
                                    lambda m: validation_sums(m.predict, va))
    return dict(val=val, hidden=hidden, slope_l2=slope_l2, seconds=time.time() - t,
                by_epoch={int(e): v for e, v in out.items()})


def run_tasks(tasks, workers):
    if workers <= 1:
        _init_worker()
        return [inner_task(t) for t in tasks]
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        return pool.map(inner_task, tasks, chunksize=1)


def tune_split(data: Spectra, train_subjects, workers) -> dict:
    """한 바깥 분할의 내부 검증 -> 후보 84개의 검증 값과 선택 결과."""
    tasks = [(tuple(train_subjects), v, h, l2) for v in train_subjects for h in GRID_HIDDEN for l2 in GRID_SLOPE_L2]
    res = run_tasks(tasks, workers)
    # Global Linear의 내부 검증 값 (epoch 0 검사 기준)
    gl = dict(sse_a=0.0, n_a=0, sse_p=0.0, n_p=0)
    gl_by_val = {}
    for v in train_subjects:
        sl = fit_slope(data.select([s for s in train_subjects if s != v]))
        r = validation_sums(lambda X, T: apply_slope(X, T, sl), data.select([v]))
        gl_by_val[v] = r
        for k in gl:
            gl[k] += r[k]
    ep0_equal = all(r["by_epoch"][0] == gl_by_val[r["val"]] for r in res)
    if not ep0_equal:
        bad = max(abs(r["by_epoch"][0]["sse_a"] - gl_by_val[r["val"]]["sse_a"]) for r in res)
        raise RuntimeError(f"epoch 0 검증 값이 Global Linear와 다름 (제곱오차 합 최대 차이 {bad:.3e})")
    cands = []
    for h, l2, e in itertools.product(GRID_HIDDEN, GRID_SLOPE_L2, GRID_EPOCH):
        rs = [r for r in res if r["hidden"] == h and r["slope_l2"] == l2]
        assert len(rs) == len(train_subjects)
        sa, na = sum(r["by_epoch"][e]["sse_a"] for r in rs), sum(r["by_epoch"][e]["n_a"] for r in rs)
        sp, npk = sum(r["by_epoch"][e]["sse_p"] for r in rs), sum(r["by_epoch"][e]["n_p"] for r in rs)
        cands.append(dict(hidden=h, slope_l2=l2, epoch=e, val_a_rmse=float(np.sqrt(sa / na)), val_peak_rmse=float(np.sqrt(sp / npk))))
    gl_val = dict(val_a_rmse=float(np.sqrt(gl["sse_a"] / gl["n_a"])), val_peak_rmse=float(np.sqrt(gl["sse_p"] / gl["n_p"])))
    ep0 = [c for c in cands if c["epoch"] == 0]
    assert all(c["val_a_rmse"] == ep0[0]["val_a_rmse"] and c["val_peak_rmse"] == ep0[0]["val_peak_rmse"] for c in ep0)
    assert ep0[0]["val_a_rmse"] == gl_val["val_a_rmse"]
    return dict(train=list(train_subjects), n_inner_folds=len(train_subjects), candidates=cands, global_linear_validation=gl_val,
                epoch0_equals_global_linear=True, chosen=select(cands), peak_constrained_chosen=select_peak_constrained(cands),
                inner_train_seconds=float(sum(r["seconds"] for r in res)))


def _key(c):                                    # 동률: 작은 hidden, 큰 slope_l2, 적은 epoch
    return (c["val_a_rmse"], c["hidden"], -c["slope_l2"], c["epoch"])


def _describe(c):
    c = dict(c)
    c["at_grid_boundary"] = dict(hidden_min=c["hidden"] == min(GRID_HIDDEN), hidden_max=c["hidden"] == max(GRID_HIDDEN),
                                 slope_l2_min=c["slope_l2"] == min(GRID_SLOPE_L2), slope_l2_max=c["slope_l2"] == max(GRID_SLOPE_L2),
                                 epoch_max=c["epoch"] == max(GRID_EPOCH))
    c["epoch0_global_linear"] = c["epoch"] == 0
    return c


def select(cands):
    return _describe(min(cands, key=_key))


def select_peak_constrained(cands):
    limit = next(c["val_peak_rmse"] for c in cands if c["epoch"] == 0)
    ok = [c for c in cands if c["val_peak_rmse"] <= limit]
    out = _describe(min(ok, key=_key))
    out["peak_limit_epoch0"] = limit
    out["n_candidates_passing"] = len(ok)
    return out


# ---------------------------------------------------------------------------
def unit_splits(unit):
    if unit == "Development10_AllSubjects":
        return [dict(split_id="all_subjects", train=load_spectra(DEV_CSV).subjects, test=[])]
    return load_splits(unit)


def refit_and_predict(unit, data, splits, chosen_by_split, out: Path, label: str):
    """선택된 설정으로 seed 0, 1, 2 재학습 + 테스트 예측과 지표."""
    pred_g, pred_f, rows, checks = [], [], [], []
    for sp in splits:
        sid, ch = sp["split_id"], chosen_by_split[sp["split_id"]]
        tr, te = data.select(sp["train"]), data.select(sp["test"])
        assert not set(tr.subjects) & set(te.subjects)
        ii, jj, subj = test_pairs(unit, sid, data)
        X, Y, T = data.a[ii], data.a[jj], data.hbi[jj]
        order = np.lexsort((te.trial, te.subject))
        A0 = te.a[order]
        gl = LinearModel().fit(tr)
        P, F = {}, {}
        for seed in SEEDS:
            m = BiGRUModel(seed=seed, epochs=ch["epoch"], cfg=make_cfg(ch["hidden"], ch["slope_l2"])).fit(tr)
            m.save(out / "models" / f"{sid}_seed_{seed}.pt")
            P[seed], F[seed] = m.predict(X, T), m.predict(A0, C.TARGET_HBI)
            ck = dict(split_id=sid, seed=seed, hbi570_exact=bool(np.array_equal(P[seed][:, HBI_INDEX], T)),
                      finite=bool(np.isfinite(P[seed]).all() and np.isfinite(F[seed]).all()),
                      slope_equals_global_linear=bool(np.array_equal(m.b2, gl.slope)))
            if ch["epoch"] == 0:
                ck["epoch0_equals_global_linear"] = bool(np.array_equal(P[seed], gl.predict(X, T)))
            checks.append(ck)
            for s in te.subjects:
                sm = subj == s
                rows.append(dict(split_id=sid, subject=s, seed=seed, **group_metrics(P[seed][sm], Y[sm])))
        for n in range(len(ii)):
            pred_g.append(dict(split_id=sid, subject=subj[n], input_trial=int(data.trial[ii[n]]), target_trial=int(data.trial[jj[n]]),
                               target_hbi=float(T[n]), predictions={f"seed{s}": P[s][n].tolist() for s in SEEDS}))
        for n, k in enumerate(order):
            pred_f.append(dict(split_id=sid, subject=te.subject[k], input_trial=int(te.trial[k]), input_hbi=float(te.hbi[k]),
                               target_hbi=C.TARGET_HBI, predictions={f"seed{s}": F[s][n].tolist() for s in SEEDS}))
        print(f"[{label} {sid}] hidden {ch['hidden']}, slope_l2 {ch['slope_l2']:g}, epoch {ch['epoch']} -> 재학습 완료", flush=True)
    if not all(all(v for k, v in c.items() if isinstance(v, bool)) for c in checks):
        raise RuntimeError("재학습 검사 실패: " + str([c for c in checks if not all(v for v in c.values() if isinstance(v, bool))][:3]))
    jdump(out / "predictions.json", dict(wavelengths_nm=WL.tolist(), model=label, general_reconstruction=pred_g,
                                         fixed_target_0p6=pred_f), indent=None)
    df = pd.DataFrame(rows)
    jdump(out / "metrics.json", dict(model=label, aggregation="피험자별 group 지표, seed별", checks=checks,
                                     subject_seed=df.to_dict(orient="records")))
    return df


def write_config(out, unit, extra):
    jdump(out / "config.json", dict(
        unit=unit, created=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        grid=dict(hidden=list(GRID_HIDDEN), slope_l2=list(GRID_SLOPE_L2), epoch=list(GRID_EPOCH)),
        fixed=dict(lr=LR, clip=CLIP, optimizer="Adam", batch="full", selection_seed=SELECT_SEED, refit_seeds=list(SEEDS),
                   training_pairs="같은 피험자 안의 모든 순서쌍 i != j"),
        selection=dict(criterion="내부 검증 쌍 전체의 absorbance RMSE (570 nm 제외)", inner="학습 피험자 안 Leave-One-Subject-Out",
                       tie_break="값이 정확히 같을 때만: 작은 hidden, 큰 slope_l2, 적은 epoch",
                       validation_prediction="BiGRUModel.predict (float64 조립)"),
        data=str(DEV_CSV.relative_to(ROOT)), external_cohort_used=False, device="cpu", torch_threads=1,
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__),
        **extra))


def run_unit(unit, workers, only=None):
    out = OUT / unit
    if (out / "selection.json").exists():
        sys.exit(f"[중단] {out / 'selection.json'} 이 이미 있음. 덮어쓰지 않음.")
    t0 = time.time()
    data = load_spectra(DEV_CSV)
    splits = unit_splits(unit)
    if only:
        splits = [sp for sp in splits if sp["split_id"] in only]
    sel = []
    for sp in splits:
        part = out / "selection_parts" / f"{sp['split_id']}.json"
        if part.exists():                                   # 중단 후 재개: 이미 끝난 분할은 다시 계산하지 않음
            r = jload(part)
        else:
            r = dict(split_id=sp["split_id"], test=sp["test"], **tune_split(data, sp["train"], workers))
            jdump(part, r)
        assert r["train"] == sp["train"] and not set(r["train"]) & set(sp["test"])
        sel.append(r)
        ch = r["chosen"]
        print(f"[{unit} {sp['split_id']}] 선택: hidden {ch['hidden']}, slope_l2 {ch['slope_l2']:g}, epoch {ch['epoch']} "
              f"(검증 RMSE {ch['val_a_rmse'] * 1e3:.3f}, Global Linear {r['global_linear_validation']['val_a_rmse'] * 1e3:.3f}) "
              f"[{time.time() - t0:.0f}s]", flush=True)
    if only:
        print("일부 분할만 실행함 (selection.json 은 쓰지 않음)")
        return
    jdump(out / "selection.json", dict(unit=unit, n_candidates=len(sel[0]["candidates"]), splits=sel,
                                       tuning_seconds=time.time() - t0))
    if unit != "Development10_AllSubjects":
        refit_and_predict(unit, data, splits, {r["split_id"]: r["chosen"] for r in sel}, out, "BiGRU")
    write_config(out, unit, dict(n_inner_trainings=sum(len(r["train"]) for r in sel) * len(GRID_HIDDEN) * len(GRID_SLOPE_L2),
                                 elapsed_seconds=time.time() - t0,
                                 note="Development10_AllSubjects 는 선택 결과만 저장 (ExternalTest6용, Phase 6 전까지 적용하지 않음)"
                                 if unit == "Development10_AllSubjects" else None))
    print(f"완료: {out} ({time.time() - t0:.0f}s)")


def run_peak_constrained(unit):
    base, out = OUT / unit, OUT / unit / "peak_constrained"
    if (out / "selection.json").exists():
        sys.exit(f"[중단] {out / 'selection.json'} 이 이미 있음. 덮어쓰지 않음.")
    sel = jload(base / "selection.json")["splits"]
    data = load_spectra(DEV_CSV)
    chosen = {r["split_id"]: select_peak_constrained(r["candidates"]) for r in sel}
    assert all(chosen[r["split_id"]] == r["peak_constrained_chosen"] for r in sel)
    jdump(out / "selection.json", dict(
        unit=unit, rule="내부 검증 Peak RMSE <= epoch 0(Global Linear)의 내부 검증 Peak RMSE 인 후보 중 내부 검증 absorbance RMSE 최소. "
                        "추가 학습 없이 selection.json 의 기록으로 선택.",
        splits=[dict(split_id=r["split_id"], train=r["train"], test=r["test"], chosen=chosen[r["split_id"]],
                     unconstrained_chosen=r["chosen"]) for r in sel]))
    if unit != "Development10_AllSubjects":
        refit_and_predict(unit, data, unit_splits(unit), chosen, out, "BiGRU (Peak-constrained selection)")
    write_config(out, unit, dict(variant="Peak-constrained selection (보조 분석, 주 결과를 대체하지 않음)"))
    for r in sel:
        c = chosen[r["split_id"]]
        print(f"[{unit} {r['split_id']}] Peak-constrained: hidden {c['hidden']}, slope_l2 {c['slope_l2']:g}, epoch {c['epoch']} "
              f"(통과 후보 {c['n_candidates_passing']}/84)")


def history_check(data, subjects, hidden, slope_l2, epochs=300):
    """새 루프의 epoch별 학습 이력이 BiGRUModel.fit 의 history 와 완전히 같은지 (다르면 최대 절대 차이와 함께 멈춤)."""
    tr = data.select(subjects)
    t = time.time()
    m, _ = train_with_checkpoints(tr, hidden, slope_l2, SELECT_SEED, [epochs])
    secs = time.time() - t
    ref = BiGRUModel(seed=SELECT_SEED, epochs=epochs, cfg=make_cfg(hidden, slope_l2)).fit(tr)
    a, b = np.array(m.history), np.array(ref.history)
    if not np.array_equal(a, b):
        raise RuntimeError(f"튜닝 루프와 BiGRUModel.fit 의 history 가 다름: 최대 절대 차이 {np.abs(a - b).max():.3e}")
    same_w = all(torch.equal(x, y) for x, y in zip(m.net.state_dict().values(), ref.net.state_dict().values()))
    if not same_w:
        raise RuntimeError("튜닝 루프와 BiGRUModel.fit 의 최종 가중치가 다름")
    return secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", choices=UNITS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--splits", nargs="*", default=None, help="일부 분할만 (검사용, selection.json 은 쓰지 않음)")
    ap.add_argument("--time-only", action="store_true")
    ap.add_argument("--verify-parallel", action="store_true")
    ap.add_argument("--peak-constrained", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(1)
    data = load_spectra(DEV_CSV)
    if a.time_only:
        subs = data.subjects
        for h, l2, n in ((16, 1e-3, 9), (16, 0.0, 8), (4, 1e-1, 6)):
            secs = history_check(data, subs[:n], h, l2)
            print(f"hidden {h}, slope_l2 {l2:g}, 학습 피험자 {n}명, 300 epoch: {secs:.1f}s, history·가중치 완전 일치")
        return
    if a.verify_parallel:
        sp = load_splits("Development10_Holdout73")[0]
        tasks = [(tuple(sp["train"]), v, h, l2) for v in sp["train"][:2] for h in (4, 16) for l2 in (0.0, 1e-2)]
        strip = lambda rs: [{k: v for k, v in r.items() if k != "seconds"} for r in rs]   # noqa: E731
        s1, s2 = strip(run_tasks(tasks, 1)), strip(run_tasks(tasks, max(2, a.workers)))
        print("단일 실행과 병렬 실행 결과 완전 일치:", s1 == s2)
        if s1 != s2:
            sys.exit("[중단] 병렬 실행 결과가 단일 실행과 다름")
        return
    if a.unit is None:
        ap.error("--unit 이 필요함")
    if a.peak_constrained:
        run_peak_constrained(a.unit)
    else:
        run_unit(a.unit, a.workers, a.splits)


if __name__ == "__main__":
    main()
