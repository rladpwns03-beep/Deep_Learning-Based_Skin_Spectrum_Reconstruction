"""Recurrent-cell ablation under the YejunKim protocol (LOSO 10 x 3 seeds, inner 7/2 epoch selection).

Faithful to YejunKim/code (spectra.py, model.py, train.py): same pairs, subject weights, B1 fit,
features, zero-initialised head, slope_l2 penalty, Adam lr 3e-3, full batch, max 150 / patience 20,
clip 1.0, epoch-0 (B1) candidate, refit on 9 subjects. Only the sequence encoder changes.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn

WL = np.arange(400, 701, 10)
HBI = int(np.flatnonzero(WL == 570)[0])
FREE = WL != 570
CFG = dict(seeds=[17, 29, 43], split_seed=20261027, inner_val=2, lr=3e-3, slope_l2=1e-3,
           max_epochs=150, patience=20, min_delta=1e-8, clip=1.0, target=0.6)


def load(data_dir):
    A, S, T = [], [], []
    for p in sorted(Path(data_dir).glob("S*.csv")):
        f = pd.read_csv(p, encoding="cp949")
        r = f[[f"{w}nm" for w in WL]].to_numpy(float) / 100
        A.append(-np.log10(r)); S += [p.stem] * len(r); T += f.iloc[:, 0].astype(int).tolist()
    return np.vstack(A), np.array(S), np.array(T)


def peak(a):
    return a[..., 9] - (a[..., 7] + a[..., 11]) / 2


def pairs(a, s):
    subs = sorted(set(s)); X, Y, W = [], [], []
    for sid in subs:
        ix = np.flatnonzero(s == sid); n = len(ix)
        for i in ix:
            for j in ix:
                if i != j:
                    X.append(a[i]); Y.append(a[j]); W.append(1 / (len(subs) * n * (n - 1)))
    return np.asarray(X), np.asarray(Y), np.asarray(W)


def fit_pre(a, s):
    ac, h = a.copy(), a[:, HBI].copy()
    for sid in set(s):
        m = s == sid; ac[m] -= ac[m].mean(0); h[m] -= h[m].mean()
    slope = h @ ac / (h @ h); slope[HBI] = 1.0
    return dict(mean=a.mean(0), std=np.maximum(a.std(0), 1e-6), b1=slope)


class Residual(nn.Module):
    def __init__(self, pre, cell, hidden, bidirectional):
        super().__init__()
        self.register_buffer("mean", torch.tensor(pre["mean"], dtype=torch.float32))
        self.register_buffer("std", torch.tensor(pre["std"], dtype=torch.float32))
        self.register_buffer("b1", torch.tensor(pre["b1"], dtype=torch.float32))
        self.register_buffer("pos", torch.linspace(-1, 1, len(WL)))
        self.register_buffer("free", torch.tensor(FREE))
        self.cell = cell
        if cell == "MLP":   # pointwise, no wavelength context
            self.enc = nn.Sequential(nn.Linear(4, hidden), nn.Tanh()); out = hidden
        else:
            rnn = {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}[cell]
            self.enc = rnn(4, hidden, num_layers=1, bidirectional=bidirectional, batch_first=True)
            out = hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(out, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, a, target):
        h = a[:, HBI]
        target = torch.as_tensor(target, dtype=a.dtype).expand(len(a))
        x = torch.stack(((a - self.mean) / self.std, self.pos.expand_as(a),
                         h[:, None].expand_as(a), target[:, None].expand_as(a)), dim=-1)
        z = self.enc(x) if self.cell == "MLP" else self.enc(x)[0]
        r = self.head(z).squeeze(-1) * self.free
        out = a + (target - h)[:, None] * (self.b1 + r)
        out = torch.where(self.free[None, :], out, target[:, None])
        return out, r


def predict(model, pre, a, target):
    """float64 assembly as in YejunKim predict(); returns prediction and residual slope."""
    model.eval()
    target = np.broadcast_to(np.asarray(target, float), (len(a),)).copy()
    with torch.no_grad():
        _, r = model(torch.tensor(a, dtype=torch.float32), torch.tensor(target, dtype=torch.float32))
    r = r.numpy().astype(np.float64)
    out = a + (target - a[:, HBI])[:, None] * (pre["b1"] + r)
    out[:, HBI] = target
    return out, r


def b1_pred(a, target, pre):
    target = np.broadcast_to(np.asarray(target, float), (len(a),))
    out = a + (target - a[:, HBI])[:, None] * pre["b1"]; out[:, HBI] = target
    return out


def objective(model, batch, l2):
    a, y, w = batch
    out, r = model(a, y[:, HBI])
    mse = (w * ((out[:, model.free] - y[:, model.free]) ** 2).mean(1)).sum()
    pen = (w * r[:, model.free].square().mean(1)).sum()
    return mse + l2 * pen, mse


def tens(a, s):
    X, Y, W = pairs(a, s)
    return tuple(torch.tensor(v, dtype=torch.float32) for v in (X, Y, W))


def fit(a, s, spec, seed, val=None, epochs=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    pre = fit_pre(a, s)
    model = Residual(pre, spec["cell"], spec["hidden"], spec["bi"])
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    batch = tens(a, s); vb = tens(*val) if val is not None else None
    best, best_ep, stale = float("inf"), 0, 0
    maximum = CFG["max_epochs"] if val is not None else int(epochs)
    for ep in range(maximum + 1):
        if ep:
            model.train(); opt.zero_grad(set_to_none=True)
            loss, _ = objective(model, batch, CFG["slope_l2"]); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"]); opt.step()
        if val is not None:
            model.eval()
            with torch.no_grad():
                v = float(objective(model, vb, 0)[1])
            if v < best - CFG["min_delta"]:
                best, best_ep, stale = v, ep, 0
            else:
                stale += 1
            if stale >= CFG["patience"]:
                break
    return model, pre, (best_ep if val is not None else maximum)


def run(spec, data_dir, out):
    a, s, t = load(data_dir); subs = sorted(set(s))
    rows, target_rows, resid_rows = [], [], []
    nparam = None
    for fold, held in enumerate(subs):
        outer = [x for x in subs if x != held]
        rng = np.random.default_rng(CFG["split_seed"] + fold)
        ival = sorted(rng.choice(outer, CFG["inner_val"], replace=False).tolist())
        itr = sorted(set(outer) - set(ival))
        m_in, m_va, m_out, m_te = np.isin(s, itr), np.isin(s, ival), np.isin(s, outer), s == held
        for seed in CFG["seeds"]:
            _, _, ep = fit(a[m_in], s[m_in], spec, seed, val=(a[m_va], s[m_va]))
            model, pre, _ = fit(a[m_out], s[m_out], spec, seed, epochs=ep)
            nparam = sum(p.numel() for p in model.parameters())
            X, Y, _ = pairs(a[m_te], s[m_te])
            tgt = Y[:, HBI]; dh = np.abs(tgt - X[:, HBI])
            preds = {"B1": b1_pred(X, tgt, pre), spec["name"]: predict(model, pre, X, tgt)[0]}
            q = np.quantile(dh, 2 / 3)
            for meth, p in preds.items():
                ae = p - Y; re = 100 * (10 ** -p - 10 ** -Y); pe = peak(p) - peak(Y)
                top = dh >= q
                rows.append(dict(model=spec["name"], method=meth, subject=held, seed=seed, epoch=ep,
                                 r_rmse_pp=np.sqrt(np.mean(re[:, FREE] ** 2)),
                                 a_rmse=np.sqrt(np.mean(ae[:, FREE] ** 2)),
                                 a_rmse_top=np.sqrt(np.mean(ae[top][:, FREE] ** 2)),
                                 peak_rmse=np.sqrt(np.mean(pe ** 2)), peak_mae=np.mean(np.abs(pe))))
            at = a[m_te]
            p06, r06 = predict(model, pre, at, CFG["target"])
            for meth, p in {"Raw": at, "B1": b1_pred(at, CFG["target"], pre), spec["name"]: p06}.items():
                target_rows.append(dict(model=spec["name"], method=meth, subject=held, seed=seed,
                                        peak_sd=peak(p).std(ddof=1)))
            for k in range(len(at)):
                resid_rows.append(dict(model=spec["name"], subject=held, seed=seed, trial=int(t[m_te][k]),
                                       **{f"r{w}": r06[k, i] for i, w in enumerate(WL)},
                                       **{f"b1_{w}": pre["b1"][i] for i, w in enumerate(WL)}))
        print(spec["name"], held, "done", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{spec['name']}_pairs.csv", index=False)
    pd.DataFrame(target_rows).to_csv(out / f"{spec['name']}_target.csv", index=False)
    pd.DataFrame(resid_rows).to_csv(out / f"{spec['name']}_resid06.csv", index=False)
    json.dump(dict(spec=spec, n_params=nparam), open(out / f"{spec['name']}_meta.json", "w"))


SPECS = {
    "BiGRU8": dict(cell="GRU", hidden=8, bi=True),
    "GRU8": dict(cell="GRU", hidden=8, bi=False),
    "BiLSTM8": dict(cell="LSTM", hidden=8, bi=True),
    "BiRNN8": dict(cell="RNN", hidden=8, bi=True),
    "LSTM8": dict(cell="LSTM", hidden=8, bi=False),
    "RNN8": dict(cell="RNN", hidden=8, bi=False),
    "MLP16": dict(cell="MLP", hidden=16, bi=False),
    "BiLSTM7": dict(cell="LSTM", hidden=7, bi=True),
    "BiRNN16": dict(cell="RNN", hidden=16, bi=True),
    "GRU12": dict(cell="GRU", hidden=12, bi=False),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results"); args = ap.parse_args()
    torch.set_num_threads(1)
    t0 = time.time(); spec = dict(SPECS[args.name], name=args.name)
    run(spec, args.data, Path(args.out)); print(args.name, f"{time.time()-t0:.0f}s")
