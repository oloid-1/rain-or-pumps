"""
The simulator model: a BiLSTM whose only rain input is the weekly sequence,
trained with a rain-response penalty so that more rain cannot predict a deeper
fall. This is the model the what-if UI runs.

Why it differs from models/train_bilstm.py, which stays as it is:

1. Rain enters through the sequence only. The BiLSTM arm also takes 23
   rain-derived tabular features (windows, lags, anomalies). A scenario that
   scales the rain would have to rebuild every one of them consistently, and the
   sensitivity check in BILSTM_RESULTS.md scaled the sequence while holding them
   fixed. Here the tabular arm is static only: well depth, specific yield, the
   interval length, the training-years normals, season, aquifer, well type and
   transition. Scaling the sequence is then the whole scenario.

2. A rain-response penalty. BILSTM_RESULTS.md found the response to a 20% rain
   change is about 1 cm and flips sign when one channel is removed. Each
   training batch is re-run with its rain scaled up and down by a random 10-40%,
   and any prediction where more rain gives a larger fall is penalised:
       relu(pred(up) - pred(base)) + relu(pred(base) - pred(down))
   delta_h_m is positive when the level falls, so the target ordering is
   pred(up) <= pred(base) <= pred(down). The penalty is soft, so the
   share of rows that still violate it is measured and reported.

3. Exports what a browser needs: ONNX weights plus every scaling statistic and
   category level, so the UI can run the model live.

    python simulator/train_sim.py                                 6 channels, data/training
    python simulator/train_sim.py --mono-weight 0                 same model, no penalty (ablation)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "models"))
from bilstm import RainBiLSTM  # noqa: E402
from train_bilstm import load, channel_stats, scale, LOG_CHANNELS, SIGNED_LOG_CHANNELS  # noqa: E402
from train_transformer import metrics, TARGET, SEED  # noqa: E402
from build_training_data import OUT_DIR  # noqa: E402
from transformer import count_params  # noqa: E402

STATIC_NUMERIC = ["well_depth_m", "sy", "days_since_prev", "rain_normal_annual_mm",
                  "rain_normal_season_mm", "rain_normal_monsoon_mm"]
STATIC_CATEGORICAL = ["season", "aquifer", "well_type", "transition"]


# --------------------------------------------------------------------------
def encode_static(tab, train_mask, stats=None):
    """Standardise the static numerics and index the categoricals with statistics
    from TRAIN rows, or with `stats` from a trained model's sim_meta.json, and
    return the statistics so the browser can repeat it exactly."""
    X = tab[STATIC_NUMERIC].to_numpy("float32")
    if stats:
        med, mu, sd = (np.asarray(stats[k], "float32") for k in ("median", "mean", "std"))
        X = np.where(np.isfinite(X), X, med)
    else:
        med = np.nanmedian(X[train_mask], axis=0)
        X = np.where(np.isfinite(X), X, med)
        mu, sd = X[train_mask].mean(0), X[train_mask].std(0)
        sd[sd < 1e-6] = 1.0
    Xn = ((X - mu) / sd).astype("float32")

    codes, levels_out = [], {}
    for c in STATIC_CATEGORICAL:
        v = tab[c].astype(str).fillna("unknown")
        levels = stats["categorical"][c] if stats else sorted(v[train_mask].unique())
        idx = pd.Index(levels).get_indexer(v)
        codes.append(np.where(idx < 0, len(levels), idx).astype("int64"))
        levels_out[c] = levels
    stats = dict(numeric=STATIC_NUMERIC, median=med.tolist(), mean=mu.tolist(),
                 std=sd.tolist(), categorical=levels_out)
    return Xn, np.stack(codes, 1), [len(levels_out[c]) + 1 for c in STATIC_CATEGORICAL], stats


class RainScaler(nn.Module):
    """Scales the rain of an already-scaled sequence batch by a factor, in torch,
    so the penalty is differentiable. Mirrors train_bilstm.rain_scaled:
    rain_mm is multiplied; rain_anom_mm moves by (f-1)*rain; the calendar,
    in_interval and wet_frac channels are unchanged."""

    def __init__(self, channels, stats):
        super().__init__()
        self.r = channels.index("rain_mm")
        self.a = channels.index("rain_anom_mm") if "rain_anom_mm" in channels else None
        self.mr, self.sr = stats["rain_mm"]
        if self.a is not None:
            self.ma, self.sa = stats["rain_anom_mm"]

    def forward(self, s, f):
        # s: (B, T, C) scaled; f: (B,) factor
        f = f[:, None]
        rain = torch.expm1(s[:, :, self.r] * self.sr + self.mr).clamp(min=0)
        out = s.clone()
        out[:, :, self.r] = (torch.log1p(rain * f) - self.mr) / self.sr
        if self.a is not None:
            u = s[:, :, self.a] * self.sa + self.ma
            anom = torch.sign(u) * torch.expm1(u.abs()) + (f - 1.0) * rain
            out[:, :, self.a] = (torch.sign(anom) * torch.log1p(anom.abs()) - self.ma) / self.sa
        return out


# --------------------------------------------------------------------------
def train(model, scaler, S, Xn, C, y, tr, va, a, device, label):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model, scaler = model.to(device), scaler.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * (len(tr) // a.bs + 1))
    loss_fn = nn.HuberLoss(delta=1.0)
    T = lambda arr, k: torch.from_numpy(arr[k]).to(device)  # noqa: E731

    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = pen_tot = 0.0
        order = np.random.permutation(tr)
        for i in range(0, len(order), a.bs):
            k = order[i:i + a.bs]
            s, x, c, t = T(S, k), T(Xn, k), T(C, k), T(y, k)
            opt.zero_grad()
            l = loss_fn(model(s, x, c), t)
            pen = torch.zeros((), device=device)
            if a.mono_weight > 0:
                # dropout off for the three passes so they differ only in the rain.
                # cuDNN refuses an RNN backward in eval mode, so these passes use
                # the native LSTM kernel instead.
                model.eval()
                u = torch.empty(len(k), device=device).uniform_(0.1, 0.4)
                with torch.backends.cudnn.flags(enabled=False):
                    p0 = model(s, x, c)
                    pu = model(scaler(s, 1 + u), x, c)
                    pd_ = model(scaler(s, 1 - u), x, c)
                pen = torch.relu(pu - p0).mean() + torch.relu(p0 - pd_).mean()
                model.train()
            (l + a.mono_weight * pen).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(l.detach()) * len(k)
            pen_tot += float(pen.detach()) * len(k)
        vm = metrics(y[va], predict(model, S, Xn, C, va, device))
        hist.append(dict(epoch=ep, train_loss=tot / len(tr), penalty=pen_tot / len(tr), **vm))
        print(f"  [{label}] epoch {ep:>2}  train {tot/len(tr):.4f}  penalty {pen_tot/len(tr):.5f}  "
              f"valid mae {vm['mae']:.4f}  r2 {vm['r2']:.4f}", flush=True)
        if vm["mae"] < best - 1e-4:
            best, bad = vm["mae"], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                print(f"  [{label}] early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def predict(model, S, Xn, C, idx, device, scaler=None, f=None):
    model.eval()
    p = []
    for i in range(0, len(idx), 4096):
        k = idx[i:i + 4096]
        s = torch.from_numpy(S[k]).to(device)
        if scaler is not None:
            s = scaler(s, torch.full((len(k),), f, device=device))
        p.append(model(s, torch.from_numpy(Xn[k]).to(device),
                       torch.from_numpy(C[k]).to(device)).cpu().numpy())
    return np.concatenate(p)


def response(model, scaler, S, Xn, C, idx, device):
    """How the prediction moves when every week's rain is scaled, on idx rows.
    mean: average change in predicted fall (m). wrong_way: share of rows where
    more rain predicts a larger fall (or less rain a smaller one)."""
    base = predict(model, S, Xn, C, idx, device)
    out = {}
    for f in (0.5, 0.8, 1.2, 1.5):
        d = predict(model, S, Xn, C, idx, device, scaler, f) - base
        wrong = d > 1e-4 if f > 1 else d < -1e-4
        out[f"x{f}"] = dict(mean=float(d.mean()), p10=float(np.percentile(d, 10)),
                           p90=float(np.percentile(d, 90)), wrong_way=float(wrong.mean()))
    return out


def export(m, S, Xn, C, channels, cs, tstats, valid, response, out):
    """ONNX weights plus every statistic the browser needs to repeat the scaling."""
    m = m.cpu().eval()
    n = 4
    torch.onnx.export(
        m, (torch.from_numpy(S[:n]), torch.from_numpy(Xn[:n]), torch.from_numpy(C[:n])),
        out / "sim.onnx", input_names=["seq", "num", "cat"], output_names=["delta_h_m"],
        dynamic_axes={"seq": {0: "b"}, "num": {0: "b"}, "cat": {0: "b"}, "delta_h_m": {0: "b"}},
        opset_version=17, dynamo=False)
    meta = dict(channels=channels, channel_scaling={
                    c: dict(kind="log1p" if c in LOG_CHANNELS else "signed_log1p",
                            mean=cs[c][0], std=cs[c][1]) for c in cs},
                weeks=int(S.shape[1]), static=tstats,
                target="delta_h_m, metres, positive when the water level fell",
                valid=valid, response_valid=response)
    (out / "sim_meta.json").write_text(json.dumps(meta, indent=2))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(OUT_DIR))
    ap.add_argument("--out", default=str(HERE / "out" / "run"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--mono-weight", type=float, default=1.0)
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-export", action="store_true")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tab, x, channels, _, _ = load(a.data)
    if a.limit:
        keep = np.sort(np.random.default_rng(SEED).choice(len(tab), a.limit, replace=False))
        tab, x = tab.iloc[keep].reset_index(drop=True), x[keep]
    y = tab[TARGET].to_numpy("float32")
    print(f"rows {len(tab):,}  channels {channels}  static {STATIC_NUMERIC + STATIC_CATEGORICAL}  "
          f"mono {a.mono_weight}  device {device}", flush=True)

    def fit(tr, va, label):
        mk = np.zeros(len(tab), bool); mk[tr] = True
        cs = channel_stats(x, channels, mk)
        S = scale(x, channels, cs)
        Xn, C, sizes, tstats = encode_static(tab, mk)
        m = RainBiLSTM(len(STATIC_NUMERIC), sizes, len(channels))
        sc = RainScaler(channels, cs)
        m, hist = train(m, sc, S, Xn, C, y, tr, va, a, device, label)
        return m, sc, S, Xn, C, hist, cs, tstats

    res = {"config": vars(a), "rows": int(len(tab)), "channels": channels,
           "static": STATIC_NUMERIC + STATIC_CATEGORICAL}
    if a.cv:
        folds = []
        for f in sorted(tab.district_fold.unique()):
            tr = np.flatnonzero((tab.district_fold != f).to_numpy())
            va = np.flatnonzero((tab.district_fold == f).to_numpy())
            m, sc, S, Xn, C, *_ = fit(tr, va, f"fold{f}")
            folds.append(metrics(y[va], predict(m, S, Xn, C, va, device)))
        res["cv_folds"] = folds
        res["cv_mean"] = {k: float(np.mean([d[k] for d in folds])) for k in folds[0]}
        print("district-fold CV mean:", res["cv_mean"])
        (out / "results_cv.json").write_text(json.dumps(res, indent=2))
        return

    tr = np.flatnonzero((tab.split == "train").to_numpy())
    va = np.flatnonzero((tab.split == "valid").to_numpy())
    te = np.flatnonzero((tab.split == "test").to_numpy())
    t0 = time.time()
    m, sc, S, Xn, C, hist, cs, tstats = fit(tr, va, "sim")
    print(f"params {count_params(m):,}  {(time.time()-t0)/60:.1f} min")
    pred = predict(m, S, Xn, C, np.arange(len(tab)), device)
    res.update(valid=metrics(y[va], pred[va]), test=metrics(y[te], pred[te]),
               train=metrics(y[tr], pred[tr]), history=hist,
               response_valid=response(m, sc, S, Xn, C, va, device))
    print("valid", res["valid"], "\ntest", res["test"])
    print("rain response on valid:", json.dumps(res["response_valid"], indent=1))
    torch.save(m.state_dict(), out / "sim.pt")
    (out / "results.json").write_text(json.dumps(res, indent=2))

    if not a.no_export:
        export(m, S, Xn, C, channels, cs, tstats, res["valid"], res["response_valid"], out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
