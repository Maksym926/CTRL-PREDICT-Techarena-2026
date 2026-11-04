"""(4 Nov, copied from the scratchpad; derived from a per-segment test, run(None) = final_blend.py protocol)
Does the final blend's which-part (conditional logit) gain from per-segment weights? Segments by the user's known
clicks (history_length) or the gap. Same data and protocol as final_blend.py with the MLP stack as third model
(package F): 2-fold by user, beta fitted on one half's clicked events, whether weights + scale by grid on that half,
scored on the other half."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

REPO = Path(r"G:/Development/HuaweiTechArenaPrep/project/CTRL-PREDICT-Techarena-2026")
import os  # noqa: E402

os.chdir(REPO)
sys.path.insert(0, str(REPO))
from src.validation import make_split  # noqa: E402

C_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/validation/nn_step7_s30_val_c4.parquet"
sp = make_split(4)
rows = sp.val_candidates.select("rowid", "event_id", "clicked").to_pandas().merge(
    sp.val_events.select("event_id", "user_id").to_pandas(), on="event_id")
d = rows.assign(rowid=rows["rowid"].astype(str))
VARS = ["", "_w512", "_lr5", "_ctx", "_ctx_lr5", "_ctx_lr5_o3", "_ctx_lr3", "_lr3", "_ctx_lr2"]
for name, path in (("a", "data/validation/safe_v2_step7_s30_val_c4.parquet"),
                   ("b", "data/validation/softmax_step7_s30_opt4_val_c4.parquet"), ("c", C_PATH)) + tuple(
                   ("nn" + v, f"data/validation/nn_step7_s30{v}_val_c4.parquet") for v in VARS):
    p = pd.read_parquet(path)
    d = d.merge(p.assign(rowid=p["rowid"].astype(str)).rename(columns={"predicted_probability": name}), on="rowid")
SEQ = REPO / "research/techarena/cache/seq"
for name, folder in (("s50", "stage3_h50_s30"), ("s20", "stage3_s5")):
    s = pd.read_parquet(SEQ / folder / f"seq_b{sp.cutoff_ts}.parquet")
    d = d.merge(s[["rowid", "seq_prob"]].assign(rowid=s["rowid"].astype(str)).rename(columns={"seq_prob": name}),
                on="rowid")
f = pd.read_parquet(REPO / "research/techarena/cache/lab/step6_val.parquet", columns=["rowid", "history_length", "gap_hours"])
d = d.merge(f.assign(rowid=f["rowid"].astype(str)), on="rowid")
assert len(d) == len(rows)
d = d.sort_values(["event_id", "rowid"]).reset_index(drop=True)
y = d["clicked"].to_numpy()
KEYS = ["a", "b", "c", "s50", "s20"]
M, Z = {}, []
for k in KEYS:
    v = np.clip(d[k].to_numpy(float), 1e-7, 1 - 1e-7).reshape(-1, 10)
    e = v.sum(1, keepdims=True)
    M[k] = np.log(e).repeat(10, 1).ravel()
    Z.append(np.log(v / e).ravel())
Z = np.stack(Z, -1)
clicked = y.reshape(-1, 10).max(1) == 1
half = pd.util.hash_array(d["user_id"].to_numpy()[::10].astype(str)) % 2 == 0
hl = d["history_length"].to_numpy()[::10]
gap = d["gap_hours"].to_numpy()[::10]
print("history_length quantiles", np.quantile(hl, [0.1, 0.25, 0.5, 0.75, 0.9]).round(1),
      "gap quantiles", np.quantile(gap, [0.1, 0.25, 0.5, 0.75, 0.9]).round(1))
STEPS = np.round(np.linspace(0, 1, 11), 1)
WEIGHTS = [(wa, wc) for wa in STEPS for wc in STEPS if wa + wc <= 1.0001]
SCALES = (0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2)


def fit_beta(ev):
    z = Z.reshape(-1, 10, Z.shape[1])[ev & clicked]
    t = y.reshape(-1, 10)[ev & clicked].argmax(1)

    def nll(b):
        u = z @ b
        u = u - u.max(1, keepdims=True)
        lse = np.log(np.exp(u).sum(1))
        p = np.exp(u - lse[:, None])
        return -(u[np.arange(len(t)), t] - lse).sum(), -(z[np.arange(len(t)), t] - (p[:, :, None] * z).sum(1)).sum(0)
    return minimize(nll, np.r_[0.7, 0.3, np.zeros(Z.shape[1] - 2)], jac=True, method="L-BFGS-B").x


def renorm(u):
    u = u.reshape(-1, 10)
    m = u.max(1, keepdims=True)
    return (u - m - np.log(np.exp(u - m).sum(1, keepdims=True))).ravel()


def whether(w):
    return w[0] * M["a"] + (1 - w[0] - w[1]) * M["b"] + w[1] * M["c"]


def auc(score, ev):
    m = np.repeat(ev, 10)
    return roc_auc_score(y[m], score[m])


def run(seg):
    """seg: per-event segment ids (or None = one segment). Returns held-out AUCs on the two halves."""
    out = []
    for h in (True, False):
        fit_ev = half == h
        u = np.zeros(len(y))
        betas = []
        for g in (np.unique(seg) if seg is not None else [None]):
            in_g = np.ones(len(fit_ev), bool) if g is None else seg == g
            b = fit_beta(fit_ev & in_g)
            betas.append(b.round(3))
            rows_g = np.repeat(in_g, 10)
            u[rows_g] = Z[rows_g] @ b
        which = renorm(u)
        w, c = max(((w, s) for w in WEIGHTS for s in SCALES), key=lambda t: auc(whether(t[0]) + t[1] * which, fit_ev))
        out.append(auc(whether(w) + c * which, ~fit_ev))
        if seg is not None:
            print("   betas per segment (a, b, c, s50, s20):", [list(b) for b in betas], "w", w, "scale", c)
    return out


NNP = {v: np.clip(d["nn" + v].to_numpy(float), 1e-7, 1 - 1e-7) for v in VARS}
ZC = KEYS.index("c")


def set_c(vs):
    v = np.mean([NNP[x] for x in vs], axis=0).reshape(-1, 10)
    e = v.sum(1, keepdims=True)
    M["c"] = np.log(e).repeat(10, 1).ravel()
    Z[:, ZC] = np.log(v / e).ravel()


OPTIONS = [("base (F)", [""]), ("plain lr3", ["_lr3"]), ("context lr2", ["_ctx_lr2"]),
           ("base + plain lr3", ["", "_lr3"]), ("base + context lr3", ["", "_ctx_lr3"]),
           ("base + context lr2", ["", "_ctx_lr2"]), ("base + plain lr3 + context lr3 + context lr2", ["", "_lr3", "_ctx_lr3", "_ctx_lr2"]),
           ("all nine", VARS)]
ref = None
for name, vs in OPTIONS:
    set_c(vs)
    r = run(None)
    ref = ref or r
    print(f"{name:48s} held-out {r[0]:.6f} / {r[1]:.6f}  ({r[0] - ref[0]:+.6f} / {r[1] - ref[1]:+.6f})", flush=True)
