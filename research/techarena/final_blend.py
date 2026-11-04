"""Final blend of the stack models: the factorised blend (blend_csv.py --factorised), with the within-event ("which")
part refitted as a conditional logit that also uses the raw sequence models' within-event shares.

    whether = w_a log E_binary + w_b log E_softmax (+ w_c log E_third)    (E = sum of the event's 10 predictions)
    u       = sum_k beta_k log share_k,  k in {binary, softmax, (third,) 50-click sequence model, 20-click sequence model}
    which   = u - logsumexp over the event's 10 candidates of u   (renormalised: a proper within-event log-share)
    score   = whether + c * which

beta: softmax over each clicked validation event's 10 candidates. c (which-vs-whether scale) is picked on global AUC;
the whether weights are fixed (w_a = --w-whether, w_b = the rest) unless a third model is given or --whether-grid, then
they are picked on global AUC from a 0.1 grid too. Everything is checked first with a 2-fold split by user (fit on one
half, score the other) against the plain factorised blend; the final values are then fitted on all validation events
and applied to the submission files.
History: on package C (3 Nov) the 50-click shares add +0.00015 held out, the 20-click ones +0.00004 (negative weight:
the stack under-uses what the older clicks add). On 4 Nov the MLP stack (nn_step7.py) as the third model added
+0.0007 / +0.0006 held out on top of binary + tuned partner.

Usage (from the repo root):
    python research/techarena/final_blend.py --val-a data/validation/safe_v2_step7_s30_val_c4.parquet \
        --val-b data/validation/softmax_step7_s30_val_c4.parquet [--val-c data/validation/nn_step7_s30_val_c4.parquet] \
        --sub-a submissions/safe_v2_step7_s30.csv --sub-b submissions/safe_v2_step7_softmax_s30.csv \
        [--sub-c submissions/safe_v2_step7_nn_s30.csv] --seq50 stage3_h50_s30 --out submissions/safe_v2_step7_s30_final.csv
Without --sub-a/--sub-b/--out it only reports the validation numbers.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from src.data import load_candidates  # noqa: E402
from src.validation import make_split  # noqa: E402

EPS = 1e-7
SCALES = (0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2)
ap = argparse.ArgumentParser()
ap.add_argument("--val-a", required=True, help="binary stack, validation predictions")
ap.add_argument("--val-b", required=True, help="softmax stack, validation predictions")
ap.add_argument("--sub-a")
ap.add_argument("--sub-b")
ap.add_argument("--seq50", default="stage3_h50_s30", help="folder under cache/seq with the 50-click features")
ap.add_argument("--seq20", default="stage3_s5", help="folder under cache/seq with the 20-click features")
ap.add_argument("--w-whether", type=float, default=0.1, help="binary's whether weight when it is not grid-searched")
ap.add_argument("--whether-grid", action="store_true", help="pick the whether weights on validation (0.1 grid)")
ap.add_argument("--val-c", help="optional third probability model (e.g. nn_step7.py), validation predictions")
ap.add_argument("--sub-c", help="the same model's submission file")
ap.add_argument("--out")
ARGS = ap.parse_args()
SEQ = HERE / "cache" / "seq"
KEYS = ["a", "b"] + (["c"] if ARGS.val_c else []) + ["s50", "s20"]
GRID = ARGS.whether_grid or bool(ARGS.val_c)
STEPS = np.round(np.linspace(0, 1, 11), 1)
WEIGHTS = ([(wa, wc) for wa in STEPS for wc in (STEPS if ARGS.val_c else [0.0]) if wa + wc <= 1.0001] if GRID
           else [(ARGS.w_whether, 0.0)])


def frame(rows: pd.DataFrame, a_path: str, b_path: str, seq_name: str, c_path: str | None = None) -> pd.DataFrame:
    """rows (rowid, event_id, ...) + a, b, (c,) s50, s20, sorted by event so that every 10 rows form one event."""
    d = rows.assign(rowid=rows["rowid"].astype(str))
    for name, path in (("a", a_path), ("b", b_path)) + ((("c", c_path),) if c_path else ()):
        p = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        d = d.merge(p.assign(rowid=p["rowid"].astype(str)).rename(columns={"predicted_probability": name}), on="rowid")
    for name, folder in (("s50", ARGS.seq50), ("s20", ARGS.seq20)):
        s = pd.read_parquet(SEQ / folder / f"seq_{seq_name}.parquet")
        d = d.merge(s[["rowid", "seq_prob"]].assign(rowid=s["rowid"].astype(str)).rename(columns={"seq_prob": name}),
                    on="rowid")
    assert len(d) == len(rows), f"{a_path}: predictions or sequence features missing for some rows"
    d = d.sort_values(["event_id", "rowid"]).reset_index(drop=True)
    assert (d.groupby("event_id").size() == 10).all(), "events without exactly 10 rows"
    return d


def parts(d: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """log event masses (per row) and the stacked log-shares (rows x components)."""
    logs, mass = [], {}
    for k in KEYS:
        v = np.clip(d[k].to_numpy(float), EPS, 1 - EPS).reshape(-1, 10)
        e = v.sum(1, keepdims=True)
        mass[k] = np.log(e).repeat(10, 1).ravel()
        logs.append(np.log(v / e).ravel())
    return mass, np.stack(logs, -1)


def whether_of(mass: dict, w: tuple[float, float]) -> np.ndarray:
    out = w[0] * mass["a"] + (1 - w[0] - w[1]) * mass["b"]
    return out + w[1] * mass["c"] if w[1] else out


def renorm(u: np.ndarray) -> np.ndarray:
    """Within-event log-softmax: without it the combination's event-level offset is arbitrary and moves whole events
    against each other in the global AUC (held out on package C: +0.00003 over the unnormalised version)."""
    u = u.reshape(-1, 10)
    m = u.max(1, keepdims=True)
    return (u - m - np.log(np.exp(u - m).sum(1, keepdims=True))).ravel()


def fit_beta(Z: np.ndarray, y: np.ndarray, ev_mask: np.ndarray) -> np.ndarray:
    """Conditional logit over the clicked events in ev_mask."""
    z = Z.reshape(-1, 10, Z.shape[1])[ev_mask]
    t = y.reshape(-1, 10)[ev_mask].argmax(1)

    def nll(beta):
        u = z @ beta
        u = u - u.max(1, keepdims=True)
        lse = np.log(np.exp(u).sum(1))
        p = np.exp(u - lse[:, None])
        val = -(u[np.arange(len(t)), t] - lse).sum()
        grad = -(z[np.arange(len(t)), t] - (p[:, :, None] * z).sum(1)).sum(0)
        return val, grad
    return minimize(nll, np.r_[0.7, 0.3, np.zeros(Z.shape[1] - 2)], jac=True, method="L-BFGS-B").x


def main() -> None:
    sp = make_split(4)
    rows = sp.val_candidates.select("rowid", "event_id", "clicked").to_pandas().merge(
        sp.val_events.select("event_id", "user_id").to_pandas(), on="event_id")
    v = frame(rows, ARGS.val_a, ARGS.val_b, f"b{sp.cutoff_ts}", ARGS.val_c)
    y = v["clicked"].to_numpy()
    mass, Z = parts(v)
    clicked = y.reshape(-1, 10).max(1) == 1
    half = pd.util.hash_array(v["user_id"].to_numpy()[::10].astype(str)) % 2 == 0
    plain = whether_of(mass, (0.1, 0.0)) + Z[:, 0] * 0.7 + Z[:, 1] * 0.3

    def auc(score, ev_mask):
        m = np.repeat(ev_mask, 10)
        return roc_auc_score(y[m], score[m])

    def pick(which, ev_mask):
        return max(((w, s) for w in WEIGHTS for s in SCALES),
                   key=lambda t: auc(whether_of(mass, t[0]) + t[1] * which, ev_mask))

    rank_blend = (v["a"].rank() + v["b"].rank()).to_numpy()
    report = {"rank_50_50": roc_auc_score(y, rank_blend), "factorised": roc_auc_score(y, plain), "keys": KEYS,
              "halves": []}
    for h in (True, False):
        beta = fit_beta(Z, y, clicked & (half == h))
        which = renorm(Z @ beta)
        w, c = pick(which, half == h)
        report["halves"].append({"beta": beta.round(4).tolist(), "whether_weights": w, "scale": c,
                                 "held_out": auc(whether_of(mass, w) + c * which, half != h),
                                 "held_out_factorised": auc(plain, half != h)})
    gains = [r["held_out"] - r["held_out_factorised"] for r in report["halves"]]
    beta = fit_beta(Z, y, clicked)
    w, c = pick(renorm(Z @ beta), np.ones(len(clicked), bool))
    if min(gains) <= 0:
        print("held-out gain not positive on both halves: using the plain factorised blend")
        beta, w, c = np.r_[0.7, 0.3, np.zeros(len(KEYS) - 2)], (0.1, 0.0), 1.0
    report.update({"held_out_gains": gains, "beta": beta.round(4).tolist(), "whether_weights": w, "scale": c,
                   "final_on_validation": roc_auc_score(y, whether_of(mass, w) + c * renorm(Z @ beta))})
    print(json.dumps(report, indent=1))
    if not ARGS.out:
        return
    ev = load_candidates("eval").select("rowid", "event_id").to_pandas()
    s = frame(ev, ARGS.sub_a, ARGS.sub_b, "full_eval", ARGS.sub_c)
    mass_s, Z_s = parts(s)
    score = whether_of(mass_s, w) + c * renorm(Z_s @ beta)
    r = pd.Series(score).rank() / len(score)
    pd.DataFrame({"rowid": s["rowid"], "predicted_probability": r.to_numpy()}).to_csv(ARGS.out, index=False)
    rho = np.corrcoef(r, (s["a"].rank() + s["b"].rank()).to_numpy())[0, 1]
    print(f"wrote {ARGS.out}: {len(s):,} rows; rank correlation with the 50/50 rank blend {rho:.5f}")
    Path(ARGS.out).with_suffix(".json").write_text(json.dumps({**report, **vars(ARGS)}, indent=1))


if __name__ == "__main__":
    main()
