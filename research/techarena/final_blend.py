"""Final blend of the binary and softmax stacks: the factorised blend (blend_csv.py --factorised), with the within-event
("which") part refitted as a conditional logit that also uses the raw sequence models' within-event shares.

    whether = W log E_binary + (1 - W) log E_softmax      (E = sum of the event's 10 predictions; W = 0.1)
    which   = sum_k beta_k log share_k,  k in {binary, softmax, 50-click sequence model, 20-click sequence model}
    score   = whether + c * which

beta: softmax over each clicked validation event's 10 candidates. c (which-vs-whether scale) is picked on global AUC.
Both are checked first with a 2-fold split by user (fit on one half, score the other), against the plain factorised
blend; the final beta and c are then fitted on all validation events and applied to the submission files.
On package C (3 Nov): the 50-click shares add +0.00015 held out, the 20-click ones a further +0.00004 (negative weight:
the stack under-uses what the older clicks add).

Usage (from the repo root):
    python research/techarena/final_blend.py --val-a data/validation/safe_v2_step7_s30_val_c4.parquet \
        --val-b data/validation/softmax_step7_s30_val_c4.parquet --sub-a submissions/safe_v2_step7_s30.csv \
        --sub-b submissions/safe_v2_step7_softmax_s30.csv --seq50 stage3_h50_s30 \
        --out submissions/safe_v2_step7_s30_final.csv
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
ap.add_argument("--w-whether", type=float, default=0.1)
ap.add_argument("--out")
ARGS = ap.parse_args()
SEQ = HERE / "cache" / "seq"
KEYS = ["a", "b", "s50", "s20"]


def frame(rows: pd.DataFrame, a_path: str, b_path: str, seq_name: str) -> pd.DataFrame:
    """rows (rowid, event_id, ...) + a, b, s50, s20, sorted by event so that every 10 rows form one event."""
    d = rows.assign(rowid=rows["rowid"].astype(str))
    for name, path in (("a", a_path), ("b", b_path)):
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


def parts(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """whether (per row) and the stacked log-shares (rows x components)."""
    logs, mass = [], {}
    for k in KEYS:
        v = np.clip(d[k].to_numpy(float), EPS, 1 - EPS).reshape(-1, 10)
        mass[k] = v.sum(1, keepdims=True)
        logs.append(np.log(v / mass[k]).ravel())
    whether = (ARGS.w_whether * np.log(mass["a"]) + (1 - ARGS.w_whether) * np.log(mass["b"])).repeat(10, 1).ravel()
    return whether, np.stack(logs, -1)


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
    return minimize(nll, np.array([0.7, 0.3, 0.0, 0.0]), jac=True, method="L-BFGS-B").x


def main() -> None:
    sp = make_split(4)
    rows = sp.val_candidates.select("rowid", "event_id", "clicked").to_pandas().merge(
        sp.val_events.select("event_id", "user_id").to_pandas(), on="event_id")
    v = frame(rows, ARGS.val_a, ARGS.val_b, f"b{sp.cutoff_ts}")
    y = v["clicked"].to_numpy()
    whether, Z = parts(v)
    clicked = y.reshape(-1, 10).max(1) == 1
    half = pd.util.hash_array(v["user_id"].to_numpy()[::10].astype(str)) % 2 == 0
    plain = Z[:, 0] * 0.7 + Z[:, 1] * 0.3

    def auc(score, ev_mask):
        m = np.repeat(ev_mask, 10)
        return roc_auc_score(y[m], score[m])

    rank_blend = (v["a"].rank() + v["b"].rank()).to_numpy()
    report = {"rank_50_50": roc_auc_score(y, rank_blend), "factorised": roc_auc_score(y, whether + plain), "halves": []}
    for h in (True, False):
        beta = fit_beta(Z, y, clicked & (half == h))
        which = Z @ beta
        c = max(SCALES, key=lambda s: auc(whether + s * which, half == h))
        report["halves"].append({"beta": beta.round(4).tolist(), "scale": c,
                                 "held_out": auc(whether + c * which, half != h),
                                 "held_out_factorised": auc(whether + plain, half != h)})
    gains = [r["held_out"] - r["held_out_factorised"] for r in report["halves"]]
    beta = fit_beta(Z, y, clicked)
    which = Z @ beta
    c = max(SCALES, key=lambda s: auc(whether + s * which, np.ones(len(clicked), bool)))
    if min(gains) <= 0:
        print("held-out gain not positive on both halves: using the plain factorised blend")
        beta, c = np.array([0.7, 0.3, 0.0, 0.0]), 1.0
    report.update({"held_out_gains": gains, "beta": beta.round(4).tolist(), "scale": c,
                   "final_on_validation": roc_auc_score(y, whether + c * (Z @ beta))})
    print(json.dumps(report, indent=1))
    if not ARGS.out:
        return
    ev = load_candidates("eval").select("rowid", "event_id").to_pandas()
    s = frame(ev, ARGS.sub_a, ARGS.sub_b, "full_eval")
    w_s, Z_s = parts(s)
    score = w_s + c * (Z_s @ beta)
    r = pd.Series(score).rank() / len(score)
    pd.DataFrame({"rowid": s["rowid"], "predicted_probability": r.to_numpy()}).to_csv(ARGS.out, index=False)
    rho = np.corrcoef(r, (s["a"].rank() + s["b"].rank()).to_numpy())[0, 1]
    print(f"wrote {ARGS.out}: {len(s):,} rows; rank correlation with the 50/50 rank blend {rho:.5f}")
    Path(ARGS.out).with_suffix(".json").write_text(json.dumps({**report, **vars(ARGS)}, indent=1))


if __name__ == "__main__":
    main()
