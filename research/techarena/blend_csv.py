"""50/50 rank blend of two submission CSVs (rowid, predicted_probability), e.g. the binary and softmax-objective models.

Usage (from the repo root):
    python research/techarena/blend_csv.py submissions/safe_v2_step5.csv submissions/safe_v2_step5_softmax.csv \
        submissions/safe_v2_step5_blend.csv [--factorised 0.1 0.7]
Both files must cover exactly the same rows; the output is the mean of the two normalised ranks (only order matters
for AUC), in (0, 1].

--factorised W_WHETHER W_WHICH: instead, split each model's predictions into the event's click mass E (sum over its
10 candidates, "whether") and the within-event shares p / E ("which"), and combine them geometrically:
    log score = W_WHETHER log E_a + (1 - W_WHETHER) log E_b + W_WHICH log share_a + (1 - W_WHICH) log share_b
On package C's validation files (binary a, softmax b) the softmax model has the better event mass and the binary model
the better shares; (0.1, 0.7) was the best grid point on both user halves (+0.00012 held out). Output = normalised
ranks of the score.
"""
import argparse

import numpy as np
import pandas as pd

EPS = 1e-7


def factorised(event_id: np.ndarray, pa: np.ndarray, pb: np.ndarray, w_whether: float, w_which: float) -> np.ndarray:
    """Factorised geometric blend (see the module docstring); returns the log score per row."""
    d = pd.DataFrame({"e": event_id, "a": np.clip(pa, EPS, 1 - EPS), "b": np.clip(pb, EPS, 1 - EPS)})
    g = d.groupby("e", sort=False)
    ea, eb = g["a"].transform("sum").to_numpy(), g["b"].transform("sum").to_numpy()
    return (w_whether * np.log(ea) + (1 - w_whether) * np.log(eb)
            + w_which * np.log(d["a"].to_numpy() / ea) + (1 - w_which) * np.log(d["b"].to_numpy() / eb))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a_path")
    ap.add_argument("b_path")
    ap.add_argument("out_path")
    ap.add_argument("--factorised", type=float, nargs=2, metavar=("W_WHETHER", "W_WHICH"))
    args = ap.parse_args()
    a, b = pd.read_csv(args.a_path), pd.read_csv(args.b_path)
    assert len(a) == len(b) and a["rowid"].is_unique and set(a["rowid"]) == set(b["rowid"]), "row sets differ"
    m = a.merge(b, on="rowid", suffixes=("_a", "_b"))
    assert m[["predicted_probability_a", "predicted_probability_b"]].notna().all().all(), "missing predictions"
    if args.factorised:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.data import load_candidates
        ev = load_candidates("eval").select("rowid", "event_id").to_pandas()
        m = m.merge(ev.assign(rowid=ev["rowid"].astype(str)), on="rowid", how="left")
        assert m["event_id"].notna().all() and m.groupby("event_id").size().eq(10).all(), "rows do not form full events"
        s = factorised(m["event_id"].to_numpy(), m["predicted_probability_a"].to_numpy(),
                       m["predicted_probability_b"].to_numpy(), *args.factorised)
        r = pd.Series(s).rank() / len(m)
    else:
        r = 0.5 * (m["predicted_probability_a"].rank() + m["predicted_probability_b"].rank()) / len(m)
    pd.DataFrame({"rowid": m["rowid"], "predicted_probability": r.to_numpy()}).to_csv(args.out_path, index=False)
    rho = np.corrcoef(m["predicted_probability_a"].rank(), m["predicted_probability_b"].rank())[0, 1]
    print(f"wrote {args.out_path}: {len(m):,} rows, rank correlation of the inputs {rho:.4f}"
          + (f", factorised {args.factorised}" if args.factorised else ""))


if __name__ == "__main__":
    main()
