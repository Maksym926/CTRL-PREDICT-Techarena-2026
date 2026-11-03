"""More seeds for the choice-model features: the stack's top features (cmf12_prob first by gain, cmfp_prob second)
come from a single seed each, and averaging seeds lifted the sequence-model feature by +0.0004 in the stack (5 -> 30
seeds, 3 Nov).

A seed run refits one choice-MF configuration at every LightGBM boundary with another seed and caches its candidate
probabilities through safe_v2_choicemf.cmf_features itself, so everything but args.seed is the original feature. One
fit per boundary serves validation and submission alike: a boundary's model sees only train events before it, and its
targets here are each user's first event from the boundary on (a superset of both scopes' targets), or all of
evaluation at the cutoff. The merge averages the original seed 0 (cache/<cache>/full, the frame's rows) with the new
seeds, and with partial averages made on another PC (--with folder:n_seeds).

Usage (from the repo root):
    python research/techarena/cmf_seeds.py --model d12 --seed 1           # cache/choicemf_d12_seeds/s1/<b>.parquet
    python research/techarena/cmf_seeds.py --model d12 --merge 1 2 3      # -> cache/choicemf_d12_avg/<b>.parquet
    python research/techarena/cmf_seeds.py --model d12 --merge 3 4 --no-base --out choicemf_d12_part   # second PC
    python research/techarena/cmf_seeds.py --model d12 --merge 1 2 --with choicemf_d12_part:2           # main PC
Then safe_v2_step7.py --swap cmf12=choicemf_d12_avg ... swaps the columns. Restart-safe (finished boundaries skip).
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["v2", "d12", "d16", "pop"], default="d12")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--merge", type=int, nargs="*", help="average these seeds (and seed 0 unless --no-base) instead of training")
ap.add_argument("--with", dest="with_", nargs="*", default=[], help="partial averages under cache/, as folder:n_seeds")
ap.add_argument("--no-base", action="store_true", help="leave out seed 0 (a partial average made on another PC)")
ap.add_argument("--out", default="", help="merge output folder under cache/ (default <cache>_avg)")
ap.add_argument("--threads", type=int, default=12)
ap.add_argument("--bounds", type=int, nargs="*", help="only these boundaries (smoke tests)")
ARGS = ap.parse_args()
sys.argv = sys.argv[:1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import leakfree_safe_v2 as lf  # noqa: E402  (sets cwd to the repo root)
from safe_v2_choicemf import cmf_features  # noqa: E402
from safe_v2_step1 import CMF2_ARGS, CMF2_EPOCHS  # noqa: E402
from src.choice_mf import code_maps, first_per_user  # noqa: E402
from src.data import load_candidates, load_events  # noqa: E402

# cache folder, settings on top of step 1's CMF2_ARGS, epochs: as steps 1 (v2), 4 (d12, d16) and 5 (pop) built them
MODELS = {"v2": ("choicemf_v2", {}, CMF2_EPOCHS),
          "d12": ("choicemf_d12", {"dim": 12, "svd_rank": 12}, CMF2_EPOCHS),
          "d16": ("choicemf_d16", {"dim": 16, "svd_rank": 16}, CMF2_EPOCHS),
          "pop": ("choicemf_pop", {"aux_pop": 1, "aux_weight": 1.0, "aux_neg": 9, "aux_ratio": 4}, 3)}  # step 5's POP_*
NAME, SETTINGS, EPOCHS = MODELS[ARGS.model]
CACHE = HERE / "cache"
ORIGINAL = CACHE / NAME / "full"  # seed 0
SEEDS = CACHE / f"{NAME}_seeds"
BOUNDS = list(lf.bounds_for(7, half=True)) + [lf.CUT]  # the 13 windows (validation's 7 and day 4 among them) + evaluation
if ARGS.bounds:
    BOUNDS = [b for b in BOUNDS if b in ARGS.bounds]


def train_seed() -> None:
    torch.set_num_threads(ARGS.threads)
    out = SEEDS / f"s{ARGS.seed}"
    args = SimpleNamespace(**{**vars(CMF2_ARGS), **SETTINGS, "seed": ARGS.seed})
    events, cands = load_events("train"), load_candidates("train")
    ev_e, ev_c = load_events("eval"), load_candidates("eval")
    umap, imap = code_maps()
    t0 = time.perf_counter()
    for b in sorted(BOUNDS, reverse=True):  # latest (largest) first, like the other window runs
        if (out / f"{b}.parquet").exists():
            lf.log(f"boundary {b}: done, skipping")
            continue
        if b == lf.CUT:
            rowids, target = ev_c["rowid"].to_list(), dict(target_events=ev_e, target_cands=ev_c)
        else:
            first = first_per_user(events.filter(pl.col("timestamp") >= b)).select("event_id")
            rowids, target = cands.join(first, on="event_id", how="semi")["rowid"].to_list(), {}
        cmf_features(pd.DataFrame({"rowid": rowids}), b, events, cands, umap, imap, cache=out, args=args,
                     epochs=EPOCHS, **target)
        lf.log(f"{ARGS.model} seed {ARGS.seed}, boundary {b}: {len(rowids):,} rows "
               f"({(time.perf_counter() - t0) / 60:.1f} min so far)")


def merge(seeds: list[int]) -> None:
    out = CACHE / (ARGS.out or f"{NAME}_avg")
    out.mkdir(parents=True, exist_ok=True)
    sources = ([] if ARGS.no_base else [(ORIGINAL, 1)]) + [(SEEDS / f"s{s}", 1) for s in seeds] + \
              [(CACHE / f.split(":")[0], int(f.split(":")[1])) for f in ARGS.with_]
    for b in BOUNDS:
        base = None
        p, n, w = 0.0, 0.0, 0
        for folder, weight in sources:
            f = pd.read_parquet(folder / f"{b}.parquet")
            f["rowid"] = f["rowid"].astype(str)
            if base is None:  # the first source fixes the rows (seed 0: exactly the frames' targets)
                base = f[["rowid"]]
            m = base.merge(f, on="rowid", how="left")
            if m["cmf_prob"].isna().any():
                raise SystemExit(f"{folder.name}, boundary {b}: {int(m['cmf_prob'].isna().sum()):,} rows missing")
            p = p + weight * m["cmf_prob"].to_numpy()
            n = n + weight * m["cmf_noclick"].to_numpy()
            w += weight
        p, n = p / w, n / w
        rank = ((-p.reshape(-1, 10)).argsort(1).argsort(1) + 1).reshape(-1).astype(np.float32)  # blocks of 10 per event
        pd.DataFrame({"rowid": base["rowid"].to_numpy(), "cmf_prob": p, "cmf_noclick": n, "cmf_rank": rank}).to_parquet(
            out / f"{b}.parquet", index=False, compression="zstd")
        lf.log(f"{ARGS.model} boundary {b}: averaged {w} seeds over {len(base):,} rows -> {out.name}")
    (out / "seeds.json").write_text(json.dumps({"model": ARGS.model, "seeds": w, "sources": [str(s) for s, _ in sources]}))


if __name__ == "__main__":
    merge(ARGS.merge) if ARGS.merge is not None else train_seed()
