# Progress report: Huawei Tech Arena 2026, Phase 1 (click prediction)

**Best leaderboard score: 0.814056** (place 41; top of the board 0.8329 as of 26 Oct 2026).
Fully reproducible and leak-free, built on the team's Parquet data and local validation.

## Results

| # | Model | Local AUC | Leaderboard | Local → LB |
|---|---|---|---|---|
| 001 | Starter solution (unmodified) | 0.6039 | 0.6200 | +0.016 |
| 002 | Monish's SAFE_v2 features, leak-free protocol | 0.7919 | 0.8077 | +0.016 |
| 003 | **SAFE_v2 + choice-model features** | **0.7965** | **0.8141** | +0.018 |

Local validation predicts the leaderboard: **leaderboard ≈ local AUC + 0.016**, confirmed on three very different
models. New ideas can be tested locally before spending a submission. The full log is in `experiments.csv`.

## What was built

### 1. Project setup
- Repo skeleton, venv, `requirements.txt` + `requirements.lock.txt` (now incl. polars, pyarrow, torch).
- `src/check_submission.py`: validates a submission against the official rules (match by rowid, exact header,
  probabilities in [0, 1]).

### 2. Parquet data (`src/convert_to_parquet.py`, `src/data.py`, `src/verify_parquet.py`)
- 7.3 GB `train.csv` → 442 MB of Parquet in 27 s: an **events** table (one row per event, history parsed once into
  lists) and a **candidates** table (one row per candidate).
- `verify_parquet.py`: 10 PASS/FAIL checks (row counts, 10 candidates per event, ID integrity, at most one click per
  event, history order, spot check against the CSV). All pass.
- Loaders: `load_events(split)`, `load_candidates(split)`, `load_joined(split)`.

### 3. Local validation (`src/validation.py`)
- `evaluation.csv` turned out to be **each user's first event after the training week**, with histories frozen at
  the end of that week. `make_split()` reproduces this inside train: fit on days 1–4, validate on each user's first
  event from day 5 on.
- The split matches eval closely: median history 14 vs 15, median 37 h vs 42 h since the last click.
- `score()` computes the local AUC of predictions in submission format.

### 4. Audit of Monish's pipeline (`research/techarena/`)
His SAFE_v2 features are strong, but his validation had **three leaks**:
1. **Training targets ran past the validation boundary.** 41–43% of validation events were also training rows,
   labels included.
2. **Early stopping on the validation set.**
3. **In-sample choice-model features** for training rows.

Rerunning his exact features on our data: leaky protocol **0.8028** local, leak-free **0.7919**. The leak inflated
his numbers by about 0.011, 4–5× larger than the thresholds he used to accept or reject ideas.
His final submission scripts were leak-free; only his measurements were affected.

- `leakfree_safe_v2.py`: reuses his feature code unchanged, fixes all three leaks, runs both protocols side by side,
  and `--submit` builds a leak-free submission (experiment 002).

### 5. Choice model (`src/choice_mf.py`) + SAFE_v2 (`research/techarena/safe_v2_choicemf.py`)
- Each event is modelled as an 11-way choice (10 candidates or no click), trained on **all impressions**, not just
  clicks, and started from the rank-11 SVD.
- Alone: 0.7858 local, versus 0.7701 for SVD.
- Added to SAFE_v2 as three LightGBM features (`cmf_prob`, `cmf_noclick`, `cmf_rank`), each window's model trained
  only on events before its boundary: **+0.0045 local, +0.0063 on the leaderboard** (experiment 003).

## What we learned about the data

**Where the signal is:**
- **User taste × item profile.** Taste has about **11 dimensions**: SVD peaks sharply at rank 11.

  | SVD rank | 8 | 10 | **11** | 12 | 16 | 32 |
  |---|---|---|---|---|---|---|
  | Local AUC | 0.730 | 0.761 | **0.770** | 0.765 | 0.749 | 0.736 |

- **Repeat clicks.** A candidate the user clicked before is clicked 31–35% of the time, against 5.7% otherwise.
- **The "no click" decision.** 37% of events have no click. Our model separates them at only 0.652 AUC; a perfect
  event-level prediction would lift global AUC to 0.88. This decision appears to be largely random.

**Tested and ruled out (no useful signal):**
| Idea | Result |
|---|---|
| Candidate position in the event | 0.501 AUC (click rate ≈ 0.063 at every position) |
| Time of day (`time_bucket`) | Click rate 0.0628–0.0633 in every bucket; item × bucket worse than item alone |
| Item popularity | 0.527 AUC alone |
| Eval-week exposure counts | 0.524 alone, 0.501 beyond our model |
| Click order / recency | Each history slot ≈ 0.68; mean of all 20 (0.796) beats the last 3 (0.742) |
| Item co-click kNN / EASE | 0.594 / 0.562 (too few clicks per item) |
| Slate composition | 0.506 alone, 0.48 beyond our model; slates only mildly personalised |

**Choice-model variants (all plateau at about 0.788 alone):**
| Variant | Local AUC |
|---|---|
| 32 dims + per-user vectors (used in the 0.814 submission) | 0.7858 |
| 11 dims | 0.7867 |
| 11 dims, time-free | 0.7873 |
| **11 dims, time-free, + auxiliary task on all past clicks (weight 0.3)** | **0.7882** |
| History-only (no per-user vectors) | 0.780–0.781 |
| Eval-matched reweighting | 0.781–0.784 |

Every variant peaks after 1–3 epochs. More tuning of this model type is unlikely to close the gap to the top teams.

## Next steps
1. **LightGBM-level gains** (est. 0.816–0.818 on the leaderboard): the improved choice model as a feature, training
   windows every 6 hours instead of 12, and averaging several models.
2. **Transformer (team task):** candidates attend to the relevant history items, plus a learned no-click token, with
   item embeddings started from the rank-11 SVD. The one untested idea that could add a new kind of signal.
3. **Outside hints:** check the competition forum or data description for how the data was generated.

## Reproduce
```bash
source .venv/Scripts/activate
python src/convert_to_parquet.py && python src/verify_parquet.py      # data (~1 min)
python research/techarena/leakfree_safe_v2.py                         # leaky vs leak-free validation (~60 min)
python research/techarena/leakfree_safe_v2.py --submit                # experiment 002 (~60 min)
python research/techarena/safe_v2_choicemf.py                         # validation, 0.7965 (~20 min)
python research/techarena/safe_v2_choicemf.py --submit                # experiment 003, submissions/safe_v2_cmf.csv (~60 min)
python src/check_submission.py submissions/safe_v2_cmf.csv data/evaluation.csv
```
The two `research/techarena` scripts reuse each other's cached features; run them in the order above.
