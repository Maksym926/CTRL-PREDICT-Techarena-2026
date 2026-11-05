# HANDOFF: full context for continuing in a new chat

> **Superseded on 3 Nov 2026: read `research/techarena/WHAT_WE_TRIED.md` first.** It has the current state (best
> local 0.807134, `submissions/safe_v2_step7_blend.csv`), all results, the code map and the next steps. Everything
> below is the 27 Oct state, kept for reference.

> New chat: "Read HANDOFF.md first, then continue." Written 27 Oct 2026. Team-facing summary: `PROGRESS.md` (covers
> results up to 0.814; this file supersedes it).

## 1. Project and current standing
- **Huawei Tech Arena 2026, Phase 1:** predict a click probability for every row of `data/evaluation.csv`
  (529,390 rows = 52,939 events × 10 candidates). Metric: **global ROC AUC** over all rows.
  Submission: CSV `rowid,predicted_probability`. **The user uploads it manually** (zipped) and reports the score.
- Phase 2 (later): a recommendation agent under a time limit. `src/model.py` has `ClickModel` (load once, predict in
  batches).
- Team name **"Ctrl + Predict"**. Teammates include Monish, who built `research/techarena/`.
- **Best leaderboard score: 0.819061** (about place 37). The board on 26 Oct had top 0.8329, 5 teams ≥ 0.8318 and
  34 teams > 0.82, so the "0.83 signal" is widely known and we haven't found it.

## 2. Environment
- Windows 11, repo `G:\Development\HuaweiTechArenaPrep\project\CTRL-PREDICT-Techarena-2026`. Shells: Git Bash
  (preferred) and PowerShell. Run scripts **from the repo root**, after `source .venv/Scripts/activate`.
- Python 3.14.7 in `.venv`: pandas, numpy, scikit-learn, lightgbm 4.7.0, matplotlib, jupyter, polars 2.0.0,
  pyarrow 26.0.0, torch 2.14.1+cpu (all in `requirements.txt` / `requirements.lock.txt`). **xgboost 3.4.1 is
  installed but deliberately not in the requirements** (it gave no gain).
- 12 CPU cores, 31.5 GB RAM, **no GPU**. Caches use about 5 GB in `research/techarena/cache/`.
- Gotchas:
  - Printing polars tables in the console crashes on cp1252; set `PYTHONIOENCODING=utf-8`.
  - `np.isin` on string arrays is quadratic (it hung a run). Use `pandas .isin(set(...))` or joins.
  - pandas 3 string arrays → polars: convert with `.tolist()` first.
  - polars 2.0: `Series.is_in(other.implode())`; use `pl.scan_csv(..., infer_schema=False)`.
  - **Never load `data/train.csv` (7.3 GB) fully.** Use the Parquet files.
  - **Caches are keyed only by boundary timestamp.** If feature code changes, delete the matching cache folder
    (see §7), or stale features are silently reused.
  - Long runs: start them in the background with logs in `logs/`. LightGBM fits take 3–10 min each.

## 3. How the user likes to work
- The user **commits and pushes themselves**. Give commit commands one per ```bash block. They have twice pasted a
  whole `git commit -m "..." -m "..."` into PyCharm's commit box, which mangled the messages (commits cea843b,
  95516ba). Remind them to run it in Git Bash, or give plain message text for the PyCharm box.
- The user approves each step ("yes run X"). Propose the next step and wait. For each run, state the expected
  outcome before it finishes and compare afterwards.
- Every submission gets a row in `experiments.csv` (local AUC + leaderboard).
- The user likes file-by-file / step-by-step explanations in plain English, with tables.

## 4. Data facts (verified)
- `train.csv`: 6,868,240 rows = 686,824 events × 10 candidates, 7 days (Mon 20 Jan 2025 00:00 UTC = 1737331200,
  to 27 Jan). Columns: `rowid, event_id, user_id, item_id, timestamp, time_bucket, day_type, history_item_ids,
  history_timestamps, clicked`. IDs are opaque strings (`row_/evt_/usr_/itm_` + hex).
- **Evaluation = each user's first event after the training week** (one event per user, 67% on day 8, then decaying).
  Its histories stop at the end of the training week (no clicks from the eval week), so they are "stale": median
  42.5 h since the last click, against 8.4 h for normal train events. Median history length is 15 in eval vs 20 in
  train.
- History = the user's past **clicks** (up to 20, oldest first), reaching back to 14 days before the training week.
  Every train click appears in that user's later histories.
- 6.29% of candidates are clicked. 62.9% of events have exactly one click; 37% have none.
- 97% of eval users appear in train; every eval candidate item was a train candidate. 50,000 items, 54,456 users.
- Days 6–7 (Sat–Sun) have `day_type = 1`. Eval is mostly weekdays.

## 5. Validation protocol (most important)
- `src/validation.py`: `make_split(cutoff_day=4)` gives fit = events before day 5; val = **each user's first event
  from day 5 on** (47,928 events). `score(val_candidates, preds_df)` gives the AUC.
- **Leaderboard ≈ local + 0.015 to 0.018**, confirmed on 4 submissions. About two thirds of a *local gain* carried
  over to the leaderboard in the latest submission.
- **Leak-free training protocol** (used everywhere since):
  - LightGBM trains on "windows": for each boundary B (every half day), targets = each user's first event after B
    (capped at the validation boundary when validating), with features computed **only from data before B**.
  - Tree count comes from an **inner holdout**: the latest window, fitting only on targets dated before it.
    **Never early-stop on validation.**
  - Choice-model features for training rows are **cross-fitted by user** (`user_id % 2`).
  - Submission mode: windows at days 0.5–6.5 (13), targets to the end of train, eval features at boundary
    CUT = day 7.

## 6. Code map
| File | Purpose |
|---|---|
| `src/data.py` | `load_events/load_candidates/load_joined(split)` from `data/parquet/` |
| `src/convert_to_parquet.py`, `src/verify_parquet.py` | CSV → Parquet (27 s), 10 checks (all pass) |
| `src/validation.py` | `make_split`, `score` (see §5) |
| `src/check_submission.py` | Official-rules checker: `python src/check_submission.py <sub.csv> data/evaluation.csv` |
| `src/evaluate.py` | `auc()` |
| `src/choice_mf.py` | **Choice model** (§8). Also `click_pairs`, `click_svd` (Monish's normalised rank-k SVD), `encode`, `taste_vectors`, `train`, `predict`. CLI flags: `--dim --svd-rank --l2 --user-emb --reweight --no-time --aux-weight --aux-ratio --aux-neg --max-epochs` |
| `research/techarena/src/` | Monish's code (60 scripts). Core: `features.py` (SAFE_v2 features), `svd_model.py`, `choice.py` (12-param choice model), `phase_c_real.shrink_extra` |
| `research/techarena/leakfree_safe_v2.py` (`lf`) | Loads Monish's package as `monish`, Parquet → his `GlobalDataset`, vectorised `click_log_before`. Helpers: `load_data, training_windows, boundary_frame, crossfit_choice, inner_holdout_trees, fit_seeds, evaluate, model_columns`. Runs leaky vs fixed protocol; `--submit` = experiment 002 |
| `research/techarena/safe_v2_choicemf.py` | + choice model v1 features (`cmf_prob/cmf_noclick/cmf_rank`); `--submit` = experiment 003 |
| `research/techarena/safe_v2_step1.py` | + choice model v2 + top-k similarity + event-level features |
| `research/techarena/safe_v2_step2.py` | + 33 rotated taste-vector features; **`--submit` = experiment 004, current best** |
| `research/techarena/safe_v2_step3_xgb.py` | XGBoost on step 2 features + blend (no gain) |
| `research/techarena/safe_v2_step3_6h.py` | Step 2 on 6-hour windows (no gain) |

## 7. Caches, outputs, runtimes
- Feature caches, all under `research/techarena/cache/`:
  - `leakfree/<boundary>.parquet`: SAFE_v2 frames, half-day 0.5–6.5, quarter-day 0.25–3.75, plus
    `evaluation.parquet`.
  - `choicemf/` (v1, validation) and `choicemf/full` (v1, submission).
  - `choicemf_v2/{val,full}`, `extra/{val,full}`, `taste_vectors/{val,full}`.
- Validation predictions: `data/validation/*_val_c4.parquet` (step1, step2, step3_xgb, step3_6h, safe_v2_cmf,
  choice_mf).
- Results JSON: `research/techarena/outputs/*.json`. Logs: `logs/`.
- Submissions in `submissions/`: `starter.csv`, `safe_v2_leakfree.csv`, `safe_v2_cmf.csv`, `safe_v2_step2.csv`.
- Runtimes:
  - `leakfree_safe_v2.py`: validation about 60 min (builds features), `--submit` about 57 min.
  - `safe_v2_step1.py`: validation about 22 min.
  - `safe_v2_step2.py`: validation about 27 min, `--submit` about 60 min (needs both leakfree caches).
- Order on a fresh machine: `convert_to_parquet` → `leakfree_safe_v2.py` → `leakfree_safe_v2.py --submit` →
  `safe_v2_step2.py` → `safe_v2_step2.py --submit`. Step 2 builds step 1's features through cache.

## 8. Results
**Submissions** (`experiments.csv` rows 001–004):
| # | Model | Local | LB | Gap |
|---|---|---|---|---|
| 001 | Starter (unmodified) | 0.6039 | 0.6200 | +0.016 |
| 002 | Monish SAFE_v2, leak-free | 0.7919 | 0.8077 | +0.016 |
| 003 | + choice model v1 (32 dims) | 0.7965 | 0.8141 | +0.018 |
| **004** | **+ choice model v2 + event/top-k features + rotated taste vectors (109 features, 485 trees × 3 seeds)** | **0.8042** | **0.8191** | +0.015 |

The team's earlier best was Monish's `SUBMISSION_EVENT_ADAPTIVE_GRAPH` at 0.8116: a blend with "graph80", which has
no source code.

**Validation-only results (not submitted):**
- Step 1 (choice model v2 + event/top-k features) = 0.7990.
- Step 3: XGBoost alone 0.804149; 6-hour windows 0.804083; blends at most 0.804432 (+0.0003). All variants have
  rank correlation 0.996–0.997 with step 2, so not worth submitting.

**Choice model** (alone, local AUC):
| Variant | AUC |
|---|---|
| v1: 32 dims, `l2` 0.1 (in 003) | 0.7858 |
| 11 dims | 0.7867 |
| 11 dims, time-free | 0.7873 |
| **v2: 11 dims, time-free, auxiliary task on all past clicks with weight 0.3, `l2` 0.01, 2 epochs** | **0.7882** |
| History-only (no per-user vector) | 0.780–0.781 |
| Eval-matched reweighting | 0.781–0.784 |

Every variant peaks at epoch 1–3 and then declines. It has plateaued.

**Step 2 feature importance:** `cmf_prob` #1, `choice_probability` #2, `cmf_rank` #3, taste products for dims 8/9
#4/#5 (taste features 15% of total gain), `user_distinct_clicked` #8, `user_event_click_rate` #15,
`hours_since_last_click` #17, `hist_top3_cos` #29. The `hist_n_cos_gt*` count features are useless (#64–73) and can
be dropped.

## 9. What we know about the signal
- **Taste (user × item) is nearly everything.** Normalised click-SVD peaks sharply at **rank 11** (8: 0.730,
  10: 0.761, **11: 0.770**, 12: 0.765, 16: 0.749, 32: 0.736). The singular values are
  `[0.938, 0.563, 0.548 … 0.539]`: a popularity axis plus about 10 **equally strong** taste dimensions (axes are
  arbitrary within that space). ICA shows mostly Gaussian structure (no topics), and k-means finds no clusters.
- **Repeats:** a candidate that is already in the history is clicked 31–35% of the time, against 5.7% otherwise,
  regardless of how recent the earlier click was.
- **No click (37% of events):** event-level AUC is only 0.652. A perfect event prediction would give 0.88 global,
  but a user's click rate in one period predicts the next only weakly (Monish: correlation 0.22).
- **Ruled out:**
  - candidate position (0.501);
  - time of day / `time_bucket` (flat click rate);
  - item popularity (0.527);
  - eval-week exposure counts;
  - click order / recency (each history slot about 0.68; the mean of all 20 history items 0.796 beats the last 3,
    0.742);
  - item co-click kNN (0.594) and EASE (0.562);
  - slate composition (0.506; slates are only mildly personalised);
  - per-dimension weighting (+0.002 alone);
  - XGBoost blending;
  - 6-hour windows.

## 10. Monish's branch (`monish`)
- He found the SVD signal and built SAFE_v2 (about 65 features + choice features + LightGBM). His validation folds
  DM22/23/24 had **3 leaks**:
  1. training targets running to the end of train, so 41–43% of validation events were also training rows;
  2. early stopping on validation;
  3. in-sample choice features.
- His leaky 0.8028 became **0.7919** leak-free on the same features. His accept/reject threshold (+0.002) was smaller
  than the leak (+0.011), so his rejected ideas were never fairly tested.
- His final submission scripts were leak-free. Many of his ~55 experiment scripts still contain the leak.
  `tests/`, `scripts/` and `requirements-gpu.txt` referenced in his README are missing. Data paths default to his
  Mac.

## 11. Git state (27 Oct)
- **Current branch: `monish`.** The suggested `leakfree-choicemf` branch was never created; commits 95516ba (mangled
  message) and e8da84e are on `monish`. Check whether they were pushed.
- **Uncommitted:**
  - modified: `experiments.csv` (row 004), `src/choice_mf.py` (`user_vector`, `taste_vectors`, `--reweight`,
    `--no-time`, aux click task), `research/techarena/safe_v2_choicemf.py` (args/epochs/no_time/aux support);
  - new: `research/techarena/safe_v2_step1.py`, `safe_v2_step2.py`, `safe_v2_step3_xgb.py`, `safe_v2_step3_6h.py`,
    `HANDOFF.md`.
- `PROGRESS.md` only covers up to the 0.814 result.

## 12. Next steps (backlog, roughly by value)
1. **Outside hints:** competition forum, FAQ or data description for how clicks were generated. Most likely route to
   the 0.83 signal.
2. **Transformer (team task, ideally on a GPU):** candidates attend to history items, plus a no-click token, item
   embeddings started from rank-11 SVD, epochs picked on the inner holdout. It must be trained **per window** (only
   data before each boundary) and output scores per `rowid`, to plug in like `cmf_features()`. Today's evidence
   (top-k, recency, slate) suggests a modest gain. CPU feasible (about 2–5 min/epoch) but slow to iterate.
3. Small items, each about +0.001 on the leaderboard:
   - Optuna tuning of LightGBM on step 2 features;
   - ICA-basis taste vectors (+0.003 alone);
   - averaging choice models (ranks 11 and 12, several seeds);
   - dropping the `hist_n_cos_gt*` features.
4. Untested leak-free ideas from Monish:
   - LightGBM trained on **all** events (not only first-after-boundary);
   - his logistic-MF and graph signals.
5. Better "no click" (event-level) modelling: the largest theoretical headroom, but it looks mostly random.
