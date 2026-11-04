# Tech Arena 2026, Phase 1: full handoff (everything we tried)

> **New session: read this file first.** Last updated 4 Nov 2026, 16:00 (§17: 4 Nov late afternoon). Deadline: 4 Nov
> midnight, Dublin time (this PC's clock).
> It replaces `HANDOFF.md` (27 Oct) and `PROGRESS.md`. Results JSONs: `research/techarena/outputs/`; logs: `logs/`.

**STATE 4 Nov 16:00 (read this first).**
- Deadline: **midnight Dublin time** (= this PC's clock, UTC+0; the second PC's clock is UTC+2). LB best still
  **0.821624** (005). The user (4 Nov 15:30): **no upload until something reaches LB 0.83**; keep searching, try
  completely different approaches, ask them to send files to the GPU PC (Google Drive) when needed.
- **Package G built and checked (15:50): `submissions/safe_v2_step7_s30_final_g.csv`** = best file. Held-out
  0.808004 / 0.810281 (F 0.807880 / 0.810090), validation 0.809212 (F 0.809056), valid (529,390 rows), Spearman with F
  0.99929; expected LB ≈ 0.8232. (F = previous best, `..._final_f.csv`.) If 0.83 is not reached, still recommend
  uploading G before midnight as insurance. After any upload: `experiments.csv` row 006 and push.
- Tests 4 Nov afternoon (§17): **the evaluation set is NOT clicked-events-only** (coverage of never-clicking users and
  delay-vs-click-rate match "first event after the cutoff"; the `final_f_share.csv` probe would lose ~0.005, dropped).
  Items shown-but-not-clicked to the same user earlier: click rate 0.055 vs 0.057 (no "future clicks as fillers" leak;
  `shown_before` features exist anyway). Within-event order by rowid / item_id: uniform.
- Mult-VAE, PLE-encoded MLPs, gap/cold-item/history-filler probes: no gain (§17).
- **Lead 16:40: `nn_step7.py --window-day`** (the row's window boundary in days as an input; targets get their own
  boundary, so at day 7 the stack behaves like its latest, most data-rich window): context MLP lr 2e-4, 3 seeds
  0.808138 alone vs 0.807954 plain (+0.00018); `--window-weight 1` −0.00009, both +0.00014. Running: second PC
  `lgb_exp17.bat` (validation: wday context seeds 3–23 + wday attention ×12) → scratch `h2_choice.py` (G vs G-wday
  mixes, rule ≥ +0.0001 both halves) and `lgb_exp18.bat` (the two wday MLP submissions, fetched to `submissions/`);
  then the partner with `softmax_step7.py --window-day --tag _s30_opt4_wday` (validation, second PC CPU). If the
  MLP mix passes: `weighted_avg.py` (2 × ctx_lr2_wday_s24 + attn1_lr2_wday_s12) → `final_blend.py --val-c/--sub-c` →
  `submissions/safe_v2_step7_s30_final_h.csv`.
- **Running 16:25 (day-6 validation, §17):** is G's blend (weights fitted at day 4) still right at a later boundary?
  New `--val6` mode in `safe_v2_step7.py`, `softmax_step7.py`, `nn_step7.py`: train on the submission frames' windows
  0.5–5.5 (targets before day 6), score window 6.0 → `data/validation/*_val6.parquet`. Jobs: binary on the second
  PC's CPU (log there `logs\lgb_val6.log`), partner here (`logs/softmax_step7_s30_opt4_val6.log`, ~70 min), MLPs
  (context ×24, attention ×12) on the second PC's GPU = `D:\HuaweiTechArena\lgb_exp16.bat` after batch 15
  (`--window-day` / `--window-weight` MLP test). Then the scratch `blend6.py` (copy in this session's scratchpad):
  rule = day-6-fitted weights ≥ +0.0002 over G's on both day-6 halves → refit the blend on day 4 + day 6 pooled.
- Main PC froze at 15:00 with the CPU MLP build + two blend scripts + a Google Drive upload: run one heavy job at a
  time here. The second PC: GitHub pull refused (deploy key), copy changed files by scp; its GPU does the MLPs
  (`--device cuda`). The Google Drive copy of the frames is also in
  `D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026\for_second_pc-20261104T151107Z-1-002\` there (can be deleted).

**Update 4 Nov 05:20 — best file to upload: `submissions/safe_v2_step7_s30_final_f.csv` (package F)**: C's binary
LightGBM + the softmax partner with tuned parameters + the MLP stack (`nn_step7.py`), combined by `final_blend.py`.
Validation 0.809056 in-sample, held-out halves 0.807880 / 0.810090 (≈ +0.0012 over package C's rank blend, 0.807766);
eval-reweighted 0.812795 (C 0.811507); checked valid (529,390 rows). Expected LB ≈ 0.823 (005 scored 0.821624 at
local 0.807134). Fallbacks, also valid: `safe_v2_step7_s30_final_e.csv` (E, no MLP: 0.808377) and
`safe_v2_step7_s30_final_c.csv` (0.808133). Not uploaded yet: the user uploads; add the `experiments.csv` row (006)
with the LB score. 0.83 is out of reach with these models (needs local ≈ 0.817).

## 0. Where we are (3 Nov, 21:50)

| | |
|---|---|
| Deadline | **Phase 1 closes 4 Nov 2026, midnight** |
| Our best leaderboard (LB) score | **0.821624** (experiment 005, 3 Nov: `safe_v2_step7_blend.csv`); previous 0.819061 (004) |
| Top of the leaderboard | 0.834673 (3 Nov). On 26 Oct: top 0.8329, 5 teams ≥ 0.8318, 34 teams > 0.82 |
| **Best built file (not uploaded yet)** | **package C, local 0.807766** (+0.00063 vs 005) → `submissions/safe_v2_step7_s30_blend_c.csv` (checked valid); expected LB ≈ 0.8222 |
| Local → LB gap | +0.0145 to +0.0176 over 5 submissions (~86% of the last local gain carried over) |
| To reach LB 0.83 | local ≈ 0.817 (+0.0098 over C). Every model-side idea now gives ≤ +0.0005 (§13–§14) |

**Packages (local AUC):** A `safe_v2_step7_blend2.csv` 0.807361 · B `safe_v2_step7_s30_blend.csv` 0.807593 ·
**C `safe_v2_step7_s30_blend_c.csv` 0.807766** (step 7 with 30-seed sequence features + softmax partner retrained on
them) · D (overnight, below). The user had not uploaded A/B/C at 21:50 (wanted ≥ 0.83 first); recommendation: upload
the best of C/D anyway. Every upload gets an `experiments.csv` row.

**Overnight jobs 3–4 Nov (running, all restart-safe; scripts copied to `research/techarena/queues/`):**
1. **Second PC** (`ssh yulia@100.103.24.101`, see the memory note / §14): one SSH chain runs `cmf_seeds.py` for cmf12
   seeds 3–9, cmf16 seeds 1–4, cmf (v2) seeds 1–4 (≈5 min per seed there), then partial averages
   `cache/choicemf_{d12,d16,v2}_part` (`--no-base`). Log there: `logs\cmf_chain.log`. Seeds 1–2 of cmf12 were done
   first (`logs\cmf_d12_s1.log`, `_s2.log`). At 21:45 cmf12 seed 8 was running.
2. **This PC, `queues/queue_d_transfer.sh`**: waits for `choicemf_v2_part\seeds.json` on the second PC, copies the
   three `_part` folders here by scp, then touches `logs/queue_d_transfer.done`.
3. **This PC, `queues/queue_d.sh`** (log `logs/queue_d.log`): cmfp seeds 1–4 here → merge all four features with seed 0
   (`cache/choicemf_*_avg`) → `safe_v2_step7.py --seq stage3_h50_s30 --swap cmf=choicemf_v2_avg cmf12=choicemf_d12_avg
   cmf16=choicemf_d16_avg cmfp=choicemf_pop_avg --tag _d` (validation) → `softmax_step7.py` with the same swaps →
   `blend_val.py` → **builds package D only if the blend ≥ 0.807866** (C + 0.0001) →
   `submissions/safe_v2_step7_d_blend.csv`, then "queue finished". Expected D: local 0.8079–0.8082, LB ≈ 0.8223–0.8225.
- **To check:** `tail logs/queue_d.log`; second PC: `ssh ... "type D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026\logs\cmf_chain.log"`.
- **Second PC and git:** it runs an untracked scp copy of `research\techarena\cmf_seeds.py` (same content as the
  committed one). Once its chain is finished, delete that file there before `git pull`, or the pull refuses.
- **If a job died** (e.g. the session closed): re-run the same `cmf_seeds.py --model M --seed S` (finished boundaries
  skip), then `bash research/techarena/queues/queue_d_transfer.sh` and `bash research/techarena/queues/queue_d.sh`
  (they wait for their inputs; queue_d skips nothing, so delete partial outputs of an interrupted step first).
  Windows: TaskStop/killing a Git Bash script can leave its python children running — check `Get-CimInstance
  Win32_Process` command lines before restarting, and never edit a running bash script.

---

## 1. Task and rules
- Predict a click probability for every row of `data/evaluation.csv`: 529,390 rows = 52,939 events × 10 candidates.
  Metric: **global ROC AUC** over all rows. Submission: CSV `rowid,predicted_probability`. **The user uploads it
  manually** and reports the score. Checker: `python src/check_submission.py <sub.csv> data/evaluation.csv`.
- Phase 2 (later): a recommendation agent under a time limit; `src/model.py` has `ClickModel` (load once, predict in
  batches).
- Team "Ctrl + Predict"; teammate Monish built `research/techarena/` (branch `monish`, which we work on).
- **Another team's CSV** (`G:\Downloads\submission_champ_dominant_free_lunch_933_67.csv`) is **analysis only, never
  blend it or train on it**. Analysis: 0.959 rank correlation with ours; their edge was better taste modelling, no leak.

## 2. Environment, machine limits, gotchas
- Windows 11, repo `G:\Development\HuaweiTechArenaPrep\project\CTRL-PREDICT-Techarena-2026`. Run everything **from
  the repo root** with `.venv/Scripts/python.exe` (Bash's `python` has no pandas); set `PYTHONIOENCODING=utf-8`
  (polars prints crash on cp1252).
- Python 3.14 `.venv`: pandas, numpy, scikit-learn, lightgbm 4.7, polars 2.0, pyarrow, torch (CPU). xgboost is
  installed but deliberately not in requirements.
- **12 logical cores, 31.5 GB RAM, commit limit ~65–72 GB, no local GPU.** A submission build (6M rows) peaks at
  23–28 GB; a validation run at 15–20 GB.
  - **Run heavy jobs one at a time.** Three at once hit 67.8 GB commit and paged the running job.
  - Below-normal-priority jobs starve almost completely while a LightGBM job uses all cores.
  - Several torch jobs that each call `set_num_threads(12)` oversubscribe the CPU; cap them at about 4.
  - **Never let two processes write the same cache file.**
- **Caches are keyed only by boundary timestamp.** If feature code changes, delete the matching cache folder, or
  stale features are silently reused.
- Never load `data/train.csv` (7.3 GB); use `data/parquet/`. `np.isin` on strings is quadratic (use joins).
  polars `join` without `how="left"` does not keep row order.
- Kaggle (GPU): dataset `maksymchechotkin/ctrl-predict-parquet`. Notebooks are built by
  `research/techarena/make_kaggle_notebook.py` (embeds the current `seq_model.py`; re-run after changing it).
  - On Kaggle: Import Notebook → Add Input (dataset) → Accelerator **GPU T4 x2**.
  - Run the first two cells (GPU and data checks), then Save Version → Save & Run All, and download the zip from
    the Output tab.
  - The user needed phone verification for GPU access (done).

## 3. How the user likes to work
- The user **commits and pushes themselves, in Git Bash**. Give one command per ```bash block; messages via `-m`,
  ending with `-m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`. Pasting `git commit -m ... -m ...`
  into PyCharm's box mangled messages twice.
- Approves each step; state the expected outcome before a run and compare after. Plain-English explanations with
  tables. Never upload or commit for them.
- **Decision rule, fixed before each validation run:** submit only if local AUC gains at least **+0.0005**.
- Every LB submission gets an `experiments.csv` row (id, date, author, change, local_auc, lb_auc, notes).

## 4. Data facts
- **Shape:**
  - Train: 686,824 events × 10 candidates, 7 days from Mon 20 Jan 2025 00:00 UTC (D0 = 1737331200).
  - Eval: 52,939 events, each user's first event after the training cutoff (CUT = D0 + 7 days). 67% fall on
    day 7 (0-based, the day right after CUT).
  - 54,456 users, **exactly 50,000 items**.
- **Columns:**
  - events: `event_id, user_id, timestamp, time_bucket (0-3), day_type (1 on days 5-6 = weekend)`, plus
    `history_item_ids` / `history_timestamps`: the user's last ≤ 20 clicks, oldest first;
  - candidates: `rowid, event_id, item_id, clicked`.
  - IDs are salted hashes (no MD5/SHA pattern up to 120k).
- 62.9% of events have one click, 37.1% none; 6.29% of candidates are clicked. Every in-period click is an event
  click and appears in the user's later histories; 581,168 pre-period clicks are known only from histories.
- **The data is synthetic**, and we measured its generator:
  1. Clicked event = the real next item + 9 fillers; no-click event = 10 fillers. Non-clicked candidates look
     identical in both.
  2. **Click or not is a fixed per-user propensity.** Click rate is flat by day (0.626–0.632), time bucket and
     weekday vs weekend. Past click rate alone gives event-level AUC 0.59; timing signals ~0.50.
  3. Taste is static: about 10 latent dims plus 1 popularity axis. The click-SVD rank sweep peaks at 11; singular
     values 0.94, then a flat 0.568…0.543. There are no time-of-day or weekday item profiles.
  4. **Explicit repeat step:** ~12% of clicks re-click a history item, flat across history lengths 3-20, picked
     by recency (~0.85× per position).
  5. **The filler sampler draws ~2% of slots from the user's own history**, with the same recency profile.
     - About 2/3 of in-history candidates are fillers. Only taste separates them from real repeats (AUC 0.677).
     - An oracle fixing this would add +0.012 global, but no extractable signal was found beyond the stack.
     - The sampler also leans slightly toward the user's taste (cosine 0.169 vs 0.155 for popularity-matched
       random items; real items 0.424). Fillers are not drawn near the real item (slate centrality 0.505).
  6. History field = exactly "the last 20 clicks": events with fewer than 20 never have older known clicks.
     35% of validation targets and 37% of eval users have more than 20 known clicks.

## 5. Validation protocol (leak-free)
- `src/validation.py`: `make_split(4)`.
  - Fit = events before day 4 (0-based; D0 + 4 days = 1737676800).
  - Validation = each user's first event at or after it: 47,928 events, 479,280 rows.
  - `score(val_candidates, preds)` gives the AUC.
- **LightGBM windows:**
  - For each boundary B every half day (validation: days 0.5–3.5, 7 windows; submission: days 0.5–6.5, 13 windows),
    target = each user's first event after B. Features come **only from data before B**.
  - The same event can be a target in several windows, so merge on (rowid, window_id).
- **Tree count from the inner holdout:** the latest window, fitting only on targets before it. **Never early-stop on
  validation.** Choice-model features for training rows are cross-fitted by user (`crossfit_choice`).
- LightGBM: Optuna parameters from `outputs/optuna_step4.json`, loaded by `safe_v2_stage3.py` at import; 3 seeds
  (`lf.FULL_SEEDS`) averaged.
- **Fast path (since 3 Nov):** finished feature tables are saved in `research/techarena/cache/lab/`.
  - `step6_train/val.parquet` = step 5's validation frames + step 6's columns.
  - `step5_full_train/test.parquet` = step 5's submission frames.
  - New feature columns can be merged in by rowid and LightGBM retrained without rebuilding everything
    (`safe_v2_step7.py`). Step 5 retrained this way gives 0.806467 vs the original 0.806442, so the path is faithful.

## 6. Code map

| File | Purpose |
|---|---|
| `src/data.py`, `src/validation.py`, `src/evaluate.py`, `src/check_submission.py` | Parquet loaders, split/score, AUC, official-format checker |
| `src/choice_mf.py` | **Choice model**: softmax over the 10 candidates + no-click, user vector + recency-weighted last-20 history, SVD-initialised, auxiliary task over all past clicks. `--aux-pop 1` draws auxiliary negatives like the fillers (step 5). `click_pairs`, `click_svd`, `encode`, `train`, `predict`. `--tag` protects the reference output |
| `research/techarena/src/` | Monish's package: `features.py` (SAFE_v2 features), `svd_model.py`, `choice.py` |
| `leakfree_safe_v2.py` (`lf`) | Leak-free protocol helpers: `training_windows, boundary_frame, crossfit_choice, inner_holdout_trees, fit_seeds, evaluate, model_columns`; `--submit` = 002 |
| `safe_v2_choicemf.py` | + choice model v1 (003); `cmf_features()` caches per boundary |
| `safe_v2_step1.py`, `safe_v2_step2.py` | + choice model v2, top-k/event features; + rotated taste vectors; `--submit` = **004** |
| `safe_v2_step3_xgb.py`, `safe_v2_step3_6h.py` | XGBoost, 6-hourly windows (no gain) |
| `safe_v2_step4.py`, `optuna_step4.py` | + 12/16-dim choice models; Optuna tuning (0.804643) |
| `seq_model.py` | **Sequence model** (transformer: candidates attend to the history; softmax over 10 + no-click). Modes `val / full / boundary / windows`. Flags `--seeds --seed --skips --augment --hist N --search-once --inner-only --fit-frac --tag` |
| `make_kaggle_notebook.py` → `kaggle/*.ipynb` | `seq_variants`, `seq_stage3` (20 clicks, 5 seeds), **`seq_h50`** (50 clicks, 5 seeds on T4 x2, ~21 min), `seq_h50_more` (seeds 5-9) |
| `safe_v2_stage3.py` | + sequence features from `cache/seq/<--seq>` (stage 3) |
| `signal_detectors.py`, `ease_detector.py` | Slate / trend / transition features; EASE (inner-holdout detectors) |
| `safe_v2_extra.py` | Stage 3 (5 seeds) + small wins (0.806163) |
| `safe_v2_step5.py` | + filler-matched choice model (0.806442); `--submit` also saves `cache/lab/step5_full_*` |
| `safe_v2_step6.py` | + "best of the other 9 / margin" columns (no gain); `--dump` saves `cache/lab/step6_*` |
| **`safe_v2_step7.py`** | **Step 5 + 50-click sequence features** from saved frames. `--baseline` reproduces step 5. Validation ~35 min, `--submit` ~65 min |
| `softmax_objective.py`, `softmax_blend_val.py` | LightGBM with an event-softmax objective (custom objective + no-click); blend check |
| `lab_ensemble.py`, `lab_submit.py` | Feature lab on saved frames; submission from saved frames (`--softmax`, `--ensemble`, `--base step5`) |
| `blend_csv.py` | 50/50 rank blend of two submission CSVs |
| `merge_seq_dirs.py` | Seed-weighted average of sequence-feature folders (e.g. h50 seeds 0-4 + 5-9) |
| `learning_curve.py`, `ceiling_test.py`, `safe_v2_dense.py` | Diagnostics; 6-hourly windows on step 4 features (stopped; expected ~0) |

## 7. Caches, outputs, runtimes
- `research/techarena/cache/`:
  - `leakfree/` (SAFE_v2 frames);
  - `choicemf/`, `choicemf_v2/`, `choicemf_d12/`, `choicemf_d16/`;
  - `choicemf_pop/{val,full}` (step 5);
  - `choicemf_pop_d12/{val, full 11 of 14}`, `choicemf_pop_d16/val`;
  - `extra/`, `taste_vectors/`;
  - `seq/stage3` (1 seed), `seq/stage3_s5` (5 seeds, 20 clicks), **`seq/stage3_h50` (5 seeds, 50 clicks)**;
  - `lab/` (saved frames, ~2.5 GB).
- Validation predictions: `data/validation/<name>_val_c4.parquet` (every step, the choice-model variants, softmax).
- **Submissions** (`submissions/`, git-ignored):

| File | Local | Note |
|---|---|---|
| `safe_v2_step2.csv` | 0.804154 | = 004, LB 0.819061 |
| `safe_v2_stage3.csv` | 0.805031 | stage 3, 1 seed (submitted? LB pending) |
| `safe_v2_stage3_s5.csv` | 0.805634 | 5 seeds |
| `safe_v2_extra.csv` | 0.806163 | + small wins |
| `safe_v2_step5.csv` | 0.806442 | + filler-matched choice model |
| `safe_v2_step5_softmax.csv` | (alone 0.806305) | softmax-objective model, 2498 trees × 3 seeds |
| `safe_v2_step5_blend.csv` | 0.806868 | step 5 + softmax |
| `safe_v2_step7.csv` | 0.806901 | + 50-click sequence features |
| **`safe_v2_step7_blend.csv`** | **0.807134** | **= 005, LB 0.821624** |

- Runtimes on this machine (alone):
  - step 5: validation 96 min, `--submit` 142 min; step 7 (saved frames): validation 35 min, `--submit` 65 min;
  - softmax submission 219 min (trees hit the 2500 cap);
  - Kaggle `seq_h50`: 21 min on T4 x2; the old per-boundary-search 20-click run took 262 GPU-min.

## 8. Results

### Main pipeline
| Step | Change | Local | LB |
|---|---|---|---|
| 001 | Starter | 0.603913 | 0.619958 |
| 002 | Monish's SAFE_v2 features, leak-free (the original leaky protocol: 0.802783) | 0.791945 | 0.807715 |
| 003 | + choice model v1 | 0.796472 | 0.814056 |
| 004 | + choice model v2, top-k/event features, rotated taste vectors | 0.804154 | **0.819061** |
| — | Step 4: + 12/16-dim choice models, Optuna | 0.804643 | — |
| — | Stage 3: + sequence model (20 clicks, 1 seed) | 0.805031 | pending |
| — | Stage 3, 5 seeds | 0.805634 | — |
| — | + small wins (slate, trends, transitions, EASE) | 0.806163 | — |
| — | Step 5: + choice model with filler-matched auxiliary negatives | 0.806442 | — |
| — | Step 7: + sequence model fed up to 50 known clicks (Kaggle, 5 seeds) | 0.806901 | — |
| 005 | **Step 7 + softmax-objective model, 50/50 rank blend** | **0.807134** | **0.821624** |

### What helped (all small)
| Idea | Effect |
|---|---|
| Slate / trend / transition features (inner holdout) | +0.00038 / +0.00032 / +0.00017; together +0.00053 |
| EASE item-item model | +0.00029 |
| 5 sequence-model seeds instead of 1 | +0.004 alone, +0.0006 in the stack |
| Choice-model auxiliary negatives drawn like the fillers | +0.0015 alone, +0.0003 in the stack (step 5) |
| **Sequence model fed up to 50 known clicks** (`--hist 50`) | alone 0.795763 vs 0.793931 (+0.0018); on the latest, most eval-like inner holdout 0.8201 vs 0.8152 (+0.0049); in the stack +0.00043 (step 7); `seq50_prob` is the #5 feature |
| Binary stack + softmax-objective stack, 50/50 rank blend | +0.0004 on step 5; step 7 + softmax = +0.00067 |

### What did not help
| Idea | Result |
|---|---|
| XGBoost instead of LightGBM | 0.804149 vs 0.804154; blend 0.804333 |
| 6-hourly training windows | 0.804083 (−0.00007) |
| Sequence-model regularisation (3 variants) | inner 0.7955 / 0.7933 / 0.7940 vs 0.7956 |
| Sequence model with skipped-item tokens | inner 0.7858 vs 0.7943 |
| Sequence model + pre-period clicks as extra training events | validation 0.7898 vs 0.7908 |
| Step 6: "best of the other 9 candidates" / margin features | +0.00004 |
| Choice models across widths (11/12/16 dims) + ensemble columns | +0.0030 alone, +0.0002 in the stack |
| Second-stage model on in-history candidates | −0.0001 to −0.0003 |
| Smarter combinations of existing models | logistic stacker +0.00065 (≈ +0.0002 over the simple blend); LightGBM stacker +0.00009; raw taste models in blends −0.0005 |
| Co-click smoothing of rare items' taste vectors (outside proposal) | rare items 0.7055 → 0.7396 alone, popular items drop equally; +0.000008 beyond step 5 |
| Per-user filler sampler / slate centrality / pseudo-labelling / 11-dim SVD geometry (outside proposals) | mild taste tilt already learned; 0.505; max eval prob 0.77 < 0.9; already in |
| Full-catalog next-click model (CPU prototype) | inconclusive: barely trained in 80 min (within-event 0.570); needs a GPU |
| Session-window taste vectors; EASE/item-kNN as the main model; per-user repeat propensity; item time profiles; deep repeats beyond 20 | 0.712 vs 0.751; too sparse; 0.515; 0.495; 0.09% of clicks |
| Ceiling test (simulated history-free world) | flawed design |
| Leak hunts: ID hashes, row order, slates, timestamps, momentum, concurrent exposure, impression vectors | nothing |

## 9. Where the headroom is
| Finding | Number |
|---|---|
| Effect on global AUC of +0.01 in "which item" (within event) | +0.0095 |
| Effect on global AUC of +0.01 in "whether they click" (event level) | +0.0008 |
| "Which item" AUC of step 5 | 0.795 (repeat clicks 0.98, new items 0.77) |
| Oracle: history-drawn fillers scored like ordinary fillers | 0.818 (+0.012), unattainable with current features |
| Within-event binary-model calibration (share temperature test) | already optimal |

"Which item" AUC by the user's total known clicks:

| Known clicks | 0–2 | 3–5 | 6–10 | 11–15 | 16–20 | 21–30 | 31–40 |
|---|---|---|---|---|---|---|---|
| AUC | 0.717 | 0.756 | 0.784 | 0.805 | 0.817 | 0.82 | 0.836 |

Weak spots: users with few clicks and rare items (items in 3–9 users' histories: 0.754). Every taste model gains a
lot alone from averaging seeds or widths; the stack captures most of that. The 50-click sequence model is the only
change that improved a taste model's *information* (more context), and its gain grows later in the week.

## 10. Monish's branch notes
- Monish built SAFE_v2 and found the SVD signal. Their validation had 3 leaks:
  1. training targets past the validation boundary;
  2. early stopping on validation;
  3. in-sample choice features.
- Fixing them turned the leaky 0.8028 into a leak-free 0.7919. Many of the ~55 experiment scripts on that branch
  still leak; the final submission scripts do not.

## 11. Git and experiments.csv (pending)
- Branch `monish`. `submissions/`, `research/techarena/cache/`, `research/techarena/outputs/` and `logs/` are
  git-ignored. Commit these in Git Bash (three commits, one command per block):

```bash
git add src/choice_mf.py research/techarena/seq_model.py research/techarena/make_kaggle_notebook.py research/techarena/kaggle/ research/techarena/safe_v2_stage3.py research/techarena/signal_detectors.py research/techarena/ease_detector.py research/techarena/ceiling_test.py research/techarena/learning_curve.py research/techarena/safe_v2_extra.py research/techarena/safe_v2_dense.py research/techarena/safe_v2_step5.py
```

```bash
git commit -m "Stage 3 seq features, small wins, step 5 and the 50-click sequence model" -m "- seq_model.py: per-window sequence features on Kaggle GPU (--mode windows, --seeds, --hist 50, --search-once); stage 3 local 0.805634
- signal_detectors.py, ease_detector.py, safe_v2_extra.py: slate, trend, transition and EASE features (local 0.806163)
- src/choice_mf.py --aux-pop: auxiliary negatives drawn like the fillers; safe_v2_step5.py: local 0.806442
- kaggle/seq_h50.ipynb, seq_h50_more.ipynb: 50-click sequence model, 5 seeds on T4 x2 in 21 min" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

```bash
git add research/techarena/safe_v2_step6.py research/techarena/safe_v2_step7.py research/techarena/softmax_objective.py research/techarena/softmax_blend_val.py research/techarena/lab_ensemble.py research/techarena/lab_submit.py research/techarena/blend_csv.py research/techarena/merge_seq_dirs.py research/techarena/WHAT_WE_TRIED.md HANDOFF.md experiments.csv
```

```bash
git commit -m "Step 7 + softmax blend: LB 0.821624 (local 0.807134), labs and full handoff" -m "- safe_v2_step7.py: step 5 + 50-click sequence features from saved frames, 0.806901; 50/50 rank blend with the softmax-objective model 0.807134, LB 0.821624 (experiments.csv row 005)
- safe_v2_step6.py (no gain), softmax_objective.py, softmax_blend_val.py, lab_ensemble.py, lab_submit.py, blend_csv.py, merge_seq_dirs.py
- WHAT_WE_TRIED.md: full handoff (results, protocol, code map, next steps)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- `experiments.csv` has rows 001–005 (005 = step 7 blend, LB 0.821624). Add one row per further upload, for example:
  `006,<date>,Maksym,"<change>",<local>,<LB>,"<file; notes>"`
  (and a row for the 1-seed stage 3 file if it was uploaded: local 0.805031).

## 12. Next steps toward 0.83 (by value)
Honest expectation: each item below adds roughly +0.0002–0.001 locally. Reaching 0.83 needs about +0.008, so
outside information is still the most likely route to the big jump.

1. **Improve the 50-click sequence model** (the newest model; it gains most where users have many clicks). Each
   variant is cheap on Kaggle: about 5 min on the inner holdout with `--mode val --inner-only --hist 50`, or 21 min
   for all 14 windows with `--mode windows --search-once`. Then locally: put the folder in `cache/seq/<name>`, run
   `safe_v2_step7.py --seq <name>` (35 min), then `--submit` (65 min), then `blend_csv.py` with the softmax CSV.
   - a. **More seeds:** `kaggle/seq_h50_more.ipynb` (seeds 5-9, ready). Merge with
     `python research/techarena/merge_seq_dirs.py research/techarena/cache/seq/stage3_h50_s10 research/techarena/cache/seq/stage3_h50 research/techarena/cache/seq/stage3_h50_more`,
     then `safe_v2_step7.py --seq stage3_h50_s10`.
   - b. **Capacity:** `--dim 128` and/or `--layers 3`, `--heads 8` (longer sequences may need more model).
   - c. **Longer still:** `--hist 100` (7.9% of train events and 3.9% of eval users hit the 50 cap).
   - d. **Right-aligned histories** (latest click always at the same position; the choice model does this), a small
     code change in `seq_model.py` (`extend_histories` / `Events`).
   - e. **Filler-matched auxiliary next-click loss**, as in the choice model (`--aux-pop` gave +0.0015 there). The
     pre-period augmentation that hurt was a different design.
   - f. Learning-rate schedule (warmup + decay) to allow a second epoch; always 1 epoch today.
   - Compare variants on the inner holdout first (a `seq_variants`-style notebook), then run the winner on all
     windows.
2. **50-click histories for the choice models** (`src/choice_mf.py` `HISTORY = 20`, right-aligned): they are the top
   stack features. Needs retraining 4 models × 22 boundaries (hours of CPU; GPU after adding device support).
3. **Softmax-objective model with the 50-click features** as the blend partner (the current partner lacks them):
   - validation: `softmax_blend_val.py` on step 7 frames (~40 min);
   - submission: ~3.5 h on CPU.
4. Logistic stacker over several models instead of the 50/50 blend (+0.0002 on validation; the weights would come
   from validation predictions).
5. Outside hints (forum, FAQ, data description) on how clicks were generated; the 26-Oct board suggests the 0.83
   signal is widely known.

Status after 3 Nov afternoon (§13): 1a (more seeds) done and used as a 20-seed average; 1e as the fixed
augmentation, 1f, and 2 tested with no gain; 3 done (+0.00023); 1b, 1c, 1d are built as notebooks but not run.

## 13. 3 Nov afternoon: hypothesis hunt, sequence-model variants, choice model with 50 clicks
**Hypotheses for a hidden "0.83 signal", all ruled out** (probe scripts were in the session scratchpad):

| Hypothesis | Finding |
|---|---|
| Items come in discrete groups (11 categories would also give "1 + 10 flat axes") | Pairwise cosine of well-observed items is one smooth hump; k-means inertia falls smoothly, no elbow at 11 |
| Evaluation histories hide clicks not seen in train | None after CUT; the 0.5% unexplained entries are pre-period clicks of 1,159 users without train events |
| Evaluation slates built differently | Same in-history share (0.0259 vs 0.0254) at every history length, same popularity |
| Fillers matched to the real item's popularity | Raw corr +0.14, within user -0.009: a user-level taste effect |
| Item lifecycles | Day-of-appearance chi-square as expected under none (median 5.3 for 6 df) |
| Several interests per user | Mean similarity to history 0.767 within-event AUC vs max 0.710 |
| Per-user time-bucket / weekend click propensity | Observed chi-square 55,903 vs 56,397 permuted (none) |
| Per-user time-bucket / weekend taste | Same-bucket click pairs cosine 0.2740 vs 0.2738 |
| Re-clicks vs history fillers | Signals exist (twice in history: real 0.53 vs once 0.37; newest position more often real) but are already features |
| Other team's file, by group | It ranks re-click candidates, popular items and cold users higher; nudging ours that way on validation: 0 gain (our model is calibrated there) |

**Sequence-model variants** (Kaggle, `--save-inner`; inner holdout of the evaluation fit, 29,940 events, 5 seeds each):

| Variant | Inner | Validation alone |
|---|---|---|
| base h50, seeds 5-9 (`seq_pair_b/more`) | 0.822349 | 0.795818 |
| LR warmup + cosine decay, 1 epoch (`cos`) | 0.822026 | 0.795595 |
| fixed pre-period augmentation (`seq_aug`) | 0.821393 | 0.794636 |
| positions counted from the latest click (`seq_pair_a/right`) | 0.821897 | — |
| LR decay over 2 epochs (`seq_pair_a/cos2`) | 0.818200 | — |
| average of more + cos + aug | 0.822957 | — |
| average of all five | 0.823159 | — |
| 20-seed average incl. stage3_h50 (`cache/seq/stage3_h50_s20`) | — | 0.796733 (vs 0.795763) |
| **30-seed average** (stage3_h50 + the five, `cache/seq/stage3_h50_s30`) | — | **0.797228** |
| bigger model dim 128 / 3 layers / 8 heads (`seq_pair_c/big`) | 0.821522 | — |
| histories up to 100 clicks (`seq_pair_c/h100`) | 0.822661 | — (left out: +0.0003 alone, +0.00014 to the average) |

**Stack results 3 Nov evening** (local; 005 = 0.807134):

| Package | Local | File |
|---|---|---|
| A: step 7 + softmax partner with 50-click features | 0.807361 | `submissions/safe_v2_step7_blend2.csv` |
| step 7 with the 30-seed sequence features (alone) | 0.807295 (+0.000394 vs step 7) | `submissions/safe_v2_step7_s30.csv` |
| **B: step 7 s30 + that partner** | **0.807593 (+0.00046)** | `submissions/safe_v2_step7_s30_blend.csv` |
| C (queued): + partner retrained on the s30 features | built only if >= 0.807693 | `submissions/safe_v2_step7_s30_blend_c.csv` |

Tuning the 50/50 blend weight: cross-validated by user halves -0.000009 (flat between 0.5 and 0.6), so 50/50 stays.

**Outside search (3 Nov evening):** nothing public about this challenge (its Agorize page needs a login). The only
organiser hint is `starter_solution.py`'s docstring: it "does not use the structure of the actual data-generating
model", a good solution should model user/item interactions directly "(e.g. matrix factorization)". Other 2026 Tech
Arena rules (France) score a public part of the test data ("indicative only") and rank on a private part; if ours is
the same, the public top may be partly noise. The challenge's own FAQ/forum is still unchecked (user only).
**Full-catalogue probe** (`fullmf_probe.py`): user/item vectors + bias trained with a softmax over all 50,000 items on
every known click; compared on validation "which item" AUC with the choice model, the 50-click sequence model and step 7.

- The old `--augment` was buggy: pseudo events had P(real | in-history candidate) 0.983 vs 0.405 in real events
  (no history fillers) and right-aligned histories. `--augment-fill 0.019` fixes both (0.404), yet it is still worse.
- **Softmax partner with the 50-click features** (`softmax_step7.py`): alone 0.806884 (old partner 0.806305); blended
  with step 7 0.807361 (+0.00023).
- **Choice model with 50-click histories** (`src/choice_mf.py --history 50`, cheap aux-pop config): validation 0.788505
  vs 0.788177 (+0.0003): the auxiliary all-clicks task already feeds older clicks into the user vector. No rebuild.

**New code:** `seq_model.py` (`--sched cos`, `--right-align`, `--save-inner`, `--augment-fill`),
`make_kaggle_notebook.py` (`kaggle/seq_aug.ipynb`, `seq_pair_a/b/c.ipynb`), `seq_inner_report.py` (compare variants
on the inner holdout), `softmax_step7.py`, `blend_val.py` (rank-blend AUC from saved validation predictions),
`safe_v2_step7.py --tag`, `merge_seq_dirs.py` (seed counts of single runs), `src/choice_mf.py --history/--threads`.

## 14. 3 Nov evening: more tests, the second PC, overnight seeds
**Full-catalogue MF on Kaggle** (`fullmf_model.py`, `kaggle/fullmf.ipynb`, SVD start + softmax over all 50,000 items,
3 seeds, ~45 min on T4): inner holdout 0.78327 (SVD start) → 0.79241 at epoch 3 (+0.0091); validation-boundary
which-AUC 0.770 (choice model 0.781, sequence model 0.789 on the same events). Both pre-set rules failed → not stacked.

**Leak / mechanism tests, all negative** (probe scripts in the session scratchpad):

| Test | Finding |
|---|---|
| Click rate by stored slate position; re-click real vs filler by position | Flat (0.0625–0.0632; 0.395–0.409) |
| Duplicate items within a slate | None |
| rowid / event_id as plain md5/sha of event, item, position, user+time | No matches |
| Timestamps | All minute-aligned; `time_bucket` = hour bins 0–5/6–11/12–17/18–23, `day_type` = days 5–6, no mismatches; event rate 1:2:3:4 across buckets (click rate flat); a click's history timestamp = its event's; eval events not clipped at the cutoff |
| Our errors (new items) | Real item top-1 in 31.6%, ranks 7–10 in 9.8% (those: taste match below the fillers', light users, median 9 distinct clicks); real item = most-shown candidate 10.5%, highest-CTR 10.8% (≈ chance): no popularity-click mechanism |
| History-filler mechanism | Odds(real) ∝ times the item is in the history (1 : 1.95 : 3.0): real re-clicks are drawn per history entry, fillers per distinct item. Already a feature (`history_count`) |
| Per-user predictability | Early vs late real-item rank, within known-click bins: corr +0.030 (cmf12) / +0.037 (seq), null 0 ± 0.007: real but weak (feature not built; worth ~+0.0001) |
| Slate reuse | No eval rows/events in train, no identical slates anywhere; 0.42% of eval candidates were in the user's last train slate (72% of those clicked there = re-click candidates, already features) |
| Other team's file | A global rank transform (values uniform in (0,1), event sums 5.0); their top pick = ours in 81.2% of events, our #2 in 14.0%; no confident-event pattern |

**Choice-model seed averaging** (`cmf_seeds.py`): each stack choice feature came from one seed. A seed refits the exact
original configuration (`safe_v2_choicemf.cmf_features`, only `args.seed` differs) at the 14 boundaries; targets are
each user's first event from the boundary (superset of both scopes), so one run serves validation and submission.
Merge self-test (seed 0 with itself) reproduced the cache exactly. Settings: v2 = step 1's CMF2 (11 dims, 2 epochs),
d12 / d16 = 12 / 16 dims, pop = step 5's filler-matched config (3 epochs). `safe_v2_step7.py` / `softmax_step7.py
--swap prefix=folder` replace the `<prefix>_prob/_noclick/_rank` columns.

**Second PC** (details in the memory note `techarena-second-pc`): Yulia-PC, `D:\HuaweiTechArena\CTRL-PREDICT-Techarena-2026`,
branch `monish`; Python 3.14.10 venv = this PC's lock file + torch 2.14.1+cu126 (GTX 1660 SUPER, CUDA works);
`data\parquet` verified; 12 cores, 31.9 GB RAM, commit limit 38.9 GB; git through a read-only deploy key (HTTPS + Git
Credential Manager cannot work over SSH). SSH copies run ~0.3 MB/s (use Google Drive for big files); a choice-model
seed takes ~5 min there on CPU. No `cache/lab` frames there, so LightGBM steps run on this PC only.

**Still untested ideas (small or speculative):** "user's mean past rank" feature (per-user predictability); a Bayesian /
posterior-sampling taste model; EM with soft labels on evaluation slates. The challenge FAQ/forum (user only) is unchecked.

## 15. 3–4 Nov night: blending, more hypotheses, the second PC as a LightGBM machine
The user went to bed at 22:00 asking to keep testing ideas and to commit/push overnight (uploads stay theirs).

**Blending (free, no retraining; scripts `blend_csv.py --factorised`, `final_blend.py`).** Package C's two models
differ: the softmax partner has the better event click mass ("whether" AUC 0.668 vs 0.662), the binary model the better
within-event shares ("which" 0.7957 vs 0.7954). All numbers on validation, held-out = fitted on one user half, scored on
the other:

| Blend of C's binary + softmax files | Validation | Held-out gain |
|---|---|---|
| 50/50 rank blend (package C) | 0.807766 | — |
| factorised: whether 0.1/0.9, which 0.7/0.3 (same weights picked on both halves) | 0.807890 | +0.00011 / +0.00013 |
| + which part as a conditional logit with the raw 50-click (30 seeds) and 20-click (5 seeds) sequence shares | 0.808107 (in-sample) | +0.00014 / +0.00025 over factorised |
| + h100 / big / right sequence variants as more components | — | +0.00003 / +0.00006 (left out) |

The logit weights (binary 0.56, softmax 0.31, 50-click +0.20, 20-click −0.12, scale 0.95) say the stack under-uses
what the older clicks add. File: `submissions/safe_v2_step7_s30_final_c.csv` (checked valid; ≈ C + 0.0003).

**Hypotheses tested tonight (all negative or tiny):**

| Idea | Result |
|---|---|
| Users drawn from a limited set of taste archetypes (would let 2–3 clicks pin a light user's taste) | No: heavy users' two half-histories find each other first 62.5% of the time (Gaussian null 65.1%); k-means inertia vs the null falls smoothly 0.97 → 0.92 (K 10 → 500), no elbow |
| Dedicated event-level "whether" model (LightGBM on event features, 2-fold by user) | 0.661 vs the softmax partner's 0.668; mixed in +0.00006 |
| Raw sequence model better than the stack for some group (known clicks × repeat/new) | No group: the stack wins everywhere; new-item which-AUC 0.650 (0–2 clicks), 0.725, 0.761, 0.782, 0.798, 0.803 (21–30), 0.817 (31–40), 0.810 (41–60): flat above 30 clicks, so heavy users are limited by the item side / generator noise |
| Taste score relative to the user's other history items, for real re-clicks vs history fillers | Plain taste 0.627, relative versions 0.61–0.62; added to the stack 0.6825 → 0.6827 (nothing) |
| Full-catalogue model rejected unfairly? | No: `fullmf_model.py` already scores log p − log q (filler correction) |
| More distinct LightGBM targets (`learning_curve.json`) | +0.0006 per doubling; "every event as a target" ≈ +0.0012 but needs per-event features for every component and ~3× the memory: not feasible before the deadline |

**Second PC as a LightGBM machine.** Tailscale now connects directly (~1.8 MB/s instead of 0.3), so the validation
frames (`cache/lab/step6_*`, 1.4 GB) were copied there in 13 min. Its commit headroom (~24 GB) allows one validation
run at a time, not a submission build. Queue (`D:\HuaweiTechArena\lgb_exp.bat`, summary `logs\lgb_exp.log` there),
step 7 with the 30-seed features (this PC's reference 0.807295): `share` (`--share-feats`: the sequence models'
within-event log-shares and their 50 − 20 difference), `lr015` (learning rate 0.015), `tm12` (trees × 1.2), `nl31`
(31 leaves, min 400), `cs06` (column sample 0.6), `base` (reproducibility check). New `safe_v2_step7.py` options:
`--lgb key=value`, `--max-trees`, `--tree-mult`, `--share-feats`.

**Package D (choice-model seed averaging, 3 Nov 23:47): STOP.** cmf (v2) 5 seeds, cmf12 10, cmf16 5, cmfp 5 averaged
and swapped into step 7 s30: binary 0.807352 (C's 0.807295, +0.00006), softmax partner 0.807197 (C's 0.807347,
−0.00015, one seed so noisy), 50/50 rank blend 0.807727 vs C 0.807766 → not built. The stack already absorbs the
choice models' seed noise (as with the width ensemble). The averaged folders stay in `cache/choicemf_*_avg`.
Second PC checks: the `base` run reproduces this PC's 0.807295 exactly (and runs ~2.2× faster: ~11 min per
validation); `--share-feats` 0.807325 (+0.00003, not used).

**4 Nov 00:00–00:35, more tests (validation; reference binary 0.807295, C final blend 0.808133):**

| Test | Result |
|---|---|
| Binary, learning rate 0.015 (2813 trees) | 0.807325 (+0.00003) |
| Binary, trees × 1.2 | 0.807319 (+0.00002) |
| Binary, 6 LightGBM seeds | 0.807293 (0) |
| Lambdarank stack (`rank_step7.py`) as a third which input of the final blend | which-AUC 0.7943; held-out gains fall slightly (+0.000134 / +0.000263 vs +0.000165 / +0.000269): dropped |
| **Softmax partner with step 4's tuned LightGBM parameters** (`softmax_step7.py --params outputs/optuna_step4.json`; the partner always ran on the untuned defaults) | **alone 0.807807 (was 0.807347), 1928 trees; rank blend 0.807933; final blend 0.808377 in-sample, held-out +0.00012 / +0.00032 over C's final blend** → package E |
| Final blend under Monish's eval-reweighting (history/gap mix of the evaluation users) | rank 0.811507 → factorised 0.811626 → final 0.811867: the gain holds for an eval-like mix |

Package E = C's binary + the tuned partner (`queues/queue_e.sh`: partner `--submit` with tree cap 4000, then
`final_blend.py` → `submissions/safe_v2_step7_s30_final_e.csv`). The Optuna re-tune of the binary on step 7's frames
(`optuna_step7.py`) runs on the second PC (optuna 5.0.0 had to be installed in its venv).

**4 Nov 01:00 — the MLP stack (`nn_step7.py`) is the night's biggest gain.** Same 140 columns as the LightGBM models,
rank-Gauss transformed; a shared MLP (140 → 256 → 128) scores each candidate, a pooled head the no-click option;
event softmax over 11; epochs from the inner holdout (peak at 4: 0.812815, LightGBM's current parameters 0.812356 on
the same holdout), 3 seeds. Validation **alone 0.807615** (binary 0.807295, tuned partner 0.807807). In the final blend
(`final_blend.py --val-c`, now a probability model in both parts, whether weights by grid):

| Final blend (held-out halves) | A | B |
|---|---|---|
| binary + tuned partner + sequence shares | 0.807147 | 0.809508 |
| + MLP stack | **0.807880** | **0.810090** |

i.e. about +0.0012 over package C's rank blend (whether: partner 0.7, MLP 0.3; which: MLP 0.43, partner 0.36, 50-click
0.18, binary 0.12, 20-click −0.11). **Package F** = C's binary + tuned partner + MLP stack (`queues/queue_f.sh` →
`submissions/safe_v2_step7_s30_final_f.csv`), built after package E on this PC. Second PC batch 3: MLP variants
(512 wide / dropout 0.2; learning rate 5e-4), then Optuna on the partner's own parameters.

**4 Nov 02:45–03:30: batch 3 results and blend checks** (validation; F's final blend held-out 0.807880 / 0.810090):

| Test | Result |
|---|---|
| MLP 512 wide, dropout 0.2 | alone 0.807646 (3 epochs; inner 0.812690) |
| MLP learning rate 5e-4 (16 epochs max) | alone 0.807653 (4 epochs; inner 0.813276) |
| The three MLPs averaged (9 fits, correlation 0.995–0.997) as the final blend's third model | held-out +0.00002 / +0.00003 over one MLP (rule +0.0001): the extra submissions are not built |
| Optuna on the partner's own parameters (`optuna_softmax_step7.py`, 15 trials) | inner +0.00006; partner alone 0.807505 (step 4's parameters 0.807807): rejected |
| Optuna on the binary (`optuna_step7.py`, 45 trials, batch 2) | inner +0.00036, validation 0.807358 (+0.00006); partner with those parameters 0.807533: rejected |
| Conditional-logit weights per segment (3 or 5 bins of known clicks; 3 bins of gap) | held-out −0.00011 to +0.00003: no (the binary/partner weights just trade off) |
| Package F's blend under Monish's eval-reweighting | 0.812795 (C rank blend 0.811507, C final blend 0.811867): the gain holds for an eval-like mix |

The inner holdout is a poor judge of LightGBM parameters now (both Optuna runs gained there and not on validation);
step 4's tuned parameters stay. New `nn_step7.py` options: `--context` (the candidate head also sees
h_i − mean_j h_j and max_j h_j, i.e. the rest of its event; default path unchanged, checked bit-identical) and `--wd`.
Second PC batch 4 (`D:\HuaweiTechArena\lgb_exp4.bat`): context MLP at learning rates 1e-3 and 5e-4, MLP 512 /
dropout 0.3 / 5e-4.

**4 Nov 03:00–04:00: batch 4 and more blend checks** (F's blend with the base MLP: held-out 0.807880 / 0.810090):

| Third model in the final blend | Alone | Held-out (A / B) | vs base MLP |
|---|---|---|---|
| base MLP (F) | 0.807615 | 0.807880 / 0.810090 | — |
| context MLP, lr 1e-3 | 0.807668 | 0.807852 / 0.810078 | −0.00003 / −0.00001 |
| context MLP, lr 5e-4 | **0.807872** | 0.807834 / 0.810054 | −0.00005 / −0.00004 |
| base + context lr 5e-4 averaged | — | 0.807915 / 0.810113 | +0.00004 / +0.00002 |
| base + context lr 1e-3 averaged | — | 0.807904 / 0.810113 | +0.00002 / +0.00002 |
| MLP 512 / dropout 0.3 / lr 5e-4 | 0.807220 (7 epochs) | not tested | |

Lesson: a better MLP alone is not a better blend partner. The context MLP learns more of what the LightGBM partner
already has (its blend gain is lower), and any average of MLP variants adds only +0.00002–0.00004 held out (seed-level
variance), so F keeps the base MLP. Also: fitting the blend parameters under the eval reweighting (weighted
conditional logit + weighted AUC grid) changes held-out eval-reweighted AUC by −0.00001 / +0.00002 (nothing). The
cell table shows the one gap validation cannot cover: 9.4% of evaluation events have a 3–7 day gap, none of the
validation events (its period after the cutoff is shorter, gaps ≤ 72 h); the submission models' training frame has 4.0%
such targets (`step5_full_train`, max 156 h; validation's training frame 0.5%). Batches 5–6 on the second PC:
context MLP with new seeds / lr 3e-4 / 512 wide, then `--binary` (per-candidate logistic loss) at two learning rates,
as a more different third model.

Batch 5 (04:00–04:40): context MLP lr 5e-4 with refit seeds 3–5 (`--seed-offset 3`) 0.807792 (seeds 0–2: 0.807872, so
seed noise ≈ ±0.00004 and the context MLP is ≈ +0.0002 better alone than the base MLP); context MLP lr 3e-4
(6 epochs) **0.807948 alone**, in the final blend +0.00002 / +0.00007 held out, averaged with the base MLP +0.00005 /
+0.00006: still below the +0.0001 rule, not built. Both MLPs as separate which-components (instead of an average):
+0.00004 / +0.00002. **Package E built 04:19**: `submissions/safe_v2_step7_s30_final_e.csv` (C's binary + tuned partner,
final blend: validation 0.808377 in-sample, held-out 0.807136 / 0.809499, checked valid; partner submission 2602 trees
× 3 seeds, 197 min).

**Package F built 05:14** (`queues/queue_f.sh`): MLP submission `submissions/safe_v2_step7_nn_s30.csv` (inner holdout
of the full frames chose 3 epochs, 0.832625; 3 seeds, 49 min), then `final_blend.py` →
**`submissions/safe_v2_step7_s30_final_f.csv`**: validation 0.809056 (same fit as the validation run: beta binary 0.12,
partner 0.36, MLP 0.43, 50-click 0.18, 20-click −0.11; whether partner 0.7 / MLP 0.3; scale 0.95), checked valid.
Sanity: the three submission models look like their validation runs (mean probability binary 0.0627, partner 0.0628,
MLP 0.0655 vs 0.0620 / 0.0623 / 0.0673; row Spearman 0.985–0.996). F vs C's final file Spearman 0.9973, vs E's
0.9979. Batch 5's last run, context MLP 512 / dropout 0.2: 0.807411 (worse). This PC then builds the context MLP lr
3e-4 submission (`--tag _s30_ctx_lr3 --submit`) only so that it is ready if a later combination passes the rule.

Batch 6 (`--binary`, per-candidate logistic loss): alone 0.806841 (lr 1e-3, 6 epochs) and 0.806404 (lr 5e-4,
9 epochs). Less correlated with the base MLP (0.986 vs ≈ 0.995 for the softmax variants) but adds nothing in the final
blend: as an extra which-component +0.00002 / +0.00001, averaged with the base MLP +0.000002 / −0.000015. Batch 7:
plain MLP at lr 3e-4 and context MLP at lr 2e-4.

Batch 7 (06:00): plain MLP lr 3e-4 0.807684 alone (8 epochs), context MLP lr 2e-4 **0.807954** alone (8 epochs). All
options as the final blend's third model, scored in one pass (scratch `multi_c_blend.py`, same protocol as
`final_blend.py`; held-out gain over F's base MLP, halves A / B):

| Third model (probabilities averaged) | Held-out gain |
|---|---|
| plain MLP lr 3e-4 | −0.00001 / +0.00006 |
| context MLP lr 2e-4 | +0.00003 / +0.00011 |
| base + plain lr 3e-4 | +0.00003 / +0.00005 |
| base + context lr 3e-4 | +0.00005 / +0.00006 |
| **base + context lr 2e-4** | **+0.00006 / +0.00010** |
| base + plain lr 3e-4 + context lr 3e-4 + context lr 2e-4 | +0.00006 / +0.00010 |
| all nine softmax MLPs | +0.00004 / +0.00006 |

The context MLP's blend value rises as its learning rate falls (1e-3: −0.00003 / −0.00001; 5e-4: −0.00005 / −0.00004;
3e-4: +0.00002 / +0.00007; 2e-4: +0.00003 / +0.00011), but nothing clears +0.0001 on both halves yet. Batch 8: context
lr 1e-4, and lr 2e-4 with refit seeds 3–5 (new `--patience`: early stop of the epoch search). This PC
(`queues/queue_g.sh`) builds the context lr 3e-4 and lr 2e-4 submissions as reserves.

Batch 8 (06:40–07:30): context MLP lr 1e-4 0.807895 alone (14 epochs, inner 0.813384), context lr 2e-4 with refit
seeds 3–5 0.807917 (seeds 0–2: 0.807954). Pre-declared options (scratch `multi_c_blend2.py`; adopt only at ≥ +0.0001
held-out on both halves):

| Third model | Held-out gain over F |
|---|---|
| context lr 2e-4, 6 seeds | +0.00007 / +0.00012 |
| context lr 1e-4 | +0.00001 / +0.00005 |
| (a) base + context lr 2e-4 (6 seeds) | +0.00008 / +0.00010 |
| (b) base + context lr 1e-4 | +0.00006 / +0.00006 |
| (c) base + context lr 1e-4 + context lr 2e-4 (6 seeds) | +0.00008 / +0.00009 |

**None clears the rule: F stays the file to upload.** The direction is consistent (≈ +0.00008, i.e. ≈ +0.00007 on the
LB), so if a spare upload is wanted later, the candidate is option (a) as a new third model: it needs the context
lr 2e-4 submission with 6 refit seeds (`queue_g.sh` builds seeds 0–2 →
`submissions/safe_v2_step7_nn_s30_ctx_lr2.csv`; seeds 3–5 = the same command with `--seed-offset 3 --tag
_s30_ctx_lr2_o3`, ≈ 1.2 h), the average of those with `safe_v2_step7_nn_s30.csv` as `--sub-c` (and the matching
validation average as `--val-c`), then `final_blend.py`. Reserve file already built:
`submissions/safe_v2_step7_nn_s30_ctx_lr3.csv` (context lr 3e-4, 5 epochs on the full frames, inner 0.833148 vs the
base MLP's 0.832625).

## 16. 4 Nov day: the 50-click sequence model on the second PC's GPU
The user asked (09:00) to keep searching, starting with the sequence model, using every machine. The second PC's
GTX 1660 SUPER runs `seq_model.py` with CUDA (torch 2.14.1+cu126, venv Python 3.14.10): one pass over the 618k
inner-fit events takes 33 s, a 3-seed inner-holdout screen ~3.5 min, so Kaggle is not needed for screening. Its
GitHub deploy key was refused from 09:43 (`Permission denied (publickey)`), so changed files go over by scp
(`git checkout -- <file>` there before the next pull). This PC has no NVIDIA GPU.

**Why it stops after one epoch:** the evaluation-fit inner holdout falls from 0.8194 (epoch 1) to 0.8143 and 0.7983
while the training loss drops 1.52 → 1.23 → 0.88: the model memorises its training events. Screens
(`queues/seq_exp1-4.bat`; `--mode full --inner-only --hist 50 --seeds 3`, inner holdout of the evaluation fit, 29,940
events; logs `logs\seq_exp.log` there):

| Variant (new options in `seq_model.py`) | 1 seed, epoch 1 | 3-seed ensemble | vs base |
|---|---|---|---|
| base | 0.819401 | 0.821739 | — |
| `--no-user` (no free user vector) | 0.818504 | 0.820863 | −0.0009 |
| `--hist-drop 0.2` (hide history clicks while training) | 0.819401 | 0.821528 | −0.0002 |
| both | 0.818717 | 0.820669 | −0.0011 |
| `--emb-l2 1.0` (item vectors anchored to the SVD start) | 0.818644 | 0.820833 | −0.0009 |
| `--dropout 0.3` | 0.819956 | 0.821780 | +0.00004 |
| `--item-lr-mult 0` / 0.1 / 0.3 / **2** / 3 (item vectors' learning rate) | 0.8156 / 0.8161 / 0.8172 / 0.8197 / 0.8174 | 0.817558 / 0.818211 / 0.819342 / **0.822031** / 0.820726 | −0.0042 … **+0.0003** … −0.0010 |
| `--dim 32` / `--dim 96` | 0.817570 / 0.818876 | 0.820780 / 0.821430 | −0.0010 / −0.0003 |
| `--lr 5e-4` (2 epochs) / `--lr 2e-3` | 0.816619 / 0.818678 | 0.819699 / 0.821170 | −0.0020 / −0.0006 |
| `--ema 0.99` / 0.995 / 0.998 (predict with a moving average of the weights) | **0.820495** / 0.818302 / 0.818450 | 0.821786 / 0.819853 / 0.820065 | +0.00005 / −0.0019 / −0.0017 |

Nothing beats the base ensemble beyond seed noise: the per-user vectors are what memorises (without them the loss
falls much more slowly) but also what the model needs; item vectors want to learn at least as fast as the rest; the
moving average helps one model (+0.0011) but not an ensemble, because it removes the same noise as more seeds. The
untrained SVD start already scores 0.8010, the trained ensemble 0.8217–0.8235: the model adds ~0.02 over plain
matrix factorisation and is at its optimum for this design.

**Extra sequence models in the final blend** (`final_blend.py --seq-extra <folders>`: their within-event shares join
the which part; checked on F, held-out halves vs F's 0.807880 / 0.810090): 100-click histories (`seq_pair_c/h100`,
5 seeds) +0.00001 / −0.00001; bigger model (`seq_pair_c/big`) +0.00002 / +0.00002. The blend already has what the
sequence models know. (For reference: the sequence model alone scores 0.797 on validation, F 0.809, so using it
alone would be far worse.)

**Known clicks, validation vs evaluation** (scratch `seg_known.py`: all of a user's known clicks before the
boundary, up to 100; the data's `history_length` is capped at 20 and hides heavy users):

| Known clicks | ≤2 | 3–5 | 6–10 | 11–20 | 21–30 | 31–50 | >50 | mean |
|---|---|---|---|---|---|---|---|---|
| validation targets | 6.0% | 11.5% | 20.5% | 26.9% | 23.8% | 10.1% | 1.1% | 16.6 |
| evaluation targets | 6.0% | 11.7% | 19.4% | 25.6% | 16.8% | 16.9% | 3.6% | 18.9 |

Evaluation has about twice as many 30+ users. Final-blend weights per known-clicks segment (2–4 bins, held out by user
halves): −0.00001 to −0.00014, and the 50-click weight does not rise with known clicks, so nothing to adjust for that
mix.

**30 more sequence seeds** (`queues/seq_extra.bat` on the second PC: base seeds 100–109, EMA 0.99 seeds 200–209,
100-click histories seeds 300–309; validation and evaluation boundaries only, ~1 h; folders
`cache/seq/seqx_{base,ema,h100}`, merged `seqx_all`, and with the 30 old seeds `seqx_s60`): alone on validation base
0.796348, EMA 0.794966 (its search chose 2 epochs), 100-click 0.796284; the new 30 together 0.797093; **60 seeds
0.797289 vs 30 seeds 0.797228 (+0.00006)**: the sequence features are saturated in seeds.
In F's final blend (held-out vs 0.807880 / 0.810090): 60-seed folder as the 50-click component 0.807879 / 0.810086
(−0.000001 / −0.000004); the new 30 seeds as an extra component 0.807881 / 0.810087 (+0.000001 / −0.000003).
**The sequence-model avenue is closed**: tuning, regularising, more seeds or more variants do not move F.

**Blend-level checks, 4 Nov midday** (F's held-out 0.807880 / 0.810090; scratch `cand_terms.py`, `whether_lr.py`):

| Test | Held-out change |
|---|---|
| Candidate terms in the which part's conditional logit: in history, log history count, recency position | −0.00004 / −0.00001 |
| … popularity: log item clicks, log times shown before | +0.00001 / +0.000004 |
| Event-level logistic "whether" model (fitted on one half) from the three event masses only | −0.00007 / −0.00010 |
| … plus best-candidate share (MLP, 50-click), MLP entropy, the 50-click model's no-click probability, user click rate, past events, gap, history length | −0.00001 / −0.00006 (whether AUC 0.6699 / 0.6703 vs F's 0.6703 / 0.6697) |
| LightGBM second level instead of the linear blend (rows: the five log-shares, three masses, within-event ranks, user/item/gap features; 15 leaves, trees by an inner user split; scratch `gbm_stacker.py`) | −0.00082 / −0.00061; rank-averaged with F −0.00021 / −0.00013 |

Single event-level signals on validation (whether AUC): partner mass 0.669, MLP 0.667, binary 0.662, 50-click
no-click 0.640, user click rate 0.598, best-candidate share 0.54–0.55, gap 0.50. The components are calibrated for
in-history and popular candidates, and their event masses already hold what the slate and the user's rate say.
Running: second PC batch 9, `nn_step7.py --attn N` (self-attention over the event's 10 candidates).

**4 Nov afternoon: NN mixes clear the rule (the first gain over F since the morning).** The second PC's GPU now runs
the MLP stack too (`nn_step7.py --device cuda`; ~4 s per epoch vs ~40 s on CPU; most of a run is CPU-side loading).
New option `--attn N` (self-attention over the event's 10 candidates). Alone on validation: attention lr 2e-4
0.807753 (3 seeds) / 0.808037 (9 more seeds); attention lr 3e-4 0.807684; 2 layers 0.807672; with context 0.807214;
512 wide 0.807500. More refit seeds matter for the MLPs: base 3 seeds 0.807615 → 9 seeds 0.807892 / 0.808088;
context lr 2e-4 0.807954 → 9 seeds 0.808196 / 0.808257. In F's final blend (third model = probability average,
held-out gain over F's 0.807880 / 0.810090):

| Third model | Held-out gain |
|---|---|
| attention lr 2e-4 (3 seeds) | −0.00002 / +0.00010 |
| base + attention lr 2e-4 | +0.00009 / +0.00013 |
| base 3 + context lr 2e-4 6 + attention 3 | +0.00011 / +0.00016 |
| base 21 seeds | +0.00005 / +0.00008 |
| context lr 2e-4, 24 seeds | +0.00012 / +0.00014 |
| base 21 + context 24 | +0.00010 / +0.00013 |
| fallback (only models this PC can build): base 3 + context 3 + attention 3 | +0.00010 / +0.00016 |
| context 24 + attention 3 | +0.00012 / +0.00018 |
| base 21 + context 24 + attention 3 | +0.00012 / +0.00018 |
| **context 24 (weight 2) + attention 3** | **+0.00013 / +0.00019** |

Diversity of MLP types (plain / context / attention) and seed count both help. Package G = F's binary + partner +
this MLP mix. Pre-declared choice: the mix with the best worse-half gain among {context 2 + attention 1, context 1 +
attention 1, base + context + attention, context 2 + attention 1 + base 1} once the attention MLP has 12 validation
seeds; G is built only if both halves gain ≥ +0.0001. The submission MLPs need the full frames on the second PC (sent
by the user via Google Drive, `for_second_pc/`; `queues/second_pc_wait_files.sh` checks the 8 sizes, then
`queues/lgb_exp13.bat` builds context × 24, attention × 12, base × 21 seeds on the GPU). Fallback: this PC's CPU builds
the attention MLP (3 seeds, `--tag _s30_attn1_lr2 --submit`), with the reserve context lr 2e-4 and F's base MLP.

Other afternoon checks, all no gain: the untuned partner as a 4th model (+0.00001 / −0.00001); same-period item
frequency (an item shown more often than its popularity suggests) as a candidate term (+0.000005 / +0.000007); binary
LightGBM with `extra_trees` alone 0.806818 (base 0.807295). "Evaluation set = clicked events only?" On validation,
clicked events only: raw scores 0.8106–0.8125, within-event shares 0.8161–0.8165 (on all events shares lose 0.005);
the other team's file keeps an event-level part (Spearman with our full score 0.969, with our shares only 0.950;
between-event share of rank variance 0.074 vs our 0.071), so probably not; a probe file is ready
(`submissions/safe_v2_step7_s30_final_f_share.csv`, F with shares only) if an upload is spent on it.

**Package G choice (4 Nov 14:42, pre-declared rule: best worse-half gain, ≥ +0.0001 on both halves):** with the
attention MLP at 12 validation seeds (3 + 9 more, `_attn1_lr2_o3s9`): context 24 × 2 + attention 12 **+0.000123 /
+0.000190** (chosen); context 24 + attention 12 +0.000111 / +0.000199; base 21 + context 24 + attention 12 +0.000115 /
+0.000181; context 24 × 2 + attention 12 + base 21 +0.000121 / +0.000177. Validation mix file:
`data/validation/nn_step7_s30_g_mix_val_c4.parquet` (= `queues/weighted_avg.py` of
`nn_step7_s30_ctx_lr2_many_val_c4.parquet`:2 and `nn_step7_s30_attn12_val_c4.parquet`:1; the "many"/"attn12" files are
seed-weighted averages of the second PC's GPU runs, fetched as float32 arrays `nn_base18.npy`, `nn_ctx18.npy`,
`nn_attn9.npy` in the validation row order, md5 4896fbe6…). Submission: `queues/lgb_exp13.bat` (GPU) then
`queues/build_g.sh`. The 15:00 freeze stopped this PC's CPU fallback build of the attention MLP (`--tag
_s30_attn1_lr2 --submit`); not needed while the GPU path works.

## 17. 4 Nov late afternoon: package G built; is the evaluation set built differently?; a Mult-VAE taste model
**Package G** (`queues/build_g.sh`, finished 15:50): held-out 0.808004 / 0.810281 (F 0.807880 / 0.810090), validation
0.809212, checked valid → `submissions/safe_v2_step7_s30_final_g.csv`. The second PC's GPU built the 24-seed context
MLP and 12-seed attention MLP submissions in 10 and 13 min (inner holdout chose 5 epochs for both; the context run's
curve tracks this PC's CPU lr 2e-4 run within 5e-5 per epoch, so the flags are applied).

**Clicked events only?** Label-free test (scratch `clicked_only_test.py`): if the evaluation set held only each user's
first *clicked* event, users who never click would be missing and rare clickers' events would come later. Simulated in
train at a day-4 cutoff:

| | Evaluation set | Sim: first event | Sim: first clicked event |
|---|---|---|---|
| Coverage of users with 0 past clicks (3–5 / 6–10 past events) | 0.93 / 1.00 (others 0.95–0.97 / 1.00) | 0.88 / 0.95 | 0.66 / 0.70 |
| Spearman(past click rate, delay to the target), within activity bins | −0.014 … +0.012 | +0.002 … +0.033 | −0.05 … −0.14 |

→ the evaluation set is "each user's first event after the cutoff", clicked or not. The shares-only probe would lose.

**Other quick leak checks (scratch `shown_before_test.py`):** a candidate shown but not clicked to the same user before
is clicked at 0.0545 / 0.0437 (shown 1 / 2 times) vs 0.0568 never shown, so fillers are not drawn from future clicks;
the clicked row's position after sorting an event by rowid or item_id is uniform (0.099–0.101).

**Mult-VAE taste model** (`mvae_model.py`, new; second PC GPU, ~2 s per epoch; `queues/mvae_screen.bat`): a VAE over
each user's set of known clicks with a softmax over all 50,000 items and a linear decoder (the generator's measured
form), click-SVD start (inner which-AUC 0.7774 before training). It overfits within ~5 epochs whatever the
regularisation; validation boundary which-AUC (choice model 0.781, 50-click sequence model 0.789, full-catalogue MF
0.770):

| Variant | Inner holdout (best, epoch 5) | Validation targets |
|---|---|---|
| base: dim 32, reconstruct all, beta 0.2, input dropout 0.5 (3 seeds) | 0.7790 | 0.7745 |
| predict only the hidden clicks | 0.7782 | 0.7724 |
| dim 12 / dim 12 + hidden only | 0.7791 / 0.7785 | 0.7735 / 0.7722 |
| weight decay 1.0 | 0.7807 | 0.7722 |
| beta 1.0 | 0.7799 | 0.7733 |
| input dropout 0.8 + hidden only | 0.7789 | 0.7739 |

In G's final blend as an extra which-component (`--seq-extra mvae`): held-out 0.807980 / 0.810269 vs G's 0.808004 /
0.810281 (weight ≈ 0.005) → nothing. **Full-catalogue models (MF and VAE) lose to the slate-trained ones and add
nothing to the blend: closed.**

**More "something else" probes, 4 Nov 16:00–16:20 (all negative; scratch scripts `gap_repeat_test.py`,
`gap_mass_test.py`, `hist_filler_probe.py`):**

| Hypothesis | Finding |
|---|---|
| Long gaps (32% of eval events are > 72 h after the last click, few in validation) are mis-scored at evaluation | Re-clicks follow history position, not elapsed time (event-level re-click rate 0.077 / 0.077 / 0.074 / 0.076 by gap). The components are calibrated by gap on validation (> 120 h: actual 0.598, binary 0.587, partner 0.587, MLP +0.04 everywhere), and their evaluation masses fall with the gap the same way (users with ~3 train events, shrunk to ~0.60). G's file is a rank transform, so its event rank-sums are not masses. |
| Items in evaluation slates never seen in train | None: every evaluation item was a train candidate; first-ever showings click at 0.060 vs 0.063 |
| Mult-VAE useful on new items only | No: which-AUC real new item vs non-history fillers 0.760 (seq50 0.775, binary 0.783, MLP mix 0.783) |
| Hidden structure separating real re-clicks from history fillers | In-history candidates (177,575, 29% real): position, count, history length/distinct, in-history candidates per event, user rate → 2-fold AUC 0.583; + events since the item's last click (incl. no-click events), position among distinct items, oldest position 0.583; + age (h), gaps, hour 0.584; + times shown unclicked since the last click 0.587. The stack reaches ~0.68 only through taste, so this pot (oracle +0.012) is taste-limited |

Where the error is (validation, clicked events): the real item is new in 26,586 events, a repeat in 3,569. Which-AUC
of the real new item vs the non-history fillers 0.783 (binary), vs all candidates 0.771: history fillers outranking
real new items cost ~0.012 which-AUC on 88% of clicked events, and only taste separates them.

**MLP stack with piecewise-linear input encoding** (`nn_step7.py --ple 16`, Gorishniy et al. 2022; batch 14, 3 seeds,
validation alone vs the same type without it): plain 0.807152 (0.807615), context lr 2e-4 0.806611 (0.807954),
attention lr 2e-4 0.807225 (0.807753). In G's third model (scratch `h_choice.py`, held-out vs G 0.808004 / 0.810281):
+ plain −0.00006 / −0.00002, + context −0.00004 / +0.00004, + attention −0.00004 / +0.00002, + all three −0.00016 /
−0.00003 → closed.

**Window-aware MLP** (`nn_step7.py --window-day`, batch 15, context lr 2e-4, 3 seeds, validation alone; plain
0.807954): `--window-day` **0.808138 (+0.00018)**, `--window-weight 1` 0.807865, both 0.808092. Batch 17 (wday
context seeds 3–23, wday attention ×12) and the partner `softmax_step7.py --window-day` validation started 17:10
(a broken wait loop left the GPU idle 16:36–17:10).
Partner with `softmax_step7.py --window-day` (validation): alone 0.807727 vs 0.807807 without (1729 vs 1928 trees) → the
window-day input helps the MLP only. Day-6 validation (window 6.0 of the submission frames, 35,138 events = users active
in the 24 h after day 6, so a heavier-user mix than the evaluation): binary 0.831233, partner 0.831668.
Also negative (4 Nov 17:20): evaluation file row order (G's top pick at stored position p: 9.7–10.3% for every p;
the train click is uniform over stored positions too), evaluation.csv has no extra columns; train timestamps at minute
level (minute of hour, parity, mod 5/7/10, gaps mod k: chi²/df ≈ 1) → no timing leak.

**Window-day MLPs at full seed counts (batch 17, scratch `h2_choice.py`):** alone, context 24 seeds wday 0.808298 vs
plain 0.808248 (+0.00005; the 3-seed +0.00018 was mostly seed noise), attention 12 seeds wday 0.808188 vs 0.808107. In
the final blend (held-out vs G 0.808004 / 0.810281): G-wday (both types wday) +0.00003 / −0.00002; wday context + plain
attention +0.00001 / −0.00002; G + wday context +0.00000 / −0.00001; G + both wday +0.00001 / +0.00000 → below the
+0.0001 rule: **package H not built**. (Its submission MLPs `safe_v2_step7_nn_s30_ctx_lr2_wday_s24.csv` /
`..._attn1_lr2_wday_s12.csv` exist on the second PC; unused.)

**Global pairwise AUC loss** (`nn_step7.py --auc-loss W`: event-softmax loss + W × RankNet loss over every
clicked/non-clicked row pair of the batch, across events; batch 19, wday context, 3 seeds; reference 0.808138):
W = 1.0 0.808150, W = 0.3 0.808133 → nothing (the event-softmax models are already calibrated for global AUC).

**Deep-history candidates** (clicked by the user before, but older than the last-20 history; 0.07% of train rows,
click rate 0.279 like last-20 items 0.293): already handled (validation: predicted 0.24–0.25 vs actual 0.236; when real,
mean within-event rank 1.21, last-20 items 1.20) via `clicked_before` and the 50-click sequence features.
