# CTRL-PREDICT-Techarena-2026

## Setup (Git Bash)
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.lock.txt
```

## Layout
- `data/` – competition files (not committed)
- `submissions/` – generated CSVs (not committed)
- `notebooks/` – exploration
- `src/` – reusable code (evaluation, submission checks, model)
- `experiments.csv` – one row per experiment: change, local AUC, leaderboard AUC

## Check a submission before uploading
```bash
python src/check_submission.py submissions/<file>.csv data/sample_submission.csv
```

## Rules
- Validate locally with a **time-based split**, never a random split.
- Every submission gets a row in `experiments.csv`.
