# TechArena CTR pipeline

This project builds the click model from the raw competition files using
strict timestamp boundaries.  The default paths are the original files in
`/Users/monishobaid/Downloads`; set `TECARENA_DATA_DIR` to use another folder.

The bundled runtime on this Mac needs the OpenMP library path when importing
LightGBM:

```bash
export PYTHONPATH=.runtime_deps:.
export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/llvm/20.1.5/lib
python3 -m tests.test_pipeline
python3 -m src.predict
```

The fake-data test runs before the large cache build.  `src.predict` writes a
checked `outputs/submission_choice_features.csv` using the low-rank SVD,
impression SVD, history, propensity, transition and no-click choice features.

## DCU GPU run

The repository is designed to be copied to the private GitLab project without
data or generated outputs. On `soc-gpu.computing.dcu.ie`, clone the repository,
copy the data into `data/`, and run:

```bash
bash scripts/setup_server.sh
bash scripts/rsync_data.sh       # run this on the laptop
python -m tests.test_pipeline
bash scripts/detect_scheduler.sh
sbatch scripts/job_slatenet.sbatch   # when SLURM is available
# otherwise:
bash scripts/run_slatenet.sh
```

The transfer script reads the username from `$USER` and never commits data,
outputs, credentials or model checkpoints. If the host has no package-network
access, run `pip download -r requirements-gpu.txt -d wheelhouse/` on a machine
with access, rsync `wheelhouse/`, and replace the install line in
`setup_server.sh` with `python -m pip install --no-index --find-links wheelhouse -r requirements-gpu.txt`.

SlateNet is implemented in `src/slatenet.py`. It uses all labelled impressions
strictly before each fold boundary, an explicit no-click class, recency-aware
history attention, candidate repeat/age/count features, dropout and AdamW. It
writes per-epoch losses and fold predictions to `outputs/slatenet_results.json`.
