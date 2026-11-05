# TechArena modelling pipeline

This directory contains the boundary-safe modelling implementation copied from the local TechArena analysis workspace.

- `src/` contains feature construction, SVD, Logistic-MF, graph diagnostics, validation, and submission code.
- `notebooks/run_all.ipynb` records the experiment path and reports.
- Competition data, caches, submissions, and generated outputs are intentionally excluded from Git.

The code expects the original competition files through the environment variables documented in `src/config.py`.
