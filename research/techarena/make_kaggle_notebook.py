"""Write the Kaggle notebooks for seq_model.py (each embeds the current seq_model.py, so re-run after changing it):

    python research/techarena/make_kaggle_notebook.py

  kaggle/seq_variants.ipynb  inner-holdout comparison at the validation boundary: baseline vs + skipped items, each
                             as 1 seed and as a 5-seed average. Output: seq_variants.zip (one JSON per run).
  kaggle/seq_stage3.ipynb    stage 3: --mode windows (14 leak-free fits) with SEQ_ARGS below. Output: seq_stage3.zip,
                             to unzip into research/techarena/cache/seq/stage3/ (then run safe_v2_stage3.py).
  kaggle/seq_h50.ipynb       stage 3 with histories of up to 50 known clicks (--hist 50), 5 seeds split over the two
                             T4s (3 seeds on one GPU). Output: seq_h50.zip, to unzip into
                             research/techarena/cache/seq/stage3_h50/ (then run safe_v2_step7.py).

On Kaggle: File -> Import Notebook, add the Parquet dataset, accelerator GPU T4 x2, run the first two cells to check
the GPU and the data, then Save Version -> Save & Run All. Download the zip from the version's Output tab.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEQ_ARGS = "--seeds 5"  # stage 3 settings: seq_variants winner (5-seed average 0.7983 inner vs 0.7943 1 seed)


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def markdown(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


SETUP = [
    code("import torch\n"
         "print(torch.__version__, torch.cuda.is_available())\n"
         "assert torch.cuda.is_available(), 'no GPU: Settings -> Accelerator -> GPU T4 x2'\n"
         "print(torch.cuda.get_device_name(0))\n"
         "print((torch.ones(4, 4, device='cuda') @ torch.ones(4, 4, device='cuda')).sum().item())  # 64.0 = GPU works"),
    code("import glob, os\n"
         "found = glob.glob('/kaggle/input/**/train_events.parquet', recursive=True)\n"
         "assert found, 'dataset not attached: Add Input -> your Parquet dataset'\n"
         "DATA = os.path.dirname(found[0])\n"
         "print(DATA, sorted(os.listdir(DATA)))"),
    code("%%writefile seq_model.py\n" + (HERE / "seq_model.py").read_text(encoding="utf-8")),
]


def zip_cell(name: str) -> dict:
    return code("import shutil\n"
                f"print(sorted(os.listdir('/kaggle/working/{name}')))\n"
                f"shutil.make_archive('/kaggle/working/{name}', 'zip', '/kaggle/working/{name}')\n"
                f"print(os.path.getsize('/kaggle/working/{name}.zip') / 1e6, 'MB')")


def hist_notebook(name: str, s0: int) -> list[dict]:
    """Stage 3 with --hist 50: 5 seeds from s0, split 3 + 2 over the two T4s, merged into /kaggle/working/<name>."""
    return [
        markdown("# Stage 3 with longer histories (up to 50 known clicks)\n"
                 "Same 14 leak-free fits as `seq_stage3`, with `--hist 50`: every event keeps its last 20 clicks and "
                 f"gets the user's older known clicks prepended. Epochs are chosen once, on the evaluation fit's inner holdout (`--search-once`). With GPU T4 x2, seeds {s0}-{s0 + 2} run on one GPU and seeds {s0 + 3}-{s0 + 4} "
                 "on the other at the same time (about 25 min); with one GPU, `SEEDS_ONE_GPU` seeds run (3 = about "
                 "40 min). Progress prints every 5 minutes. Restart-safe: finished boundaries are skipped. "
                 f"Output: `/kaggle/working/{name}.zip`."),
        *SETUP,
        code("import subprocess, time\n"
             "SEEDS_ONE_GPU = 3  # 5 = same as the 20-click features, about 1 h on one T4\n"
             "n_gpu = torch.cuda.device_count()\n"
             f"runs = [('a', {s0}, 3, '0'), ('b', {s0 + 3}, 2, '1')] if n_gpu >= 2 else [('a', {s0}, SEEDS_ONE_GPU, '0')]\n"
             "print(n_gpu, 'GPU(s):', [(tag, f'seeds {s}..{s + n - 1}', 'GPU ' + g) for tag, s, n, g in runs])\n"
             "procs = []\n"
             "for tag, seed, n, gpu in runs:\n"
             f"    cmd = ['python', 'seq_model.py', '--data', DATA, '--out', f'/kaggle/working/{name}_{{tag}}',\n"
             "           '--mode', 'windows', '--hist', '50', '--search-once', '--seed', str(seed), '--seeds', str(n)]\n"
             f"    log = open(f'/kaggle/working/{name}_{{tag}}.log', 'w')\n"
             "    procs.append((tag, n, subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),\n"
             "                                           stdout=log, stderr=subprocess.STDOUT)))\n"
             "t0 = time.time()\n"
             "while any(p.poll() is None for _, _, p in procs):\n"
             "    time.sleep(300)\n"
             "    for tag, _, p in procs:\n"
             f"        lines = open(f'/kaggle/working/{name}_{{tag}}.log').read().splitlines()\n"
             "        print(f'{(time.time() - t0) / 60:5.0f} min  run {tag}:', lines[-1] if lines else '(starting)', "
             "flush=True)\n"
             "for tag, _, p in procs:\n"
             f"    assert p.returncode == 0, f'run {{tag}} failed: see /kaggle/working/{name}_{{tag}}.log'\n"
             "print('all runs finished in', round((time.time() - t0) / 3600, 1), 'h')"),
        code("import json\n"
             "import numpy as np, pandas as pd\n"
             f"OUT = f'/kaggle/working/{name}'\n"
             "os.makedirs(OUT, exist_ok=True)\n"
             f"parts = [(f'/kaggle/working/{name}_{{tag}}', n) for tag, n, _ in procs]\n"
             "w = np.array([n for _, n in parts], float) / sum(n for _, n in parts)  # seed-weighted average\n"
             "names = sorted(f for f in os.listdir(parts[0][0]) if f.endswith('.parquet'))\n"
             "for f in names:\n"
             "    frames = [pd.read_parquet(f'{d}/{f}') for d, _ in parts]\n"
             "    for fr in frames[1:]:\n"
             "        assert (fr['rowid'].to_numpy() == frames[0]['rowid'].to_numpy()).all(), f'{f}: rows differ'\n"
             "    prob = sum(wi * fr['seq_prob'].to_numpy() for wi, fr in zip(w, frames))\n"
             "    noclick = sum(wi * fr['seq_noclick'].to_numpy() for wi, fr in zip(w, frames))\n"
             "    rank = ((-prob.reshape(-1, 10)).argsort(1).argsort(1) + 1).ravel().astype(np.float32)\n"
             "    pd.DataFrame({'rowid': frames[0]['rowid'], 'seq_prob': prob, 'seq_noclick': noclick,\n"
             "                  'seq_rank': rank}).to_parquet(f'{OUT}/{f}', index=False)\n"
             "    info = {'seeds': [n for _, n in parts], 'parts': [json.load(open(f'{d}/{f[:-8]}.json')) for d, _ in parts]}\n"
             "    json.dump(info, open(f'{OUT}/{f[:-8]}.json', 'w'), indent=1)\n"
             "print(len(names), 'files merged (14 expected):', names)\n"
             "assert len(names) == 14, 'some boundaries are missing'"),
        zip_cell(name),
    ]


NOTEBOOKS = {
    "seq_variants": [
        markdown("# Sequence model variants (inner holdout only)\n"
                 "Baseline vs + skipped items (shown but not clicked in the last 5 events), each with 5 seeds. "
                 "Validation is never scored here. Output: `/kaggle/working/seq_variants.zip`."),
        *SETUP,
        code("import json, sys\n"
             "from pathlib import Path\n"
             "sys.path.insert(0, '.')\n"
             "import seq_model as sm\n"
             "OUT = '/kaggle/working/seq_variants'\n"
             "loaded = sm.load(Path(DATA), 5)  # once, with skipped items; the baseline simply ignores them\n"
             "VARIANTS = {'_base': [], '_skip5': ['--skips', '5']}\n"
             "for tag, extra in VARIANTS.items():\n"
             "    sm.main(['--data', DATA, '--out', OUT, '--mode', 'val', '--inner-only', '--seeds', '5',\n"
             "             '--tag', tag, *extra], loaded)"),
        code("rows = []\n"
             "for tag in VARIANTS:\n"
             "    r = json.load(open(f'{OUT}/seq_val{tag}_inner.json'))\n"
             "    rows.append((tag, r['epochs'], r['best_inner_auc'], r['ensemble_inner_auc']))\n"
             "print(f\"{'variant':10s} {'epochs':>6s} {'1 seed':>9s} {'5 seeds':>9s}   (reference 1 seed: 0.7956 CPU)\")\n"
             "for tag, ep, one, five in rows:\n"
             "    print(f'{tag:10s} {ep:6d} {one:9.6f} {five:9.6f}')"),
        zip_cell("seq_variants"),
    ],
    "seq_stage3": [
        markdown("# Stage 3: sequence-model features per LightGBM window\n"
                 f"14 leak-free fits (evaluation + 13 windows), settings `{SEQ_ARGS or 'baseline'}`. "
                 "Output: `/kaggle/working/seq_stage3.zip`. Restart-safe: finished boundaries are skipped."),
        *SETUP,
        code(f"!python seq_model.py --data {{DATA}} --out /kaggle/working/seq_stage3 --mode windows {SEQ_ARGS}".rstrip()),
        zip_cell("seq_stage3"),
    ],
    "seq_h50": hist_notebook("seq_h50", 0),
    "seq_h50_more": hist_notebook("seq_h50_more", 5),
}

for name, cells in NOTEBOOKS.items():
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}}}
    out = HERE / "kaggle" / f"{name}.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out}")
