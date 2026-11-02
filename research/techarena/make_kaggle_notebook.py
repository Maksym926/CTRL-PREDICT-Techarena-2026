"""Write research/techarena/kaggle/seq_stage3.ipynb: a Kaggle notebook that runs seq_model.py --mode windows on a GPU.

    python research/techarena/make_kaggle_notebook.py

The notebook embeds the current seq_model.py, so re-run this after changing it. On Kaggle: File -> Import Notebook,
add the Parquet dataset, accelerator GPU T4 x2, then Save Version -> Save & Run All. Download seq_stage3.zip from the
version's Output tab and unzip it into research/techarena/cache/seq/stage3/ (then run safe_v2_stage3.py).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "kaggle" / "seq_stage3.ipynb"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def markdown(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


cells = [
    markdown("# Stage 3: sequence-model features per LightGBM window\n"
             "14 leak-free fits (evaluation + 13 windows). Output: `/kaggle/working/seq_stage3.zip`. "
             "Restart-safe: finished boundaries are skipped."),
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
    code("!python seq_model.py --data {DATA} --out /kaggle/working/seq_stage3 --mode windows --log-every 0"),
    code("import shutil\n"
         "print(sorted(os.listdir('/kaggle/working/seq_stage3')))\n"
         "shutil.make_archive('/kaggle/working/seq_stage3', 'zip', '/kaggle/working/seq_stage3')\n"
         "print(os.path.getsize('/kaggle/working/seq_stage3.zip') / 1e6, 'MB')"),
]
nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}}}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {OUT}")
