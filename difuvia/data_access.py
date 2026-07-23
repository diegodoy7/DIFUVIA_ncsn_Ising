"""
Helpers for fetching the large data/checkpoints from Hugging Face on demand.

The Monte Carlo dataset, ablation samples, comparison samples, and checkpoints
live at the public HF dataset repo (see scripts/download_data.py). Experiment
scripts call `ensure_data([...])` so a first-time user can transparently pull
only the folders they need instead of committing gigabytes to git.
"""

import os

REPO_ID = "diegodoy7/difuvia-ncsn-ising-data"


def ensure_data(subdirs, repo_id: str = REPO_ID, local_dir: str = ".") -> None:
    """
    Ensure each folder in `subdirs` exists locally; download the missing ones
    from the HF dataset repo. No-op for folders that already have files (e.g.
    freshly generated `ablation_samples_nophysics/`).
    """
    missing = [d for d in subdirs
               if not (os.path.isdir(os.path.join(local_dir, d))
                       and os.listdir(os.path.join(local_dir, d)))]
    if not missing:
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is required to download data. Install it with "
            "`pip install -r requirements.txt`, or run scripts/download_data.py."
        )

    patterns = [f"{d}/*" for d in missing]
    print(f"[data] Fetching from {repo_id}: {', '.join(missing)}")
    snapshot_download(repo_id=repo_id, repo_type="dataset",
                      local_dir=local_dir, allow_patterns=patterns)


def load_avg_gen_time_ms(directory: str, temperatures: list) -> float:
    """Average per-sample 'gen_time_ms' recorded in per-temperature .pt files in
    `directory` (e.g. gen_data/DDPM). Returns NaN if unavailable. Used to compute
    the DDPM baseline for the Speedup column."""
    import os
    import numpy as np
    import torch

    times = []
    for T in temperatures:
        for fname in (f"T{T:.2f}.pt", f"samples_T{T:.2f}.pt"):
            fpath = os.path.join(directory, fname)
            if os.path.exists(fpath):
                data = torch.load(fpath, map_location="cpu", weights_only=False)
                if isinstance(data, dict) and "gen_time_ms" in data:
                    times.append(data["gen_time_ms"])
                break
    return float(np.mean(times)) if times else float("nan")
