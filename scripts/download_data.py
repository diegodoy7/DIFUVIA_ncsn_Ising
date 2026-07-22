"""Download the Monte Carlo dataset, pre-generated samples, and model checkpoints
from Hugging Face Hub into the repo's expected local paths.

Usage: python scripts/download_data.py
"""
from huggingface_hub import snapshot_download

REPO_ID = "diegodoy7/difuvia-ncsn-ising-data"

if __name__ == "__main__":
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=".",
        allow_patterns=["train_data/*", "ablation_samples/*", "gen_data/*", "networks/*"],
    )
    print("Downloaded train_data/, ablation_samples/, gen_data/, networks/")
