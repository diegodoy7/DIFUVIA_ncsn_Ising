"""One-time script to upload large data/checkpoint folders to the Hugging Face
dataset repo. Not needed by end users cloning the repo — see download_data.py."""
from huggingface_hub import HfApi

REPO_ID = "diegodoy7/difuvia-ncsn-ising-data"

api = HfApi()
for folder in ["train_data", "ablation_samples", "gen_data", "networks"]:
    print(f"Uploading {folder}/ ...")
    api.upload_folder(
        folder_path=folder,
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo=folder,
    )
    print(f"Done: {folder}/")
