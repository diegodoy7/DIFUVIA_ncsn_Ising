"""
BLOCK 2: Noise Scale Selection via Marchenko-Pastur Criterion

Loads the full training dataset, computes the Song & Ermon σ₁ heuristic
(max pairwise distance), then runs the two-phase MP criterion search to find
the optimal σ₁* for the NCSN noise schedule.

Usage
-----
python experiments/r2_sigma_selection.py \
    --config     ncsnv2/configs/ising.yml \
    --fig_dir    figures \
    --table_dir  tables
"""

import os
import argparse
import torch

from difuvia.model_utils import load_config, get_device
from difuvia.analysis import (
    marchenko_pastur_criterion,
    get_max_pairwise_distance,
    find_optimal_sigma_max,
)
from difuvia.viz import plot_eigenvalue_spectra


def print_comparison_table(sigma_mp: float, sigma_song_ermon: float) -> None:
    ratio = sigma_mp / sigma_song_ermon
    print("\n" + "=" * 58)
    print("TABLE 2: σ₁ Selection — MP Criterion vs Song & Ermon (2020)")
    print("=" * 58)
    print(f"  {'Method':<35} {'σ₁':>10}")
    print("-" * 58)
    print(f"  {'Song & Ermon (max pairwise distance)':<35} {sigma_song_ermon:>10.4f}")
    print(f"  {'Marchenko-Pastur criterion (proposed)':<35} {sigma_mp:>10.4f}")
    print("-" * 58)
    print(f"  {'Ratio (MP / Song & Ermon)':<35} {ratio:>10.4f}")
    print("=" * 58)


def main():
    parser = argparse.ArgumentParser(description="R2: Marchenko-Pastur σ₁ selection")
    parser.add_argument("--config",     default="ncsnv2/configs/ising.yml")
    parser.add_argument("--fig_dir",    default="figures")
    parser.add_argument("--table_dir",  default="tables")
    args = parser.parse_args()

    os.makedirs(args.fig_dir,   exist_ok=True)
    os.makedirs(args.table_dir, exist_ok=True)

    device = get_device()
    print("=" * 60)
    print("BLOCK 2: MARCHENKO-PASTUR NOISE SCALE SELECTION")
    print("=" * 60)

    # 1. Load dataset
    print("\n[1/4] Loading dataset...")
    from data.ising_adapter_continuous import get_dataloader

    config = load_config(args.config)
    config.data.num_workers = 0
    data_loader   = get_dataloader(config)
    ising_dataset = torch.cat([batch[0] for batch in data_loader], dim=0)
    print(f"  Dataset shape: {ising_dataset.shape}")

    # 2. Song & Ermon σ₁ (max pairwise distance)
    print("\n[2/4] Computing Song & Ermon σ₁ (max pairwise distance)...")
    sigma_song_ermon = get_max_pairwise_distance(ising_dataset)
    print(f"  σ₁ (Song & Ermon) = {sigma_song_ermon:.4f}")

    # 3. MP criterion search
    print("\n[3/4] Running MP criterion search...")
    sigma_mp = find_optimal_sigma_max(
        ising_dataset,
        sigma_start=0.01,
        sigma_end=sigma_song_ermon,
        steps=40,
        rounds=3,
        visualize=True,
        save_fig=True,
    )

    # 4. Eigenvalue spectrum comparison figure
    print("\n[4/4] Generating eigenvalue spectrum figure...")
    plot_eigenvalue_spectra(
        ising_dataset,
        sigma_star=sigma_mp,
        save_path=os.path.join(args.fig_dir, "Fig_MP_eigenvalue_spectra.pdf"),
    )

    # Print and save comparison table
    print_comparison_table(sigma_mp, sigma_song_ermon)

    import csv
    with open(os.path.join(args.table_dir, "Table2_sigma_comparison.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "sigma_1"])
        writer.writerow(["Song & Ermon (max pairwise distance)", f"{sigma_song_ermon:.4f}"])
        writer.writerow(["Marchenko-Pastur criterion (proposed)", f"{sigma_mp:.4f}"])
        writer.writerow(["Ratio (MP / Song & Ermon)", f"{sigma_mp/sigma_song_ermon:.4f}"])
    print(f"  Table saved: {args.table_dir}/Table2_sigma_comparison.csv")

    print("\n" + "=" * 60)
    print("BLOCK 2 COMPLETE.")
    print(f"  σ₁* (MP criterion)  = {sigma_mp:.4f}")
    print(f"  σ₁  (Song & Ermon)  = {sigma_song_ermon:.4f}")
    print(f"  Figures saved in    : {args.fig_dir}/")
    print(f"  Table saved in      : {args.table_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
