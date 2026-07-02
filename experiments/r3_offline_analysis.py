"""
BLOCK 3: PC Sampler Ablation — Offline Analysis (loads pre-generated samples)

Loads ablation samples already saved by r3_ablation.py, recomputes observables,
prints TABLE 3 (nRMSE, IPS, Wasserstein-1, generation time), and generates:
  - Fig3_ablation_observables.pdf  — 4-panel thermodynamic observables vs T
  - Fig3_ablation_correlation.pdf  — log-log G(r) at Tc
  - Fig3_pareto_IPS.pdf           — Pareto frontier (time vs IPS)
  - Fig3_pareto_W1.pdf            — Pareto frontier (time vs W1)

Usage
-----
python experiments/r3_offline_analysis.py \
    --save_dir   ablation_samples \
    --data_dir   train_data \
    --fig_dir    figures \
    --table_dir  tables
"""

import os
import argparse
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from difuvia.thermodynamics import load_mc_ground_truth, load_mc_correlation_at_tc
from difuvia.sampling import load_offline_samples_and_compute
from difuvia.analysis import print_ablation_table
from difuvia.viz import plot_all_observables, plot_correlation_at_tc, plot_pareto


HYPERPARAMETER_GRID = [
    {"id": "E1",  "M": 100, "K": 2, "lambda0": 0.2},
    {"id": "E2",  "M": 100, "K": 4, "lambda0": 0.2},
    {"id": "E3",  "M": 150, "K": 2, "lambda0": 0.2},
    {"id": "E4",  "M": 150, "K": 3, "lambda0": 0.2},
    {"id": "E5",  "M": 150, "K": 2, "lambda0": 0.4},
    {"id": "E6",  "M": 170, "K": 1, "lambda0": 0.3},
    {"id": "E7",  "M": 200, "K": 4, "lambda0": 0.2},
    {"id": "E8",  "M": 200, "K": 3, "lambda0": 0.3},
    {"id": "E9",  "M": 200, "K": 1, "lambda0": 0.2},
    {"id": "E10", "M": 250, "K": 2, "lambda0": 0.2},
    {"id": "E11", "M": 250, "K": 3, "lambda0": 0.2},
    {"id": "E12", "M": 250, "K": 3, "lambda0": 0.4},
]


def main():
    parser = argparse.ArgumentParser(description="R3: Offline ablation analysis from saved samples")
    parser.add_argument("--save_dir",   default="ablation_samples",
                        help="Directory containing pre-generated .pt files")
    parser.add_argument("--data_dir",   default="train_data")
    parser.add_argument("--fig_dir",    default="figures")
    parser.add_argument("--table_dir",  default="tables")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Subset of experiment IDs to load (e.g. E1 E3 E7). Default: all.")
    args = parser.parse_args()

    TEMPERATURES = [round(1.77 + i * 0.1, 2) for i in range(11)]
    os.makedirs(args.fig_dir,   exist_ok=True)
    os.makedirs(args.table_dir, exist_ok=True)

    grid = HYPERPARAMETER_GRID
    if args.experiments:
        grid = [p for p in grid if p["id"] in args.experiments]
        print(f"Filtered to {len(grid)} experiments: {[p['id'] for p in grid]}")

    print("=" * 60)
    print("BLOCK 3: ABLATION — OFFLINE ANALYSIS")
    print(f"  Experiments : {len(grid)}")
    print(f"  Samples dir : {args.save_dir}/")
    print("=" * 60)

    # 1. MC ground truth
    print("\n[1/5] Loading Monte Carlo ground truth...")
    mc_gt = load_mc_ground_truth(args.data_dir, TEMPERATURES, L=args.image_size)

    # 2. MC G(r) at Tc
    print("\n[2/5] Loading MC spin correlation at Tc...")
    T_crit, mc_corr_tc = load_mc_correlation_at_tc(args.data_dir, TEMPERATURES, L=args.image_size)
    if T_crit:
        print(f"  MC G(r) at T={T_crit:.2f}")

    # 3. Load saved samples and compute observables
    print("\n[3/5] Loading saved samples and computing observables...")
    results = load_offline_samples_and_compute(
        hyperparameter_grid=grid,
        temperatures=TEMPERATURES,
        mc_ground_truth=mc_gt,
        image_size=args.image_size,
        save_dir=args.save_dir,
    )
    loaded = [r["exp_id"] for r in results if r["obs_per_temp"]]
    print(f"\n  Loaded {len(loaded)} experiments: {loaded}")

    if not results:
        print("[ERROR] No results loaded. Run r3_ablation.py first to generate samples.")
        return

    # 4. Figures
    print("\n[4/5] Generating figures...")
    plot_all_observables(
        results, mc_gt, TEMPERATURES,
        save_path=os.path.join(args.fig_dir, "Fig3_ablation_observables.pdf"),
    )
    if T_crit:
        plot_correlation_at_tc(
            results, mc_corr_tc, T_crit, L=args.image_size,
            save_path=os.path.join(args.fig_dir, "Fig3_ablation_correlation.pdf"),
        )

    # 5. Table and Pareto plots
    print("\n[5/5] Generating results table...")
    df = print_ablation_table(results, mc_gt, TEMPERATURES, save_csv=True)

    # Pareto: IPS vs time
    if "Time(ms)" in df.columns:
        for metric in ["IPS", "W1_Dist"]:
            try:
                df_plot = df[["Exp", metric, "Time(ms)"]].copy()
                df_plot[metric]     = df_plot[metric].astype(float)
                df_plot["Time(ms)"] = df_plot["Time(ms)"].astype(float)
                plot_pareto(
                    df_plot, metric=metric,
                    save_path=os.path.join(args.fig_dir, f"Fig3_pareto_{metric}.pdf"),
                )
            except Exception as e:
                print(f"  [Warning] Pareto plot ({metric}) skipped: {e}")

    print("\n" + "=" * 60)
    print("BLOCK 3 OFFLINE ANALYSIS COMPLETE.")
    print(f"  Figures saved in : {args.fig_dir}/")
    print(f"  Table saved in   : {args.table_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
