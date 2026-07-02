"""
BLOCK 3: PC Sampler Ablation Study

By default, loads the pre-generated E1-E12 ablation samples (12 hyperparameter
configurations x 11 temperatures, full experimentation-phase output, 1000
samples each) from --save_dir and recomputes figures/table directly. No model
checkpoint is needed for this path.

To regenerate the grid from scratch with a trained NCSNv2 checkpoint instead
(e.g. to try new (M, K, lambda0) values not in E1-E12), see the commented-out
"GENERATE FROM SCRATCH" block in main() below and uncomment it. That path
re-runs the PC sampler for every (config, temperature) pair, which is
expensive — easily hours at full n_samples scale — so it is opt-in.

Usage
-----
python experiments/r3_ablation.py \
    --save_dir   ablation_samples \
    --data_dir   train_data \
    --fig_dir    figures \
    --table_dir  tables
"""

import os
import argparse

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
    parser = argparse.ArgumentParser(description="R3: PC sampler ablation study")
    parser.add_argument("--checkpoint",
                        default="networks/ckpt_epoch_300.pth",
                        help="Only used by the commented-out generate-from-scratch path below.")
    parser.add_argument("--config",    default="ncsnv2/configs/ising.yml")
    parser.add_argument("--data_dir",  default="train_data")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Only used by the commented-out generate-from-scratch path below.")
    parser.add_argument("--save_dir",  default="ablation_samples")
    parser.add_argument("--fig_dir",   default="figures")
    parser.add_argument("--table_dir", default="tables")
    args = parser.parse_args()

    TEMPERATURES = [round(1.77 + i * 0.1, 2) for i in range(11)]
    IMAGE_SIZE   = 64
    os.makedirs(args.fig_dir,   exist_ok=True)
    os.makedirs(args.table_dir, exist_ok=True)

    print("=" * 60)
    print("BLOCK 3: ABLATION STUDY — PC SAMPLER HYPERPARAMETERS")
    print(f"  Experiments  : {len(HYPERPARAMETER_GRID)}")
    print(f"  Temperatures : {len(TEMPERATURES)}")
    print(f"  Samples dir  : {args.save_dir}/  (pre-generated, E1-E12)")
    print("=" * 60)

    # 1. Load MC ground truth
    print("\n[1/5] Loading Monte Carlo ground truth...")
    mc_gt = load_mc_ground_truth(args.data_dir, TEMPERATURES, L=IMAGE_SIZE)

    # 2. MC G(r) at Tc
    print("\n[2/5] Computing MC G(r) at Tc...")
    T_crit, mc_corr_tc = load_mc_correlation_at_tc(args.data_dir, TEMPERATURES, L=IMAGE_SIZE)
    if T_crit:
        print(f"  MC G(r) computed at T={T_crit:.2f}")

    # 3. Load pre-generated E1-E12 samples (principal path — no model needed)
    print("\n[3/5] Loading pre-generated ablation samples...")
    results = load_offline_samples_and_compute(
        hyperparameter_grid=HYPERPARAMETER_GRID,
        temperatures=TEMPERATURES,
        mc_ground_truth=mc_gt,
        image_size=IMAGE_SIZE,
        save_dir=args.save_dir,
    )

    # ------------------------------------------------------------------
    # ALTERNATIVE: generate a fresh hyperparameter grid from scratch with
    # a trained checkpoint instead of loading pre-generated samples.
    # Uncomment to use (and comment out the load_offline_samples_and_compute
    # call above, or just let `results` be overwritten below).
    #
    # from difuvia.physics import SoftIsingEnergy
    # from difuvia.model_utils import load_model, get_device
    # from difuvia.sampling import run_experiment
    #
    # CUSTOM_GRID = [
    #     {"id": "E1", "M": 100, "K": 2, "lambda0": 0.2},
    #     # ... add or replace (M, K, lambda0) configs here
    # ]
    #
    # device = get_device()
    # score_model, sigmas, config = load_model(args.checkpoint, args.config, device)
    # energy_fn = SoftIsingEnergy(J=1.0, double_well_strength=0.6)
    # results = run_experiment(
    #     score_model=score_model,
    #     sigmas=sigmas,
    #     energy_fn=energy_fn,
    #     mc_ground_truth=mc_gt,
    #     temperatures=TEMPERATURES,
    #     hyperparameter_grid=CUSTOM_GRID,
    #     n_samples_per_temp=args.n_samples,
    #     image_size=IMAGE_SIZE,
    #     device=device,
    #     save_dir=args.save_dir,
    #     save_samples=True,
    # )
    # ------------------------------------------------------------------

    if not results:
        print("[ERROR] No results loaded. Check --save_dir contains E1-E12 .pt files.")
        return

    # 4. Figures
    print("\n[4/5] Generating figures...")
    plot_all_observables(
        results, mc_gt, TEMPERATURES,
        save_path=os.path.join(args.fig_dir, "Fig3_ablation_observables.pdf"),
    )
    if T_crit:
        plot_correlation_at_tc(
            results, mc_corr_tc, T_crit, L=IMAGE_SIZE,
            save_path=os.path.join(args.fig_dir, "Fig3_ablation_correlation.pdf"),
        )

    # 5. Results table
    print("\n[5/5] Generating results table...")
    df = print_ablation_table(results, mc_gt, TEMPERATURES, save_csv=True)

    # Pareto plot (if time information is available)
    if "Time(ms)" in df.columns:
        try:
            df_plot = df[["Exp", "IPS", "W1_Dist", "Time(ms)"]].copy()
            df_plot["IPS"]     = df_plot["IPS"].astype(float)
            df_plot["W1_Dist"] = df_plot["W1_Dist"].astype(float)
            df_plot["Time(ms)"] = df_plot["Time(ms)"].astype(float)
            plot_pareto(
                df_plot, metric="IPS",
                save_path=os.path.join(args.fig_dir, "Fig3_pareto_IPS.pdf"),
            )
        except Exception as e:
            print(f"  [Warning] Pareto plot skipped: {e}")

    print("\n" + "=" * 60)
    print("BLOCK 3 COMPLETE.")
    print(f"  Figures saved in : {args.fig_dir}/")
    print(f"  Table saved in   : {args.table_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
