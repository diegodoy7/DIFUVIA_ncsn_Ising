"""
Quantitative evaluation metrics, statistical tests, and summary tables.

Contains:
- Marchenko-Pastur noise scale selection (R2)
- nRMSE, IPS, Wasserstein-1 evaluation metrics (R3, R4)
- Pareto frontier computation (R3.2_bm)
- print_ablation_table, print_comparison_table (R3, R4)
"""

import csv
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

UV_GREEN = (0.157, 0.678, 0.337)
UV_BLUE  = (0.094, 0.322, 0.616)


# ──────────────────────────────────────────────────────────────
# Marchenko-Pastur noise scale selection (R2)
# ──────────────────────────────────────────────────────────────

def marchenko_pastur_criterion(dataset, plot: bool = False, ax=None, title: str = None):
    """
    Test whether a batch of configurations is statistically indistinguishable
    from pure noise using the Marchenko-Pastur law.

    Parameters
    ----------
    dataset : np.ndarray or torch.Tensor, shape (N, ...)
    plot    : whether to plot the eigenvalue spectrum
    ax      : matplotlib Axes for subplot integration (optional)
    title   : plot title override

    Returns
    -------
    is_pure_noise : bool   (True if λ_max < λ+)
    signal_ratio  : float  (λ_max / λ+)
    """
    if hasattr(dataset, "cpu"):
        X = dataset.cpu().numpy()
    else:
        X = np.array(dataset)

    N_samples  = X.shape[0]
    N_features = int(np.prod(X.shape[1:]))
    X_flat = X.reshape(N_samples, N_features)
    X_norm = (X_flat - X_flat.mean(axis=0)) / (X_flat.std(axis=0) + 1e-8)

    try:
        _, s, _ = np.linalg.svd(X_norm, full_matrices=False)
        eigvals = (s ** 2) / (N_samples - 1)
    except np.linalg.LinAlgError:
        return False, 9999.0

    Q = N_samples / N_features
    lambda_plus  = (1 + np.sqrt(1 / Q)) ** 2
    lambda_max   = np.max(eigvals)
    signal_ratio = lambda_max / lambda_plus
    is_pure_noise = signal_ratio < 1.0

    if plot:
        import matplotlib.pyplot as plt
        own_fig = ax is None
        if own_fig:
            _, ax = plt.subplots(figsize=(8, 5))
        ax.hist(eigvals, bins=50, density=True, alpha=0.7,
                color=UV_BLUE, edgecolor="white", linewidth=0.3,
                label="Empirical spectrum")
        ax.axvline(lambda_plus, color="red", linestyle="--", linewidth=2,
                   label=f"$\\lambda_+$ = {lambda_plus:.3f}  (MP bound)")
        ax.axvline(lambda_max, color="orange", linestyle="-", linewidth=2,
                   label=f"$\\lambda_{{\\max}}$ = {lambda_max:.3f}")
        status = "Pure noise ✓" if is_pure_noise else f"Signal present (ratio={signal_ratio:.2f})"
        ax.set_title(title or f"Eigenvalue Spectrum — {status}", fontsize=12)
        ax.set_xlabel("Eigenvalue $\\lambda$", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        if own_fig:
            plt.tight_layout()
            plt.show()

    return is_pure_noise, signal_ratio


def get_max_pairwise_distance(data) -> float:
    """
    Compute σ₁ as the maximum pairwise Euclidean distance between
    all training configurations (Song & Ermon 2020, Technique 1).

    data : torch.Tensor (N, 1, L, L) or (N, D)
    """
    from scipy.spatial.distance import pdist
    X_flat = data.view(data.shape[0], -1).cpu().numpy()
    return float(np.max(pdist(X_flat, metric="euclidean")))


def find_optimal_sigma_max(
    clean_data,
    sigma_start: float = 0.01,
    sigma_end: float = 100,
    steps: int = 50,
    rounds: int = 3,
    visualize: bool = True,
    save_fig: bool = True,
) -> float:
    """
    Two-phase σ₁ selection via the Marchenko-Pastur criterion.

    Phase 2 — Global search : geometric spacing over [sigma_start, sigma_end]
    Phase 3 — Local refinement: uniform spacing over the bracketed transition

    Returns
    -------
    best_sigma : float  (optimal σ₁*)
    """
    import matplotlib.pyplot as plt

    X_clean = clean_data.view(clean_data.shape[0], -1).cpu().numpy()
    test_sigmas   = np.geomspace(sigma_start, sigma_end, num=steps)
    ratios_global = []
    ref_min, ref_max = sigma_start, sigma_end
    best_sigma = sigma_end

    print("Phase 2: Global search (geometric spacing)...")
    for i, sigma in enumerate(test_sigmas):
        noise = np.random.normal(0, 1, X_clean.shape)
        is_noise, ratio = marchenko_pastur_criterion(X_clean + sigma * noise)
        ratios_global.append(ratio)
        if is_noise:
            best_sigma = sigma
            ref_min = test_sigmas[max(i - 2, 0)]
            ref_max = test_sigmas[min(i + 1, len(test_sigmas) - 1)]
            print(f"  Transition detected at: {best_sigma:.4f}")
            print(f"  Refinement bracket:     [{ref_min:.4f}, {ref_max:.4f}]")
            break

    if best_sigma == sigma_end:
        print("  [WARNING] No transition detected. Consider increasing sigma_end.")

    print(f"\nPhase 3: Local refinement (uniform spacing)...")
    ratios_refined, sigmas_refined = [], []

    for r in range(rounds):
        if ref_max - ref_min < 1e-6:
            print(f"  [Round {r+1}] Interval collapsed — stopping.")
            break
        test_sigmas_ref  = np.linspace(ref_min, ref_max, num=steps)
        transition_found = False
        print(f"  [Round {r+1}] [{ref_min:.5f}, {ref_max:.5f}]")

        for i, sigma in enumerate(test_sigmas_ref):
            noise = np.random.normal(0, 1, X_clean.shape)
            is_noise, ratio = marchenko_pastur_criterion(X_clean + sigma * noise)
            if r == rounds - 1:
                ratios_refined.append(ratio)
                sigmas_refined.append(sigma)
            if is_noise and not transition_found:
                best_sigma       = sigma
                transition_found = True
                ref_min = test_sigmas_ref[max(i - 2, 0)]
                ref_max = test_sigmas_ref[min(i + 1, len(test_sigmas_ref) - 1)]
                print(f"    Transition at {best_sigma:.5f} → [{ref_min:.5f}, {ref_max:.5f}]")
                break
        if not transition_found:
            print(f"    No transition found — keeping bracket.")

    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(test_sigmas[:len(ratios_global)], ratios_global,
                     "o-", color=UV_BLUE, markersize=5, linewidth=1.5, label="Signal ratio $\\rho$")
        axes[0].axhline(1.0, color="red", linestyle="--", linewidth=1.5,
                        label="Pure noise limit ($\\rho = 1$)")
        axes[0].axvline(best_sigma, color=UV_GREEN, linestyle="--", linewidth=1.5,
                        label=f"$\\sigma_1^* = {best_sigma:.2f}$")
        axes[0].set_xscale("log")
        axes[0].set_xlabel("$\\sigma$", fontsize=12)
        axes[0].set_ylabel("$\\rho = \\lambda_{{\\max}} / \\lambda_+$", fontsize=12)
        axes[0].set_title("Phase 2: Global Search", fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, which="both", alpha=0.4)

        if sigmas_refined:
            axes[1].plot(sigmas_refined, ratios_refined,
                         "o-", color=UV_GREEN, markersize=5, linewidth=1.5,
                         label="Signal ratio $\\rho$")
            axes[1].axhline(1.0, color="red", linestyle="--", linewidth=1.5,
                            label="Pure noise limit ($\\rho = 1$)")
            axes[1].axvline(best_sigma, color=UV_BLUE, linestyle="--", linewidth=1.5,
                            label=f"$\\sigma_1^* = {best_sigma:.4f}$")
            axes[1].set_xlabel("$\\sigma$", fontsize=12)
            axes[1].set_ylabel("$\\rho = \\lambda_{{\\max}} / \\lambda_+$", fontsize=12)
            axes[1].set_title("Phase 3: Local Refinement", fontsize=12)
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.4)

        plt.suptitle("$\\sigma_1$ Selection via Marchenko-Pastur Criterion",
                     fontsize=13, y=1.01)
        plt.tight_layout()
        if save_fig:
            plt.savefig("figures/Fig_MP_search_curves.pdf", bbox_inches="tight", dpi=150)
            print("  Figure saved: figures/Fig_MP_search_curves.pdf")
        plt.show()

    print(f"\nFinal optimal σ₁* = {best_sigma:.4f}")
    return best_sigma


# ──────────────────────────────────────────────────────────────
# Evaluation metrics (R3, R4)
# ──────────────────────────────────────────────────────────────

def compute_nrmse(obs_gen: dict, obs_mc: dict, temperatures: list) -> dict:
    """
    Normalized RMSE per observable.
    Normalization: RMSE_X / (max(X_MC) − min(X_MC)) over all temperatures.
    """
    T_vals = [T for T in sorted(temperatures) if T in obs_mc and T in obs_gen]
    ranges = {}
    for key in ["M_mean", "E_mean", "Cv", "Chi"]:
        mc_vals    = [obs_mc[T][key] for T in T_vals]
        ranges[key] = max(mc_vals) - min(mc_vals)

    sq = {k: [] for k in ["M_mean", "E_mean", "Cv", "Chi"]}
    for T in T_vals:
        for key in sq:
            sq[key].append((obs_gen[T][key] - obs_mc[T][key]) ** 2)

    return {
        "nRMSE_M":   np.sqrt(np.mean(sq["M_mean"])) / ranges["M_mean"],
        "nRMSE_E":   np.sqrt(np.mean(sq["E_mean"])) / ranges["E_mean"],
        "nRMSE_Cv":  np.sqrt(np.mean(sq["Cv"]))     / ranges["Cv"],
        "nRMSE_Chi": np.sqrt(np.mean(sq["Chi"]))    / ranges["Chi"],
    }


def compute_ips(nrmse_dict: dict, weights: dict = None) -> float:
    """
    Integrated Physical Score (IPS): weighted sum of nRMSEs. Lower is better.

    weights are: M=0.30, E=0.30, Cv=0.20, χ=0.20n due to first-order or second-order quantities. 
    """
    if weights is None:
        weights = {"nRMSE_M": 0.30, "nRMSE_E": 0.30, "nRMSE_Cv": 0.20, "nRMSE_Chi": 0.20}
    return float(sum(weights[k] * nrmse_dict[k] for k in weights))


def compute_wasserstein(obs_gen: dict, obs_mc: dict, temperatures: list) -> float:
    """
    Mean Wasserstein-1 distance on magnetization and energy distributions
    across all temperatures.
    """
    w1_list = []
    for T in sorted(temperatures):
        if T not in obs_gen or T not in obs_mc:
            continue
        gt, gen = obs_mc[T], obs_gen[T]
        if "m_vals" in gt and "m_vals" in gen:
            w1_list.append(wasserstein_distance(gt["m_vals"], gen["m_vals"]))
        if "e_vals" in gt and "e_vals" in gen:
            w1_list.append(wasserstein_distance(gt["e_vals"], gen["e_vals"]))
    return float(np.nanmean(w1_list)) if w1_list else float("nan")


def pareto_frontier(times, ips_vals, ids) -> list:
    """
    Compute the Pareto frontier (minimize both inference time and quality metric).

    Returns list of (time, score, id) for Pareto-optimal points,
    sorted by increasing time.
    """
    pts = sorted(zip(times, ips_vals, ids), key=lambda p: p[0])
    frontier, min_score = [], float("inf")
    for t, score, eid in pts:
        if score < min_score:
            frontier.append((t, score, eid))
            min_score = score
    return frontier


# ──────────────────────────────────────────────────────────────
# Summary tables (R3, R4)
# ──────────────────────────────────────────────────────────────

def print_ablation_table(
    results: list,
    mc_ground_truth: dict,
    temperatures: list,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Print TABLE 3: sampler sensitivity analysis — nRMSE, W1, IPS, generation time.

    results: list of dicts from run_experiment or load_offline_samples_and_compute.
    """
    T_vals = sorted([T for T in temperatures if T in mc_ground_truth])
    mc_ranges = {}
    for key in ["M_mean", "E_mean", "Cv", "Chi"]:
        vals = [mc_ground_truth[T][key] for T in T_vals]
        mc_ranges[key] = max(vals) - min(vals)

    rows = []
    for res in results:
        obs = res["obs_per_temp"]
        sq_err = {"M": [], "E": [], "Cv": [], "Chi": []}
        w1_m_list, w1_e_list = [], []

        valid_T = [T for T in T_vals if T in obs]
        for T in valid_T:
            sq_err["M"].append((obs[T]["M_mean"] - mc_ground_truth[T]["M_mean"]) ** 2)
            sq_err["E"].append((obs[T]["E_mean"] - mc_ground_truth[T]["E_mean"]) ** 2)
            sq_err["Cv"].append((obs[T]["Cv"]    - mc_ground_truth[T]["Cv"])     ** 2)
            sq_err["Chi"].append((obs[T]["Chi"]  - mc_ground_truth[T]["Chi"])    ** 2)
            if "m_vals" in mc_ground_truth[T] and "m_vals" in obs[T]:
                w1_m_list.append(wasserstein_distance(mc_ground_truth[T]["m_vals"], obs[T]["m_vals"]))
            if "e_vals" in mc_ground_truth[T] and "e_vals" in obs[T]:
                w1_e_list.append(wasserstein_distance(mc_ground_truth[T]["e_vals"], obs[T]["e_vals"]))

        nrmse_M   = np.sqrt(np.mean(sq_err["M"]))   / mc_ranges["M_mean"]
        nrmse_E   = np.sqrt(np.mean(sq_err["E"]))   / mc_ranges["E_mean"]
        nrmse_Cv  = np.sqrt(np.mean(sq_err["Cv"]))  / mc_ranges["Cv"]
        nrmse_Chi = np.sqrt(np.mean(sq_err["Chi"])) / mc_ranges["Chi"]
        w1_int    = float(np.nanmean(w1_m_list + w1_e_list))
        ips       = 0.30 * nrmse_M + 0.30 * nrmse_E + 0.20 * nrmse_Cv + 0.20 * nrmse_Chi
        gen_time  = res.get("gen_time_ms", float("nan"))

        rows.append({
            "Exp":       res["exp_id"],
            "M":         res["M"],
            "K":         res["K"],
            "λ₀":        res["lambda0"],
            "nRMSE_M":   f"{nrmse_M:.4f}",
            "nRMSE_E":   f"{nrmse_E:.4f}",
            "nRMSE_Cv":  f"{nrmse_Cv:.4f}",
            "nRMSE_Chi": f"{nrmse_Chi:.4f}",
            "W1_Dist":   f"{w1_int:.4f}",
            "IPS":       f"{ips:.4f}",
            "Time(ms)":  f"{gen_time:.2f}",
            "_IPS":      ips,
            "_Time":     gen_time,
        })

    df = pd.DataFrame(rows).sort_values(by=["_IPS", "_Time"], ascending=True)
    display_cols = ["Exp", "M", "K", "λ₀", "nRMSE_M", "nRMSE_E",
                    "nRMSE_Cv", "nRMSE_Chi", "W1_Dist", "IPS", "Time(ms)"]
    print("\n" + "=" * 105)
    print("TABLE 3: Sampler Sensitivity Analysis — Normalized & Distributional Metrics")
    print("=" * 105)
    print(df[display_cols].to_string(index=False))
    print("=" * 105)
    if save_csv:
        df.to_csv("tables/Table_ablation_normalized.csv", index=False)
        print("  Saved: tables/Table_ablation_normalized.csv")
    return df


def print_comparison_table(
    obs_mc: dict,
    obs_ncsn: dict,
    obs_ddpm: dict,
    temperatures: list,
    save_csv: bool = True,
) -> None:
    """
    Print TABLE 4: thermodynamic fidelity comparison — NCSNv2 vs DDPM.
    """
    nrmse_ncsn = compute_nrmse(obs_ncsn, obs_mc, temperatures)
    nrmse_ddpm = compute_nrmse(obs_ddpm, obs_mc, temperatures)
    ips_ncsn   = compute_ips(nrmse_ncsn)
    ips_ddpm   = compute_ips(nrmse_ddpm)

    print("\n" + "=" * 65)
    print("TABLE 4: Thermodynamic Fidelity — NCSNv2 vs DDPM")
    print("  nRMSE_X = RMSE_X / range(X_MC) | IPS: M=0.30, E=0.30, Cv=0.20, χ=0.20")
    print("=" * 65)
    print(f"  {'Metric':<20} {'NCSNv2':>12} {'DDPM':>12}")
    print("-" * 65)
    metrics = [
        ("nRMSE_M",   "nRMSE ⟨|m|⟩"),
        ("nRMSE_E",   "nRMSE ⟨e⟩"),
        ("nRMSE_Cv",  "nRMSE Cv"),
        ("nRMSE_Chi", "nRMSE χ"),
    ]
    for key, label in metrics:
        print(f"  {label:<20} {nrmse_ncsn[key]:>12.5f} {nrmse_ddpm[key]:>12.5f}")
    print("-" * 65)
    print(f"  {'IPS (weighted)':<20} {ips_ncsn:>12.5f} {ips_ddpm:>12.5f}")
    print("=" * 65)

    if save_csv:
        with open("tables/Table4_fidelity_comparison.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Metric", "NCSNv2", "DDPM"])
            for key, label in metrics:
                w.writerow([label, f"{nrmse_ncsn[key]:.5f}", f"{nrmse_ddpm[key]:.5f}"])
            w.writerow(["IPS", f"{ips_ncsn:.5f}", f"{ips_ddpm:.5f}"])
        print("  Saved: tables/Table4_fidelity_comparison.csv")
