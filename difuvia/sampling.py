"""
Sampling algorithms for the physics-informed NCSN.

Contains:
- pc_physics_guided_sampler: Predictor-Corrector + physical force injection
- generate_samples: standard annealed Langevin dynamics (no physics)
- run_experiment: grid search loop that generates and saves ablation samples
- load_offline_samples_and_compute: reload saved .pt files without re-running the model
"""

import os
import time
import numpy as np
import torch

from difuvia.physics import Tc
from difuvia.thermodynamics import compute_observables, compute_spin_correlation


@torch.no_grad()
def pc_physics_guided_sampler(
    score_model,
    energy_fn,
    sigmas,
    target_temp: float = 2.27,
    image_size: int = 64,
    batch_size: int = 16,
    num_steps: int = 100,
    corrector_steps: int = 1,
    guidance_scale: float = 0.2,
    snr: float = 0.16,
    device=None,
) -> torch.Tensor:
    """
    Physics-Guided Predictor-Corrector (PC) Sampler.

    Combines the learned score estimate with an analytical physical energy
    gradient (e.g. SoftIsingEnergy) at each denoising step.

    Parameters
    ----------
    score_model    : NCSNv2 with temperature conditioning
    energy_fn      : object with .get_grad(x) method (e.g. SoftIsingEnergy)
    sigmas         : noise schedule tensor (from get_sigmas)
    num_steps      : number of PC denoising steps (M in ablation notation)
    corrector_steps: number of Langevin corrector steps per noise level (K)
    guidance_scale : weight of the physics force term (λ₀)
    snr            : signal-to-noise ratio controlling corrector step size

    Returns
    -------
    torch.Tensor (batch_size, 1, image_size, image_size) in {-1, +1}
    """
    if device is None:
        device = next(score_model.parameters()).device

    sigmas_np = sigmas.data.cpu().numpy()
    timesteps = np.linspace(0, len(sigmas_np) - 1, num_steps).astype(int)
    used_sigmas = sigmas_np[timesteps]
    x = torch.randn(batch_size, 1, image_size, image_size, device=device)
    temp_tensor = torch.ones(batch_size, device=device) * target_temp

    for i, sigma in enumerate(used_sigmas):
        sigma_curr = sigma
        sigma_next = used_sigmas[i + 1] if i < len(used_sigmas) - 1 else 0.0
        labels = torch.ones(batch_size, device=device, dtype=torch.long) * timesteps[i]
        step_size = 2 * (snr * sigma_curr) ** 2

        # Corrector: K Langevin steps with physics guidance
        for _ in range(corrector_steps):
            with torch.no_grad():
                score_net = score_model(x, labels, y_temp=temp_tensor)
            force = energy_fn.get_grad(x)
            norm_s = torch.norm(score_net.reshape(batch_size, -1), dim=-1).mean()
            norm_f = torch.norm(force.reshape(batch_size, -1), dim=-1).mean()
            scale = guidance_scale * (norm_s / (norm_f + 1e-6))
            total = score_net + scale * force
            noise = torch.randn_like(x)
            x = x + step_size * total + np.sqrt(2 * step_size) * noise

        # Predictor: Euler step
        with torch.no_grad():
            score_net = score_model(x, labels, y_temp=temp_tensor)
        force = energy_fn.get_grad(x)
        norm_s = torch.norm(score_net.reshape(batch_size, -1), dim=-1).mean()
        norm_f = torch.norm(force.reshape(batch_size, -1), dim=-1).mean()
        scale = guidance_scale * (norm_s / (norm_f + 1e-6))
        total = score_net + scale * force
        x = x + (sigma_curr - sigma_next) * total

    return torch.sign(x)


@torch.no_grad()
def generate_samples(
    score_model,
    sigmas,
    n_samples: int = 2,
    target_temp: float = 2.27,
    image_size: int = 64,
    step_lr: float = 0.00002,
    n_steps_each: int = 100,
    discrete_output: bool = True,
    device=None,
) -> torch.Tensor:
    """
    Standard annealed Langevin dynamics sampler (no physics guidance).

    Follows Algorithm 1 from Song & Ermon (2020) / NCSNv2.
    Useful as a baseline or for quick qualitative inspection.
    """
    if device is None:
        device = next(score_model.parameters()).device

    x = torch.randn(n_samples, 1, image_size, image_size, device=device)
    temp_tensor = torch.full((n_samples,), target_temp, device=device, dtype=torch.float32)

    for c, sigma in enumerate(sigmas):
        labels = torch.ones(n_samples, device=device, dtype=torch.long) * c
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for _ in range(n_steps_each):
            z = torch.randn_like(x)
            grad = score_model(x, labels, y_temp=temp_tensor)
            x = x + step_size * grad + torch.sqrt(2 * step_size) * z

    return x.sign() if discrete_output else x.clamp(-1, 1)


def run_experiment(
    score_model,
    sigmas,
    energy_fn,
    mc_ground_truth: dict,
    temperatures: list,
    hyperparameter_grid: list,
    n_samples_per_temp: int = 1000,
    image_size: int = 64,
    device=None,
    save_dir: str = "ablation_samples",
    save_samples: bool = True,
) -> list:
    """
    Grid search over PC sampler hyperparameters (M, K, λ₀).

    For each configuration in hyperparameter_grid (list of dicts with
    keys 'id', 'M', 'K', 'lambda0'), generates n_samples_per_temp samples
    at each temperature and optionally saves to save_dir/{exp_id}_T{T:.2f}.pt.

    Returns
    -------
    list of result dicts, each with keys:
      exp_id, M, K, lambda0, gen_time_ms, obs_per_temp, corr_at_tc
    """
    if device is None:
        device = next(score_model.parameters()).device

    os.makedirs(save_dir, exist_ok=True)
    T_crit = min(temperatures, key=lambda t: abs(t - Tc))
    all_results = []

    for params in hyperparameter_grid:
        M, K, lambda0 = params["M"], params["K"], params["lambda0"]
        exp_id = params.get("id", "A?")

        print(f"\n{'='*55}")
        print(f"Experiment {exp_id}: M={M}, K={K}, λ₀={lambda0}")
        print(f"{'='*55}")

        obs_per_temp = {}
        corr_at_tc = None
        total_time = 0.0
        n_generated = 0

        for T in temperatures:
            print(f"  T={T:.2f} ...", end=" ", flush=True)

            batch_size = 50
            n_batches = n_samples_per_temp // batch_size
            samples = []

            t0 = time.time()
            for _ in range(n_batches):
                s = pc_physics_guided_sampler(
                    score_model=score_model,
                    energy_fn=energy_fn,
                    sigmas=sigmas,
                    target_temp=T,
                    image_size=image_size,
                    batch_size=batch_size,
                    num_steps=M,
                    corrector_steps=K,
                    guidance_scale=lambda0,
                    device=device,
                )
                samples.append(s.cpu())
            t1 = time.time()

            configs = torch.cat(samples, dim=0)
            gen_time = (t1 - t0) / len(configs)
            total_time += t1 - t0
            n_generated += len(configs)

            obs = compute_observables(configs, T, image_size)
            obs["gen_time"] = gen_time
            obs_per_temp[T] = obs

            if abs(T - T_crit) < 1e-9:
                r_vals, G_r = compute_spin_correlation(configs)
                corr_at_tc = (r_vals, G_r)

            print(
                f"<|m|>={obs['M_mean']:.4f}  <e>={obs['E_mean']:.4f}  "
                f"Cv={obs['Cv']:.4f}  χ={obs['Chi']:.4f}  "
                f"t={gen_time * 1000:.1f}ms/sample"
            )

            if save_samples:
                torch.save(
                    {"samples": configs, "T": T, "params": params,
                     "gen_time_ms": gen_time * 1000},
                    os.path.join(save_dir, f"{exp_id}_T{T:.2f}.pt"),
                )

        all_results.append({
            "exp_id": exp_id,
            "M": M,
            "K": K,
            "lambda0": lambda0,
            "gen_time_ms": (total_time / n_generated) * 1000 if n_generated > 0 else 0.0,
            "obs_per_temp": obs_per_temp,
            "corr_at_tc": corr_at_tc,
        })

    return all_results


def load_offline_samples_and_compute(
    hyperparameter_grid: list,
    temperatures: list,
    mc_ground_truth: dict,
    image_size: int = 64,
    save_dir: str = "ablation_samples",
) -> list:
    """
    Load pre-saved .pt files and recompute observables without re-running the model.

    Each file must be at save_dir/{exp_id}_T{T:.2f}.pt and contain
    {'samples': torch.Tensor (N, 1, L, L)}.

    Returns the same structure as run_experiment, compatible with all
    analysis and visualization functions.
    """
    T_crit = min(temperatures, key=lambda t: abs(t - Tc))
    all_results = []

    for params in hyperparameter_grid:
        M, K, lambda0 = params["M"], params["K"], params["lambda0"]
        exp_id = params.get("id", "A?")
        obs_per_temp = {}
        corr_at_tc = None
        gen_times = []

        for T in temperatures:
            filepath = os.path.join(save_dir, f"{exp_id}_T{T:.2f}.pt")
            if not os.path.exists(filepath):
                print(f"  [!] Missing: {filepath}")
                continue
            saved_data = torch.load(filepath, map_location="cpu", weights_only=False)
            configs = saved_data["samples"]
            t_ms = saved_data.get("gen_time_ms", float("nan"))
            gen_times.append(t_ms)
            obs = compute_observables(configs, T, image_size)
            obs["gen_time"] = t_ms / 1000.0 if t_ms == t_ms else 0.0
            obs_per_temp[T] = obs

            if abs(T - T_crit) < 1e-9:
                r_vals, G_r = compute_spin_correlation(configs)
                corr_at_tc = (r_vals, G_r)

        all_results.append({
            "exp_id": exp_id,
            "M": M,
            "K": K,
            "lambda0": lambda0,
            "gen_time_ms": float(np.nanmean(gen_times)) if gen_times else 0.0,
            "obs_per_temp": obs_per_temp,
            "corr_at_tc": corr_at_tc,
        })
        print(f"Loaded experiment {exp_id} ({len(obs_per_temp)} temperatures)")

    return all_results
