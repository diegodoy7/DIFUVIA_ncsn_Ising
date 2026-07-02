"""
Physical model definitions for the 2D Ising system.

Contains:
- Module-level constants (J, kB, Tc)
- SoftIsingEnergy: Ising Hamiltonian + double-well potential for physics-guided sampling
- Exact analytical results (Yang 1952, Onsager 1944) for validation
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import ellipk

J  = 1.0
kB = 1.0
Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))  # ≈ 2.2692


class SoftIsingEnergy:
    """
    Ising Hamiltonian + double-well potential energy functional.

    Used by pc_physics_guided_sampler to compute physical forces that
    guide the diffusion process toward low-energy Ising configurations.

    Parameters
    ----------
    J                   : Ising coupling constant
    double_well_strength: weight of the V(x) = λ(x²-1)² binarization term
    """

    def __init__(self, J: float = 1.0, double_well_strength: float = 0.6):
        self.J = J
        self.lambda_dw = double_well_strength

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Total energy per sample: interaction (Ising) + double-well potential.

        x : (B, 1, H, W) continuous spin field
        Returns: (B,) energy tensor
        """
        kernel = torch.tensor(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]], device=x.device, dtype=x.dtype
        ).view(1, 1, 3, 3)
        neighbors = F.conv2d(x, kernel, padding=1)
        interaction = -0.5 * self.J * (x * neighbors).sum(dim=(1, 2, 3))
        double_well = self.lambda_dw * ((x ** 2 - 1) ** 2).sum(dim=(1, 2, 3))
        return interaction + double_well

    def get_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Physical force: −∇E (negative energy gradient)."""
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            e = self.energy(x_in)
            grad = torch.autograd.grad(e.sum(), x_in)[0]
        return -grad


# ──────────────────────────────────────────────────────────────
# Exact analytical results for 2D Ising model
# ──────────────────────────────────────────────────────────────

def theory_magnetization(T: float) -> float:
    """Yang (1952) exact spontaneous magnetization per spin."""
    if T >= Tc:
        return 0.0
    arg = np.sinh(2.0 / T)
    try:
        return (1.0 - arg ** (-4)) ** (1.0 / 8.0)
    except Exception:
        return 0.0


def theory_energy(T: float) -> float:
    """Onsager (1944) exact energy per spin."""
    beta = 1.0 / T
    k = 2.0 * np.sinh(2.0 * beta) / np.cosh(2.0 * beta) ** 2
    K_val = ellipk(k ** 2)
    return (-1.0 / np.tanh(2.0 * beta)) * (
        1.0 + (2.0 / np.pi) * (2.0 * np.tanh(2.0 * beta) ** 2 - 1.0) * K_val
    )


def theory_specific_heat(T: float, dT: float = 1e-5) -> float:
    """Specific heat via numerical derivative of exact energy."""
    return (theory_energy(T + dT) - theory_energy(T - dT)) / (2 * dT)


def theory_susceptibility(T: float, gamma: float = 1.75) -> float:
    """Power-law divergence of susceptibility near Tc (Onsager/Fisher)."""
    t = abs((T - Tc) / Tc)
    if t < 1e-6:
        return float("nan")
    C = 0.96258 if T < Tc else 0.02554
    return C * t ** (-gamma)


def get_theory_curves(T_min: float = 1.5, T_max: float = 3.0, n: int = 600) -> dict:
    """
    Generate dense exact theory curves, avoiding the divergence at Tc.

    Returns dict with keys 'T', 'M', 'E', 'Cv', 'Chi'.
    """
    T_below = np.linspace(T_min, Tc - 0.003, n // 2)
    T_above = np.linspace(Tc + 0.003, T_max, n // 2)
    T_th = np.concatenate([T_below, T_above])
    return {
        "T":   T_th,
        "M":   np.array([theory_magnetization(t)   for t in T_th]),
        "E":   np.array([theory_energy(t)           for t in T_th]),
        "Cv":  np.array([theory_specific_heat(t)    for t in T_th]),
        "Chi": np.array([theory_susceptibility(t)   for t in T_th]),
    }
