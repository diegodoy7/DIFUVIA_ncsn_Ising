"""
NFE (Number of Function Evaluations) instrumentation and verification.

NFE counts the number of forward passes through the *learned* neural network
during one generation trajectory. The analytical Ising force
(`SoftIsingEnergy.get_grad`) is pure autograd and does NOT evaluate the network,
so it does not count.

Closed forms
------------
- PIPC (pc_physics_guided_sampler): per noise level, the corrector does K network
  calls and the predictor 1, over M noise levels  =>  NFE = M * (K + 1).
- DDPM (Diffusion.sample): the reverse loop is `range(1, noise_steps)`, one call
  each, with NO classifier-free guidance  =>  NFE = noise_steps - 1  (= 999).

`ForwardCounter` wraps the network *externally* so the samplers themselves are
left untouched (the existing E1-E12 logic is not modified).
"""

import numpy as np


class ForwardCounter:
    """Transparent wrapper that counts calls to a torch model without changing
    its numerics. Delegates attribute access so `next(model.parameters())`,
    `.eval()`, `.train()` etc. still work for callers that introspect the model."""

    def __init__(self, model):
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "count", 0)

    def __call__(self, *args, **kwargs):
        object.__setattr__(self, "count", self.count + 1)
        return self._model(*args, **kwargs)

    def reset(self):
        object.__setattr__(self, "count", 0)

    # Delegate everything else (parameters, eval, train, attributes) to the model.
    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_model"), name)


def nfe_pipc(M: int, K: int) -> int:
    """Closed-form PIPC NFE: M predictor calls + M*K corrector calls."""
    return M * (K + 1)


def nfe_ddpm(noise_steps: int = 1000) -> int:
    """Closed-form DDPM NFE: reverse loop range(1, noise_steps), one call each.
    No classifier-free guidance, so it is NOT 2*T."""
    return noise_steps - 1


def verify_pipc_nfe(score_model, energy_fn, sigmas, grid, device=None,
                    image_size: int = 64, probe_batch: int = 2,
                    target_temp: float = 2.27) -> list:
    """
    Empirically count network calls for one PIPC trajectory per config and
    compare against the closed form M*(K+1).

    Returns a list of dicts: {id, M, K, nfe_formula, nfe_empirical, match}.
    Does NOT raise on mismatch — the caller decides (per user rule: stop and
    report discrepancies rather than silently "fixing" them).
    """
    from difuvia.sampling import pc_physics_guided_sampler

    rows = []
    for params in grid:
        M, K = params["M"], params["K"]
        lam = params["lambda0"]
        counter = ForwardCounter(score_model)
        pc_physics_guided_sampler(
            score_model=counter,
            energy_fn=energy_fn,
            sigmas=sigmas,
            target_temp=target_temp,
            image_size=image_size,
            batch_size=probe_batch,
            num_steps=M,
            corrector_steps=K,
            guidance_scale=lam,
            device=device,
        )
        formula = nfe_pipc(M, K)
        rows.append({
            "id": params.get("id", "?"),
            "M": M, "K": K, "lambda0": lam,
            "nfe_formula": formula,
            "nfe_empirical": counter.count,
            "match": counter.count == formula,
        })
    return rows


def verify_ddpm_nfe(model, diffusion, device=None, target_temp: float = 2.27) -> dict:
    """
    Empirically count network calls for one DDPM trajectory (n=1) and compare
    against noise_steps-1. Returns {noise_steps, nfe_formula, nfe_empirical,
    match, uses_cfg=False}.
    """
    import torch

    counter = ForwardCounter(model)
    labels = torch.full((1,), target_temp, dtype=torch.float, device=diffusion.device)
    diffusion.sample(counter, n=1, labels=labels)
    formula = nfe_ddpm(diffusion.noise_steps)
    return {
        "noise_steps": diffusion.noise_steps,
        "nfe_formula": formula,
        "nfe_empirical": counter.count,
        "match": counter.count == formula,
        "uses_cfg": False,
    }


def print_nfe_verification(rows: list) -> bool:
    """Pretty-print a PIPC verification table. Returns True iff all rows match."""
    all_ok = all(r["match"] for r in rows)
    print("\n" + "=" * 60)
    print("NFE VERIFICATION — empirical count vs M*(K+1)")
    print("=" * 60)
    print(f"  {'Exp':<5}{'M':>5}{'K':>4}{'formula':>9}{'empirical':>11}{'':>4}")
    for r in rows:
        flag = "OK" if r["match"] else "MISMATCH"
        print(f"  {r['id']:<5}{r['M']:>5}{r['K']:>4}"
              f"{r['nfe_formula']:>9}{r['nfe_empirical']:>11}   {flag}")
    print("=" * 60)
    if not all_ok:
        print("  [STOP] Empirical NFE does not match the formula for some configs.")
        print("         Reporting the discrepancy — NOT auto-fixing.")
    return all_ok
