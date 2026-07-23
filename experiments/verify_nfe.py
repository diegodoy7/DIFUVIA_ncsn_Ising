"""
Verify the NFE (Number of Function Evaluations) metric empirically.

Counts the actual forward passes through the learned network during ONE
generation trajectory and checks them against the closed forms:
  - PIPC : NFE = M*(K+1)        (E1–E12 physics-guided + A1–A3 physics-free)
  - DDPM : NFE = noise_steps-1  (= 999, no classifier-free guidance)

If any empirical count disagrees with the formula, the script prints the
discrepancy and exits non-zero WITHOUT altering anything (per project rule:
report, do not auto-fix). Uses tiny probe batches, so it is cheap, but it does
load the checkpoints and run the models.

Usage
-----
python experiments/verify_nfe.py
python experiments/verify_nfe.py --skip_ddpm
"""

import argparse
import sys

from difuvia.data_access import ensure_data
from difuvia.nfe import verify_pipc_nfe, verify_ddpm_nfe, print_nfe_verification


# Same grids the ablation scripts use.
PIPC_GRID = [
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
    {"id": "A1",  "M": 150, "K": 1, "lambda0": 0.0},
    {"id": "A2",  "M": 150, "K": 2, "lambda0": 0.0},
    {"id": "A3",  "M": 150, "K": 3, "lambda0": 0.0},
]


def main():
    parser = argparse.ArgumentParser(description="Empirical NFE verification")
    parser.add_argument("--ncsn_ckpt", default="networks/ckpt_epoch_300.pth")
    parser.add_argument("--ddpm_ckpt", default="networks/ising_ddpm_11t.pt")
    parser.add_argument("--config",    default="ncsnv2/configs/ising.yml")
    parser.add_argument("--probe_batch", type=int, default=2)
    parser.add_argument("--skip_ddpm", action="store_true")
    args = parser.parse_args()

    from difuvia.model_utils import load_model, get_device
    from difuvia.physics import SoftIsingEnergy

    ensure_data(["networks"])
    device = get_device()
    print(f"Device: {device}")

    # PIPC verification (E1–E12 + A1–A3).
    score_model, sigmas, _ = load_model(args.ncsn_ckpt, args.config, device)
    energy_fn = SoftIsingEnergy(J=1.0, double_well_strength=0.6)
    rows = verify_pipc_nfe(score_model, energy_fn, sigmas, PIPC_GRID,
                           device=device, probe_batch=args.probe_batch)
    pipc_ok = print_nfe_verification(rows)

    ddpm_ok = True
    if not args.skip_ddpm:
        from difuvia.ddpm import Diffusion, UNet_conditional
        import torch

        ddpm = UNet_conditional(device=device).to(device)
        ddpm.load_state_dict(torch.load(args.ddpm_ckpt, map_location=device))
        diffusion = Diffusion(img_size=64, device=device)
        info = verify_ddpm_nfe(ddpm, diffusion, device=device)
        print("\n" + "=" * 60)
        print("DDPM NFE")
        print("=" * 60)
        print(f"  noise_steps        : {info['noise_steps']}")
        print(f"  classifier-free    : {info['uses_cfg']}  (cfg_scale is unused dead code)")
        print(f"  formula (steps-1)  : {info['nfe_formula']}")
        print(f"  empirical          : {info['nfe_empirical']}")
        print(f"  match              : {info['match']}")
        print("=" * 60)
        ddpm_ok = info["match"]

    if not (pipc_ok and ddpm_ok):
        print("\n[STOP] NFE mismatch detected — reporting, not fixing. See tables above.")
        sys.exit(1)
    print("\nAll NFE counts match their closed forms.")


if __name__ == "__main__":
    main()
