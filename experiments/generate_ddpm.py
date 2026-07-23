"""
Generate DDPM baseline samples for all 11 temperatures.

Loads the conditional DDPM checkpoint (networks/ising_ddpm_11t.pt) and samples
n configurations per temperature into --output (default gen_data/DDPM/), using
the same per-file schema as the pre-generated baseline:
  {"samples": (n,1,64,64) in {-1,+1}, "temperature": T, "n_samples": n,
   "gen_time_ms": per-sample time}.

Sampling uses noise_steps=1000 (999 network evaluations/sample, NO
classifier-free guidance). This is the reproducible source of the DDPM samples
compared in Block 4.

Usage
-----
python experiments/generate_ddpm.py \
    --model    networks/ising_ddpm_11t.pt \
    --n        1000 \
    --batch_size 4 \
    --output   gen_data/DDPM
"""
import argparse
import os
import time

import torch

from difuvia.ddpm import Diffusion, UNet_conditional
from difuvia.model_utils import get_device

TEMPERATURES = [round(1.77 + i * 0.1, 2) for i in range(11)]


def main():
    parser = argparse.ArgumentParser(description="Generate DDPM baseline samples")
    parser.add_argument("--model",      default="networks/ising_ddpm_11t.pt")
    parser.add_argument("--n",          type=int, default=1000, help="Samples per temperature")
    parser.add_argument("--batch_size", type=int, default=4, help="Samples per diffusion call")
    parser.add_argument("--output",     default="gen_data/DDPM")
    parser.add_argument("--img_size",   type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model = UNet_conditional(device=device).to(device)
    diffusion = Diffusion(img_size=args.img_size, device=device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Loaded DDPM checkpoint: {args.model}  (noise_steps={diffusion.noise_steps}, "
          f"NFE/sample={diffusion.noise_steps - 1}, no CFG)")

    os.makedirs(args.output, exist_ok=True)
    print("=" * 60)
    print(f"Temperatures : {TEMPERATURES}")
    print(f"Samples/T    : {args.n}   batch_size={args.batch_size}")
    print(f"Output dir   : {args.output}")
    print("=" * 60)

    for T in TEMPERATURES:
        out_path = os.path.join(args.output, f"T{T:.2f}.pt")
        if os.path.exists(out_path):
            print(f"\nT={T:.2f}  ->  already exists, skipping.")
            continue

        print(f"\nT = {T:.2f} — generating {args.n} samples ...")
        t_start = time.time()
        all_samples = []
        n_batches = args.n // args.batch_size
        remainder = args.n % args.batch_size

        with torch.no_grad():
            for _ in range(n_batches):
                labels = torch.full((args.batch_size,), T, dtype=torch.float, device=device)
                sampled = diffusion.sample(model, n=args.batch_size, labels=labels)
                all_samples.append(torch.sign(sampled).cpu())
            if remainder > 0:
                labels = torch.full((remainder,), T, dtype=torch.float, device=device)
                sampled = diffusion.sample(model, n=remainder, labels=labels)
                all_samples.append(torch.sign(sampled).cpu())

        configs = torch.cat(all_samples, dim=0)
        gen_time = (time.time() - t_start) / len(configs)
        torch.save(
            {"samples": configs, "temperature": T, "n_samples": len(configs),
             "gen_time_ms": gen_time * 1000},
            out_path,
        )
        print(f"  Saved {len(configs)} configs -> {out_path}  "
              f"[{gen_time * 1000:.1f} ms/sample]")

    print("\nDDPM generation complete.")


if __name__ == "__main__":
    main()
