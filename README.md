# DIFUVIA: Physics-Informed Score-Based Diffusion for the 2D Ising Model

**DIFUVIA**  is a score-based generative model trained on 2D Ising spin configurations. The model extends NCSNv2 (Song & Ermon, 2020) with a temperature-conditional Fourier feature embedding and is sampled via a physics-guided Predictor-Corrector (PIPC) algorithm that injects the Ising Hamiltonian gradient as a corrector force during inference. The result faithfully reproduces exact thermodynamic observables — ⟨|m|⟩, ⟨e⟩, C_v, χ, and G(r) — across the ferro–paramagnetic phase transition, including critical scaling at T_c ≈ 2.269.

## Repository structure

```
difuvia-ncsn-pipc/
├── difuvia/                        ← core Python package
│   ├── physics.py                  ← SoftIsingEnergy, exact theory (Yang 1952, Onsager 1944)
│   ├── thermodynamics.py           ← observables, spin correlation G(r), MC data loaders
│   ├── sampling.py                 ← PC sampler, annealed Langevin, ablation grid runner
│   ├── ddpm.py                     ← DDPM baseline (conditional U-Net + diffusion)
│   ├── nfe.py                      ← NFE metric: counter, formulas, verification
│   ├── model_utils.py              ← model loading, YAML config, device selection
│   ├── data_access.py              ← on-demand Hugging Face data fetch
│   ├── analysis.py                 ← nRMSE, IPS, W1, NFE table, time-vs-NFE fit
│   └── viz.py                      ← all matplotlib figures
├── experiments/
│   ├── r1_dataset_validation.py    ← Block 1: MC dataset vs exact theory
│   ├── r2_sigma_selection.py       ← Block 2: σ₁ selection via Marchenko-Pastur criterion
│   ├── r3_ablation.py              ← Block 3: PC sampler grid (--source download|generate)
│   ├── r3_offline_analysis.py      ← Block 3: offline analysis of saved samples
│   ├── r3b_ablation_nophysics.py   ← Block 3c: physics-free ablation (λ₀=0, A1–A3)
│   ├── verify_nfe.py               ← empirical NFE verification (PIPC + DDPM)
│   ├── generate_ddpm.py            ← regenerate DDPM baseline samples
│   └── r4_comparative.py           ← Block 4: NCSNv2-PIPC vs DDPM full comparison
├── ncsnv2/                         ← NCSNv2 architecture (Song & Ermon 2020)
│   ├── configs/ising.yml           ← model and training configuration
│   ├── models/ncsnv2_difuvia.py    ← temperature-conditional score network (DIFUVIA)
│   ├── losses/dsm.py               ← Denoising Score Matching training loss
│   └── runners/ncsn_runner.py      ← training loop
├── data/
│   └── ising_adapter_continuous.py ← Monte Carlo dataset adapter
├── scripts/
│   └── download_data.py            ← fetches data/checkpoints from Hugging Face Hub
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/diegodoy7/DIFUVIA_ncsn_Ising.git
cd DIFUVIA_ncsn_Ising

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # makes the difuvia package importable from anywhere
```

> **Hardware:** experiments were run on Apple Silicon (MPS backend). CUDA and CPU are also supported — `difuvia.model_utils.get_device()` selects automatically.

---

## Data & checkpoints (Hugging Face)

The Monte Carlo dataset, pre-generated ablation samples, comparison samples, and model checkpoints are hosted on Hugging Face Hub at [diegodoy7/difuvia-ncsn-ising-data](https://huggingface.co/datasets/diegodoy7/difuvia-ncsn-ising-data) — **no training or sample generation is required to reproduce any of the four research blocks below.**

| Path | Contents | Size |
|---|---|---|
| `train_data/` | Monte Carlo (Wolff) Ising configurations, 11 temperatures | ~110 MB |
| `ablation_samples/` | PC sampler ablation grid, 12 configs (E1–E12) × 11 temperatures | ~2.0 GB |
| `gen_data/NCSN/`, `gen_data/DDPM/` | Pre-generated samples for the Block 4 comparison | ~345 MB |
| `networks/` | Trained NCSNv2-PIPC and DDPM checkpoints | ~175 MB |

**Total: ~2.6 GB.** After installing dependencies (see above), fetch the data with:

```bash
python scripts/download_data.py
```

This downloads `train_data/`, `ablation_samples/`, `gen_data/`, and `networks/` into the repo root via the `huggingface_hub` library — no account or token needed, the dataset repo is public.

---

## Dataset

The model is trained on a Monte Carlo dataset generated with the Wolff cluster algorithm:
- **Lattice:** L = 64 × 64, periodic boundary conditions
- **Temperatures:** 11 values from T* = 1.77 to T* = 2.77 (spacing 0.1), bracketing T_c ≈ 2.269
- **Samples:** N = 1000 per temperature

Ships at `train_data/` via Hugging Face Hub (see above) — no separate download needed beyond `scripts/download_data.py`. Each temperature has a corresponding file readable by `data/ising_adapter_continuous.py`.

---

## Reproducing results

Run the four research blocks in order. Each script saves figures to `figures/` and tables to `tables/`.

### Block 1 — Dataset validation

Validates the Monte Carlo dataset against exact 2D Ising theory (Yang 1952, Onsager 1944).

```bash
python experiments/r1_dataset_validation.py \
    --data_dir train_data \
    --fig_dir  figures \
    --table_dir tables
```

Outputs: `figures/Fig1_representative_lattices.pdf`, `figures/Fig2_thermodynamic_validation.pdf`, `tables/Table1_validation_wolff.csv`

---

### Block 2 — Noise scale selection (σ₁)

Selects the maximum noise level σ₁ for the NCSNv2 noise schedule using the Marchenko-Pastur eigenvalue criterion, providing a more principled alternative to the Song & Ermon max-pairwise-distance heuristic.

```bash
python experiments/r2_sigma_selection.py \
    --config    ncsnv2/configs/ising.yml \
    --fig_dir   figures \
    --table_dir tables
```

Outputs: `figures/Fig_MP_eigenvalue_spectra.pdf`, `tables/Table2_sigma_comparison.csv`

---

### Block 3 — PC sampler ablation study

Sweeps 12 configurations (E1–E12) of the physics-guided Predictor-Corrector sampler over (M, K, λ₀). All 132 sample files (E1–E12 × 11 temperatures) ship in `ablation_samples/` via Git LFS — **skip straight to Step 3b** unless you want to regenerate or extend the grid.

**Step 3a — use available data or generate.** `r3_ablation.py` takes `--source`:
`download` (default; pulls the pre-generated E1–E12 from Hugging Face if not present)
or `generate` (re-runs the score model from the checkpoint — may take hours on CPU/MPS).

```bash
# use the shipped samples
python experiments/r3_ablation.py --source download

# or regenerate from the trained checkpoint
python experiments/r3_ablation.py --source generate \
    --checkpoint networks/ckpt_epoch_300.pth --n_samples 1000 --save_dir ablation_samples
```

**Step 3b** — offline analysis (loads saved `.pt` files; fast):

```bash
python experiments/r3_offline_analysis.py \
    --save_dir ablation_samples --data_dir train_data --ddpm_dir gen_data/DDPM \
    --fig_dir figures --table_dir tables
```

The results table now reports, per configuration, **M, K, λ₀, NFE, per-sample time, and
speedup vs the DDPM baseline**, and a consistency fit `time ≈ a·NFE + b` (with R²).
Outputs: `figures/Fig3_ablation_observables.pdf`, `figures/Fig3_ablation_correlation.pdf`, `figures/Fig3_pareto_IPS.pdf`, `tables/Table_ablation_normalized.csv`

**Step 3c — physics-free ablation (λ₀ = 0).** Isolates the contribution of the physical
guidance by running the *same* sampler and weights with guidance off, at M=150, K∈{1,2,3}
(configs A1–A3), kept separate under `ablation_samples_nophysics/`:

```bash
python experiments/r3b_ablation_nophysics.py --source generate   # or --source download
```

Outputs: `tables/Table_ablation_nophysics.csv`, `figures/Fig3b_nophysics_*.pdf`.

**NFE verification.** Confirm the closed-form NFE (`M·(K+1)` for PIPC, `999` for DDPM)
against an empirical forward-pass counter:

```bash
python experiments/verify_nfe.py
```

---

### Block 4 — Comparative study (NCSNv2-PIPC vs DDPM)

Compares NCSNv2-PIPC against a DDPM baseline across all temperatures and thermodynamic observables. Pre-generated samples for both models ship in `gen_data/NCSN/` and `gen_data/DDPM/` via Git LFS, so this runs immediately with no generation step:

```bash
python experiments/r4_comparative.py \
    --ncsn_dir       gen_data/NCSN \
    --ddpm_dir       gen_data/DDPM \
    --data_dir       train_data \
    --n_bootstrap    0 \
    --fig_dir        figures \
    --table_dir      tables \
    --experiment_name E9
```

If `--ncsn_dir` is omitted, samples are instead generated on-the-fly from the checkpoint.

The DDPM baseline is fully reproducible in-repo: `difuvia/ddpm.py` holds the conditional
U-Net + diffusion, and `experiments/generate_ddpm.py --output gen_data/DDPM` regenerates the
DDPM samples from `networks/ising_ddpm_11t.pt` (noise_steps=1000, **NFE = 999**, no
classifier-free guidance).

Outputs: `figures/Fig4_thermodynamic_comparison.pdf`, `figures/Fig5_lattice_comparison.pdf`, `figures/Fig6_correlation_comparison.pdf`, `tables/Table4_fidelity_comparison.csv`


---

## Evaluation metrics

| Metric | Description |
|---|---|
| **nRMSE_X** | RMSE(X_gen, X_MC) / range(X_MC) — normalized per observable |
| **IPS** | Integrated Physical Score = 0.30·nRMSE_M + 0.30·nRMSE_E + 0.20·nRMSE_Cv + 0.20·nRMSE_χ |
| **NFE** | Number of Function Evaluations = network forward passes per sample. PIPC: M·(K+1); DDPM: 999 (noise_steps−1, no CFG). The analytical Ising force does not count. |
| **Speedup** | DDPM per-sample time ÷ PIPC per-sample time (same hardware). Consistency: `time ≈ a·NFE + b` fit with R². |
| **W1** | Mean Wasserstein-1 distance on m and e sample distributions across temperatures |

Lower is better for all three metrics.

---

## References

```bibtex
@inproceedings{song2020improved,
  title     = {Improved Techniques for Training Score-Based Generative Models},
  author    = {Yang Song and Stefano Ermon},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2020}
}

@inproceedings{song2019generative,
  title     = {Generative Modeling by Estimating Gradients of the Data Distribution},
  author    = {Yang Song and Stefano Ermon},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2019}
}

@article{lee2025thermodynamic,
  title={Thermodynamic fidelity of generative models for Ising system},
  author={Lee, Brian H and Nykiel, Kat and Hallberg, Ava E and Rider, Brice and Strachan, Alejandro},
  journal={Journal of Applied Physics},
  volume={137},
  number={12},
  year={2025},
  publisher={AIP Publishing}
}

@article{onsager1944crystal,
  title   = {Crystal Statistics. {I}. A Two-Dimensional Model with an Order-Disorder Transition},
  author  = {Lars Onsager},
  journal = {Physical Review},
  volume  = {65},
  number  = {3--4},
  pages   = {117--149},
  year    = {1944}
}
```
