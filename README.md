# DIFUVIA: Physics-Informed Score-Based Diffusion for the 2D Ising Model

**DIFUVIA**  is a score-based generative model trained on 2D Ising spin configurations. The model extends NCSNv2 (Song & Ermon, 2020) with a temperature-conditional Fourier feature embedding and is sampled via a physics-guided Predictor-Corrector (PIPC) algorithm that injects the Ising Hamiltonian gradient as a corrector force during inference. The result faithfully reproduces exact thermodynamic observables — ⟨|m|⟩, ⟨e⟩, C_v, χ, and G(r) — across the ferro–paramagnetic phase transition, including critical scaling at T_c ≈ 2.269.

## Repository structure

```
difuvia-ncsn-pipc/
├── difuvia/                        ← core Python package
│   ├── physics.py                  ← SoftIsingEnergy, exact theory (Yang 1952, Onsager 1944)
│   ├── thermodynamics.py           ← observables, spin correlation G(r), MC data loaders
│   ├── sampling.py                 ← PC sampler, annealed Langevin, ablation grid runner
│   ├── model_utils.py              ← model loading, YAML config, device selection
│   ├── analysis.py                 ← nRMSE, IPS, Wasserstein-1, Marchenko-Pastur, tables
│   └── viz.py                      ← all matplotlib figures
├── experiments/
│   ├── r1_dataset_validation.py    ← Block 1: MC dataset vs exact theory
│   ├── r2_sigma_selection.py       ← Block 2: σ₁ selection via Marchenko-Pastur criterion
│   ├── r3_ablation.py              ← Block 3: PC sampler hyperparameter grid (runs model)
│   ├── r3_offline_analysis.py      ← Block 3: offline analysis of saved samples
│   └── r4_comparative.py           ← Block 4: NCSNv2-PIPC vs DDPM full comparison
├── ncsnv2/                         ← NCSNv2 architecture (Song & Ermon 2020)
│   ├── configs/ising.yml           ← model and training configuration
│   ├── models/ncsnv2_difuvia.py    ← temperature-conditional score network (DIFUVIA)
│   ├── losses/dsm.py               ← Denoising Score Matching training loss
│   └── runners/ncsn_runner.py      ← training loop
├── data/
│   └── ising_adapter_continuous.py ← Monte Carlo dataset adapter
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

## Data & checkpoints (Git LFS)

This repository ships its Monte Carlo dataset, pre-generated ablation samples, comparison samples, and model checkpoints via [Git LFS](https://git-lfs.com/) — **no training or sample generation is required to reproduce any of the four research blocks below.**

| Path | Contents | Size |
|---|---|---|
| `train_data/` | Monte Carlo (Wolff) Ising configurations, 11 temperatures | ~110 MB |
| `ablation_samples/` | PC sampler ablation grid, 12 configs (E1–E12) × 11 temperatures | ~2.0 GB |
| `gen_data/NCSN/`, `gen_data/DDPM/` | Pre-generated samples for the Block 4 comparison | ~345 MB |
| `networks/` | Trained NCSNv2-PIPC and DDPM checkpoints | ~175 MB |

**Total: ~2.6 GB.** Install Git LFS *before* cloning, otherwise these paths will download as small text pointer files instead of real data:

```bash
# macOS
brew install git-lfs
# Debian/Ubuntu
sudo apt install git-lfs

git lfs install          # one-time, per machine
git clone https://github.com/diegodoy7/DIFUVIA_ncsn_Ising.git
```

If you already cloned without Git LFS installed, install it and run `git lfs pull` from inside the repo to fetch the real files.


---

## Dataset

The model is trained on a Monte Carlo dataset generated with the Wolff cluster algorithm:
- **Lattice:** L = 64 × 64, periodic boundary conditions
- **Temperatures:** 11 values from T* = 1.77 to T* = 2.77 (spacing 0.1), bracketing T_c ≈ 2.269
- **Samples:** N = 1000 per temperature

Ships at `train_data/` via Git LFS (see above) — no separate download needed. Each temperature has a corresponding file readable by `data/ising_adapter_continuous.py`.

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

**Step 3a (optional)** — regenerate samples from scratch (runs the score model; may take several hours on CPU):

```bash
python experiments/r3_ablation.py \
    --checkpoint networks/ckpt_epoch_300.pth \
    --data_dir   train_data \
    --n_samples  1000 \
    --save_dir   ablation_samples \
    --fig_dir    figures \
    --table_dir  tables
```

**Step 3b** — offline analysis (loads saved `.pt` files; fast):

```bash
python experiments/r3_offline_analysis.py \
    --save_dir   ablation_samples \
    --data_dir   train_data \
    --fig_dir    figures \
    --table_dir  tables
```

Outputs: `figures/Fig3_ablation_observables.pdf`, `figures/Fig3_ablation_correlation.pdf`, `figures/Fig3_pareto_IPS.pdf`, `tables/Table_ablation_normalized.csv`

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

Outputs: `figures/Fig4_thermodynamic_comparison.pdf`, `figures/Fig5_lattice_comparison.pdf`, `figures/Fig6_correlation_comparison.pdf`, `tables/Table4_fidelity_comparison.csv`


---

## Evaluation metrics

| Metric | Description |
|---|---|
| **nRMSE_X** | RMSE(X_gen, X_MC) / range(X_MC) — normalized per observable |
| **IPS** | Integrated Physical Score = 0.30·nRMSE_M + 0.30·nRMSE_E + 0.20·nRMSE_Cv + 0.20·nRMSE_χ |
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
