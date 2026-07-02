"""
Model loading, configuration management, and device selection.

dict2namespace is re-exported from data.ising_adapter_continuous (canonical source)
to provide a single import path without duplicating the implementation.
"""

import os
import yaml
import torch

from data.ising_adapter_continuous import dict2namespace  # canonical definition


def load_config(config_path: str = "ncsnv2/configs/ising.yml"):
    """Load and parse a YAML configuration file into an argparse Namespace."""
    with open(config_path, "r") as f:
        return dict2namespace(yaml.safe_load(f))


def get_device() -> torch.device:
    """Select MPS (Apple Silicon), CUDA, or CPU — in that priority order."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: str, config_path: str, device: torch.device):
    """
    Load a trained NCSNv2 checkpoint.

    Parameters
    ----------
    checkpoint_path : path to .pth checkpoint file
    config_path     : path to YAML config (default: ncsnv2/configs/ising.yml)
    device          : torch.device

    Returns
    -------
    (score_model, sigmas, config)
    """
    from ncsnv2.models import get_sigmas
    from ncsnv2.models.ncsnv2_difuvia import NCSNv2

    config = load_config(config_path)
    config.device = device
    sigmas = get_sigmas(config).to(device)
    config.training.sigmas = sigmas
    score = NCSNv2(config).to(device)
    states = torch.load(checkpoint_path, map_location=device, weights_only=False)
    score.load_state_dict(states["model_state"])
    score.eval()
    print(f"Model loaded: {checkpoint_path}")
    return score, sigmas, config


def save_simulation_data(
    samples: torch.Tensor,
    temperature: float,
    L: int,
    folder: str = "gen_data",
) -> None:
    """Save generated Ising spin configurations to a .pt file."""
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"genSamples_L{L}_T{temperature:.2f}.pt")
    torch.save({"temperature": temperature, "L": L, "samples": samples.cpu()}, filename)
    print(f"Data saved: {filename}")
