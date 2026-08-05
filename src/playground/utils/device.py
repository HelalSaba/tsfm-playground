from __future__ import annotations

import torch

DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    if device_arg == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps was requested, but MPS is not available")
    return torch.device(device_arg)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_generator(seed: int) -> torch.Generator:
    # MPS does not support torch.Generator on the MPS device.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def randperm(n: int, *, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.randperm(n, generator=generator).to(device)


def randn(size: int, *, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.randn(size, generator=generator).to(device)
