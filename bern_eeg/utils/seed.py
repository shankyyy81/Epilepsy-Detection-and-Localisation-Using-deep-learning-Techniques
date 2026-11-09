"""Reproducibility helpers."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, PyTorch (CPU, CUDA, MPS)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        # benchmark=True is faster on fixed-size inputs; results remain seed-reproducible
        # up to non-deterministic cuDNN kernels.
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s + worker_id)
    random.seed(s + worker_id)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
