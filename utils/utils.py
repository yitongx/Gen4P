"""Numerical helpers used by the D4RT decoder."""

import torch
import torch.nn.functional as F


def soft_clamp_max(
    x: torch.Tensor,
    max_val: float,
    beta: float = 1.0,
) -> torch.Tensor:
    assert beta > 0, f"beta must be > 0, got beta={beta}"
    return x - F.softplus(beta * (x - float(max_val))) / beta
