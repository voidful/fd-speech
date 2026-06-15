"""Differentiable moment utilities for SR-FD.

All functions force fp32 internally so that statistics are not corrupted by
bf16 accumulation. Mask-aware helpers ignore padded frames so they do not
contaminate per-utterance statistics with silence.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def batch_mean_and_second_moment(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the per-batch mean and raw second moment of a 2-D feature matrix.

    Args:
        x: ``[B, C]`` float tensor.

    Returns:
        ``(mu, M)`` where ``mu`` is ``[C]`` and ``M = E[x x^T]`` is ``[C, C]``.
    """
    if x.dim() != 2:
        raise ValueError(f"batch_mean_and_second_moment expects 2-D input, got shape {tuple(x.shape)}")
    if x.size(0) == 0:
        raise ValueError("batch_mean_and_second_moment requires non-empty batch")
    x32 = x.to(torch.float32)
    n = x32.size(0)
    mu = x32.mean(dim=0)
    # Outer-product accumulation; division by n yields E[xx^T].
    second_moment = (x32.transpose(0, 1) @ x32) / float(n)
    second_moment = 0.5 * (second_moment + second_moment.transpose(0, 1))
    return mu, second_moment


def covariance_from_mean_and_second_moment(
    mu: torch.Tensor,
    second_moment: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert raw moments to a numerically safe covariance matrix.

    cov = E[xx^T] - mu mu^T, symmetrized and regularized with ``eps * I``.
    """
    if mu.dim() != 1:
        raise ValueError(f"mu must be 1-D, got {tuple(mu.shape)}")
    if second_moment.dim() != 2 or second_moment.size(0) != second_moment.size(1):
        raise ValueError(f"second_moment must be square 2-D, got {tuple(second_moment.shape)}")
    if mu.size(0) != second_moment.size(0):
        raise ValueError(
            f"mu/second_moment dim mismatch: mu={mu.size(0)}, M={second_moment.size(0)}"
        )

    mu32 = mu.to(torch.float32)
    M32 = second_moment.to(torch.float32)
    cov = M32 - mu32.unsqueeze(1) @ mu32.unsqueeze(0)
    cov = 0.5 * (cov + cov.transpose(0, 1))
    eye = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    cov = cov + eps * eye
    return cov


def masked_time_mean_std(
    z: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pool a per-frame feature sequence into a per-utterance representation.

    Output is ``[B, 2D]`` formed by concatenating mean and std along the time
    axis. Padding frames (where ``mask == 0``) are ignored.

    Args:
        z: ``[B, D, L]`` or ``[B, L, D]``. The orientation is auto-detected by
            looking at where ``mask`` aligns; if ``mask`` is omitted we assume
            ``[B, D, L]`` (which matches AudioVAE latents).
        mask: optional ``[B, L]`` or ``[B, 1, L]``.
        eps: variance floor for numerical stability.
    """
    if z.dim() != 3:
        raise ValueError(f"masked_time_mean_std expects 3-D z, got {tuple(z.shape)}")

    z32 = z.to(torch.float32)

    if mask is None:
        # Assume [B, D, L].
        B, D, L = z32.shape
        mu = z32.mean(dim=2)
        var = z32.var(dim=2, unbiased=False).clamp_min(eps)
        std = var.sqrt()
        return torch.cat([mu, std], dim=-1)

    if mask.dim() == 3:
        if mask.size(1) != 1:
            raise ValueError(f"3-D mask must have second dim 1, got {tuple(mask.shape)}")
        mask = mask.squeeze(1)
    if mask.dim() != 2:
        raise ValueError(f"mask must be 2-D after squeeze, got {tuple(mask.shape)}")

    B, A, Bdim = z32.shape  # We don't yet know which axis is time.
    # Decide layout by matching mask length.
    L = mask.size(1)
    if Bdim == L and A != L:
        # [B, D, L]
        z_bdl = z32
    elif A == L and Bdim != L:
        # [B, L, D] -> transpose to [B, D, L]
        z_bdl = z32.transpose(1, 2).contiguous()
    elif Bdim == L:
        z_bdl = z32  # square ambiguity: prefer [B, D, L]
    else:
        raise ValueError(
            f"Cannot align mask length {L} with feature shape {tuple(z32.shape)};"
            f" expected one of [B, D, L] or [B, L, D]."
        )

    mask32 = mask.to(torch.float32)  # [B, L]
    mask_bdl = mask32.unsqueeze(1)  # [B, 1, L]

    masked_sum = (z_bdl * mask_bdl).sum(dim=2)  # [B, D]
    counts = mask32.sum(dim=1).clamp_min(1.0)  # [B]
    mu = masked_sum / counts.unsqueeze(1)
    diff = (z_bdl - mu.unsqueeze(2)) * mask_bdl
    var = (diff * diff).sum(dim=2) / counts.unsqueeze(1)
    std = var.clamp_min(eps).sqrt()
    return torch.cat([mu, std], dim=-1)


def accumulate_moments(
    state: dict,
    x: torch.Tensor,
) -> dict:
    """Stream-friendly accumulator for offline reference statistics.

    ``state`` keys: ``n``, ``sum`` (``[C]``), ``second_moment_sum`` (``[C, C]``).
    Returns the updated state dict (same object).
    """
    if x.dim() != 2:
        raise ValueError(f"accumulate_moments expects 2-D input, got {tuple(x.shape)}")
    x32 = x.to(torch.float64).detach()  # use fp64 for offline accumulation
    n_new = x32.size(0)
    sum_new = x32.sum(dim=0)
    sm_new = x32.transpose(0, 1) @ x32

    if state.get("n", 0) == 0:
        state["n"] = int(n_new)
        state["sum"] = sum_new
        state["second_moment_sum"] = sm_new
    else:
        state["n"] += int(n_new)
        state["sum"] = state["sum"] + sum_new
        state["second_moment_sum"] = state["second_moment_sum"] + sm_new
    return state


def finalize_accumulated_moments(state: dict, eps: float = 1e-6) -> dict:
    """Convert an ``accumulate_moments`` state into ``(mu, cov, second_moment)``."""
    n = int(state["n"])
    if n == 0:
        raise RuntimeError("Cannot finalize empty moment state.")
    sum_ = state["sum"].to(torch.float64)
    sm_sum = state["second_moment_sum"].to(torch.float64)
    mu = sum_ / float(n)
    second_moment = sm_sum / float(n)
    second_moment = 0.5 * (second_moment + second_moment.transpose(0, 1))
    cov = second_moment - mu.unsqueeze(1) @ mu.unsqueeze(0)
    cov = 0.5 * (cov + cov.transpose(0, 1))
    eye = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    cov = cov + eps * eye
    return {
        "n": n,
        "mu": mu.to(torch.float32),
        "cov": cov.to(torch.float32),
        "second_moment": second_moment.to(torch.float32),
        "feature_dim": int(mu.numel()),
    }
