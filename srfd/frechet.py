"""Differentiable Fréchet distance between two Gaussian moments.

Uses an eigendecomposition rather than scipy's ``sqrtm`` so the computation is
fully differentiable on GPU and stays in PyTorch.
"""

from __future__ import annotations

import torch


def _symmetrize(m: torch.Tensor) -> torch.Tensor:
    return 0.5 * (m + m.transpose(-1, -2))


def trace_sqrt_product_symmetric(
    cov_r: torch.Tensor,
    cov_g: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable ``Tr( (cov_r^{1/2} cov_g cov_r^{1/2})^{1/2} )``.

    All matrices are symmetrized and regularized with ``eps * I`` to avoid
    negative eigenvalues from numerical error.
    """
    if cov_r.shape != cov_g.shape:
        raise ValueError(f"Covariance shape mismatch: {tuple(cov_r.shape)} vs {tuple(cov_g.shape)}")
    if cov_r.dim() != 2 or cov_r.size(0) != cov_r.size(1):
        raise ValueError(f"Covariances must be square 2-D, got {tuple(cov_r.shape)}")

    cov_r = _symmetrize(cov_r.to(torch.float32))
    cov_g = _symmetrize(cov_g.to(torch.float32))
    C = cov_r.size(0)
    eye = torch.eye(C, device=cov_r.device, dtype=cov_r.dtype)
    cov_r = cov_r + eps * eye
    cov_g = cov_g + eps * eye

    eigvals_r, eigvecs_r = torch.linalg.eigh(cov_r)
    sqrt_eigvals_r = eigvals_r.clamp_min(eps).sqrt()
    sqrt_cov_r = eigvecs_r @ torch.diag(sqrt_eigvals_r) @ eigvecs_r.transpose(-1, -2)
    sqrt_cov_r = _symmetrize(sqrt_cov_r)

    middle = sqrt_cov_r @ cov_g @ sqrt_cov_r
    middle = _symmetrize(middle)
    middle_eigvals = torch.linalg.eigvalsh(middle)
    trace_sqrt = middle_eigvals.clamp_min(eps).sqrt().sum()
    return trace_sqrt


def frechet_distance(
    mu_r: torch.Tensor,
    cov_r: torch.Tensor,
    mu_g: torch.Tensor,
    cov_g: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable Fréchet distance between two Gaussians.

    ``FD = ||mu_r - mu_g||^2 + Tr(cov_r) + Tr(cov_g) - 2 Tr( (cov_r^{1/2} cov_g cov_r^{1/2})^{1/2} )``

    Args:
        mu_r, mu_g: 1-D tensors of equal length.
        cov_r, cov_g: square 2-D tensors with matching shape.
        eps: regularizer.

    Returns:
        Scalar tensor (>= 0). Floats are upcast to fp32 internally.
    """
    if mu_r.shape != mu_g.shape:
        raise ValueError(f"mu shape mismatch: {tuple(mu_r.shape)} vs {tuple(mu_g.shape)}")
    if mu_r.dim() != 1:
        raise ValueError(f"mu must be 1-D, got {tuple(mu_r.shape)}")

    mu_r = mu_r.to(torch.float32)
    mu_g = mu_g.to(torch.float32)
    cov_r = cov_r.to(torch.float32)
    cov_g = cov_g.to(torch.float32)

    diff = mu_r - mu_g
    mu_term = diff.dot(diff)

    trace_sqrt = trace_sqrt_product_symmetric(cov_r, cov_g, eps=eps)

    fd = mu_term + torch.trace(cov_r) + torch.trace(cov_g) - 2.0 * trace_sqrt
    return fd.clamp_min(0.0)
