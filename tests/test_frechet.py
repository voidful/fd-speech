"""Tests for the differentiable Fréchet distance utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from srfd.frechet import frechet_distance, trace_sqrt_product_symmetric  # noqa: E402


def _random_psd(d: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    A = torch.randn(d, d)
    return A @ A.t() + 0.1 * torch.eye(d)


def test_fd_zero_for_identical_stats():
    d = 8
    mu = torch.randn(d)
    cov = _random_psd(d, seed=0)
    fd = frechet_distance(mu, cov, mu, cov)
    assert fd.item() < 1e-3


def test_fd_nonneg():
    mu_r = torch.randn(8)
    mu_g = torch.randn(8)
    cov_r = _random_psd(8, seed=1)
    cov_g = _random_psd(8, seed=2)
    fd = frechet_distance(mu_r, cov_r, mu_g, cov_g)
    assert fd.item() >= 0.0


def test_fd_mean_shift_matches_squared_distance():
    # With equal covariances, FD reduces to ||mu_r - mu_g||^2.
    d = 4
    cov = torch.eye(d)
    mu = torch.zeros(d)
    fd = frechet_distance(mu, cov, mu + 1.0, cov)
    assert abs(fd.item() - float(d)) < 1e-3


def test_fd_is_symmetric():
    mu_r = torch.randn(6)
    mu_g = torch.randn(6)
    cov_r = _random_psd(6, seed=3)
    cov_g = _random_psd(6, seed=4)
    fd_rg = frechet_distance(mu_r, cov_r, mu_g, cov_g)
    fd_gr = frechet_distance(mu_g, cov_g, mu_r, cov_r)
    assert torch.allclose(fd_rg, fd_gr, atol=1e-3)


def test_fd_grad_flows_to_generated():
    d = 5
    mu_r = torch.randn(d)
    cov_r = _random_psd(d, seed=5)

    mu_g = torch.randn(d, requires_grad=True)
    A = torch.randn(d, d, requires_grad=True)
    cov_g = A @ A.t() + 0.1 * torch.eye(d)

    fd = frechet_distance(mu_r, cov_r, mu_g, cov_g)
    fd.backward()
    assert mu_g.grad is not None
    assert torch.isfinite(mu_g.grad).all()
    assert A.grad is not None
    assert torch.isfinite(A.grad).all()


def test_trace_sqrt_product_handles_near_singular():
    d = 4
    mu = torch.zeros(d)
    cov_r = torch.eye(d) * 1e-12
    cov_g = torch.eye(d)
    fd = frechet_distance(mu, cov_r, mu, cov_g)
    assert torch.isfinite(fd).all()
    assert fd.item() >= 0.0


def test_trace_sqrt_symmetric():
    d = 6
    cov_r = _random_psd(d, seed=10)
    cov_g = _random_psd(d, seed=11)
    t = trace_sqrt_product_symmetric(cov_r, cov_g)
    assert t.item() >= 0.0
