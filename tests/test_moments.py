"""Tests for FDSpeech moment utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from srfd.moments import (  # noqa: E402
    accumulate_moments,
    batch_mean_and_second_moment,
    covariance_from_mean_and_second_moment,
    finalize_accumulated_moments,
    masked_time_mean_std,
)


def test_batch_mean_and_second_moment_shapes():
    x = torch.randn(8, 16)
    mu, M = batch_mean_and_second_moment(x)
    assert mu.shape == (16,)
    assert M.shape == (16, 16)
    assert torch.allclose(M, M.transpose(0, 1))
    assert torch.isfinite(mu).all()
    assert torch.isfinite(M).all()


def test_batch_mean_matches_torch_mean():
    x = torch.randn(64, 8, dtype=torch.float64).float()
    mu, _ = batch_mean_and_second_moment(x)
    assert torch.allclose(mu, x.mean(dim=0), atol=1e-5)


def test_covariance_is_symmetric_and_psd():
    x = torch.randn(128, 12)
    mu, M = batch_mean_and_second_moment(x)
    cov = covariance_from_mean_and_second_moment(mu, M)
    assert torch.allclose(cov, cov.transpose(0, 1), atol=1e-6)
    eigs = torch.linalg.eigvalsh(cov)
    assert eigs.min().item() > -1e-5  # tiny numerical slack
    assert torch.isfinite(cov).all()


def test_masked_time_mean_std_bdl():
    z = torch.randn(2, 4, 16)
    out = masked_time_mean_std(z)
    assert out.shape == (2, 8)
    expected_mu = z.mean(dim=2)
    assert torch.allclose(out[:, :4], expected_mu, atol=1e-5)


def test_masked_time_mean_std_bld_with_mask():
    z = torch.randn(2, 16, 4)  # [B, L, D]
    mask = torch.zeros(2, 16)
    mask[:, :8] = 1
    out = masked_time_mean_std(z, mask=mask)
    assert out.shape == (2, 8)
    expected_mu = z[:, :8, :].mean(dim=1)
    assert torch.allclose(out[:, :4], expected_mu, atol=1e-4)


def test_masked_time_mean_std_no_nan_with_zero_mask():
    z = torch.randn(1, 4, 8)
    mask = torch.zeros(1, 8)  # all-zero mask
    out = masked_time_mean_std(z, mask=mask)
    assert torch.isfinite(out).all()


def test_accumulate_and_finalize():
    state: dict = {}
    torch.manual_seed(0)
    chunks = [torch.randn(32, 6) for _ in range(5)]
    for c in chunks:
        accumulate_moments(state, c)
    finalized = finalize_accumulated_moments(state)
    assert finalized["feature_dim"] == 6
    assert finalized["mu"].shape == (6,)
    assert finalized["cov"].shape == (6, 6)
    all_x = torch.cat(chunks, dim=0)
    assert torch.allclose(finalized["mu"], all_x.mean(dim=0), atol=1e-4)


def test_batch_mean_rejects_bad_shape():
    with pytest.raises(ValueError):
        batch_mean_and_second_moment(torch.randn(4))
    with pytest.raises(ValueError):
        batch_mean_and_second_moment(torch.empty(0, 4))
