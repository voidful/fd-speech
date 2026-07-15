"""Tests for ``SRFDEmaLoss``.

These use a tiny in-memory dummy extractor so the loss math (queue / EMA
moment estimation, multi-target Fréchet aggregation, normalization, warmup
gating, gradient flow) is exercised without downloading Whisper or wav2vec2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from srfd import SRFDEmaLoss  # noqa: E402
from srfd.extractors import BaseSRFDExtractor, SRFDExtractorConfig  # noqa: E402
from srfd.moments import accumulate_moments, finalize_accumulated_moments  # noqa: E402


class _FeatExtractor(BaseSRFDExtractor):
    """Returns ``batch['feat']`` directly as the [B, C] representation."""

    def __init__(self, name: str, dim: int):
        super().__init__(SRFDExtractorConfig(name=name, type="dummy"))
        self._dim = dim

    def feature_dim(self):
        return self._dim

    def forward(self, batch):
        return batch["feat"]


def _reference_stats(name: str, samples: torch.Tensor) -> dict:
    state: dict = {}
    accumulate_moments(state, samples)
    finalized = finalize_accumulated_moments(state)
    return {"version": 1, "reps": {name: finalized}}


def test_warmup_returns_zero_and_does_not_update():
    torch.manual_seed(0)
    ext = _FeatExtractor("dummy", 4)
    stats = _reference_stats("dummy", torch.randn(500, 4) + 2.0)
    loss_fn = SRFDEmaLoss(extractors=[ext], real_stats=stats, warmup_steps=10, stats_mode="queue")
    out = loss_fn({"feat": torch.randn(16, 4)}, step=0)
    assert float(out["loss/srfd"]) == 0.0


def test_queue_loss_reduces_fd_and_moves_generator():
    torch.manual_seed(0)
    ext = _FeatExtractor("dummy", 6)
    target_mean = 3.0
    real = torch.randn(3000, 6) * 1.2 + target_mean
    stats = _reference_stats("dummy", real)
    loss_fn = SRFDEmaLoss(
        extractors=[ext],
        real_stats=stats,
        stats_mode="queue",
        queue_size=2048,
        queue_warmup_size=64,
        normalize=True,
        warmup_steps=0,
    )
    # Affine generator z @ W^T + b parameterizes a Gaussian, so FDSpeech can drive
    # both its mean (b) and covariance (W W^T) toward the reference.
    W = torch.eye(6) * 0.3 + 0.01 * torch.randn(6, 6)
    W.requires_grad_(True)
    b = torch.zeros(6, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.05)
    first_fd = last_fd = None
    for step in range(400):
        feat = torch.randn(32, 6) @ W.t() + b
        out = loss_fn({"feat": feat}, step=step)
        fd = out.get("srfd/fd_dummy")
        if fd is not None:
            if first_fd is None:
                first_fd = float(fd)
            last_fd = float(fd)
        loss = out["loss/srfd"]
        if loss.requires_grad:
            opt.zero_grad()
            loss.backward()
            opt.step()
    assert first_fd is not None and last_fd is not None
    assert last_fd < 0.5 * first_fd  # FD clearly reduced
    assert float(b.detach().mean()) > 1.5  # mean moved from 0 toward target (3.0)


def test_multi_target_aggregation_runs():
    torch.manual_seed(1)
    ext = _FeatExtractor("dummy", 5)
    t1 = finalize_accumulated_moments(_accum(torch.randn(800, 5) + 1.0))
    t2 = finalize_accumulated_moments(_accum(torch.randn(800, 5) - 1.0))
    real_stats = [
        {"name": "a", "weight": 1.0, "stats": {"version": 1, "reps": {"dummy": t1}}},
        {"name": "b", "weight": 0.5, "stats": {"version": 1, "reps": {"dummy": t2}}},
    ]
    loss_fn = SRFDEmaLoss(
        extractors=[ext], real_stats=real_stats, stats_mode="queue",
        queue_size=1024, queue_warmup_size=32, warmup_steps=0,
    )
    out = loss_fn({"feat": torch.randn(64, 5)}, step=0)
    # Both per-target Fréchet logs should be present.
    assert "srfd/fd_dummy_target_target0_a" in out
    assert "srfd/fd_dummy_target_target1_b" in out
    assert torch.isfinite(out["loss/srfd"]).all()


def _accum(x: torch.Tensor) -> dict:
    state: dict = {}
    accumulate_moments(state, x)
    return state
