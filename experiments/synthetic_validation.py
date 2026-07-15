#!/usr/bin/env python3
"""Synthetic validation of FDSpeech on a problem with known ground truth.

This is a CPU-only controlled check that the FDSpeech machinery (differentiable
Fréchet distance + queue/EMA moment estimation) actually steers a generator's
distribution toward a target distribution. It uses a tiny in-memory feature
extractor instead of Whisper/CTC, so it depends only on ``torch``.

Setup
-----
* Real distribution: a fixed Gaussian over D-dim feature vectors. We sample a
  large reference set and summarize it with the same moment code the trainer
  uses (``accumulate_moments`` + ``finalize_accumulated_moments``).
* Generator: an affine map ``z W^T + b`` (z ~ N(0, I)), initialized far from
  the target so it parameterizes a Gaussian whose mean/covariance FDSpeech moves.

Three runs are compared:

1. ``baseline``  : generator frozen at init; FD measured but never trained.
2. ``srfd``      : generator trained with FDSpeech only (queue moments).
3. ``supervised``: generator trained directly against the real mean via MSE
   (a perfect-supervision upper bound).

Success: run 2 must reduce the *true* Fréchet distance well below init and
below the frozen baseline.

Usage::

    python experiments/synthetic_validation.py --steps 1500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from srfd import SRFDEmaLoss, frechet_distance  # noqa: E402
from srfd.extractors import BaseSRFDExtractor, SRFDExtractorConfig  # noqa: E402
from srfd.moments import accumulate_moments, finalize_accumulated_moments  # noqa: E402


class FeatureExtractor(BaseSRFDExtractor):
    """Returns the batch's [B, D] ``feat`` tensor as the representation."""

    def __init__(self, dim: int):
        super().__init__(SRFDExtractorConfig(name="feat", type="dummy"))
        self._dim = dim

    def feature_dim(self):
        return self._dim

    def forward(self, batch):
        return batch["feat"]


def build_target(dim: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, dim, generator=g)
    cov = A @ A.t() / dim + 0.5 * torch.eye(dim)
    mu = torch.randn(dim, generator=g) * 0.5
    L = torch.linalg.cholesky(cov)
    return {"mu": mu, "L": L}


def sample_real(target: dict, batch: int, device) -> torch.Tensor:
    d = target["mu"].numel()
    eps = torch.randn(batch, d, device=device)
    return eps @ target["L"].to(device).t() + target["mu"].to(device)


class LinearGenerator(nn.Module):
    """Affine generator ``x = z W^T + b`` with ``z ~ N(0, I)``.

    This directly parameterizes a Gaussian (mean ``b``, covariance ``W W^T``),
    so FDSpeech has a smooth target to drive toward and the ground-truth FD is
    well-defined at every step.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Parameter(0.3 * torch.eye(dim) + 0.01 * torch.randn(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, batch: int, device) -> torch.Tensor:
        z = torch.randn(batch, self.W.size(0), device=device)
        return z @ self.W.t() + self.b


def compute_real_stats(target, n, batch, device, extractor) -> dict:
    state: dict = {}
    seen = 0
    while seen < n:
        feat = extractor({"feat": sample_real(target, batch, device)})
        accumulate_moments(state, feat)
        seen += feat.size(0)
    return finalize_accumulated_moments(state)


@torch.no_grad()
def measure_true_fd(generator, real_finalized, extractor, device, batches=64, batch=64) -> float:
    state: dict = {}
    for _ in range(batches):
        feat = extractor({"feat": generator(batch, device)})
        accumulate_moments(state, feat)
    g = finalize_accumulated_moments(state)
    return float(
        frechet_distance(
            real_finalized["mu"].to(device), real_finalized["cov"].to(device),
            g["mu"].to(device), g["cov"].to(device),
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--ref_samples", type=int, default=8192)
    ap.add_argument("--queue_size", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "experiments" / "synthetic_validation.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    extractor = FeatureExtractor(args.dim).to(device).eval()

    target = build_target(args.dim, seed=args.seed)
    real = compute_real_stats(target, args.ref_samples, args.batch_size, device, extractor)
    real_stats = {"version": 1, "reps": {"feat": real}}
    print(f"[ref] n={real['n']} dim={real['feature_dim']} ||mu||={real['mu'].norm():.3f}", file=sys.stderr)

    def fresh():
        torch.manual_seed(args.seed + 1)
        return LinearGenerator(args.dim).to(device)

    # Run 1: frozen baseline.
    gen0 = fresh()
    init_fd = measure_true_fd(gen0, real, extractor, device)
    print(f"[baseline] init/frozen FD = {init_fd:.4f}", file=sys.stderr)

    # Run 2: FDSpeech (queue moments).
    gen1 = fresh()
    loss_fn = SRFDEmaLoss(
        extractors=[FeatureExtractor(args.dim).to(device)],
        real_stats=real_stats, stats_mode="queue",
        queue_size=args.queue_size, queue_warmup_size=256,
        normalize=False, warmup_steps=10,
    ).to(device)
    opt = torch.optim.Adam(gen1.parameters(), lr=args.lr)
    srfd_fds = {}
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            srfd_fds[step] = measure_true_fd(gen1, real, extractor, device)
        if step == args.steps:
            break
        out = loss_fn({"feat": gen1(args.batch_size, device)}, step=step)
        loss = out["loss/srfd"]
        if loss.requires_grad:
            opt.zero_grad(); loss.backward(); opt.step()
    srfd_final = srfd_fds[args.steps]
    print(f"[srfd] final FD = {srfd_final:.4f}", file=sys.stderr)

    # Run 3: supervised mean-MSE upper bound.
    gen2 = fresh()
    target_mu = real["mu"].to(device)
    opt = torch.optim.Adam(gen2.parameters(), lr=args.lr)
    sup_fds = {}
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            sup_fds[step] = measure_true_fd(gen2, real, extractor, device)
        if step == args.steps:
            break
        feat = extractor({"feat": gen2(args.batch_size, device)})
        loss = F.mse_loss(feat.mean(0), target_mu)
        opt.zero_grad(); loss.backward(); opt.step()
    sup_final = sup_fds[args.steps]
    print(f"[supervised] final FD = {sup_final:.4f}", file=sys.stderr)

    reduction = (init_fd - srfd_final) / max(init_fd, 1e-9) * 100.0
    result = {
        "config": vars(args),
        "init_fd": init_fd,
        "srfd_final_fd": srfd_final,
        "supervised_final_fd": sup_final,
        "srfd_reduction_pct": reduction,
        "srfd_fds": {int(k): float(v) for k, v in srfd_fds.items()},
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== Summary ===", file=sys.stderr)
    print(f"  init FD       : {init_fd:.4f}", file=sys.stderr)
    print(f"  FDSpeech FD   : {srfd_final:.4f}  ({reduction:.1f}% reduction)", file=sys.stderr)
    print(f"  supervised FD : {sup_final:.4f}", file=sys.stderr)
    if srfd_final >= 0.5 * init_fd:
        print("  WARNING: FDSpeech did not produce a meaningful reduction.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
