"""Save/load reference statistics for SR-FD.

The on-disk format is a torch ``.pt`` file containing a single dict::

    {
        "version": 1,
        "reps": {
            "<extractor_name>": {
                "n": int,
                "mu": Tensor[C],
                "cov": Tensor[C, C],
                "second_moment": Tensor[C, C],
                "feature_dim": int,
            },
            ...
        },
        "metadata": { ... },
    }

This is small (hundreds of KB even for D=128, C=256) and easy to inspect.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

STATS_SCHEMA_VERSION = 1


def _validate_rep_entry(name: str, entry: Dict[str, Any]) -> None:
    required = {"n", "mu", "cov", "feature_dim"}
    missing = required - set(entry.keys())
    if missing:
        raise ValueError(f"Stats entry '{name}' missing keys: {sorted(missing)}")
    mu = entry["mu"]
    cov = entry["cov"]
    if not isinstance(mu, torch.Tensor) or mu.dim() != 1:
        raise ValueError(f"Stats entry '{name}': mu must be 1-D Tensor")
    if not isinstance(cov, torch.Tensor) or cov.dim() != 2 or cov.size(0) != cov.size(1):
        raise ValueError(f"Stats entry '{name}': cov must be square 2-D Tensor")
    if mu.size(0) != cov.size(0):
        raise ValueError(
            f"Stats entry '{name}': mu/cov shape mismatch ({mu.size(0)} vs {cov.size(0)})"
        )
    if int(entry["feature_dim"]) != mu.size(0):
        raise ValueError(
            f"Stats entry '{name}': feature_dim {entry['feature_dim']} != mu length {mu.size(0)}"
        )
    cond = entry.get("cond", None)
    if cond is not None:
        if not isinstance(cond, dict):
            raise ValueError(f"Stats entry '{name}': cond must be a dict")
        for cond_key, cond_entry in cond.items():
            _validate_rep_entry(f"{name}.cond[{cond_key}]", cond_entry)


def save_stats(path: str, stats: Dict[str, Any]) -> None:
    """Save reference statistics to ``path``."""
    if "reps" not in stats:
        raise ValueError("save_stats: stats dict must contain 'reps'")
    for name, entry in stats["reps"].items():
        _validate_rep_entry(name, entry)
    payload = {
        "version": STATS_SCHEMA_VERSION,
        "reps": {
            name: {
                "n": int(entry["n"]),
                "mu": entry["mu"].detach().to(torch.float32).cpu(),
                "cov": entry["cov"].detach().to(torch.float32).cpu(),
                "second_moment": entry["second_moment"].detach().to(torch.float32).cpu()
                if "second_moment" in entry
                else None,
                "feature_dim": int(entry["feature_dim"]),
                "cond": {
                    cond_key: {
                        "n": int(cond_entry["n"]),
                        "mu": cond_entry["mu"].detach().to(torch.float32).cpu(),
                        "cov": cond_entry["cov"].detach().to(torch.float32).cpu(),
                        "second_moment": cond_entry["second_moment"].detach().to(torch.float32).cpu()
                        if "second_moment" in cond_entry
                        else None,
                        "feature_dim": int(cond_entry["feature_dim"]),
                    }
                    for cond_key, cond_entry in entry.get("cond", {}).items()
                },
            }
            for name, entry in stats["reps"].items()
        },
        "metadata": dict(stats.get("metadata", {})),
    }
    torch.save(payload, path)


def load_stats(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Load and validate reference statistics from ``path``."""
    raw = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(raw, dict) or "reps" not in raw:
        raise ValueError(f"Stats file {path} has invalid schema (missing 'reps').")
    version = int(raw.get("version", 0))
    if version != STATS_SCHEMA_VERSION:
        raise ValueError(
            f"Stats file {path}: schema version {version} != expected {STATS_SCHEMA_VERSION}"
        )
    for name, entry in raw["reps"].items():
        _validate_rep_entry(name, entry)
    return raw
