#!/usr/bin/env python3
"""Utterance-level paired bootstrap on per-utterance error.

Chains from ``score_seed_tts_eval.py``, which writes ``per_utt_wer.jsonl``
(one JSON object per utterance with a ``gen_wav`` key and a ``wer`` / ``cer``
field). Given two such files for two systems evaluated on the *same* prompts,
this reports:

  * mean paired difference (system B − system A)
  * a bootstrap 95% confidence interval over the per-utterance differences
  * a Wilcoxon signed-rank p-value (no normality assumption)

Utterances are aligned by an id derived from the ``gen_wav`` filename stem so
the two systems' rows match even if their output directories differ.

Example::

    python scripts/paired_bootstrap.py \
        --a runs/base4/per_utt_wer.jsonl \
        --b runs/srfd/per_utt_wer.jsonl \
        --metric wer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="per_utt JSONL for system A (e.g. baseline).")
    p.add_argument("--b", required=True, help="per_utt JSONL for system B (e.g. FDSpeech).")
    p.add_argument("--metric", default="wer", help="Field name to compare (wer or cer).")
    p.add_argument("--id_field", default="gen_wav", help="Field used to align utterances.")
    p.add_argument("--n_bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    return p.parse_args()


def _load(path: str, metric: str, id_field: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if metric not in row or row[metric] is None:
            continue
        key = Path(str(row[id_field])).stem
        out[key] = float(row[metric])
    return out


def bootstrap_ci(diffs: np.ndarray, n: int, seed: int, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, len(diffs), size=len(diffs))
        means[i] = diffs[idx].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def wilcoxon_p(diffs: np.ndarray) -> float:
    nz = diffs[diffs != 0]
    if nz.size < 2:
        return float("nan")
    try:
        from scipy import stats
    except Exception:
        return float("nan")
    return float(stats.wilcoxon(nz, alternative="two-sided", zero_method="zsplit").pvalue)


def main() -> int:
    args = parse_args()
    a = _load(args.a, args.metric, args.id_field)
    b = _load(args.b, args.metric, args.id_field)
    common: List[str] = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("No overlapping utterance ids between the two files.")

    m_a = np.array([a[k] for k in common])
    m_b = np.array([b[k] for k in common])
    diffs = m_b - m_a  # negative -> system B better

    lo, hi = bootstrap_ci(diffs, n=args.n_bootstrap, seed=args.seed)
    pval = wilcoxon_p(diffs)
    result = {
        "n_prompts": len(common),
        "metric": args.metric,
        "a_mean": float(m_a.mean()),
        "b_mean": float(m_b.mean()),
        "diff_mean": float(diffs.mean()),
        "diff_median": float(np.median(diffs)),
        "bootstrap_ci_95_lo": lo,
        "bootstrap_ci_95_hi": hi,
        "wilcoxon_pvalue": pval,
        "n_better_for_b": int((diffs < 0).sum()),
        "n_worse_for_b": int((diffs > 0).sum()),
        "n_tie": int((diffs == 0).sum()),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
