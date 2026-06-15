#!/usr/bin/env python3
"""Score generated speech with speechmos DNSMOS.

This script intentionally keeps DNSMOS as an auxiliary non-intrusive proxy.
It requires the optional `speechmos` package and is not part of the default
test environment. Install it in a disposable target/venv before running.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--listing", help="wav_res_ref_text path")
    source.add_argument("--wav_dir", help="Directory containing generated wav files")
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--max_items", type=int, default=0)
    return parser.parse_args()


def load_listing(path: Path, max_items: int = 0) -> list[Path]:
    wavs: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        wavs.append(Path(line.split("|", 1)[0]))
    if max_items > 0:
        wavs = wavs[:max_items]
    return wavs


def load_wav_dir(path: Path, max_items: int = 0) -> list[Path]:
    wavs = sorted(path.glob("*.wav"))
    if max_items > 0:
        wavs = wavs[:max_items]
    return wavs


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) <= 1:
        return 0.0 if values else None
    return statistics.stdev(values)


def _as_float(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def main() -> int:
    args = parse_args()
    if args.listing:
        wavs = load_listing(Path(args.listing), args.max_items)
    else:
        wavs = load_wav_dir(Path(args.wav_dir), args.max_items)
    print(f"[dnsmos] {len(wavs)} wavs", file=sys.stderr)

    try:
        import librosa
        from speechmos import dnsmos
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise SystemExit(
            "speechmos DNSMOS dependencies are missing. Install in a disposable "
            "environment, for example: pip install --target /tmp/speechmos_target "
            "speechmos==0.0.1.1 librosa==0.9.1 onnxruntime pandas tqdm"
        ) from exc

    per_utt: list[dict[str, Any]] = []
    n_skip = 0
    for idx, wav in enumerate(wavs, start=1):
        try:
            audio, _ = librosa.load(str(wav), sr=16_000, mono=True)
            # speechmos rejects samples outside [-1, 1]. Some generated wavs
            # contain small overshoots, so clamp at metric input time instead
            # of dropping the utterance.
            audio = audio.clip(-1.0, 1.0)
            scores = dnsmos.run(audio, 16_000, return_df=False, verbose=False)
            row = {
                "wav": str(wav),
                "ovrl_mos": _as_float(scores["ovrl_mos"]),
                "sig_mos": _as_float(scores["sig_mos"]),
                "bak_mos": _as_float(scores["bak_mos"]),
                "p808_mos": _as_float(scores["p808_mos"]),
            }
            per_utt.append(row)
        except Exception as exc:  # pragma: no cover - data-dependent path
            n_skip += 1
            print(f"[dnsmos-skip] {wav}: {exc}", file=sys.stderr)
        if idx % 50 == 0 or idx == len(wavs):
            print(f"[dnsmos] {idx}/{len(wavs)}", file=sys.stderr)

    values = {
        key: [float(row[key]) for row in per_utt]
        for key in ("ovrl_mos", "sig_mos", "bak_mos", "p808_mos")
    }
    out = {
        "n_items": len(wavs),
        "n_scored": len(per_utt),
        "n_skipped": n_skip,
        "metric": "speechmos_dnsmos",
        "sample_rate": 16_000,
        "max_items": args.max_items,
        "ovrl_mos_mean": _mean(values["ovrl_mos"]),
        "ovrl_mos_std": _std(values["ovrl_mos"]),
        "sig_mos_mean": _mean(values["sig_mos"]),
        "sig_mos_std": _std(values["sig_mos"]),
        "bak_mos_mean": _mean(values["bak_mos"]),
        "bak_mos_std": _std(values["bak_mos"]),
        "p808_mos_mean": _mean(values["p808_mos"]),
        "p808_mos_std": _std(values["p808_mos"]),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_path.parent / f"{out_path.stem}_per_utt_dnsmos_objective.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in per_utt) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
