#!/usr/bin/env python3
"""Score generated speech with UTMOS.

This is an auxiliary non-intrusive objective proxy. It uses the public
``tarepan/SpeechMOS`` torch.hub model, writes per-utterance JSONL, and keeps the
metric separate from human MOS/CMOS evidence.
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
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--ext", default="wav")
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values)


def _wav_paths(audio_dir: Path, ext: str, max_items: int) -> list[Path]:
    wavs = sorted(audio_dir.rglob(f"*.{ext}"))
    if max_items > 0:
        wavs = wavs[:max_items]
    return wavs


def main() -> int:
    args = parse_args()
    wavs = _wav_paths(args.audio_dir, args.ext, args.max_items)
    print(f"[utmos] {len(wavs)} wavs from {args.audio_dir}", file=sys.stderr)

    try:
        import librosa
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise SystemExit("UTMOS dependencies are missing: install torch and librosa.") from exc

    device = args.device
    if not device:
        device = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu"

    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    predictor = predictor.to(device).eval()

    rows: list[dict[str, Any]] = []
    n_skip = 0
    with torch.no_grad():
        for idx, wav in enumerate(wavs, start=1):
            try:
                audio, sr = librosa.load(str(wav), sr=None, mono=True)
                wav_tensor = torch.from_numpy(audio).to(device).unsqueeze(0)
                score = predictor(wav_tensor, sr)
                score_float = float(score.item() if hasattr(score, "item") else score)
                rows.append(
                    {
                        "wav": str(wav),
                        "item_id": wav.stem,
                        "utmos": score_float,
                    }
                )
            except Exception as exc:  # pragma: no cover - data-dependent path
                n_skip += 1
                print(f"[utmos-skip] {wav}: {exc}", file=sys.stderr)
            if idx % 25 == 0 or idx == len(wavs):
                print(f"[utmos] {idx}/{len(wavs)}", file=sys.stderr)

    values = [float(row["utmos"]) for row in rows]
    out = {
        "audio_dir": str(args.audio_dir),
        "device": device,
        "max_items": args.max_items,
        "metric": "utmos22_strong",
        "model": "tarepan/SpeechMOS:v1.2.0 utmos22_strong",
        "n_items": len(wavs),
        "n_scored": len(rows),
        "n_skipped": n_skip,
        "utmos_mean": _mean(values),
        "utmos_std": _std(values),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    per_utt_path = args.out_json.parent / f"{args.out_json.stem}_per_utt_utmos_objective.jsonl"
    per_utt_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
