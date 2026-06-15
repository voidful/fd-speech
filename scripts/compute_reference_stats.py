#!/usr/bin/env python3
"""Compute SR-FD reference statistics from an audio manifest.

This runs the *frozen* SR-FD extractors directly on reference waveforms and
stores only the first- and second-order moments (mean + covariance) of the
resulting feature vectors. The reference audio is never used again at training
time; only the stored moments are loaded.

Each of the paper's three targets is one invocation of this script:

    # 1. low-step Whisper anchor (ASR-verified four-step generations)
    python scripts/compute_reference_stats.py \
        --manifest data/ref/asr_true4_good.jsonl \
        --config configs/srfd_compact3.yaml --reps whisper_anchor8_p64 \
        --out stats/ref_whisper_anchor_asr_true4_good.pt

    # 2. teacher CTC target (ten-step teacher generations)
    python scripts/compute_reference_stats.py \
        --manifest data/ref/teacher_t10.jsonl \
        --config configs/srfd_compact3.yaml --reps ctc_content_p64 \
        --out stats/ref_ctc_content_teacher_t10.pt

    # 3. real-speech CTC target (real LibriTTS voice-cloning speech)
    python scripts/compute_reference_stats.py \
        --manifest data/ref/real_voiceclone.jsonl \
        --config configs/srfd_compact3.yaml --reps ctc_content_p64 \
        --out stats/ref_ctc_content_real_voiceclone.pt

The manifest is JSONL with one object per line containing at least an
``audio`` field (path to a wav/flac file). Any other fields are ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from srfd.extractors import build_srfd_extractors
from srfd.moments import accumulate_moments, finalize_accumulated_moments
from srfd.stats_io import save_stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute SR-FD reference statistics.")
    p.add_argument("--manifest", required=True, help="JSONL manifest with an 'audio' field.")
    p.add_argument("--config", required=True, help="YAML config; reads srfd.reps for extractors.")
    p.add_argument("--out", required=True, help="Output .pt path.")
    p.add_argument(
        "--reps",
        nargs="*",
        default=None,
        help="Subset of extractor names to compute (default: all enabled in the config).",
    )
    p.add_argument("--audio_field", default="audio", help="Manifest field with the audio path.")
    p.add_argument("--input_sample_rate", type=int, default=0,
                   help="Override input sample rate (0 = use each file's native rate).")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=0, help="0 = process all rows.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dry_run", action="store_true", help="Process 2 batches and print shapes only.")
    return p.parse_args()


def _load_reps(config_path: str, keep: Optional[List[str]]) -> List[Dict]:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    reps = (data.get("srfd", {}) or {}).get("reps", []) if isinstance(data, dict) else []
    if not reps:
        raise ValueError(f"No srfd.reps found in {config_path}")
    if keep:
        keep_set = set(keep)
        reps = [r for r in reps if r.get("name") in keep_set]
        if not reps:
            raise ValueError(f"None of {keep} matched srfd.reps names in {config_path}")
    return reps


def _read_manifest(path: str, audio_field: str, limit: int) -> List[str]:
    paths: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            audio = row[audio_field]
            if isinstance(audio, dict):  # {"path": ...} style
                audio = audio.get("path") or audio.get("array")
            paths.append(str(audio))
            if limit and len(paths) >= limit:
                break
    return paths


def _load_audio(path: str) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    # Downmix to mono: [T, C] -> [T]
    mono = torch.from_numpy(wav).mean(dim=1)
    return mono, int(sr)


def _collate(batch_paths: List[str], override_sr: int) -> Dict[str, torch.Tensor]:
    wavs: List[torch.Tensor] = []
    lengths: List[int] = []
    sr_seen: Optional[int] = None
    for p in batch_paths:
        wav, sr = _load_audio(p)
        if override_sr:
            sr = override_sr
        if sr_seen is None:
            sr_seen = sr
        elif sr_seen != sr:
            raise ValueError(
                f"Mixed sample rates in one batch ({sr_seen} vs {sr}); "
                "pre-resample the manifest or use --batch_size 1."
            )
        wavs.append(wav)
        lengths.append(wav.numel())
    max_len = max(lengths)
    B = len(wavs)
    waveform = torch.zeros(B, max_len, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, (wav, n) in enumerate(zip(wavs, lengths)):
        waveform[i, :n] = wav
        mask[i, :n] = True
    return {
        "waveform": waveform,
        "waveform_mask": mask,
        "waveform_sample_rate": int(sr_seen or 16000),
    }


def main() -> int:
    args = parse_args()

    reps_config = _load_reps(args.config, args.reps)
    extractors = build_srfd_extractors(reps_config)
    if not extractors:
        raise ValueError("No enabled extractors after filtering.")
    extractors = [e.to(args.device).eval() for e in extractors]
    print(f"Extractors: {[e.name for e in extractors]}", file=sys.stderr)

    paths = _read_manifest(args.manifest, args.audio_field, args.max_samples)
    print(f"Manifest rows: {len(paths)}", file=sys.stderr)

    accumulators: Dict[str, dict] = {ext.name: {} for ext in extractors}
    n_batches = 0
    with torch.no_grad():
        for start in range(0, len(paths), args.batch_size):
            batch_paths = paths[start : start + args.batch_size]
            batch = _collate(batch_paths, args.input_sample_rate)
            batch = {
                k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            for ext in extractors:
                rep = ext(batch).to(torch.float32)
                if rep.dim() != 2:
                    raise RuntimeError(f"{ext.name} returned {tuple(rep.shape)}; expected [B, C]")
                accumulate_moments(accumulators[ext.name], rep)
                if args.dry_run and n_batches < 2:
                    print(f"[dry_run] {ext.name}: rep {tuple(rep.shape)}", file=sys.stderr)
            n_batches += 1
            if args.dry_run and n_batches >= 2:
                print("[dry_run] stopping after 2 batches.", file=sys.stderr)
                return 0

    reps_out = {}
    for ext in extractors:
        finalized = finalize_accumulated_moments(accumulators[ext.name])
        reps_out[ext.name] = finalized
        print(
            f"[stats] {ext.name}: n={finalized['n']}, dim={finalized['feature_dim']}, "
            f"||mu||={finalized['mu'].norm().item():.4f}",
            file=sys.stderr,
        )

    metadata = {
        "manifest": str(args.manifest),
        "config": str(args.config),
        "n_samples_processed": int(min(len(paths), args.max_samples) if args.max_samples else len(paths)),
        "extractor_names": [e.name for e in extractors],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_stats(str(out_path), {"reps": reps_out, "metadata": metadata})
    print(f"Saved SR-FD reference stats to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
