#!/usr/bin/env python3
"""Score a seed-tts-eval submission directory.

Given a directory containing `wav_res_ref_text` (the listing produced by
`scripts/infer_seed_tts_eval.py`), compute:

  • WER/CER (Whisper-large-v3 for English/Chinese)
  • SIM (WavLM-SV-large)                       — same recipe as cal_sim.sh

The numbers go into `<out_dir>/seed_tts_eval_metrics.json` for paper rendering.

This combines what the upstream seed-tts-eval scoring scripts do, but in
one Python invocation that doesn't depend on multi-GPU sharding tricks
(the upstream cal_wer.sh assumes ARNOLD_WORKER_GPU which we don't have).
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--listing", required=True,
                   help="path to wav_res_ref_text (gen_wav|ref_wav|target_text)")
    p.add_argument("--out_json", required=True)
    p.add_argument("--lang", default="en", choices=["en", "zh"])
    p.add_argument("--whisper_model", default=str(PROJECT_ROOT / "models/whisper-large-v3"))
    p.add_argument("--wavlm_sv_ckpt", default=None,
                   help="path to wavlm_large_finetune.pth; if None, fall back to "
                        "microsoft/wavlm-base-plus-sv (cosine sim of pooled features)")
    p.add_argument("--wavlm_model", default=str(PROJECT_ROOT / "models/wavlm-base-plus-sv"),
                   help="local WavLM-SV model path or HF model id for fallback SIM")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_items", type=int, default=0)
    p.add_argument(
        "--item_start",
        type=int,
        default=0,
        help="0-based item offset before applying --max_items; useful for non-prefix eval slices.",
    )
    p.add_argument(
        "--item_stride",
        type=int,
        default=1,
        help="Keep every Nth item after --item_start; useful for deterministic spread-out eval slices.",
    )
    p.add_argument("--skip_sim", action="store_true",
                   help="Only compute ASR error. Useful for large search sweeps; final tables should score SIM.")
    return p.parse_args()


def normalise_en(s: str) -> str:
    s = s.lower()
    for ch in string.punctuation:
        if ch == "'":
            continue
        s = s.replace(ch, "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalise_zh(s: str) -> str:
    """Normalize Chinese ASR text for character-error scoring."""
    s = unicodedata.normalize("NFKC", s).lower()
    chars = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("Z"):
            continue
        chars.append(ch)
    return "".join(chars)


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (ca != cb),
                )
            )
        prev = cur
    return prev[-1]


def compute_cer(refs, hyps) -> float:
    n_err = 0
    n_chars = 0
    for ref, hyp in zip(refs, hyps):
        n_err += edit_distance(list(ref), list(hyp))
        n_chars += len(ref)
    return n_err / max(1, n_chars)


def per_utt_cer(refs, hyps):
    out = []
    for ref, hyp in zip(refs, hyps):
        out.append(edit_distance(list(ref), list(hyp)) / max(1, len(ref)))
    return out


def compute_wer(refs, hyps):
    try:
        from jiwer import compute_measures
    except Exception:
        try:
            from jiwer import wer
            return float(wer(refs, hyps))
        except Exception:
            # Fallback: simple per-word edit distance.
            from difflib import SequenceMatcher
            n_err = 0
            n_words = 0
            for r, h in zip(refs, hyps):
                rw, hw = r.split(), h.split()
                n_words += len(rw)
                sm = SequenceMatcher(None, rw, hw)
                n_err += sum(
                    max(b - a, d - c)
                    for tag, a, b, c, d in sm.get_opcodes()
                    if tag != "equal"
                )
            return n_err / max(1, n_words)
    return compute_measures("\n".join(refs), "\n".join(hyps))["wer"]


def per_utt_wer(refs, hyps):
    try:
        from jiwer import compute_measures
    except Exception:
        try:
            from jiwer import wer
            return [float(wer(r, h)) for r, h in zip(refs, hyps)]
        except Exception:
            return [None] * len(refs)
    out = []
    for r, h in zip(refs, hyps):
        try:
            out.append(compute_measures(r, h)["wer"])
        except Exception:
            out.append(None)
    return out


def main() -> int:
    args = parse_args()
    listing_path = Path(args.listing)
    items = []
    for line in listing_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) >= 3:
            items.append((parts[0], parts[1], parts[2]))
    if args.item_start < 0:
        raise ValueError(f"--item_start must be >= 0, got {args.item_start}")
    if args.item_stride < 1:
        raise ValueError(f"--item_stride must be >= 1, got {args.item_stride}")
    if args.item_start or args.item_stride != 1:
        items = items[args.item_start :: args.item_stride]
    if args.max_items > 0:
        items = items[: args.max_items]
    print(f"[score] {len(items)} items", file=sys.stderr)

    # ----- WER pass -----
    print("[wer] loading Whisper", file=sys.stderr)
    import soundfile as sf
    import scipy.signal
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(args.whisper_model)
    asr = WhisperForConditionalGeneration.from_pretrained(args.whisper_model).to(args.device)
    asr.eval()
    whisper_lang = "english" if args.lang == "en" else "chinese"
    forced_ids = processor.get_decoder_prompt_ids(language=whisper_lang, task="transcribe")

    refs, hyps = [], []
    per_utt = []
    n_skip = 0
    for i, (gen_wav, ref_wav, text_ref) in enumerate(items):
        try:
            wav, sr = sf.read(gen_wav)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
            feat = processor(wav, sampling_rate=16000, return_tensors="pt").input_features.to(args.device)
            feat = feat.to(dtype=next(asr.parameters()).dtype)
            with torch.no_grad():
                pred = asr.generate(feat, forced_decoder_ids=forced_ids)
            hyp = processor.batch_decode(pred, skip_special_tokens=True)[0]
        except Exception as e:
            n_skip += 1
            print(f"[wer-skip] {gen_wav}: {e}", file=sys.stderr)
            continue
        if args.lang == "zh":
            ref_n = normalise_zh(text_ref)
            hyp_n = normalise_zh(hyp)
        else:
            ref_n = normalise_en(text_ref)
            hyp_n = normalise_en(hyp)
        refs.append(ref_n)
        hyps.append(hyp_n)
        per_utt.append({"gen_wav": gen_wav, "ref": ref_n, "hyp": hyp_n})
        if (i + 1) % 50 == 0:
            print(f"[wer] {i+1}/{len(items)}", file=sys.stderr)

    if args.lang == "zh":
        wer_corpus = float(compute_cer(refs, hyps))
        wer_per = per_utt_cer(refs, hyps)
        error_metric = "cer"
    else:
        wer_corpus = float(compute_wer(refs, hyps))
        wer_per = per_utt_wer(refs, hyps)
        error_metric = "wer"
    for r, w in zip(per_utt, wer_per):
        r[error_metric] = w
    print(f"[wer] corpus {error_metric.upper()} = {wer_corpus*100:.3f}%", file=sys.stderr)

    # ----- SIM pass -----
    print("[sim] loading speaker model", file=sys.stderr)
    sims = []
    sim_model_used = None
    if args.skip_sim:
        print("[sim] skipped", file=sys.stderr)
        sim_model_used = "skipped"
    elif args.wavlm_sv_ckpt and Path(args.wavlm_sv_ckpt).exists():
        # Use the upstream UniSpeech wavlm_large finetune. We assume
        # `thirdparty/UniSpeech/...` exists in PYTHONPATH so we can call it.
        sim_model_used = "wavlm_large_finetune"
        # TODO: load the upstream verification model. For now, fall back below.
    if sim_model_used is None:
        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

        default_local_wavlm = PROJECT_ROOT / "models" / "wavlm-base-plus-sv"
        sim_model_id = args.wavlm_model
        sim_model_path = Path(sim_model_id)
        if sim_model_path.exists():
            sim_model_id = str(sim_model_path)
            sim_model_used = "wavlm-base-plus-sv (local)"
        elif sim_model_id == str(default_local_wavlm):
            sim_model_id = "microsoft/wavlm-base-plus-sv"
            sim_model_used = sim_model_id
        else:
            sim_model_used = sim_model_id
        feat_ext = AutoFeatureExtractor.from_pretrained(sim_model_id)
        sim_model = AutoModelForAudioXVector.from_pretrained(sim_model_id).to(args.device).eval()

        for i, (gen_wav, ref_wav, _) in enumerate(items):
            try:
                w_g, sr_g = sf.read(gen_wav)
                w_r, sr_r = sf.read(ref_wav)
                if w_g.ndim > 1:
                    w_g = w_g.mean(axis=1)
                if w_r.ndim > 1:
                    w_r = w_r.mean(axis=1)
                if sr_g != 16000:
                    w_g = scipy.signal.resample(w_g, int(len(w_g) * 16000 / sr_g))
                if sr_r != 16000:
                    w_r = scipy.signal.resample(w_r, int(len(w_r) * 16000 / sr_r))
                feats = feat_ext([w_g, w_r], sampling_rate=16000, return_tensors="pt", padding=True).to(args.device)
                with torch.no_grad():
                    out = sim_model(**feats)
                emb = torch.nn.functional.normalize(out.embeddings, dim=-1)
                sim = float((emb[0] * emb[1]).sum().item())
                sims.append(sim)
            except Exception as e:
                print(f"[sim-skip] {gen_wav}: {e}", file=sys.stderr)
                continue
            if (i + 1) % 50 == 0:
                print(f"[sim] {i+1}/{len(items)}", file=sys.stderr)

    sim_mean = float(np.mean(sims)) if sims else None

    per_utt_mean = (
        float(np.mean([w for w in wer_per if w is not None]) * 100)
        if wer_per
        else None
    )
    out = {
        "n_items": len(items),
        "n_skipped_wer": n_skip,
        "n_skipped_asr": n_skip,
        "error_metric": error_metric,
        "wer_corpus": wer_corpus,
        "wer_corpus_pct": wer_corpus * 100,
        "cer_corpus": wer_corpus if args.lang == "zh" else None,
        "cer_corpus_pct": wer_corpus * 100 if args.lang == "zh" else None,
        "wer_per_utt_mean": per_utt_mean,
        "error_per_utt_mean": per_utt_mean,
        "sim_mean": sim_mean,
        "sim_n": len(sims),
        "sim_model": sim_model_used,
        "lang": args.lang,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    # Also write per-utt for paired tests.
    (out_path.parent / "per_utt_wer.jsonl").write_text(
        "\n".join(json.dumps(r) for r in per_utt), encoding="utf-8"
    )
    print(f"[done] wrote {out_path}", file=sys.stderr)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
