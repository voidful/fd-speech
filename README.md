# SR-FD: Fréchet Distance Loss on Speech Representations for Text-to-Speech

**Speech Representation Fréchet Distance (SR-FD)** is a training-time
distributional regularizer for tokenizer-free, few-step flow-matching TTS.
Few-step diffusion / flow-matching TTS models are trained with *local*
objectives (conditional flow matching, reconstruction, stop prediction) that
never ask whether sampled speech follows the distribution of high-quality
speech. When such a sampler is compressed to four steps, its output
distribution can drift in content even while training loss looks healthy.

SR-FD closes this gap. During fine-tuning the model synthesizes speech with the
**same few-step sampler used at deployment**, frozen Whisper and CTC encoders
map that speech to features, and SR-FD matches the **mean and covariance** of
those features to reference statistics computed offline from three
complementary content targets — via a differentiable Fréchet distance (the same
quantity behind FID/FAD). The loss requires **no discriminator** and adds **no
inference-time computation**: at test time it is removed entirely.

> Base model: the external tokenizer-free flow-matching TTS model
> [`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2). SR-FD is added on
> top of LoRA fine-tuning; the deployed artifact is a four-step model plus a
> LoRA adapter.

## Headline result

On Seed-TTS English (upstream scorer, 11,805 reference words), four-step SR-FD
fine-tuning reverses the usual step-count trade-off: it beats not only the
four-step baseline but also the ten-step baseline.

| System | Steps | Upstream WER ↓ | SIM ↑ | UTMOS / DNSMOS OVRL / P808 ↑ |
|---|:---:|---|---|---|
| Base | 4 | 263/11805 = 2.2279% | 0.7433 | 3.2974 / 2.8950 / 3.5296 |
| Base | 10 | 205/11805 = 1.7366% | 0.7610 | 3.8072 / 3.0866 / 3.6689 |
| **+ SR-FD (ours)** | **4** | **167/11805 = 1.4147%** | **0.7613** | 3.7637 / 3.0711 / 3.6507 |
| ARCHI-TTS (reported) | 4 | 1.47% | – | – |

Four-step SR-FD reduces WER by **36.5% relative** to the four-step baseline and
**18.5% relative** to the ten-step baseline; both gaps are significant under an
utterance-level paired bootstrap, and speaker similarity is preserved. The gain
consists mainly of content-substitution reductions across all prompt lengths.
SIM, UTMOS, and DNSMOS are objective proxies, not human MOS.

## Pipeline

```
                         gradient (training only)
   text / prompt                  ▲
        │                         │
        ▼                         │
  ┌───────────────┐   gen   ┌─────────────┐   moments   ┌──────────────┐
  │ few-step      │ speech  │ frozen      │  μ_g, Σ_g   │ Fréchet      │
  │ sampler       ├────────►│ extractors  ├────────────►│ distance     ├──► L_srfd
  │ (VoxCPM2+LoRA)│         │ Whisper,CTC │             │              │
  └───────────────┘         └─────────────┘             └──────▲───────┘
                                                               │ μ_r, Σ_r
                                                   ┌───────────┴───────────┐
                                                   │ offline reference     │
                                                   │ moments (3 targets)   │
                                                   └───────────────────────┘
```

The model samples speech with the deployment-time four-step sampler; frozen
Whisper and CTC extractors map it to features whose mean/covariance are matched
to offline reference moments via a Fréchet distance. Gradients flow only into
the LoRA weights. See [docs/method.md](docs/method.md) for the full method and
[docs/integration.md](docs/integration.md) for how the loss plugs into the base
model.

## What's in this repository

```
srfd/                       # the SR-FD loss package (the contribution)
  frechet.py                #   differentiable Fréchet distance on GPU
  moments.py                #   batch / streaming / queue moment utilities
  extractors.py             #   frozen Whisper-encoder + wav2vec2-CTC extractors
  loss.py                   #   SRFDEmaLoss (queue/EMA moments, multi-target FD)
  stats_io.py               #   save/load reference statistics
  conditions.py             #   (optional) conditional statistics keys
configs/
  srfd_compact3.yaml        # paper main model: 3 targets, 2 extractors, λ=2e-4
scripts/
  compute_reference_stats.py  # offline reference moments (one run per target)
  score_seed_tts_eval.py      # Seed-TTS WER/CER + WavLM speaker similarity
  score_utmos_objective.py    # UTMOS objective proxy
  score_speechmos_dnsmos.py   # DNSMOS objective proxy
  paired_bootstrap.py         # utterance-level paired bootstrap significance
experiments/
  synthetic_validation.py   # CPU-only controlled check (known ground-truth FD)
tests/                      # unit tests for the FD math, moments, and loss
docs/                       # method + integration notes
demo/                       # anonymized demo: static site, results, LoRA adapter
```

This is a method-focused repository: it ships the SR-FD loss, a paper-aligned
config, reproduction/evaluation scripts, tests, and a self-contained demo. The
base TTS model is an external dependency.

## Quick start

```bash
pip install -e .            # core loss (needs torch)
pip install -e ".[test]"    # + pytest, pyyaml

pytest -q                   # 22 tests: FD math, moments, loss, config

# CPU-only sanity check: SR-FD drives a generator toward a target distribution
python experiments/synthetic_validation.py --steps 1500
# -> SR-FD reduces a known ground-truth Fréchet distance by ~99% (mean-only
#    supervision cannot, because it never fixes the covariance).
```

The package is import-safe without the heavy encoders; only the Whisper/CTC
extractors require `transformers` + `torchaudio` (`pip install -e ".[extractors]"`).

## Reproduction outline

1. **Reference statistics** (offline, once) — compute moments for the three
   targets. Each is one invocation with a different source manifest:

   ```bash
   # 1. low-step Whisper anchor   (ASR-verified four-step generations)
   python scripts/compute_reference_stats.py --config configs/srfd_compact3.yaml \
       --reps whisper_anchor8_p64 --manifest data/ref/asr_true4_good.jsonl \
       --out stats/ref_whisper_anchor_asr_true4_good.pt
   # 2. teacher CTC target        (ten-step teacher generations)
   python scripts/compute_reference_stats.py --config configs/srfd_compact3.yaml \
       --reps ctc_content_p64 --manifest data/ref/teacher_t10.jsonl \
       --out stats/ref_ctc_content_teacher_t10.pt
   # 3. real-speech CTC target    (real voice-cloning speech)
   python scripts/compute_reference_stats.py --config configs/srfd_compact3.yaml \
       --reps ctc_content_p64 --manifest data/ref/real_voiceclone.jsonl \
       --out stats/ref_ctc_content_real_voiceclone.pt
   ```

2. **Fine-tune with SR-FD** — wire `loss/srfd` into the base model's training
   step with `configs/srfd_compact3.yaml` (see
   [docs/integration.md](docs/integration.md)). Two stages: a supervised LoRA
   adaptation, then 1600 steps with SR-FD enabled.

3. **Evaluate** on Seed-TTS English:

   ```bash
   python scripts/score_seed_tts_eval.py --listing runs/srfd/wav_res_ref_text \
       --out_json runs/srfd/metrics.json --lang en
   python scripts/score_utmos_objective.py  --audio_dir runs/srfd/wav --out_json runs/srfd/utmos.json
   python scripts/score_speechmos_dnsmos.py --wav_dir   runs/srfd/wav --out_json runs/srfd/dnsmos.json
   python scripts/paired_bootstrap.py --a runs/base4/per_utt_wer.jsonl \
       --b runs/srfd/per_utt_wer.jsonl --metric wer
   ```

## Demo

`demo/` is a self-contained, anonymized bundle: a static website
(`demo/site/`), the Seed-TTS English results (`demo/data/results.json`), and the
packaged compact 3-target SR-FD LoRA adapter (`demo/model/`, 167 upstream word
errors). View it locally:

```bash
python3 -m http.server 8080 --directory demo/site   # then open http://localhost:8080
```

The site aligns the same Seed-TTS prompts across the base 4-step / 10-step
models, matched 4-step fine-tuning without SR-FD, FT + SR-FD, and the three
leave-one-out ablations, with per-utterance reference / ASR transcripts and a
Negative-Cases tab that surfaces remaining failure modes.

## License

Apache-2.0 (see [LICENSE](LICENSE)). The base model `openbmb/VoxCPM2` is
released by its authors under its own terms; consult its model card. Only
fine-tune on consented voices; do not use for impersonation, fraud, voice-print
bypass, or non-consensual content, and label synthetic audio.
