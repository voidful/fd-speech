<div align="center">
<h1>FDSpeech: Fréchet-Distance-Guided Few-Step Speech Synthesis</h1>

**A training-time distributional regularizer for intelligible few-step TTS**

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2607.06027-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.06027)
[![Model](https://img.shields.io/badge/Model-FDSpeech--VoxCPM2-yellow?logo=huggingface)](https://huggingface.co/voidful/FDSpeech-VoxCPM2)
[![Hugging Face Space](https://img.shields.io/badge/Space-live%20comparison-orange?logo=huggingface)](https://huggingface.co/spaces/voidful/fd-speech-demo)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue)](LICENSE)

</div>

## Overview

**FDSpeech** is a four-step VoxCPM2 adapter and training recipe built around a
Fréchet-distance loss on speech representations. It is a training-time
distributional regularizer for tokenizer-free, few-step flow-matching TTS.
Standard objectives such as conditional flow matching, reconstruction, and
stop prediction supervise local behavior, but do not directly constrain the
distribution of complete utterances produced by the low-step sampler.

FDSpeech operates on sampled speech. During fine-tuning, the model synthesizes an
utterance with the **same four-step sampler used at deployment**. Frozen
Whisper and wav2vec 2.0 CTC encoders map that speech to content features, and a
differentiable Fréchet distance matches their mean and covariance to offline
reference statistics from three complementary targets. FDSpeech needs **no
discriminator**, is removed after training, and adds **no inference-time
parameters or computation**.

> **Base model.** The experiments use the external tokenizer-free
> flow-matching TTS model [`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2).
> The released artifact is the
> [`voidful/FDSpeech-VoxCPM2`](https://huggingface.co/voidful/FDSpeech-VoxCPM2)
> LoRA adapter for that base model, not a standalone checkpoint.

## Main result

On the 1,088-prompt Seed-TTS English `test-en` set (11,805 reference words),
four-step FDSpeech improves the original four-step VoxCPM2 WER from
2.2279% to 1.4147%, a **36.5% relative reduction**, and also improves on the
original ten-step baseline by **18.5% relative**.

| System | Steps | Upstream WER ↓ | SIM ↑ | UTMOS / DNSMOS OVRL / P808 ↑ |
|---|:---:|---:|---:|---:|
| VoxCPM2 | 4 | 263/11805 = 2.2279% | 0.7433 | 3.2974 / 2.8950 / 3.5296 |
| VoxCPM2 | 10 | 205/11805 = 1.7366% | 0.7610 | 3.8072 / 3.0866 / 3.6689 |
| **FDSpeech (ours)** | **4** | **167/11805 = 1.4147%** | **0.7613** | **3.7637 / 3.0711 / 3.6507** |
| F5-TTS (reported) | 32 | 1.83% | – | – |
| ARCHI-TTS (reported) | 32 | 1.47% | – | – |

The two paired WER improvements are significant under utterance-level
bootstrap. A blinded comparison against the ten-step baseline collected 229
judgments from 13 listeners; the decisive preference split was near even
(61 FDSpeech vs. 67 ten-step), with equivalence supported under the paper's
pre-specified 10-point margin. SIM, UTMOS, and DNSMOS are objective proxies,
not human MOS. See the [paper](https://arxiv.org/abs/2607.06027) for confidence
intervals, ablations, and the complete evaluation protocol.

## Method

```text
                         gradient (training only)
   text / prompt                  ▲
        │                         │
        ▼                         │
  ┌───────────────┐   gen   ┌─────────────┐   moments   ┌──────────────┐
  │ four-step     │ speech  │ frozen      │  μ_g, Σ_g   │ Fréchet      │
  │ sampler       ├────────►│ extractors  ├────────────►│ distance     ├──► L_FD
  │ VoxCPM2+LoRA  │         │ Whisper, CTC│             │              │
  └───────────────┘         └─────────────┘             └──────▲───────┘
                                                               │ μ_r, Σ_r
                                                   ┌───────────┴───────────┐
                                                   │ offline reference     │
                                                   │ moments (3 targets)   │
                                                   └───────────────────────┘
```

The three targets are:

| Target | Source | Extractor | Role |
|---|---|---|---|
| Low-step Whisper anchor | ASR-verified four-step generations | Whisper | Deployment-matched content anchor |
| Teacher CTC target | Ten-step teacher generations | wav2vec 2.0 CTC | Higher-step content transfer |
| Real-speech CTC target | Real LibriTTS speech | wav2vec 2.0 CTC | Natural-speech grounding |

Generated moments are estimated with a detached feature queue, so only the
current mini-batch retains gradients. See [docs/method.md](docs/method.md) for
the full loss and [docs/integration.md](docs/integration.md) for the base-model
integration points.

## What this public release contains

| Artifact | Included | Notes |
|---|:---:|---|
| Differentiable FD loss and moment utilities | Yes | `srfd/` compatibility namespace |
| Frozen Whisper/CTC extractor wrappers | Yes | Heavy models download separately |
| Paper-aligned three-target config | Yes | Paths are local placeholders |
| Reference-statistics and evaluation scripts | Yes | External audio/benchmarks required |
| Selected FDSpeech four-step LoRA adapter | Yes | Hosted as [`FDSpeech-VoxCPM2`](https://huggingface.co/voidful/FDSpeech-VoxCPM2) |
| Aligned audio demo and result metadata | Yes | Ten prompts and seven systems |
| VoxCPM2 source and base weights | No | Use the upstream release |
| Training manifests, reference audio, and precomputed moments | No | Not redistributed |
| Turnkey end-to-end training entrypoint | No | Integrate the loss into the base recipe as documented |

The repository is method-focused. It supports the public loss implementation,
the paper configuration, adapter inference, evaluation, and controlled tests;
it does not claim that the private training corpus layout or a patched base
training stack is bundled.

## Installation

Python 3.10 or newer and PyTorch 2.5 or newer are required for the FDSpeech
package.

```bash
git clone https://github.com/voidful/fd-speech.git
cd fd-speech

conda create -n fdspeech python=3.10 -y
conda activate fdspeech
python -m pip install -U pip
pip install -e ".[test]"
```

Verify the loss implementation with the unit tests and the CPU-only synthetic
experiment:

```bash
pytest -q
python experiments/synthetic_validation.py --steps 1500
```

The package is import-safe without the speech encoders. Install their optional
dependencies only when computing reference statistics or training with audio:

```bash
pip install -e ".[extractors]"
```

## Inference with the released adapter

[`voidful/FDSpeech-VoxCPM2`](https://huggingface.co/voidful/FDSpeech-VoxCPM2)
is a LoRA adapter. A current VoxCPM installation can download it and load it
together with the upstream VoxCPM2 base model. The first run downloads the
2B-parameter base checkpoint; a CUDA GPU is recommended.

```bash
pip install -U voxcpm huggingface_hub soundfile
```

```python
import json
import os

import soundfile as sf
from huggingface_hub import snapshot_download
from voxcpm import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig

adapter_dir = snapshot_download("voidful/FDSpeech-VoxCPM2")
with open(os.path.join(adapter_dir, "lora_config.json"), encoding="utf-8") as handle:
    adapter_info = json.load(handle)

model = VoxCPM.from_pretrained(
    hf_model_id="openbmb/VoxCPM2",
    load_denoiser=False,
    optimize=True,
    lora_config=LoRAConfig(**adapter_info["lora_config"]),
    lora_weights_path=adapter_dir,
)

wav = model.generate(
    text="The quick brown fox jumps over the lazy dog.",
    cfg_value=2.35,
    inference_timesteps=4,
    normalize=True,
    denoise=False,
    seed=0,
)
sf.write("fdspeech.wav", wav, model.tts_model.sample_rate)
```

For voice cloning, also pass `prompt_wav_path` and its exact `prompt_text` as
described by the [VoxCPM documentation](https://github.com/OpenBMB/VoxCPM).
Read the [model card](MODEL_CARD.md) before deployment.

## FDSpeech post-training and reproduction

### 1. Prepare reference manifests

Each JSONL manifest must contain an `audio` field pointing to a local WAV or
FLAC file. Build separate manifests for the three reference targets. The
source audio is not included in this repository.

### 2. Compute offline reference moments

Run one invocation per target:

```bash
# Low-step Whisper anchor
python scripts/compute_reference_stats.py \
  --config configs/srfd_compact3.yaml \
  --reps whisper_anchor8_p64 \
  --manifest data/ref/asr_true4_good.jsonl \
  --out stats/ref_whisper_anchor_asr_true4_good.pt

# Ten-step teacher CTC target
python scripts/compute_reference_stats.py \
  --config configs/srfd_compact3.yaml \
  --reps ctc_content_p64 \
  --manifest data/ref/teacher_t10.jsonl \
  --out stats/ref_ctc_content_teacher_t10.pt

# Real-speech CTC target
python scripts/compute_reference_stats.py \
  --config configs/srfd_compact3.yaml \
  --reps ctc_content_p64 \
  --manifest data/ref/real_voiceclone.jsonl \
  --out stats/ref_ctc_content_real_voiceclone.pt
```

### 3. Integrate and fine-tune

The base training step must expose a differentiable four-step sampler, decode
the generated latent to waveform, apply the duration gate, and add
the configured FD-loss term to the existing objective. The public API and a
training-step sketch are in [docs/integration.md](docs/integration.md).

The paper recipe first performs supervised LoRA adaptation, then runs 1,600
additional FDSpeech steps using [configs/srfd_compact3.yaml](configs/srfd_compact3.yaml).
The pretrained base weights remain frozen.

The existing `srfd` package, config block, and `loss/srfd` metric key are kept
as compatibility identifiers for released checkpoints and training logs. New
Python code can import the public aliases from `fdspeech`.

## Evaluation

Install the evaluation dependencies:

```bash
pip install -e ".[eval]"
```

The Seed-TTS benchmark audio and the generated-system listings are external
inputs and are not bundled. Given a `generated_wav|reference_wav|target_text`
listing, the included scripts compute WER/CER, speaker similarity, objective
quality proxies, and paired significance:

```bash
python scripts/score_seed_tts_eval.py \
  --listing runs/fdspeech/wav_res_ref_text \
  --out_json runs/fdspeech/metrics.json --lang en

python scripts/score_utmos_objective.py \
  --audio_dir runs/fdspeech/wav --out_json runs/fdspeech/utmos.json

python scripts/score_speechmos_dnsmos.py \
  --wav_dir runs/fdspeech/wav --out_json runs/fdspeech/dnsmos.json

python scripts/paired_bootstrap.py \
  --a runs/base4/per_utt_wer.jsonl \
  --b runs/fdspeech/per_utt_wer.jsonl --metric wer
```

## Audio demo

The [Hugging Face Space](https://huggingface.co/spaces/voidful/fd-speech-demo)
provides a three-column, same-prompt comparison of the original VoxCPM2 at
four and ten steps and FDSpeech at four steps. It includes ASR transcript diffs,
per-sample WER, benchmark results, prompt/reference audio, and deliberately
surfaced negative cases. The Space serves checked-in aligned audio, so it runs
on CPU hardware without loading three 2B-parameter systems.

Run the Gradio Space locally:

```bash
pip install -r requirements.txt
python app.py
```

The earlier static demo is still available:

```bash
python3 -m http.server 8080 --directory demo/site
# Open http://localhost:8080
```

See [demo/README.md](demo/README.md) for the bundle layout and selection
metadata.

## Repository layout

```text
fdspeech/                     canonical public import aliases
srfd/                         compatibility implementation namespace
configs/srfd_compact3.yaml    paper main configuration
scripts/                      reference-statistics and evaluation utilities
experiments/                  CPU-only synthetic validation
tests/                        unit and config tests
docs/                         method and integration notes
demo/model/                   selected LoRA adapter
demo/site/                    static aligned-audio demo
app.py                        Hugging Face Spaces comparison app
MODEL_CARD.md                 model scope, metrics, limitations, and usage
```

## Limitations and responsible use

FDSpeech was evaluated as an **English intelligibility regularizer** for a
four-step VoxCPM2 setting. The paper does not establish improvements for every
language, base model, sampler, perceptual attribute, or prompt. Lower
representation FD is not a reliable standalone model-selection metric, and
individual utterances can regress even when aggregate WER improves.

Use only consented voices. Do not use the adapter for impersonation, fraud,
voice-print bypass, or non-consensual content, and clearly label synthetic
audio.

## License

The FDSpeech code and adapter are released under
[Apache-2.0](LICENSE). VoxCPM2, pretrained extractors, datasets, and evaluation
tools remain subject to their respective licenses and terms.

## Acknowledgements

This work builds on [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) and uses
representations from [Whisper](https://github.com/openai/whisper) and
[wav2vec 2.0](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec).
Evaluation follows [Seed-TTS](https://github.com/BytedanceSpeech/seed-tts-eval),
and the training/reference recipe uses LibriTTS-derived voice-cloning material.

## Citation

If you find FDSpeech useful, please cite:

```bibtex
@article{chung2026fdspeech,
  title   = {Fr\'{e}chet Distance Loss on Speech Representations for Text-to-Speech Synthesis},
  author  = {Chung, Ho-Lam and Huang, Kuan-Po and Lu, Bo-Ru and Lee, Hung-yi},
  journal = {arXiv preprint arXiv:2607.06027},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.06027}
}
```
