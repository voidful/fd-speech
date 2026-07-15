---
license: apache-2.0
language:
  - en
pipeline_tag: text-to-speech
base_model: openbmb/VoxCPM2
library_name: voxcpm
tags:
  - text-to-speech
  - flow-matching
  - frechet-distance
  - few-step-generation
  - lora
  - fdspeech
---

# FDSpeech-VoxCPM2

**FDSpeech-VoxCPM2** is the selected compact three-target LoRA adapter from the paper
[Fréchet Distance Loss on Speech Representations for Text-to-Speech
Synthesis](https://arxiv.org/abs/2607.06027). It is trained with a
Fréchet-distance loss on speech representations for intelligible four-step
generation.

It is **not a standalone TTS checkpoint**. Use it with the external
[`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2) base model. The FD
loss is used only during fine-tuning; inference is the ordinary four-step base
model with the LoRA adapter loaded.

## Model details

- **Base model:** `openbmb/VoxCPM2`, approximately 2B parameters
- **Artifact type:** LoRA adapter
- **Adapter:** rank 32, alpha 32, q/k/v/o projections in the language model and DiT
- **Inference sampler:** four Euler steps, CFG 2.35 in the paper evaluation
- **Selected checkpoint:** `srfd_compact3/step_0001600`
- **FDSpeech targets:** low-step Whisper anchor, ten-step teacher CTC, real-speech CTC
- **Primary evaluation:** Seed-TTS English `test-en`

## Files

```text
lora_config.json
lora_weights.safetensors
training_state.json
adapters/compact3_balanced/
ablations/
configs/
reports/
```

Optimizer, scheduler, reference statistics, and FD-loss feature-queue state are
not included because they are not needed for inference. Base-model weights are
downloaded separately.

## Usage

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

The first run downloads the base model. A CUDA GPU is recommended. For
continuation-style voice cloning, provide a consented `prompt_wav_path` and its
exact `prompt_text`.

## Training data and reference statistics

The paper fine-tunes on a 767-row manifest derived from LibriTTS voice-cloning
material. Offline FDSpeech reference moments are computed from ASR-verified four-step
generations, ten-step teacher generations, and real LibriTTS speech. The
training manifest, source/reference audio, and precomputed moments are not
redistributed in this repository.

See the [training config](https://github.com/voidful/fd-speech/blob/main/configs/srfd_compact3.yaml)
for the released recipe and the
[integration guide](https://github.com/voidful/fd-speech/blob/main/docs/integration.md)
for implementation details. The `srfd` path and config key are retained as
compatibility identifiers for the released training artifacts.

## Evaluation

Results use the upstream Seed-TTS English scorer over 1,088 prompts and 11,805
reference words.

| System | Steps | Upstream WER ↓ | SIM ↑ | UTMOS / DNSMOS OVRL / P808 ↑ |
|---|:---:|---:|---:|---:|
| VoxCPM2 | 4 | 263/11805 = 2.2279% | 0.7433 | 3.2974 / 2.8950 / 3.5296 |
| VoxCPM2 | 10 | 205/11805 = 1.7366% | 0.7610 | 3.8072 / 3.0866 / 3.6689 |
| **FDSpeech-VoxCPM2** | **4** | **167/11805 = 1.4147%** | **0.7613** | **3.7637 / 3.0711 / 3.6507** |

The WER reductions against both original baselines are significant under an
utterance-level paired bootstrap. SIM, UTMOS, and DNSMOS are objective proxies,
not human MOS. A blinded comparison with the ten-step baseline produced a near
even decisive preference split, with equivalence supported within the paper's
pre-specified 10-point margin. See the paper for the complete protocol and
confidence intervals.

## Intended use

- Research on few-step flow-matching TTS and distributional regularization
- Reproduction and analysis of the paper's four-step English setting
- Evaluation of the released adapter on consented speech prompts

## Limitations and risks

- Evidence is concentrated on English Seed-TTS; multilingual gains are not established.
- FDSpeech primarily targets intelligibility and is not a general perceptual-quality objective.
- Aggregate WER improves, but individual prompts can still regress or contain substitutions.
- Raw representation FD should not be used as a standalone quality or checkpoint-selection metric.
- Voice cloning can enable impersonation and fraud. Use only consented voices, label synthetic audio, and do not use it for identity or access-control bypass.

## License

The adapter and FDSpeech code are released under Apache-2.0. The base model,
pretrained extractors, datasets, and evaluation tools remain subject to their
own terms.

## Citation

```bibtex
@article{chung2026fdspeech,
  title   = {Fr\'{e}chet Distance Loss on Speech Representations for Text-to-Speech Synthesis},
  author  = {Chung, Ho-Lam and Huang, Kuan-Po and Lu, Bo-Ru and Lee, Hung-yi},
  journal = {arXiv preprint arXiv:2607.06027},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.06027}
}
```
