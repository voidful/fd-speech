---
title: FDSpeech TTS Comparison
emoji: 🎧
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
python_version: "3.10"
pinned: false
license: apache-2.0
short_description: Compare VoxCPM2 4/10-step and FDSpeech 4-step speech
models:
  - voidful/FDSpeech-VoxCPM2
---

# FDSpeech: four-step VoxCPM2 comparison

This Space provides aligned Seed-TTS English listening examples for:

- original VoxCPM2 at four steps;
- original VoxCPM2 at ten steps; and
- **FDSpeech** at four steps.

FDSpeech is the released VoxCPM2 LoRA adapter trained with a
Fréchet-distance loss on speech representations. It improves upstream
Seed-TTS English WER from 2.2279% for the original four-step sampler to
1.4147%, without adding an inference-time module.

- Model: [`voidful/FDSpeech-VoxCPM2`](https://huggingface.co/voidful/FDSpeech-VoxCPM2)
- Code: [`voidful/fd-speech`](https://github.com/voidful/fd-speech)
- Paper: [Fréchet Distance Loss on Speech Representations for Text-to-Speech Synthesis](https://arxiv.org/abs/2607.06027)

The Space serves checked-in, aligned audio so every system uses the same prompt
and evaluation seed. Negative cases are included alongside improvements.
