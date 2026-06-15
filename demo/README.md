# SR-FD demo bundle

A self-contained, anonymized demo for the four-step SR-FD model. It packages:

- a static demo website under `site/`
- aligned Seed-TTS English audio comparisons under `site/audio/`
- Seed-TTS English benchmark results under `data/results.json`
- the compact 3-target SR-FD LoRA adapter under `model/`

The model is a LoRA adapter for `openbmb/VoxCPM2`, **not** a standalone full
checkpoint. Use it together with the base model. Training / evaluation code and
the reproduction config live at the repository root (`srfd/`, `configs/`,
`scripts/`).

## Verified results

The packaged adapter obtains **167 word errors** on upstream Seed-TTS English
(paper denominator 11,805).

| System | Steps | FT | SR-FD | Upstream WER | UTMOS / DNSMOS OVRL / P808 |
|---|---:|:---:|:---:|---:|---:|
| Base VoxCPM2 | 4 | No | No | 263/11805 = 2.2279% | 3.2974 / 2.8950 / 3.5296 |
| Base VoxCPM2 | 10 | No | No | 205/11805 = 1.7366% | 3.8072 / 3.0866 / 3.6689 |
| Matched FT | 4 | Yes | No | 174/11805 = 1.4740% | 3.7615 / 3.0729 / 3.6522 |
| FT + SR-FD | 4 | Yes | Yes | 167/11805 = 1.4147% | 3.7637 / 3.0711 / 3.6507 |
| ARCHI-TTS reported | 4 | – | – | 1.47% | – |

## Compact 3-target SR-FD

The compact target set keeps only content-centered SR-FD targets:

1. `asr_true4_good_whisper` — Whisper content statistics from ASR-reranked good
   four-step generations.
2. `teacher_t10_ctc_content` — CTC posterior statistics from ten-step teacher
   generations.
3. `real_ctc_content` — CTC posterior statistics from real voice-cloning speech.

This keeps the story focused on few-step intelligibility. The selected compact
checkpoint is `srfd_compact3/step_0001600`; among the promoted compact variants
it is the only one that reaches 167 upstream word errors (others score 173,
176, 182).

## Local demo

```bash
python3 -m http.server 8080 --directory site   # then open http://localhost:8080
```

The Audio section compares the same Seed-TTS prompts across the base 4-step and
10-step models, matched 4-step fine-tuning without SR-FD, FT + SR-FD, and three
compact leave-one-out ablations. Each card shows the generated waveform,
reference transcript, ASR transcript, per-utterance WER, and full-set WER. A
Negative Cases tab surfaces rows where FT + SR-FD still makes a word-level ASR
mistake or regresses against matched fine-tuning, so both the frontier behavior
and the remaining failure modes are inspectable.

## Adapter layout

```text
model/
  lora_config.json
  lora_weights.safetensors
  training_state.json
  selected_checkpoint.json
```

Only inference-relevant adapter files are included; optimizer, scheduler, and
SR-FD queue state are excluded.

## License

Apache-2.0, following the base model's terms. See `openbmb/VoxCPM2` for the
original model card and usage restrictions.
