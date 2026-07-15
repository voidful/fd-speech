# Model Adapter

This directory contains the selected compact 3-target FDSpeech LoRA adapter.

Included files:

```text
lora_config.json
lora_weights.safetensors
training_state.json
selected_checkpoint.json
```

The main three-target run is the step-1600 checkpoint selected on the full-set
WER frontier, matching the paper's main table. The leave-one-target-out
ablations use a separate protocol: select on the fixed 200-prompt gate subset,
then report the selected checkpoints on the full Seed-TTS English `test-en`
set. Upstream Seed-TTS WER is the primary metric; UTMOS and DNSMOS are
objective quality checks.

Selected checkpoint:

- Source: `srfd_compact3/step_0001600` (config: `configs/srfd_compact3.yaml`)
- Primary metric: upstream Seed-TTS English WER
- WER: `167/11805 = 1.4147%`
- UTMOS / DNSMOS OVRL / P808: `3.7637 / 3.0711 / 3.6507`
- Base model: `openbmb/VoxCPM2`

Optimizer, scheduler, and FD-loss queue state are intentionally excluded from this
demo model bundle because they are not needed for inference.
