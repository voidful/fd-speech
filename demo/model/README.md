# Model Adapter

This directory contains the selected compact 3-target SR-FD LoRA adapter.

Included files:

```text
lora_config.json
lora_weights.safetensors
training_state.json
selected_checkpoint.json
```

The adapter is selected by the same protocol used in the paper-facing table:

1. Train compact 3-target SR-FD variants to 1600 steps.
2. Select checkpoints on the 200-prompt Seed-TTS gate subset.
3. Promote selected checkpoints to the full Seed-TTS English test-en set.
4. Use upstream Seed-TTS WER as the primary selection metric.
5. Check UTMOS and DNSMOS as objective quality proxies.

Selected checkpoint:

- Source: `srfd_compact3/step_0001600` (config: `configs/srfd_compact3.yaml`)
- Primary metric: upstream Seed-TTS English WER
- WER: `167/11805 = 1.4147%`
- UTMOS / DNSMOS OVRL / P808: `3.7637 / 3.0711 / 3.6507`
- Base model: `openbmb/VoxCPM2`

Optimizer, scheduler, and SR-FD queue state are intentionally excluded from this
demo model bundle because they are not needed for inference.
