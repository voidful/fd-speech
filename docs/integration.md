# Integrating SR-FD into a few-step flow-matching TTS model

SR-FD is model-agnostic: it only needs (1) a **differentiable** few-step
sampler and (2) the generated waveform. This repository implements the loss and
extractors; the base model is the external, tokenizer-free flow-matching TTS
model VoxCPM2 (`openbmb/VoxCPM2`), used through a LoRA adapter. This document
shows the three integration points so the method can be reproduced or ported.

## 1. A differentiable few-step sampler

Many flow-matching decoders wrap their sampling loop in
`@torch.inference_mode()`, which blocks gradients. SR-FD needs gradients to
flow from the loss, through the generated latent, back into the (LoRA) weights.
Add a sibling `sample()` method with identical numerics but **without** the
inference-mode decorator. In VoxCPM2 this lives on the flow-matching decoder
(`UnifiedCFM`):

```python
def sample(self, mu, n_timesteps, patch_size, cond,
           temperature=1.0, cfg_value=1.0, sway_sampling_coef=1.0,
           use_cfg_zero_star=True, initial_noise=None, return_trajectory=False):
    """Differentiable sampling path used by SR-FD.

    Identical in numerics to `forward` but without `inference_mode`, so
    gradients can flow back through the produced latent into `mu`.
    """
    # ... same Euler/Heun/RK integration as the deployment sampler ...
```

The model's `forward` then exposes a `sample_with_grad` flag that routes
generation through `sample()` instead of the inference-mode path, using the
**same** step count and sampler settings as deployment (four Euler steps,
guidance 2.45, sway 1.0). This is what makes the loss act on the distribution
the sampler will actually produce, not on a teacher-forced trajectory.

## 2. Building the loss

Build the extractors and the loss once, from the `srfd` block of the config:

```python
import yaml, torch
from srfd import SRFDEmaLoss, build_srfd_extractors, load_stats

cfg = yaml.safe_load(open("configs/srfd_compact3.yaml"))["srfd"]

extractors = build_srfd_extractors(cfg["reps"])           # Whisper + CTC
targets = [                                               # three reference targets
    {"name": t["name"], "weight": t["weight"], "stats": load_stats(t["path"])}
    for t in cfg["reference_stats_paths"]
]
srfd_loss = SRFDEmaLoss(
    extractors=extractors,
    real_stats=targets,
    stats_mode=cfg["stats_mode"],          # "queue"
    queue_size=cfg["queue_size"],          # 50000
    normalize=cfg["normalize"],            # per-term FD normalization
    normalize_total_weight=cfg["normalize_total_weight"],
    warmup_steps=cfg["warmup_steps"],
)
```

## 3. The training step

On each step: (a) sample a short utterance with the differentiable four-step
sampler, (b) decode it to a waveform, (c) apply the length gate, (d) call the
SR-FD loss, and (e) add it to the base objective. Sketch:

```python
# (a) differentiable few-step generation (same settings as deployment)
gen_latent = model(batch, sample_with_grad=True, sample_n_timesteps=4)
# (b) decode to waveform via the (frozen) AudioVAE decoder
wav = model.audio_vae.decode(gen_latent)

# (c) length gate: only keep samples whose duration ratio is in [0.92, 1.08]
ratio = generated_duration / target_duration
keep = (ratio >= 0.92) & (ratio <= 1.08)

# (d) SR-FD reads the generated waveform (+ mask + sample rate)
srfd_batch = {
    "waveform": wav[keep],
    "waveform_mask": wav_mask[keep],
    "waveform_sample_rate": out_sample_rate,
}
out = srfd_loss(srfd_batch, step=global_step)   # {"loss/srfd": ...}

# (e) total objective
loss = (w_fm   * out_fm["loss/diff"]
        + w_stop * out_stop["loss/stop"]
        + L_aux
        + lambda_srfd * out["loss/srfd"])       # lambda_srfd = 2e-4
loss.backward()
```

### Numerical notes

* The Fréchet term uses `torch.linalg.eigh`, which has no bf16 CUDA kernel.
  Wrap the SR-FD call in `torch.amp.autocast(device_type="cuda", enabled=False)`
  so the eigendecomposition runs in fp32 while the rest of the step stays bf16.
* The queue detaches features from previous steps, so the autograd graph never
  grows across steps; only the current mini-batch carries gradient.
* SR-FD activates after `warmup_steps`, so the base losses stabilize training
  before the distributional term turns on.

## 4. Inference (deployment)

At test time SR-FD is gone entirely — the deployed model is the base four-step
model plus the LoRA adapter. Loading the adapter and generating:

With a current upstream `voxcpm` installation, load the adapter when the base
model is constructed and use the public `inference_timesteps` argument:

```python
import soundfile as sf
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    load_denoiser=False,
    lora_weights_path="demo/model",
)
wav = model.generate(
    text="The quick brown fox jumps over the lazy dog.",
    inference_timesteps=4,
    cfg_value=2.35,
    normalize=True,
    denoise=False,
    seed=0,
)
sf.write("srfd.wav", wav, model.tts_model.sample_rate)
```

No extractors, queues, reference moments, or Fréchet computation are involved at
inference, so there is no added inference cost.
