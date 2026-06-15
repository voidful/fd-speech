# SR-FD: Method

SR-FD (Speech Representation Fréchet Distance loss) is one extra training loss
added to a standard few-step TTS fine-tuning recipe. During fine-tuning, the
model synthesizes speech with the **same few-step sampler used at deployment**,
and frozen speech encoders turn the generated audio into feature vectors. Each
set of feature vectors is summarized by its mean and covariance (first- and
second-order moments), and the moments of generated speech are pushed toward
reference moments computed offline from desirable speech. The distance between
the two moment sets is a Fréchet distance — the same quantity behind FID and
FAD — used here as a differentiable loss. The loss needs no discriminator,
adds no parameters, and is removed at test time, so inference is unchanged.

## 1. Base objective

The base model (an external, tokenizer-free flow-matching autoregressive TTS
model — VoxCPM2) decodes continuous acoustic latents with a diffusion
transformer trained by conditional flow matching. Given a real latent `x1` and
noise `z ~ N(0, I)`, draw `t ∈ [0,1]`, form `y = (1-t) x1 + t z` with constant
velocity `v = z - x1`, and regress the predicted velocity:

```
L_fm = E ‖ u_θ(y, t) − v ‖²
```

At inference the decoder integrates the velocity field with a few Euler steps;
with only four steps the integration is very coarse — exactly the regime SR-FD
targets. The full fine-tuning objective is

```
L = w_fm · L_fm + w_stop · L_stop + L_aux + λ_srfd · L_srfd
```

`L_stop` is a stop-prediction loss; `L_aux` collects three small auxiliary
losses inherited from the underlying recipe (teacher-endpoint, preference-
feature, Whisper-text) with fixed small weights. **SR-FD is the `L_srfd` term.**

## 2. Matching the sampled-speech distribution

Standard fine-tuning supervises teacher-forced frames, so the few-step sampler
never appears in the loss and training can look healthy while the four-step
sampler drifts. SR-FD operates directly on sampled speech: during each update
the model synthesizes a complete short utterance with the deployment-time
four-step sampler, keeping the computation differentiable (see
[integration.md](integration.md)). Each frozen extractor `φ_k` maps the
generated audio `g_θ(x_b)` to one utterance-level feature vector
`h_b^k = φ_k(g_θ(x_b)) ∈ R^{d_k}`.

## 3. Two extractors, three targets

| Target | Source | Extractor | Role |
|---|---|---|---|
| Low-step Whisper anchor | ASR-verified 4-step generations | Whisper | Low-step content anchor |
| Teacher CTC target | 10-step teacher generations | CTC | Higher-step content transfer |
| Real-speech CTC target | Real LibriTTS speech | CTC | Natural-speech grounding |

The two frozen extractors are a Whisper-large-v3 encoder (semantic content,
`srfd/extractors.py::WhisperEncoderAnchorExtractor`) and a wav2vec2 CTC model
(phonetic content, `CTCPosteriorContentStatsExtractor`). All three targets
describe content, because content drift dominates the four-step failures: the
audio stays speech-like but an ASR system no longer recovers the intended
words. The Whisper anchor describes good low-step outputs, the teacher target
imports higher-step behavior, and the real-speech target grounds everything in
natural speech.

## 4. Reference and generated moments

**Reference moments** are precomputed offline from each target corpus
(`scripts/compute_reference_stats.py`) and only the moments are stored — the
reference audio is never used again. The CTC features are low-dimensional with
a well-conditioned covariance; the Whisper features are high-dimensional
relative to the sample count, so their covariance is rank-deficient and is
regularized with a small `ε I` before the matrix square root. Because the
absolute Whisper Fréchet value is biased by this rank deficiency, models are
never selected by raw FD (see §7).

**Generated moments** are estimated from a feature queue. A covariance from a
few utterances is meaningless, while generating hundreds of utterances per
update is unaffordable, so for each extractor a queue `Q_t^k` of features from
recent updates is kept. At step `t` the generated moments are computed over the
queue together with the current mini-batch. Features from earlier steps are
detached; only the current mini-batch keeps gradient — a large-sample moment
estimate at the memory cost of a single batch. (`srfd/loss.py::SRFDEmaLoss`,
`stats_mode="queue"`.)

## 5. The SR-FD loss

For each extractor `k` and target `j`, SR-FD computes a Fréchet distance
between the generated and reference Gaussian moment estimates
(`srfd/frechet.py`):

```
FD = ‖ μ_g − μ_r ‖² + Tr(Σ_g + Σ_r − 2 (Σ_r^{1/2} Σ_g Σ_r^{1/2})^{1/2})
```

Different feature spaces have different natural scales, so each term is divided
by its own detached value:

```
FD̃ = FD / stopgrad(FD + ε)
```

Each normalized term has magnitude near one, but its gradient still points in
the FD-reducing direction, so targets are balanced by gradient scale rather
than raw distance. The total loss is a weighted average of the normalized
terms, first across targets within each extractor and then across extractors;
with the paper weights the Whisper and CTC branches contribute equally and the
two CTC targets split the CTC half.

A **length gate** admits a sample into the loss only when its
generated-to-target duration ratio is close to one (`[0.92, 1.08]`), since
strongly mismatched samples usually contain truncation or runaway speech and
matching moments on them injects noise.

At test time the extractors, queues, reference moments, and Fréchet computation
all disappear: the deployed model is a plain four-step model with LoRA
adapters, with no added parameters and no added inference computation.

## 6. Hyperparameters

See `configs/srfd_compact3.yaml`. Key values: LoRA rank 32 / alpha 32 on the
q,k,v,o projections of the LM and DiT; `λ_srfd = 2e-4`; raw target weights
1.0 / 0.5 / 0.5; both extractor weights 1.0; feature queue of 50,000 vectors;
length gate `[0.92, 1.08]`; four-step Euler sampling with guidance 2.45 during
training; 1600 fine-tuning steps with AdamW (weight decay 0.01, grad-norm clip
0.03), bf16, batch size 1, cosine LR `3e-8 → 0` with no warmup.

## 7. Is the Fréchet distance a good diagnostic?

No. SR-FD trains the model to reduce representation FD, but a smaller raw FD
does not imply lower WER: across saved checkpoints the correlation between raw
FD and WER is weak, and an external CTC FD pass confirms training moves
generated features toward the reference while the raw value does not track WER.
Reference targets are therefore validated through a WER ablation, not through
absolute FD, and representation FD is not used to select checkpoints or as a
standalone quality claim.
