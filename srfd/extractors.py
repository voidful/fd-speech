"""Representation extractors for SR-FD.

The final SR-FD configuration uses two frozen, content-sensitive extractors:

* ``WhisperEncoderAnchorExtractor`` — a frozen Whisper encoder pooled into a
  fixed utterance-level vector via soft time anchors, giving a semantic /
  ASR-sensitive representation.
* ``CTCPosteriorContentStatsExtractor`` — a frozen wav2vec2 CTC head whose
  non-blank posteriors are randomly projected and pooled, giving a phonetic /
  content representation, plus blank/timing statistics.

Each extractor takes a ``batch`` dict and returns a ``[B, C]`` representation
vector. Extractors are ``nn.Module``s so they can be moved to GPU and hold
frozen sub-models; they contribute no trainable parameters. Both encoders are
frozen and used only during training; at deployment they are removed entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .moments import masked_time_mean_std  # noqa: F401  (re-exported util)


@dataclass
class SRFDExtractorConfig:
    """Single extractor entry from YAML.

    YAML form (one of these per representation):

    ```yaml
    - name: ctc_content_p64
      type: ctc_content_stats
      model_path: models/wav2vec2-base-960h
      projection_dim: 64
      weight: 1.0
    ```
    """

    name: str
    type: str
    weight: float = 1.0
    enabled: bool = True
    eps: float = 1e-6
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SRFDExtractorConfig":
        known = {"name", "type", "weight", "enabled", "eps"}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(
            name=d["name"],
            type=d["type"],
            weight=float(d.get("weight", 1.0)),
            enabled=bool(d.get("enabled", True)),
            eps=float(d.get("eps", 1e-6)),
            extra=extra,
        )


class BaseSRFDExtractor(nn.Module):
    """Base class. All representation extractors must subclass this."""

    requires_waveform: bool = False

    def __init__(self, config: SRFDExtractorConfig):
        super().__init__()
        self.config = config
        self.name = config.name
        self.weight = float(config.weight)
        self.enabled = bool(config.enabled)
        self.eps = float(config.eps)

    def feature_dim(self) -> Optional[int]:
        """Return the feature dimension if known statically, else ``None``."""
        return None

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Shared waveform / masking helpers
# --------------------------------------------------------------------------- #
def _as_2d_mask(mask: torch.Tensor, *, name: str) -> torch.Tensor:
    if mask.dim() == 3 and mask.size(1) == 1:
        mask = mask.squeeze(1)
    if mask.dim() != 2:
        raise ValueError(f"{name} must be [B, T], got {tuple(mask.shape)}")
    return mask


def _time_positions(length: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if length <= 1:
        return torch.zeros(length, device=device, dtype=dtype)
    return torch.linspace(0.0, 1.0, length, device=device, dtype=dtype)


def _resolve_waveform(batch: Dict[str, torch.Tensor], *, owner: str) -> torch.Tensor:
    for key in ("waveform", "waveform_pred", "audio", "audio_pred"):
        if key in batch and batch[key] is not None:
            wav = batch[key]
            if wav.dim() == 3 and wav.size(1) == 1:
                wav = wav.squeeze(1)
            if wav.dim() != 2:
                raise ValueError(f"{owner} waveform must be [B, T], got {tuple(wav.shape)}")
            return wav
    raise KeyError(f"{owner} requires one of {{'waveform', 'waveform_pred', 'audio', 'audio_pred'}}.")


def _sample_rate_from_batch(batch: Dict[str, torch.Tensor], fallback: int) -> int:
    sr = batch.get("waveform_sample_rate", batch.get("sample_rate", fallback))
    if isinstance(sr, torch.Tensor):
        sr = int(sr.detach().flatten()[0].item())
    return int(sr)


def _resample_mask_by_length(mask: torch.Tensor, old_len: int, new_len: int) -> torch.Tensor:
    if old_len == new_len:
        return mask
    lengths = mask.to(torch.float32).sum(dim=1)
    new_lengths = torch.ceil(lengths * float(new_len) / float(max(old_len, 1))).to(torch.long)
    arange = torch.arange(new_len, device=mask.device).unsqueeze(0)
    return arange < new_lengths.clamp_min(1).unsqueeze(1)


def _prepare_waveform(
    waveform: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    sample_rate: int,
    target_sample_rate: int,
    max_seconds: float,
    normalize: bool,
    eps: float,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    wav = waveform.to(torch.float32)
    if mask is not None:
        mask = _as_2d_mask(mask, name="waveform_mask").to(device=wav.device, dtype=torch.bool)
        if mask.size(1) != wav.size(1):
            raise ValueError(f"waveform_mask length {mask.size(1)} != waveform length {wav.size(1)}")
        wav = wav.masked_fill(~mask, 0.0)

    if sample_rate != target_sample_rate:
        import torchaudio.functional as AF

        old_len = wav.size(1)
        wav = AF.resample(wav, sample_rate, target_sample_rate)
        if mask is not None:
            mask = _resample_mask_by_length(mask, old_len=old_len, new_len=wav.size(1))

    if max_seconds > 0:
        max_len = int(round(max_seconds * target_sample_rate))
        if max_len > 0 and wav.size(1) > max_len:
            wav = wav[:, :max_len]
            if mask is not None:
                mask = mask[:, :max_len]

    if normalize:
        if mask is None:
            mean = wav.mean(dim=1, keepdim=True)
            std = wav.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
        else:
            m = mask.to(torch.float32)
            count = m.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean = (wav * m).sum(dim=1, keepdim=True) / count
            var = ((wav - mean) * m).pow(2).sum(dim=1, keepdim=True) / count
            std = var.sqrt().clamp_min(eps)
        wav = (wav - mean) / std
        if mask is not None:
            wav = wav.masked_fill(~mask, 0.0)

    return wav, mask


def _frame_mask_from_waveform_mask(
    mask: Optional[torch.Tensor],
    n_frames: int,
    hop_length: int,
) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    lengths = mask.to(torch.float32).sum(dim=1).clamp_min(1.0)
    centers = torch.arange(n_frames, device=mask.device, dtype=torch.float32) * float(hop_length)
    return centers.unsqueeze(0) < lengths.unsqueeze(1)


def _masked_sequence_mean_std(
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    eps: float,
    *,
    pooling: str = "meanstd",
) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected [B, T, C] sequence, got {tuple(x.shape)}")
    pooling = pooling.lower()
    if pooling not in {"mean", "meanstd"}:
        raise ValueError(f"Unsupported sequence pooling: {pooling}")
    if mask is None:
        mean = x.mean(dim=1)
        if pooling == "mean":
            return mean
        std = x.std(dim=1, unbiased=False).clamp_min(eps)
        return torch.cat([mean, std], dim=-1)
    m = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
    count = m.sum(dim=1).clamp_min(1.0)
    mean = (x * m).sum(dim=1) / count
    if pooling == "mean":
        return mean
    var = ((x - mean.unsqueeze(1)) * m).pow(2).sum(dim=1) / count
    std = var.clamp_min(eps).sqrt()
    return torch.cat([mean, std], dim=-1)


# --------------------------------------------------------------------------- #
# CTC posterior extractors (phonetic / content space)
# --------------------------------------------------------------------------- #
class CTCBlankStatsExtractor(BaseSRFDExtractor):
    """Frozen wav2vec2 CTC posterior statistics for length/blank behavior."""

    requires_waveform = True

    def __init__(self, config: SRFDExtractorConfig):
        super().__init__(config)
        try:
            from transformers import Wav2Vec2ForCTC
        except Exception as exc:  # pragma: no cover - dependency failure path
            raise ImportError("CTCBlankStatsExtractor requires transformers with Wav2Vec2ForCTC.") from exc

        self.model_path = str(config.extra.get("model_path", "models/wav2vec2-base-960h"))
        self.target_sample_rate = int(config.extra.get("target_sample_rate", 16000))
        self.input_sample_rate = int(config.extra.get("input_sample_rate", self.target_sample_rate))
        self.max_seconds = float(config.extra.get("max_seconds", 12.0))
        self.normalize_waveform = bool(config.extra.get("normalize_waveform", False))
        self.blank_token_id = config.extra.get("blank_token_id", None)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_path)
        self.model.eval()
        self.model.requires_grad_(False)
        if self.blank_token_id is None:
            self.blank_token_id = int(getattr(self.model.config, "pad_token_id", 0) or 0)
        else:
            self.blank_token_id = int(self.blank_token_id)

    def feature_dim(self) -> Optional[int]:
        return 8

    def _logit_mask(self, raw_mask: Optional[torch.Tensor], logit_len: int) -> Optional[torch.Tensor]:
        if raw_mask is None:
            return None
        lengths = raw_mask.to(torch.float32).sum(dim=1)
        new_lengths = torch.ceil(lengths * float(logit_len) / float(raw_mask.size(1))).to(torch.long)
        arange = torch.arange(logit_len, device=raw_mask.device).unsqueeze(0)
        return arange < new_lengths.clamp_min(1).unsqueeze(1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        wav = _resolve_waveform(batch, owner="CTCBlankStatsExtractor")
        sample_rate = _sample_rate_from_batch(batch, self.input_sample_rate)
        wav, wav_mask = _prepare_waveform(
            wav,
            batch.get("waveform_mask", None),
            sample_rate=sample_rate,
            target_sample_rate=self.target_sample_rate,
            max_seconds=self.max_seconds,
            normalize=self.normalize_waveform,
            eps=self.eps,
        )
        out = self.model(
            input_values=wav,
            attention_mask=wav_mask.to(torch.long) if wav_mask is not None else None,
        )
        probs = torch.softmax(out.logits.to(torch.float32), dim=-1)
        blank = probs[..., self.blank_token_id]
        nonblank = (1.0 - blank).clamp_min(self.eps)
        entropy = -(probs.clamp_min(self.eps) * probs.clamp_min(self.eps).log()).sum(dim=-1)
        mask = self._logit_mask(wav_mask, blank.size(1))
        m = torch.ones_like(blank) if mask is None else mask.to(blank.dtype)
        count = m.sum(dim=1).clamp_min(1.0)
        pos = _time_positions(blank.size(1), blank.device).unsqueeze(0)
        blank_mean = (blank * m).sum(dim=1) / count
        nonblank_mean = (nonblank * m).sum(dim=1) / count
        entropy_mean = (entropy * m).sum(dim=1) / count
        entropy_std = (((entropy - entropy_mean.unsqueeze(1)) * m).pow(2).sum(dim=1) / count).sqrt()
        nonblank_sum = (nonblank * m).sum(dim=1).clamp_min(self.eps)
        nonblank_centroid = (nonblank * m * pos).sum(dim=1) / nonblank_sum
        blank_tail = (blank * m * (pos >= 0.8).to(blank.dtype)).sum(dim=1) / (
            m * (pos >= 0.8).to(m.dtype)
        ).sum(dim=1).clamp_min(1.0)
        nonblank_max = (nonblank * m).max(dim=1).values
        valid_ratio = count / float(max(blank.size(1), 1))
        return torch.stack(
            [
                blank_mean,
                nonblank_mean,
                entropy_mean,
                entropy_std,
                nonblank_centroid,
                blank_tail,
                nonblank_max,
                valid_ratio,
            ],
            dim=-1,
        )


class CTCPosteriorContentStatsExtractor(CTCBlankStatsExtractor):
    """Frozen CTC posterior content statistics.

    ``CTCBlankStatsExtractor`` mainly tells SR-FD whether the generated speech
    has plausible blank/non-blank timing. This variant adds a deterministic
    random projection of non-blank posterior probabilities, so lexical/content
    substitutions are visible to the Fréchet term without exploding covariance
    dimensionality to the full CTC vocabulary size.
    """

    def __init__(self, config: SRFDExtractorConfig):
        super().__init__(config)
        self.projection_dim = int(config.extra.get("projection_dim", 64))
        self.projection_seed = int(config.extra.get("projection_seed", 260121386))
        self.pooling = str(config.extra.get("pooling", "meanstd")).lower()
        self.normalize_nonblank = bool(config.extra.get("normalize_nonblank", True))
        self.include_blank_stats = bool(config.extra.get("include_blank_stats", True))
        if self.pooling not in {"mean", "meanstd"}:
            raise ValueError(f"Unsupported CTC posterior pooling: {self.pooling}")

        vocab_size = int(getattr(self.model.config, "vocab_size", 0) or 0)
        if vocab_size <= 0 and getattr(self.model, "lm_head", None) is not None:
            vocab_size = int(self.model.lm_head.out_features)
        if vocab_size <= 0:
            raise ValueError("CTCPosteriorContentStatsExtractor could not infer vocab size.")
        out_dim = self.projection_dim if 0 < self.projection_dim < vocab_size else vocab_size
        if out_dim < vocab_size:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.projection_seed)
            proj = torch.randn(vocab_size, out_dim, generator=gen, dtype=torch.float32)
            proj = proj / math.sqrt(float(out_dim))
            self.register_buffer("posterior_projection", proj, persistent=False)
        else:
            self.register_buffer("posterior_projection", torch.empty(0), persistent=False)
        content_dim = int(out_dim) * (2 if self.pooling == "meanstd" else 1)
        self._feature_dim = content_dim + (8 if self.include_blank_stats else 0)

    def feature_dim(self) -> Optional[int]:
        return self._feature_dim

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        wav = _resolve_waveform(batch, owner="CTCPosteriorContentStatsExtractor")
        sample_rate = _sample_rate_from_batch(batch, self.input_sample_rate)
        wav, wav_mask = _prepare_waveform(
            wav,
            batch.get("waveform_mask", None),
            sample_rate=sample_rate,
            target_sample_rate=self.target_sample_rate,
            max_seconds=self.max_seconds,
            normalize=self.normalize_waveform,
            eps=self.eps,
        )
        out = self.model(
            input_values=wav,
            attention_mask=wav_mask.to(torch.long) if wav_mask is not None else None,
        )
        probs = torch.softmax(out.logits.to(torch.float32), dim=-1)
        mask = self._logit_mask(wav_mask, probs.size(1))

        content_probs = probs
        if 0 <= self.blank_token_id < probs.size(-1):
            blank_mask = torch.ones(probs.size(-1), device=probs.device, dtype=probs.dtype)
            blank_mask[self.blank_token_id] = 0.0
            content_probs = content_probs * blank_mask.view(1, 1, -1)
        if self.normalize_nonblank:
            content_probs = content_probs / content_probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        if self.posterior_projection.numel() > 0:
            content_seq = content_probs @ self.posterior_projection.to(
                device=content_probs.device,
                dtype=content_probs.dtype,
            )
        else:
            content_seq = content_probs
        pieces = [_masked_sequence_mean_std(content_seq, mask, self.eps, pooling=self.pooling)]

        if self.include_blank_stats:
            blank = probs[..., self.blank_token_id]
            nonblank = (1.0 - blank).clamp_min(self.eps)
            entropy = -(probs.clamp_min(self.eps) * probs.clamp_min(self.eps).log()).sum(dim=-1)
            m = torch.ones_like(blank) if mask is None else mask.to(blank.dtype)
            count = m.sum(dim=1).clamp_min(1.0)
            pos = _time_positions(blank.size(1), blank.device).unsqueeze(0)
            blank_mean = (blank * m).sum(dim=1) / count
            nonblank_mean = (nonblank * m).sum(dim=1) / count
            entropy_mean = (entropy * m).sum(dim=1) / count
            entropy_std = (((entropy - entropy_mean.unsqueeze(1)) * m).pow(2).sum(dim=1) / count).sqrt()
            nonblank_sum = (nonblank * m).sum(dim=1).clamp_min(self.eps)
            nonblank_centroid = (nonblank * m * pos).sum(dim=1) / nonblank_sum
            blank_tail = (blank * m * (pos >= 0.8).to(blank.dtype)).sum(dim=1) / (
                m * (pos >= 0.8).to(m.dtype)
            ).sum(dim=1).clamp_min(1.0)
            nonblank_max = (nonblank * m).max(dim=1).values
            valid_ratio = count / float(max(blank.size(1), 1))
            pieces.append(
                torch.stack(
                    [
                        blank_mean,
                        nonblank_mean,
                        entropy_mean,
                        entropy_std,
                        nonblank_centroid,
                        blank_tail,
                        nonblank_max,
                        valid_ratio,
                    ],
                    dim=-1,
                )
            )
        return torch.cat(pieces, dim=-1)


# --------------------------------------------------------------------------- #
# Whisper encoder extractors (semantic / ASR-sensitive space)
# --------------------------------------------------------------------------- #
class WhisperEncoderMeanStdExtractor(BaseSRFDExtractor):
    """Frozen Whisper encoder embedding extractor for ASR-sensitive SR-FD.

    Whisper encoder features are close to ASR content, so matching their
    utterance statistics gives SR-FD a content-sensitive signal while keeping
    the encoder frozen.
    """

    requires_waveform = True

    def __init__(self, config: SRFDExtractorConfig):
        super().__init__(config)
        try:
            from transformers import WhisperModel
        except Exception as exc:  # pragma: no cover - dependency failure path
            raise ImportError(
                "WhisperEncoderMeanStdExtractor requires transformers with WhisperModel."
            ) from exc
        try:
            import torchaudio.functional as AF
        except Exception as exc:  # pragma: no cover - dependency failure path
            raise ImportError("WhisperEncoderMeanStdExtractor requires torchaudio.") from exc

        self.model_path = str(config.extra.get("model_path", "models/whisper-large-v3"))
        self.target_sample_rate = int(config.extra.get("target_sample_rate", 16000))
        self.input_sample_rate = int(config.extra.get("input_sample_rate", self.target_sample_rate))
        self.layer_pool = str(config.extra.get("layer_pool", "last4")).lower()
        self.pooling = str(config.extra.get("pooling", "meanstd")).lower()
        self.projection_dim = int(config.extra.get("projection_dim", 128))
        self.projection_seed = int(config.extra.get("projection_seed", 260121386))
        self.max_seconds = float(config.extra.get("max_seconds", 8.0))
        self.normalize_waveform = bool(config.extra.get("normalize_waveform", False))
        self.n_fft = int(config.extra.get("n_fft", 400))
        self.hop_length = int(config.extra.get("hop_length", 160))
        self.win_length = int(config.extra.get("win_length", 400))
        self.f_min = float(config.extra.get("f_min", 0.0))
        self.f_max = float(config.extra.get("f_max", self.target_sample_rate / 2.0))
        self.mel_norm = config.extra.get("mel_norm", "slaney")
        self.mel_scale = str(config.extra.get("mel_scale", "slaney"))
        self.pad_to_max_source_positions = bool(config.extra.get("pad_to_max_source_positions", True))
        if self.layer_pool not in {"last4", "last", "all"}:
            raise ValueError(f"Unsupported Whisper layer_pool: {self.layer_pool}")
        if self.pooling not in {"mean", "meanstd"}:
            raise ValueError(f"Unsupported Whisper pooling: {self.pooling}")

        model = WhisperModel.from_pretrained(self.model_path)
        self.encoder = model.encoder
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        self.hidden_size = int(getattr(model.config, "d_model", 0) or getattr(model.config, "hidden_size", 0))
        self.n_mels = int(config.extra.get("n_mels", getattr(model.config, "num_mel_bins", 80)))
        self.max_source_positions = int(getattr(model.config, "max_source_positions", 1500))
        del model
        if self.hidden_size <= 0:
            raise ValueError("WhisperEncoderMeanStdExtractor could not infer hidden size.")

        n_freqs = self.n_fft // 2 + 1
        mel_filters = AF.melscale_fbanks(
            n_freqs=n_freqs,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            sample_rate=self.target_sample_rate,
            norm=self.mel_norm,
            mel_scale=self.mel_scale,
        )
        self.register_buffer("mel_filters", mel_filters.to(torch.float32), persistent=False)
        self.register_buffer("stft_window", torch.hann_window(self.win_length), persistent=False)

        base_dim = self.hidden_size * (2 if self.pooling == "meanstd" else 1)
        out_dim = self.projection_dim if 0 < self.projection_dim < base_dim else base_dim
        self._feature_dim = int(out_dim)
        if out_dim < base_dim:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.projection_seed)
            proj = torch.randn(base_dim, out_dim, generator=gen, dtype=torch.float32)
            proj = proj / math.sqrt(float(out_dim))
            self.register_buffer("projection_matrix", proj, persistent=False)
        else:
            self.register_buffer("projection_matrix", torch.empty(0), persistent=False)

    def feature_dim(self) -> Optional[int]:
        return self._feature_dim

    def _log_mel_features(self, wav: torch.Tensor) -> torch.Tensor:
        window = self.stft_window.to(device=wav.device, dtype=wav.dtype)
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        power = spec.abs().pow(2).to(torch.float32)
        mel_filters = self.mel_filters.to(device=power.device, dtype=power.dtype)
        mel = torch.einsum("bft,fm->bmt", power, mel_filters).clamp_min(1.0e-10)
        log_mel = torch.log10(mel)
        log_mel = torch.maximum(log_mel, log_mel.amax(dim=(1, 2), keepdim=True) - 8.0)
        log_mel = (log_mel + 4.0) / 4.0
        max_input_frames = max(self.max_source_positions * 2, 1)
        if log_mel.size(-1) > max_input_frames:
            log_mel = log_mel[..., :max_input_frames]
        elif self.pad_to_max_source_positions and log_mel.size(-1) < max_input_frames:
            log_mel = torch.nn.functional.pad(log_mel, (0, max_input_frames - log_mel.size(-1)))
        return log_mel

    def _select_hidden_states(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self.layer_pool == "last":
            return hidden_states[-1]
        if self.layer_pool == "all":
            selected = hidden_states[1:] if len(hidden_states) > 1 else hidden_states
        else:
            selected = hidden_states[-4:]
        return torch.stack(tuple(selected), dim=0).mean(dim=0)

    def _encode(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        wav = _resolve_waveform(batch, owner="WhisperEncoderMeanStdExtractor")
        sample_rate = _sample_rate_from_batch(batch, self.input_sample_rate)
        wav, wav_mask = _prepare_waveform(
            wav,
            batch.get("waveform_mask", None),
            sample_rate=sample_rate,
            target_sample_rate=self.target_sample_rate,
            max_seconds=self.max_seconds,
            normalize=self.normalize_waveform,
            eps=self.eps,
        )
        log_mel = self._log_mel_features(wav)
        frame_mask = _frame_mask_from_waveform_mask(wav_mask, log_mel.size(-1), self.hop_length)
        device_type = "cuda" if log_mel.is_cuda else "cpu"
        encoder_param = next(self.encoder.parameters(), None)
        encoder_dtype = encoder_param.dtype if encoder_param is not None else torch.float32
        with torch.autocast(device_type=device_type, enabled=False):
            out = self.encoder(
                input_features=log_mel.to(dtype=encoder_dtype),
                attention_mask=None,
                output_hidden_states=True,
            )
        hs = self._select_hidden_states(out.hidden_states).to(torch.float32)
        feature_mask = None
        if frame_mask is not None:
            feature_mask = _resample_mask_by_length(frame_mask, old_len=frame_mask.size(1), new_len=hs.size(1))
        return hs, feature_mask

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        hs, feature_mask = self._encode(batch)
        rep = _masked_sequence_mean_std(hs, feature_mask, self.eps, pooling=self.pooling)
        if self.projection_matrix.numel() > 0:
            rep = rep @ self.projection_matrix.to(device=rep.device, dtype=rep.dtype)
        return rep


class WhisperEncoderAnchorExtractor(WhisperEncoderMeanStdExtractor):
    """Whisper local-time anchor extractor for lexical order/content FD.

    Instead of one global mean/std, the encoder sequence is pooled at a set of
    soft time anchors spread across the utterance, so the representation keeps
    coarse lexical order. This is the Whisper extractor used in the paper.
    """

    def __init__(self, config: SRFDExtractorConfig):
        super().__init__(config)
        self.num_anchors = max(int(config.extra.get("num_anchors", 8)), 1)
        self.anchor_min = float(config.extra.get("anchor_min", 0.03))
        self.anchor_max = float(config.extra.get("anchor_max", 0.97))
        self.bandwidth = float(config.extra.get("bandwidth", 0.08))
        self.include_deltas = bool(config.extra.get("include_deltas", True))
        self.project_each_anchor = bool(config.extra.get("project_each_anchor", True))
        if not self.project_each_anchor and self.projection_dim > 0:
            raise ValueError(
                "WhisperEncoderAnchorExtractor with projection_dim requires project_each_anchor=true."
            )

        base_dim = self.hidden_size
        if self.project_each_anchor and 0 < self.projection_dim < base_dim:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.projection_seed)
            proj = torch.randn(base_dim, self.projection_dim, generator=gen, dtype=torch.float32)
            proj = proj / math.sqrt(float(self.projection_dim))
            self.projection_matrix = proj
            base_dim = int(self.projection_dim)
        else:
            self.projection_matrix = torch.empty(0)
        multiplier = self.num_anchors
        if self.include_deltas and self.num_anchors > 1:
            multiplier += self.num_anchors - 1
        self._feature_dim = int(base_dim * multiplier)

    def _anchor_pool(
        self,
        hs: torch.Tensor,
        feature_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, T, _C = hs.shape
        if feature_mask is None:
            valid = torch.ones(B, T, device=hs.device, dtype=torch.float32)
        else:
            valid = feature_mask.to(device=hs.device, dtype=torch.float32)

        ranks = valid.cumsum(dim=1) - 1.0
        denom = (valid.sum(dim=1, keepdim=True) - 1.0).clamp_min(1.0)
        rank01 = (ranks / denom).clamp(0.0, 1.0)

        if self.num_anchors == 1:
            anchors = hs.new_tensor([0.5], dtype=torch.float32)
        else:
            lo = min(max(self.anchor_min, 0.0), 1.0)
            hi = min(max(self.anchor_max, 0.0), 1.0)
            if hi < lo:
                lo, hi = hi, lo
            anchors = torch.linspace(lo, hi, self.num_anchors, device=hs.device, dtype=torch.float32)

        reps = []
        width = max(self.bandwidth, self.eps)
        for anchor in anchors:
            distance = (rank01 - anchor).abs()
            weights = (1.0 - distance / width).clamp_min(0.0) * valid
            empty = weights.sum(dim=1, keepdim=True) <= self.eps
            nearest = (distance + (1.0 - valid) * 10.0).argmin(dim=1)
            fallback = torch.zeros_like(weights)
            fallback.scatter_(1, nearest.unsqueeze(1), 1.0)
            weights = torch.where(empty, fallback, weights)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
            reps.append((hs * weights.unsqueeze(-1)).sum(dim=1))
        return torch.stack(reps, dim=1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        hs, feature_mask = self._encode(batch)
        if self.project_each_anchor and self.projection_matrix.numel() > 0:
            hs = hs @ self.projection_matrix.to(device=hs.device, dtype=hs.dtype)
        anchors = self._anchor_pool(hs, feature_mask)
        pieces = [anchors.reshape(anchors.size(0), -1)]
        if self.include_deltas and anchors.size(1) > 1:
            pieces.append((anchors[:, 1:] - anchors[:, :-1]).reshape(anchors.size(0), -1))
        return torch.cat(pieces, dim=-1)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_EXTRACTOR_REGISTRY = {
    "ctc_blank_stats": CTCBlankStatsExtractor,
    "ctc_posterior_content_stats": CTCPosteriorContentStatsExtractor,
    "ctc_content_stats": CTCPosteriorContentStatsExtractor,
    "whisper": WhisperEncoderMeanStdExtractor,
    "whisper_encoder_meanstd": WhisperEncoderMeanStdExtractor,
    "whisper_audio_meanstd": WhisperEncoderMeanStdExtractor,
    "whisper_encoder_anchor": WhisperEncoderAnchorExtractor,
    "whisper_sequence_anchor": WhisperEncoderAnchorExtractor,
    "whisper_anchor": WhisperEncoderAnchorExtractor,
}


def build_srfd_extractors(
    reps_config: List[Dict[str, Any]],
) -> List[BaseSRFDExtractor]:
    """Instantiate the extractors listed under ``srfd.reps`` in YAML.

    Disabled extractors are silently skipped. Unknown extractor types raise
    ``KeyError`` with a list of registered types so the message is actionable.
    """
    extractors: List[BaseSRFDExtractor] = []
    for entry in reps_config:
        cfg = SRFDExtractorConfig.from_dict(entry)
        if not cfg.enabled:
            continue
        if cfg.type not in _EXTRACTOR_REGISTRY:
            raise KeyError(
                f"Unknown SR-FD extractor type '{cfg.type}'. "
                f"Registered types: {sorted(_EXTRACTOR_REGISTRY.keys())}"
            )
        extractor_cls = _EXTRACTOR_REGISTRY[cfg.type]
        extractors.append(extractor_cls(cfg))
    return extractors
