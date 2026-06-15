"""Speech Representation Fréchet Distance (SR-FD) loss.

SR-FD is a training-time distributional regularizer for tokenizer-free
few-step flow-matching TTS. During fine-tuning the model synthesizes speech
with the same few-step sampler used at deployment; frozen Whisper and CTC
encoders map that speech to feature vectors whose mean and covariance are
matched to offline reference statistics via a differentiable Fréchet distance.
The loss needs no discriminator, adds no parameters, and is removed at test
time, so inference is unchanged.

Building blocks:

* ``moments`` / ``frechet`` — differentiable moment and Fréchet-distance ops.
* ``extractors`` — frozen content extractors: a Whisper encoder anchor for the
  semantic space and a wav2vec2 CTC head for the phonetic space.
* ``loss`` — ``SRFDEmaLoss``, the trainer-side auxiliary loss with an EMA /
  feature-queue estimate of the generated distribution.
* ``stats_io`` — save/load precomputed reference statistics.

The package is import-safe even when optional dependencies (e.g. transformers,
torchaudio) are missing; only the extractors that need them fail at
construction.
"""

from .moments import (
    batch_mean_and_second_moment,
    covariance_from_mean_and_second_moment,
    masked_time_mean_std,
    accumulate_moments,
    finalize_accumulated_moments,
)
from .frechet import frechet_distance, trace_sqrt_product_symmetric
from .extractors import (
    BaseSRFDExtractor,
    SRFDExtractorConfig,
    CTCBlankStatsExtractor,
    CTCPosteriorContentStatsExtractor,
    WhisperEncoderMeanStdExtractor,
    WhisperEncoderAnchorExtractor,
    build_srfd_extractors,
)
from .conditions import build_condition_keys_from_batch, build_condition_key
from .loss import SRFDEmaLoss
from .stats_io import load_stats, save_stats

__all__ = [
    "batch_mean_and_second_moment",
    "covariance_from_mean_and_second_moment",
    "masked_time_mean_std",
    "accumulate_moments",
    "finalize_accumulated_moments",
    "frechet_distance",
    "trace_sqrt_product_symmetric",
    "BaseSRFDExtractor",
    "SRFDExtractorConfig",
    "CTCBlankStatsExtractor",
    "CTCPosteriorContentStatsExtractor",
    "WhisperEncoderMeanStdExtractor",
    "WhisperEncoderAnchorExtractor",
    "build_srfd_extractors",
    "build_condition_key",
    "build_condition_keys_from_batch",
    "SRFDEmaLoss",
    "load_stats",
    "save_stats",
]
