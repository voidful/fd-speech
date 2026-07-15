"""Canonical public API for FDSpeech.

The implementation remains in :mod:`srfd` so existing checkpoints, configs,
training logs, and imports continue to work. New integrations should import
the FDSpeech aliases from this module.
"""

from srfd import *  # noqa: F401,F403
from srfd import (
    BaseSRFDExtractor,
    SRFDEmaLoss,
    SRFDExtractorConfig,
    __all__ as _COMPAT_ALL,
    build_srfd_extractors,
)

FDSpeechLoss = SRFDEmaLoss
FDSpeechExtractor = BaseSRFDExtractor
FDSpeechExtractorConfig = SRFDExtractorConfig
build_fdspeech_extractors = build_srfd_extractors

__all__ = [
    *_COMPAT_ALL,
    "FDSpeechLoss",
    "FDSpeechExtractor",
    "FDSpeechExtractorConfig",
    "build_fdspeech_extractors",
]
