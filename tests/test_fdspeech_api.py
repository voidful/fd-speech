"""Public FDSpeech API compatibility checks."""

from fdspeech import (
    FDSpeechExtractor,
    FDSpeechExtractorConfig,
    FDSpeechLoss,
    build_fdspeech_extractors,
)
from srfd import (
    BaseSRFDExtractor,
    SRFDEmaLoss,
    SRFDExtractorConfig,
    build_srfd_extractors,
)


def test_fdspeech_public_aliases_keep_checkpoint_api_compatible():
    assert FDSpeechLoss is SRFDEmaLoss
    assert FDSpeechExtractor is BaseSRFDExtractor
    assert FDSpeechExtractorConfig is SRFDExtractorConfig
    assert build_fdspeech_extractors is build_srfd_extractors
