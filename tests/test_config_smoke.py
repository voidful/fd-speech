"""Smoke test: the paper config parses and references known extractor types.

Does not instantiate the extractors (which would download Whisper / wav2vec2);
it only checks that the YAML is well-formed and every ``srfd.reps`` type is in
the registry, with the paper's three reference targets and hyperparameters.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from srfd.extractors import SRFDExtractorConfig, _EXTRACTOR_REGISTRY  # noqa: E402

CONFIG = PROJECT_ROOT / "configs" / "srfd_compact3.yaml"


def _load():
    import yaml

    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_config_parses():
    cfg = _load()
    assert isinstance(cfg, dict)
    assert cfg["srfd"]["enabled"] is True


def test_paper_hyperparameters():
    cfg = _load()
    assert cfg["lambdas"]["loss/srfd"] == 2.0e-4
    assert cfg["lambdas"]["loss/diff"] == 0.006
    assert cfg["lambdas"]["loss/stop"] == 0.08
    assert cfg["lora"]["r"] == 32 and cfg["lora"]["alpha"] == 32
    assert cfg["srfd"]["queue_size"] == 50000
    assert cfg["srfd"]["length_gate"]["min_ratio"] == 0.92
    assert cfg["srfd"]["length_gate"]["max_ratio"] == 1.08
    assert cfg["sampler"]["n_timesteps"] == 4


def test_three_reference_targets():
    cfg = _load()
    targets = cfg["srfd"]["reference_stats_paths"]
    names = [t["name"] for t in targets]
    assert names == ["asr_true4_good_whisper", "teacher_t10_ctc_content", "real_ctc_content"]
    weights = [t["weight"] for t in targets]
    assert weights == [1.0, 0.5, 0.5]


def test_extractor_types_are_registered():
    cfg = _load()
    reps = cfg["srfd"]["reps"]
    types = {r["type"] for r in reps}
    assert types == {"whisper_encoder_anchor", "ctc_content_stats"}
    for r in reps:
        c = SRFDExtractorConfig.from_dict(r)
        assert c.type in _EXTRACTOR_REGISTRY
