"""Condition keys for conditional FDSpeech/CFS-FD statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence


DEFAULT_CONDITION_KEYS = ("language", "prompt_mode", "text_len_bucket")


@dataclass(frozen=True)
class ConditionKey:
    language: str = "unk"
    prompt_mode: str = "zero_shot"
    text_len_bucket: str = "unk"
    duration_bucket: str = "unk"
    speaker_bucket: Optional[str] = None
    nfe: Optional[int] = None

    def encode(self, fields: Sequence[str] = DEFAULT_CONDITION_KEYS, include_nfe: bool = False) -> str:
        values: List[str] = []
        for field in fields:
            value = getattr(self, field)
            if value is None:
                value = "none"
            values.append(f"{field}={_clean(str(value))}")
        if include_nfe and self.nfe is not None:
            values.append(f"nfe={int(self.nfe)}")
        return "|".join(values) if values else "global"


def _clean(value: str) -> str:
    return value.replace("|", "_").replace("=", "-").replace("/", "_").strip() or "unk"


def text_len_bucket(n_tokens: int, short_max: int = 40, medium_max: int = 90) -> str:
    if n_tokens <= short_max:
        return "short"
    if n_tokens <= medium_max:
        return "medium"
    return "long"


def duration_bucket(duration_seconds: Optional[float], short_max: float = 3.0, medium_max: float = 8.0) -> str:
    if duration_seconds is None:
        return "unk"
    try:
        duration_value = float(duration_seconds)
    except (TypeError, ValueError):
        return "unk"
    if duration_value <= 0.0:
        return "unk"
    if duration_value <= short_max:
        return "short"
    if duration_value <= medium_max:
        return "medium"
    return "long"


def _to_list(value: Any) -> Any:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu()
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    return value


def _infer_len(value: Any) -> int:
    value = _to_list(value)
    if isinstance(value, int):
        return 1
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return len(value)
        return len(value)
    return 0


def _get_list(meta: Mapping[str, Any], key: str, n: int, default: Any) -> List[Any]:
    value = _to_list(meta.get(key, default))
    if isinstance(value, (list, tuple)):
        if len(value) == n:
            return list(value)
        if len(value) == 1:
            return [value[0]] * n
    return [value] * n


def _duration_list_from_meta(
    meta: Mapping[str, Any],
    n: int,
    *,
    frames_per_second: float = 25.0,
) -> List[Optional[float]]:
    for key in ("target_durations", "audio_durations", "duration_seconds", "durations"):
        if key in meta:
            values = _get_list(meta, key, n, None)
            return [None if v is None else float(v) for v in values]

    loss_mask = meta.get("loss_mask")
    if loss_mask is None:
        return [None] * n
    loss_mask = _to_list(loss_mask)
    if not isinstance(loss_mask, (list, tuple)):
        return [None] * n
    durations: List[Optional[float]] = []
    fps = float(meta.get("duration_frames_per_second", meta.get("feature_fps", frames_per_second)))
    fps = fps if fps > 0.0 else frames_per_second
    for row in loss_mask[:n]:
        if isinstance(row, (list, tuple)):
            valid_frames = sum(1.0 for value in row if float(value) > 0.0)
            durations.append(valid_frames / fps)
        else:
            try:
                durations.append(float(row) / fps)
            except (TypeError, ValueError):
                durations.append(None)
    if len(durations) < n:
        durations.extend([None] * (n - len(durations)))
    return durations


def build_condition_key(
    *,
    language: str = "en",
    prompt_mode: str = "zero_shot",
    text_length: int = 0,
    duration_seconds: Optional[float] = None,
    speaker_id: str = "",
    nfe: Optional[int] = None,
    fields: Sequence[str] = DEFAULT_CONDITION_KEYS,
    short_max: int = 40,
    medium_max: int = 90,
    duration_short_max: float = 3.0,
    duration_medium_max: float = 8.0,
) -> str:
    key = ConditionKey(
        language=language or "unk",
        prompt_mode=prompt_mode or "zero_shot",
        text_len_bucket=text_len_bucket(int(text_length), short_max=short_max, medium_max=medium_max),
        duration_bucket=duration_bucket(
            duration_seconds,
            short_max=duration_short_max,
            medium_max=duration_medium_max,
        ),
        speaker_bucket=(str(speaker_id) if speaker_id else None),
        nfe=nfe,
    )
    return key.encode(fields=fields, include_nfe=False)


def build_condition_keys_from_batch(
    meta: Mapping[str, Any],
    *,
    nfe: Optional[int] = None,
    fields: Sequence[str] = DEFAULT_CONDITION_KEYS,
    short_max: int = 40,
    medium_max: int = 90,
    duration_short_max: float = 3.0,
    duration_medium_max: float = 8.0,
    duration_frames_per_second: float = 25.0,
) -> List[str]:
    text_lengths = meta.get("text_token_lengths", None)
    n = _infer_len(text_lengths)
    if n == 0:
        text_lengths = meta.get("text_lengths", [])
        n = _infer_len(text_lengths)
    if n == 0:
        return []

    languages = _get_list(meta, "languages", n, "en")
    prompt_modes = _get_list(meta, "prompt_modes", n, "zero_shot")
    speaker_ids = _get_list(meta, "speaker_ids", n, "")
    lengths = _get_list(meta, "text_token_lengths", n, 0)
    if not any(lengths):
        lengths = _get_list(meta, "text_lengths", n, 0)
    durations = _duration_list_from_meta(
        meta,
        n,
        frames_per_second=duration_frames_per_second,
    )

    return [
        build_condition_key(
            language=str(languages[i]),
            prompt_mode=str(prompt_modes[i]),
            text_length=int(lengths[i]),
            duration_seconds=durations[i],
            speaker_id=str(speaker_ids[i]),
            nfe=nfe,
            fields=fields,
            short_max=short_max,
            medium_max=medium_max,
            duration_short_max=duration_short_max,
            duration_medium_max=duration_medium_max,
        )
        for i in range(n)
    ]


def unique_condition_keys(keys: Iterable[str]) -> List[str]:
    return sorted({k for k in keys if k})
