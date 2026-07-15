"""Hugging Face Space for aligned SR-FD listening comparisons.

This Space intentionally serves the paper's checked-in audio instead of
loading three 2B-parameter model variants. Every column therefore uses the
same prompt and target, starts instantly on CPU hardware, and remains
reproducible across Space restarts.
"""

from __future__ import annotations

import difflib
import html
import json
import random
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
DEMO_ROOT = ROOT / "demo" / "site"
SAMPLES_PATH = DEMO_ROOT / "samples.json"
RESULTS_PATH = DEMO_ROOT / "results.json"

with SAMPLES_PATH.open(encoding="utf-8") as handle:
    DEMO_DATA = json.load(handle)
with RESULTS_PATH.open(encoding="utf-8") as handle:
    RESULTS_DATA = json.load(handle)

SAMPLES = DEMO_DATA["samples"]
SYSTEMS = {system["key"]: system for system in DEMO_DATA["systems"]}
CORE_KEYS = ("base4", "base10", "srfd3")


def _shorten(text: str, width: int = 74) -> str:
    return text if len(text) <= width else f"{text[: width - 1].rstrip()}…"


def _choice_label(index: int, sample: dict) -> str:
    marker = "Improvement" if sample["case"] == "positive" else "Remaining failure"
    return f"{index + 1:02d} · {marker} · {_shorten(sample['reference'])}"


CHOICES = [_choice_label(index, sample) for index, sample in enumerate(SAMPLES)]
CHOICE_TO_INDEX = {label: index for index, label in enumerate(CHOICES)}
NEGATIVE_INDICES = [index for index, sample in enumerate(SAMPLES) if sample["case"] == "negative"]


def _audio_path(relative_path: str) -> str:
    relative_path = relative_path.removeprefix("./")
    path = (DEMO_ROOT / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing demo audio: {path}")
    return str(path)


def _sample_system(sample: dict, key: str) -> dict:
    for system in sample["systems"]:
        if system["key"] == key:
            return system
    raise KeyError(f"Sample {sample['id']} has no system {key}")


def _word_diff(reference: str, hypothesis: str) -> str:
    """Render a compact, escaped word-level transcript difference."""

    ref_words = reference.split()
    hyp_words = hypothesis.split()
    matcher = difflib.SequenceMatcher(a=[word.lower() for word in ref_words], b=[word.lower() for word in hyp_words])
    rendered: list[str] = []
    for operation, ref_start, ref_end, hyp_start, hyp_end in matcher.get_opcodes():
        if operation == "equal":
            rendered.extend(html.escape(word) for word in hyp_words[hyp_start:hyp_end])
        elif operation in {"replace", "insert"}:
            phrase = " ".join(html.escape(word) for word in hyp_words[hyp_start:hyp_end])
            if phrase:
                rendered.append(f'<span class="word-error">{phrase}</span>')
        elif operation == "delete":
            phrase = " ".join(html.escape(word) for word in ref_words[ref_start:ref_end])
            rendered.append(f'<span class="word-missing">missing: {phrase}</span>')
    return " ".join(rendered)


def _transcript_card(sample: dict, key: str) -> str:
    output = _sample_system(sample, key)
    per_utterance_wer = float(output["wer"]) * 100.0
    status = "Exact ASR match" if per_utterance_wer == 0 else "ASR content error"
    status_class = "ok" if per_utterance_wer == 0 else "warn"
    return f"""
    <div class="transcript-card">
      <div class="transcript-meta">
        <span class="status-pill {status_class}">{status}</span>
        <span class="sample-wer">sample WER {per_utterance_wer:.1f}%</span>
      </div>
      <div class="transcript-label">ASR transcript</div>
      <div class="transcript-text">{_word_diff(sample['reference'], output['hyp'])}</div>
    </div>
    """


def _insight(sample: dict) -> str:
    wers = {key: float(_sample_system(sample, key)["wer"]) for key in CORE_KEYS}
    ours = wers["srfd3"]
    best_baseline = min(wers["base4"], wers["base10"])
    if ours == 0 and best_baseline > 0:
        title = "SR-FD recovers the target text"
        body = "The four-step SR-FD output has an exact ASR match while both original samplers retain a content error."
        tone = "positive"
    elif ours == 0 and wers["base4"] > 0:
        title = "SR-FD closes the four-to-ten-step gap"
        body = "SR-FD fixes the original four-step error and matches the stronger ten-step baseline on this prompt."
        tone = "positive"
    elif ours < best_baseline:
        title = "SR-FD reduces this prompt's error"
        body = "The SR-FD transcript has lower per-utterance WER than both original sampling settings."
        tone = "positive"
    elif ours > best_baseline:
        title = "A remaining SR-FD failure"
        body = "This deliberately surfaced negative case shows that aggregate WER improves, but individual prompts can regress."
        tone = "negative"
    else:
        title = "No transcript-level separation"
        body = "The systems have the same per-utterance WER here; listen for acoustic and prosodic differences instead."
        tone = "neutral"
    return f'<div class="insight {tone}"><strong>{title}</strong><span>{body}</span></div>'


def _case_meta(sample: dict, index: int) -> str:
    kind = "Improvement case" if sample["case"] == "positive" else "Negative case"
    kind_class = "positive" if sample["case"] == "positive" else "negative"
    return f"""
    <div class="case-meta">
      <span class="case-index">Example {index + 1} of {len(SAMPLES)}</span>
      <span class="case-pill {kind_class}">{kind}</span>
    </div>
    """


def _target_text(sample: dict) -> str:
    return f"""
    <div class="target-card">
      <div class="eyebrow">Target text</div>
      <div class="target-copy">{html.escape(sample['reference'])}</div>
    </div>
    """


def render_sample(choice: str):
    index = CHOICE_TO_INDEX.get(choice, 0)
    sample = SAMPLES[index]
    system_outputs = {key: _sample_system(sample, key) for key in CORE_KEYS}
    return (
        _case_meta(sample, index),
        _target_text(sample),
        _insight(sample),
        _audio_path(sample["prompt_audio"]),
        _audio_path(sample["target_audio"]),
        _audio_path(system_outputs["base4"]["audio"]),
        _transcript_card(sample, "base4"),
        _audio_path(system_outputs["base10"]["audio"]),
        _transcript_card(sample, "base10"),
        _audio_path(system_outputs["srfd3"]["audio"]),
        _transcript_card(sample, "srfd3"),
    )


def _navigate(choice: str, offset: int):
    current = CHOICE_TO_INDEX.get(choice, 0)
    next_index = (current + offset) % len(SAMPLES)
    next_choice = CHOICES[next_index]
    return (next_choice, *render_sample(next_choice))


def previous_sample(choice: str):
    return _navigate(choice, -1)


def next_sample(choice: str):
    return _navigate(choice, 1)


def random_sample(choice: str):
    current = CHOICE_TO_INDEX.get(choice, 0)
    candidates = [index for index in range(len(SAMPLES)) if index != current]
    next_choice = CHOICES[random.choice(candidates)]
    return (next_choice, *render_sample(next_choice))


def next_negative_sample(choice: str):
    current = CHOICE_TO_INDEX.get(choice, -1)
    next_index = next((index for index in NEGATIVE_INDICES if index > current), NEGATIVE_INDICES[0])
    next_choice = CHOICES[next_index]
    return (next_choice, *render_sample(next_choice))


def _system_header(key: str, accent: str, claim: str) -> str:
    system = SYSTEMS[key]
    return f"""
    <div class="system-head {accent}">
      <div class="eyebrow">{html.escape(claim)}</div>
      <h3>{html.escape(system['name'])}</h3>
      <div class="system-detail">{html.escape(system['detail'])}</div>
      <div class="global-wer">Full-set WER <strong>{html.escape(system['metrics']['wer'].split('=')[-1].strip())}</strong></div>
    </div>
    """


def _validate_assets() -> None:
    for sample in SAMPLES:
        _audio_path(sample["prompt_audio"])
        _audio_path(sample["target_audio"])
        for key in CORE_KEYS:
            _audio_path(_sample_system(sample, key)["audio"])


_validate_assets()
INITIAL = render_sample(CHOICES[0])

RESULTS_TABLE = """
<div class="results-wrap">
  <div class="results-caption">Seed-TTS English test-en · upstream scorer · 1,088 prompts</div>
  <table class="results-table">
    <thead><tr><th>System</th><th>Steps</th><th>Errors / words</th><th>WER ↓</th><th>SIM ↑</th><th>UTMOS ↑</th><th>DNSMOS OVRL ↑</th><th>P808 ↑</th></tr></thead>
    <tbody>
      <tr><td>VoxCPM2</td><td>4</td><td>263 / 11805</td><td>2.2279%</td><td>0.7433</td><td>3.2974</td><td>2.8950</td><td>3.5296</td></tr>
      <tr><td>VoxCPM2</td><td>10</td><td>205 / 11805</td><td>1.7366%</td><td>0.7610</td><td>3.8072</td><td>3.0866</td><td>3.6689</td></tr>
      <tr class="ours-row"><td>VoxCPM2 + SR-FD</td><td>4</td><td>167 / 11805</td><td><strong>1.4147%</strong></td><td>0.7613</td><td>3.7637</td><td>3.0711</td><td>3.6507</td></tr>
    </tbody>
  </table>
</div>
"""

HERO = """
<section class="hero">
  <div class="hero-kicker">Speech Representation Fréchet Distance</div>
  <h1>Hear what changes at four steps.</h1>
  <p>Aligned Seed-TTS English comparisons between the original VoxCPM2 at 4 and 10 steps and our SR-FD model at 4 steps.</p>
  <div class="hero-links">
    <a href="https://arxiv.org/abs/2607.06027" target="_blank">Paper ↗</a>
    <a href="https://github.com/voidful/fd-speech" target="_blank">Code ↗</a>
    <a href="https://huggingface.co/openbmb/VoxCPM2" target="_blank">Base model ↗</a>
  </div>
  <div class="hero-stats">
    <div><strong>1.4147%</strong><span>SR-FD 4-step WER</span></div>
    <div><strong>−36.5%</strong><span>relative vs. base 4-step</span></div>
    <div><strong>−18.5%</strong><span>relative vs. base 10-step</span></div>
    <div><strong>0</strong><span>added inference-time modules</span></div>
  </div>
</section>
"""

CSS = """
:root {
  --ink: #172033;
  --muted: #64748b;
  --line: #dfe5ef;
  --panel: #ffffff;
  --soft: #f5f7fb;
  --base4: #e96b4c;
  --base10: #6b7c93;
  --ours: #5a54d6;
}
.gradio-container { max-width: 1500px !important; margin: 0 auto !important; }
.hero { color: white; border-radius: 26px; padding: 42px 46px 34px; margin: 8px 0 22px; background: radial-gradient(circle at 85% 15%, rgba(123, 116, 255, .7), transparent 32%), linear-gradient(135deg, #111a32, #342e79 65%, #554bc2); box-shadow: 0 24px 55px rgba(38, 35, 97, .24); }
.hero-kicker, .eyebrow { text-transform: uppercase; letter-spacing: .12em; font-size: 11px; font-weight: 800; }
.hero-kicker { color: #c9c6ff; }
.hero h1 { color: white; font-size: clamp(36px, 5vw, 68px); line-height: .98; letter-spacing: -.045em; margin: 11px 0 16px; max-width: 900px; }
.hero p { color: #e8e8ff; max-width: 830px; font-size: 17px; line-height: 1.55; }
.hero-links { display: flex; gap: 10px; flex-wrap: wrap; margin: 22px 0 28px; }
.hero-links a { color: white !important; border: 1px solid rgba(255,255,255,.3); background: rgba(255,255,255,.09); border-radius: 999px; padding: 8px 14px; text-decoration: none !important; font-weight: 700; }
.hero-stats { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 10px; }
.hero-stats div { background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.14); border-radius: 15px; padding: 15px; }
.hero-stats strong, .hero-stats span { display: block; }
.hero-stats strong { font-size: 23px; color: white; }
.hero-stats span { color: #d9daf4; font-size: 12px; margin-top: 3px; }
.section-intro { margin: 8px 0 16px; color: var(--muted); }
.case-meta { display: flex; justify-content: space-between; align-items: center; margin: 2px 0 10px; }
.case-index { color: var(--muted); font-size: 12px; font-weight: 700; }
.case-pill, .status-pill { font-size: 11px; padding: 5px 9px; border-radius: 999px; font-weight: 800; }
.case-pill.positive, .status-pill.ok { color: #087a55; background: #dcf8ec; }
.case-pill.negative, .status-pill.warn { color: #b64732; background: #ffebe6; }
.target-card { border: 1px solid var(--line); border-radius: 16px; background: var(--panel); padding: 17px; margin-bottom: 10px; }
.target-card .eyebrow { color: var(--muted); }
.target-copy { color: var(--ink); font-size: 18px; line-height: 1.45; font-weight: 650; margin-top: 8px; }
.insight { display: flex; flex-direction: column; gap: 4px; border-radius: 14px; padding: 14px 15px; margin: 9px 0 13px; border-left: 4px solid; }
.insight.positive { background: #ecfbf5; border-color: #20a879; }
.insight.negative { background: #fff0ec; border-color: #df6048; }
.insight.neutral { background: #f1f4f9; border-color: #7b8798; }
.insight span { color: #4b596c; font-size: 13px; line-height: 1.45; }
.insight strong { color: var(--ink); }
.system-head { border-radius: 17px; padding: 17px 18px; color: white; min-height: 145px; }
.system-head.base4 { background: linear-gradient(135deg, #bd4d37, var(--base4)); }
.system-head.base10 { background: linear-gradient(135deg, #405069, var(--base10)); }
.system-head.ours { background: linear-gradient(135deg, #3d379d, var(--ours)); box-shadow: 0 13px 28px rgba(90,84,214,.22); }
.system-head h3 { color: white; margin: 8px 0 2px; font-size: 21px; }
.system-head .eyebrow { color: rgba(255,255,255,.72); }
.system-detail { color: rgba(255,255,255,.85); font-size: 13px; }
.global-wer { margin-top: 13px; font-size: 12px; color: rgba(255,255,255,.78); }
.global-wer strong { color: white; font-size: 19px; margin-left: 4px; }
.transcript-card { min-height: 154px; border-radius: 15px; padding: 14px 15px; border: 1px solid var(--line); background: var(--panel); }
.transcript-meta { display: flex; gap: 8px; align-items: center; justify-content: space-between; }
.sample-wer { color: var(--muted); font-size: 11px; font-weight: 750; }
.transcript-label { color: var(--muted); font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .09em; margin: 15px 0 6px; }
.transcript-text { color: var(--ink); line-height: 1.6; font-size: 14px; }
.word-error { color: #b63f2c; background: #ffe2dc; border-radius: 4px; padding: 1px 3px; font-weight: 750; }
.word-missing { color: #8e4c00; background: #fff0c7; border-radius: 4px; padding: 1px 3px; font-size: 12px; font-style: italic; }
.metric-band { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 12px; margin: 10px 0 22px; }
.metric-band div { border: 1px solid var(--line); border-radius: 17px; padding: 19px; background: white; }
.metric-band strong { display: block; font-size: 27px; color: var(--ink); }
.metric-band span { color: var(--muted); font-size: 13px; }
.results-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 17px; background: white; }
.results-caption { padding: 15px 17px; color: var(--muted); font-size: 12px; font-weight: 750; border-bottom: 1px solid var(--line); }
.results-table { width: 100%; border-collapse: collapse; min-width: 850px; color: #172033; }
.results-table th, .results-table td { padding: 13px 15px; text-align: left; border-bottom: 1px solid var(--line); font-size: 13px; }
.results-table th { background: var(--soft); color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
.results-table tbody td { background: #ffffff; color: #172033 !important; }
.results-table tbody tr:last-child td { border-bottom: 0; }
.results-table tbody .ours-row td { background: #f0efff; color: #29235f !important; }
.results-table tbody .ours-row strong { color: #4038b3 !important; }
.method-flow { display: grid; grid-template-columns: 1fr auto 1fr auto 1fr; align-items: center; gap: 12px; margin: 20px 0; }
.method-flow .node { padding: 19px; border-radius: 16px; background: white; border: 1px solid var(--line); text-align: center; }
.method-flow .arrow { color: var(--ours); font-size: 25px; font-weight: 900; }
.footnote { color: var(--muted); font-size: 12px; line-height: 1.55; }
@media (max-width: 800px) {
  .hero { padding: 30px 23px 24px; border-radius: 20px; }
  .hero-stats, .metric-band { grid-template-columns: repeat(2,minmax(0,1fr)); }
  .method-flow { grid-template-columns: 1fr; }
  .method-flow .arrow { transform: rotate(90deg); text-align: center; }
}
"""

THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")

with gr.Blocks(
    title="SR-FD · Four-step TTS comparison",
    analytics_enabled=False,
) as demo:
    gr.HTML(HERO)

    with gr.Tabs():
        with gr.Tab("Listen & compare", id="compare"):
            gr.HTML(
                '<p class="section-intro"><strong>One prompt, three matched outputs.</strong> '
                "Changed words in the ASR transcript are highlighted. Use the negative-case button to inspect remaining failures.</p>"
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=320):
                    picker = gr.Dropdown(
                        choices=CHOICES,
                        value=CHOICES[0],
                        label="Listening example",
                        interactive=True,
                    )
                    with gr.Row():
                        previous_button = gr.Button("← Previous", size="sm", min_width=0)
                        random_button = gr.Button("Random", size="sm", min_width=0)
                        next_button = gr.Button("Next →", size="sm", min_width=0)
                    negative_button = gr.Button("Show next negative case", variant="secondary")
                    case_meta = gr.HTML(INITIAL[0])
                    target_text = gr.HTML(INITIAL[1])
                    insight = gr.HTML(INITIAL[2])
                    with gr.Accordion("Prompt and reference audio", open=False):
                        prompt_audio = gr.Audio(value=INITIAL[3], label="Voice prompt", interactive=False)
                        target_audio = gr.Audio(value=INITIAL[4], label="Reference target", interactive=False)

                with gr.Column(scale=3, min_width=700):
                    with gr.Row(equal_height=False):
                        with gr.Column(min_width=235):
                            gr.HTML(_system_header("base4", "base4", "Original low-step sampler"))
                            base4_audio = gr.Audio(value=INITIAL[5], label="Base · 4 steps", interactive=False)
                            base4_transcript = gr.HTML(INITIAL[6])
                        with gr.Column(min_width=235):
                            gr.HTML(_system_header("base10", "base10", "Original stronger sampler"))
                            base10_audio = gr.Audio(value=INITIAL[7], label="Base · 10 steps", interactive=False)
                            base10_transcript = gr.HTML(INITIAL[8])
                        with gr.Column(min_width=235):
                            gr.HTML(_system_header("srfd3", "ours", "Ours · same four-step budget"))
                            srfd_audio = gr.Audio(value=INITIAL[9], label="SR-FD · 4 steps", interactive=False)
                            srfd_transcript = gr.HTML(INITIAL[10])

            compare_outputs = [
                case_meta,
                target_text,
                insight,
                prompt_audio,
                target_audio,
                base4_audio,
                base4_transcript,
                base10_audio,
                base10_transcript,
                srfd_audio,
                srfd_transcript,
            ]
            navigation_outputs = [picker, *compare_outputs]
            picker.change(render_sample, inputs=picker, outputs=compare_outputs)
            previous_button.click(previous_sample, inputs=picker, outputs=navigation_outputs)
            next_button.click(next_sample, inputs=picker, outputs=navigation_outputs)
            random_button.click(random_sample, inputs=picker, outputs=navigation_outputs)
            negative_button.click(next_negative_sample, inputs=picker, outputs=navigation_outputs)

        with gr.Tab("Benchmark", id="benchmark"):
            gr.HTML(
                """
                <div class="metric-band">
                  <div><strong>2.2279%</strong><span>Original VoxCPM2 · 4 steps</span></div>
                  <div><strong>1.7366%</strong><span>Original VoxCPM2 · 10 steps</span></div>
                  <div><strong>1.4147%</strong><span>VoxCPM2 + SR-FD · 4 steps</span></div>
                </div>
                """
            )
            gr.HTML(RESULTS_TABLE)
            gr.Markdown(
                """
### What the aggregate result says

- Four-step SR-FD reduces WER by **36.5% relative** to the original four-step sampler and **18.5% relative** to the original ten-step sampler.
- Both WER differences are supported by utterance-level paired bootstrap tests.
- Speaker similarity and objective quality proxies return to approximately the ten-step level.
- In a blinded test with 13 listeners and 229 judgments, the decisive split was 61 vs. 67; equivalence was supported within the pre-specified 10-point margin.

SIM, UTMOS, and DNSMOS are objective proxies, not human MOS. Reported values and claims follow [arXiv:2607.06027](https://arxiv.org/abs/2607.06027).
                """
            )

        with gr.Tab("How SR-FD works", id="method"):
            gr.HTML(
                """
                <div class="method-flow">
                  <div class="node"><strong>Four-step sampler</strong><br><span>VoxCPM2 + trainable LoRA</span></div>
                  <div class="arrow">→</div>
                  <div class="node"><strong>Frozen content encoders</strong><br><span>Whisper + wav2vec 2.0 CTC</span></div>
                  <div class="arrow">→</div>
                  <div class="node"><strong>Fréchet loss</strong><br><span>Match mean and covariance to 3 targets</span></div>
                </div>
                """
            )
            gr.Markdown(
                """
SR-FD matches complete **sampled speech**, using the same four-step sampler used at deployment. The reference mixture combines:

1. an ASR-verified four-step **Whisper anchor**,
2. a ten-step teacher **CTC target**, and
3. a real-speech **CTC target**.

The extractors, feature queue, reference moments, and Fréchet computation are training-only. The deployed system is still VoxCPM2 plus a LoRA adapter, with no extra inference-time module.

This demo uses checked-in, aligned audio rather than live synthesis. That design keeps the input prompt and evaluation seed fixed, starts on CPU Spaces, and makes every comparison repeatable. It also exposes negative cases instead of presenting only successes.
                """
            )

        with gr.Tab("Model & citation", id="model"):
            gr.Markdown(
                """
### Released artifact

The repository includes the selected compact three-target LoRA adapter at `demo/model/`. It is **not** a standalone model and must be loaded with [`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2).

```python
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    load_denoiser=False,
    lora_weights_path="demo/model",
)
wav = model.generate(
    text="The quick brown fox jumps over the lazy dog.",
    cfg_value=2.35,
    inference_timesteps=4,
    seed=0,
)
```

### Citation

```bibtex
@article{chung2026srfd,
  title   = {Fr\\'{e}chet Distance Loss on Speech Representations for Text-to-Speech Synthesis},
  author  = {Chung, Ho-Lam and Huang, Kuan-Po and Lu, Bo-Ru and Lee, Hung-yi},
  journal = {arXiv preprint arXiv:2607.06027},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.06027}
}
```

<span class="footnote">Use only consented voices. Do not use this work for impersonation, fraud, voice-print bypass, or non-consensual content.</span>
                """
            )

if __name__ == "__main__":
    demo.queue(max_size=32).launch(theme=THEME, css=CSS, ssr_mode=False)
