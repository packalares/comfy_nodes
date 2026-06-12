"""Omni Audio Video Scenes (Director) — deterministic-time edition.

Design (v2 — corrected after the v1 model-writes-times disaster):

  Python computes ALL the times. The model never writes a number.
  The model only writes creative content per pre-numbered slot.

Flow:
  1. Pass A (1 GGUF call) — produces the initial treatment paragraph
     using the song's first 30 s + analysis.

  2. For each 30 s audio chunk (8 chunks for a 4-min song):
       a) Python pre-computes the exact shot windows for THIS chunk
          based on shot_seconds + chunk_seconds.
          e.g. chunk 0 (0-30s) with shot_seconds=10 → 3 shots:
            shot 1: 0.00–10.00, shot 2: 10.00–20.00, shot 3: 20.00–30.00
       b) ONE GGUF call sends: analysis + chunk audio + current treatment
          + the pre-numbered slot block.
       c) Model returns a single JSON object:
            {
              "shots": [ {description, image_prompt, video_prompt,
                          key_visual}, … ],
              "updated_treatment": "<new rolling memory>"
            }
          NO times in the model output. Python stamps them.
       d) Python pairs each model shot with its slot, fills missing slots
          with placeholders, and updates the running treatment.

Total GGUF calls = 1 + num_chunks (≈9 for a 4-min song).

Per-shot output schema:
  { scene_id, start, end, description, image_prompt, video_prompt,
    key_visual, chunk_idx }
"""
import gc
import json
import math
import os
import re
import tempfile

import folder_paths
import numpy as np
import soundfile as sf
import torch

from .nodes_analyze import (
    NO_MODEL_SENTINEL,
    _audio_to_temp_wav,
    _free_vram,
    _list_gguf_models,
    _list_mmproj_files,
    _run_captioner_gguf,
    _safe_format,
)
from .nodes_prompts import get_prompt


_AUDIO_CHUNK_SECONDS_MAX = 30.0


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_DEFAULT_TREATMENT_PROMPT = """You are a senior music-video director writing the PROJECT BIBLE for a new
music video. This treatment is the visual law of the film: every shot we
later generate must obey it. Future LLM calls will read THIS document to
stay on-style.

Write the treatment in MARKDOWN. It must commit, in concrete terms, to:

## Concept
The one-sentence pitch + the emotional through-line.

## Visual Style
Specific cinematography choices: lens/format (e.g. anamorphic 35 mm, T2.8,
shallow depth of field), grain, motion blur, camera handling
(handheld / steadicam / locked), grade (e.g. teal-amber, bleach-bypass,
pushed reds). Reference a film or DP if it helps.

## Color Palette
Name the 4-6 dominant colors AND where they appear (light source, costume,
environment). Be specific: "neon-red key from practical signs" not "warm
lighting".

## Character(s)
Each lead: clothing (specific garment, color, condition), hair, makeup,
build, expression, attitude. Background extras only if recurring.

## Setting
The primary location and 1-2 secondary spaces. Architecture, materials,
era, weather, time of day. Specific objects we should expect to see again
(a glass of amber liquor, a neon sign, a curtain).

## Mood Arc
How the visual energy maps to the song's structure (intro → outro), in
3-4 bullet beats.

## Recurring Visual Motifs
2-4 objects, colors, or compositions that should reappear across shots
to anchor continuity.

## What's been generated
*(empty for now — future chunks will append here)*

---

# Audio analysis
{analysis_block}

# Style hint (free-form, from the user / project)
{style_hint}

Now write the treatment markdown. Be specific. Director-grade prose,
not a list of clichés."""


_DEFAULT_CHUNK_SCENES_PROMPT = """You are a cinematographer + art director writing shot briefs for a
music-video pipeline. Each `image_prompt` you produce feeds a still-image
diffusion model (Flux / SDXL / Qwen-Image class) DIRECTLY. Each
`video_prompt` then feeds an image-to-video model (Wan 2.1 / LTX-Video
class) DIRECTLY. Treat them as the final prompts — not drafts.

The system pre-computes shot times. You write ONLY creative content.

# CURRENT TREATMENT (project bible — obey it; future chunks inherit your update)
{treatment}

# SONG-LEVEL AUDIO ANALYSIS
{analysis_json}

# STYLE HINT (user override; may be empty)
{style_hint}

# THIS CHUNK
Chunk {chunk_idx_label} of {num_chunks}, covering song seconds {chunk_start:.2f}–{chunk_end:.2f}.

{position_clause}

# SHOTS TO FILL — write exactly {num_shots} shot objects, in this order
{slot_block}

================================================================
HOW TO WRITE image_prompt  (60–100 words, natural-language sentences)
================================================================
Fill these SIX SLOTS in order. Skipping any one makes the model improvise.

  (1) FRAME + SUBJECT   — shot size + subject appearance (age, clothing item by item, hair, expression, position in frame)
  (2) ENVIRONMENT       — specific location + time of day + named props the viewer should see
  (3) LIGHTING          — single key source + direction + quality (hard / soft / practical), fill ratio
  (4) COLOR + GRADE     — 2-3 named dominant colors + grade reference (teal-amber, bleach-bypass, golden halation, etc.)
  (5) LENS + DOF        — focal length + aperture / depth-of-field commitment (35mm anamorphic at T2, shallow DOF)
  (6) MOOD / ATMOSPHERE — fog / rain / dust / haze / silence-before-motion type cue

❌ NEVER USE in image_prompt:
- Narrative verbs: "the chorus erupts", "the music swells", "feels lonely"
- Quality boosters: "masterpiece, 8K, ultra detailed, best quality, award-winning"
- Comma-tag lists (modern T5/Qwen-based diffusion ignores them; write sentences)
- Contradictions: "warm cool-toned", "sharp soft focus"
- (weight:1.3) syntax — these models ignore it

✅ GOOD image_prompt EXAMPLE (~85 words):
"Medium close-up of a man in his late 20s, sharp black blazer over a white tee, slicked-back dark hair, tired but defiant eyes, center frame. Standing under a flickering pink neon sign in a fog-filled nightclub alcove, velvet drapes faintly visible behind. Single hard practical neon from screen left, deep ambient blue fill stage-right, no top light. Saturated magenta highlights, deep teal shadows, bleach-bypass grade. 35mm anamorphic at T2, shallow DOF blurring the drapes. Charged stillness, faint cigarette smoke drifting through the frame."

================================================================
HOW TO WRITE video_prompt  (70–120 words, natural-language sentences)
================================================================
The first frame of the video is ALREADY described by image_prompt — don't
re-describe it. Describe what CHANGES over time.

Formula (Alibaba Wan + LTX official):
  Subject motion  +  Camera move  +  Atmospheric/in-frame motion  +  Technical tag

  • Subject motion          — physical action verbs ("slowly raises chin", "exhales", "turns 90° right"), with rough timing
  • Camera move             — exactly ONE: dolly in / dolly out / arc left / arc right / pan / tilt / handheld push / locked-off static / rack focus
  • Atmospheric / in-frame  — smoke drifts, fog rolls, rain streaks, neon pulses, drapes sway, dust catches light
  • Technical tag           — include "180-degree shutter, natural motion blur" for any shot with movement (Wan/LTX reliability flag)
  • End-state               — one short clause saying what the final frame shows

✅ GOOD video_prompt EXAMPLE (~95 words):
"Over 4 seconds, the man slowly tilts his chin upward and exhales smoke that catches the magenta key light. Camera handheld pushes in from medium-close to tight close-up at a steady pace. Velvet drapes behind him sway 5cm in ambient air, neon sign flickers once mid-shot, smoke drifts diagonally left-to-right across the frame, anamorphic lens flare stretches as the camera nears. 180-degree shutter, natural motion blur, fine film grain. End frame: tight close-up of his face, smoke fully framing his head, neon at full intensity."

================================================================
OUTPUT — emit ONLY this JSON object (no prose, no markdown fence)
================================================================
{{
  "shots": [
    {{
      "description":  "<one short human-readable sentence — what happens visually>",
      "image_prompt": "<60–100 words, six slots in order, sentences not tags>",
      "video_prompt": "<70–120 words, change-from-keyframe, one camera move, technical tag, end-state>",
      "key_visual":   "<one short noun phrase — the anchor that ties this shot to neighbors (e.g. 'pink neon sign', 'man in black blazer')>"
    }}
    /* exactly {num_shots} shot objects total */
  ],
  "updated_treatment": "<the FULL treatment markdown, preserving every section from the input. Append ONE concise bullet under '## What's been generated' summarizing chunk {chunk_idx_label} ({chunk_start:.0f}–{chunk_end:.0f}s): what setting, what camera language, what continuity hooks. Future chunks read THIS, not the original.>"
}}

CONSTRAINTS:
- Exactly {num_shots} shot objects, in the slot order given.
- Every image_prompt obeys the 6 slots + word range + bans above.
- Every video_prompt begins from the keyframe + one camera move + technical tag.
- updated_treatment preserves the markdown structure; only append the new bullet.
- Continuity from the treatment is non-negotiable: same character clothing, same palette, same grade.

Begin. JSON object only."""


# If prompts.md is present, override the literals above with whatever the
# user has authored there. This is the single source of truth for prompts;
# the literals are kept as a fallback for missing-file / missing-section.
_DEFAULT_TREATMENT_PROMPT = get_prompt("director.treatment", _DEFAULT_TREATMENT_PROMPT)
_DEFAULT_CHUNK_SCENES_PROMPT = get_prompt("director.chunk_scenes", _DEFAULT_CHUNK_SCENES_PROMPT)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _audio_to_mono_numpy(audio: dict) -> tuple:
    waveform = audio["waveform"]
    sr = audio["sample_rate"]
    if isinstance(waveform, torch.Tensor):
        wf = waveform.cpu().numpy()
    else:
        wf = np.asarray(waveform)
    if wf.ndim == 3:
        wf = wf[0]
    if wf.ndim > 1 and wf.shape[0] > 1:
        wf = np.mean(wf, axis=0)
    elif wf.ndim > 1:
        wf = wf[0]
    return wf.astype(np.float32), int(sr)


def _write_audio_chunk(mono: np.ndarray, sr: int, start_s: float, end_s: float) -> str:
    start_sample = max(0, int(start_s * sr))
    end_sample = min(len(mono), int(end_s * sr))
    if end_sample <= start_sample:
        end_sample = start_sample + 1
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="omni_scenes_chunk_")
    os.close(fd)
    sf.write(path, mono[start_sample:end_sample], sr)
    return path


# ---------------------------------------------------------------------------
# Analysis → prompt-friendly block
# ---------------------------------------------------------------------------

def _format_analysis_for_prompt(analysis: dict) -> str:
    """Compact, prompt-friendly rendering of the analyze JSON."""
    lines = []
    duration = analysis.get("duration")
    if duration is not None:
        lines.append(f"Duration: {duration}s ({int(duration) // 60:02d}:{int(duration) % 60:02d}).")
    if analysis.get("bpm"):
        lines.append(f"Tempo: {analysis['bpm']} BPM.")
    if analysis.get("keyscale"):
        lines.append(f"Key: {analysis['keyscale']}.")
    for field in (
        "language", "genre", "style", "short_description", "full_description",
        "mood", "vocals", "era_feel", "narrative_arc", "subject", "setting_hint",
    ):
        val = analysis.get(field)
        if isinstance(val, str) and val.strip():
            lines.append(f"{field.replace('_', ' ').capitalize()}: {val}")
    for field in ("keywords", "instruments", "color_palette"):
        val = analysis.get(field)
        if isinstance(val, list) and val:
            lines.append(f"{field.replace('_', ' ').capitalize()}: {', '.join(str(x) for x in val)}")
    lyrics = analysis.get("lyrics")
    if isinstance(lyrics, str) and lyrics.strip():
        lines.append("Lyrics (with section labels):")
        lines.append(lyrics)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic time-grid computation
# ---------------------------------------------------------------------------

def _compute_shots_per_chunk(chunk_duration: float, target_shot_seconds: float) -> int:
    """How many shots fit in this chunk, rounded to the nearest whole number.
    Always at least 1."""
    if chunk_duration <= 0:
        return 1
    n = round(chunk_duration / max(0.1, target_shot_seconds))
    return max(1, int(n))


def _compute_shot_grid(duration: float, chunk_seconds: float, target_shot_seconds: float):
    """Return a list of chunk descriptors, each holding the pre-computed
    shot windows for that chunk.

      [
        {
          "chunk_idx": 0,
          "chunk_start": 0.00,
          "chunk_end":   30.00,
          "windows": [(scene_id, start, end), …]
        },
        …
      ]
    """
    if duration <= 0:
        return []
    num_chunks = max(1, int(math.ceil(duration / chunk_seconds)))
    grid = []
    next_scene_id = 1
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_seconds
        chunk_end = min(duration, chunk_start + chunk_seconds)
        chunk_duration = chunk_end - chunk_start
        if chunk_duration < 0.1:
            continue
        shots_count = _compute_shots_per_chunk(chunk_duration, target_shot_seconds)
        slot_size = chunk_duration / shots_count
        windows = []
        for i in range(shots_count):
            s = round(chunk_start + i * slot_size, 2)
            e = round(chunk_start + (i + 1) * slot_size, 2)
            windows.append((next_scene_id, s, e))
            next_scene_id += 1
        grid.append({
            "chunk_idx": chunk_idx,
            "chunk_start": round(chunk_start, 2),
            "chunk_end": round(chunk_end, 2),
            "chunk_duration": round(chunk_duration, 2),
            "windows": windows,
        })
    return grid


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
# Qwen3 thinking-capable variants emit <think>…</think> reasoning before the
# real output. Strip those blocks so the JSON parser is never confused by
# JSON examples that may appear inside the scratchpad. Safe on models that
# don't emit thinking — the regex matches nothing.
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think_blocks(text: str) -> str:
    if not text:
        return text
    return _THINK_BLOCK_RE.sub("", text).strip()


def _safe_parse_chunk_object(text: str):
    """Pull the {"shots": [...], "updated_treatment": "..."} object out of a
    noisy GGUF response. Returns None on failure (caller logs + falls back).
    Always strips <think>...</think> blocks first."""
    if not text:
        return None
    cleaned = _CODE_FENCE_RE.sub("", _strip_think_blocks(text)).strip()
    obj_match = _JSON_OBJECT_RE.search(cleaned)
    if not obj_match:
        return None
    try:
        parsed = json.loads(obj_match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def _coerce_str(v) -> str:
    if v is None:
        return ""
    return str(v).strip()


# ---------------------------------------------------------------------------
# Tweak B — programmatic shot validation + single retry.
#
# Research-backed (DeCRIM, EMNLP 2024 Findings): self-critique inside the
# same model doesn't reliably fix mistakes — but an EXTERNAL Python check
# with surgical failure feedback gives +7.3% IFEval / +8.0% RealInstruct
# improvement. Cost: ~25-35% of shots fail validation and trigger one
# extra GGUF call. Cheap insurance for hard-rule compliance (word counts,
# required technical tag, required End frame: clause).
# ---------------------------------------------------------------------------

_MIN_IMAGE_PROMPT_WORDS = 60
_MIN_VIDEO_PROMPT_WORDS = 70
_REQUIRED_VIDEO_TAG = "180-degree shutter"
_REQUIRED_END_FRAME_PREFIX = "End frame:"


def _validate_shots(shots, expected_count: int) -> list:
    """Return a list of human-readable failure messages.

    Empty list = everything passed. Each failure is a single string the
    model can read and act on directly. Order is shot-first so retry
    feedback reads naturally.
    """
    failures: list = []
    if not isinstance(shots, list):
        return [f"Expected a JSON array of shots, got {type(shots).__name__}."]
    if len(shots) != expected_count:
        failures.append(
            f"Wrong shot count: got {len(shots)}, need exactly {expected_count}."
        )
    for i, s in enumerate(shots, start=1):
        if not isinstance(s, dict):
            failures.append(f"shot {i}: not a JSON object (got {type(s).__name__}).")
            continue
        ip = (s.get("image_prompt") or "").strip()
        vp = (s.get("video_prompt") or "").strip()
        ip_words = len(ip.split())
        vp_words = len(vp.split())
        if ip_words < _MIN_IMAGE_PROMPT_WORDS:
            failures.append(
                f"shot {i}: image_prompt is {ip_words} words, need at least {_MIN_IMAGE_PROMPT_WORDS}. "
                f"Expand with more sensory + framing detail."
            )
        if vp_words < _MIN_VIDEO_PROMPT_WORDS:
            failures.append(
                f"shot {i}: video_prompt is {vp_words} words, need at least {_MIN_VIDEO_PROMPT_WORDS}. "
                f"Expand the motion description and atmospheric detail."
            )
        if _REQUIRED_VIDEO_TAG not in vp:
            failures.append(
                f"shot {i}: video_prompt is missing the required technical tag "
                f"'180-degree shutter, natural motion blur'. Add it verbatim."
            )
        if _REQUIRED_END_FRAME_PREFIX not in vp:
            failures.append(
                f"shot {i}: video_prompt is missing the required 'End frame:' clause "
                f"at the end. Add a sentence beginning with 'End frame:' describing "
                f"the final frame after the motion completes."
            )
    return failures


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class OMNI_AUDIO_VIDEO_SCENES:
    """Deterministic-time director: Python computes shot windows; the model
    only writes the creative content + rolling treatment per chunk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Original audio — same one fed to Analyze."}),
                "analysis_json": ("STRING", {"multiline": True, "default": "",
                                              "tooltip": "JSON produced by `Omni Audio — Music Analyze`."}),
                "captioner_gguf": (_list_gguf_models(), {
                    "default": _list_gguf_models()[0],
                }),
                "captioner_mmproj": (_list_mmproj_files(), {
                    "default": _list_mmproj_files()[0],
                }),
                "shot_seconds": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 30.0, "step": 0.5,
                                            "tooltip": "Target shot/scene length in seconds. Python picks the closest whole number of shots that fits the 30s audio chunk evenly."}),
                "style_hint": ("STRING", {"multiline": True, "default": "",
                                            "tooltip": "Free-form direction passed by the user — palette, era, references."}),
                "prompt_treatment": ("STRING", {
                    "default": _DEFAULT_TREATMENT_PROMPT,
                    "multiline": True,
                    "tooltip": (
                        "Pass A prompt: runs ONCE at the start. Writes the music-video treatment paragraph.\n"
                        "Template variables:\n"
                        "  {analysis_block} — compact rendering of the analysis JSON\n"
                        "  {style_hint}     — your free-form direction"
                    ),
                }),
                "prompt_chunk_scenes": ("STRING", {
                    "default": _DEFAULT_CHUNK_SCENES_PROMPT,
                    "multiline": True,
                    "tooltip": (
                        "Pass B prompt: runs ONCE PER CHUNK. Writes the shots' creative content + an updated treatment for the next chunk.\n"
                        "Template variables:\n"
                        "  {treatment}           — current rolling treatment\n"
                        "  {analysis_json}       — full analysis JSON\n"
                        "  {style_hint}          — free-form direction\n"
                        "  {chunk_idx_label}     — e.g. '3' (1-indexed)\n"
                        "  {num_chunks}          — total chunk count\n"
                        "  {chunk_start}, {chunk_end} — song-seconds covered by THIS chunk\n"
                        "  {num_shots}           — exact shot count to write\n"
                        "  {slot_block}          — pre-numbered slots ('Shot 7: 60.00–70.00\\n…')\n"
                        "  {position_clause}     — emphatic 'NOT the start' clause for chunk > 0"
                    ),
                }),
            },
            "optional": {
                "identifier": ("STRING", {"default": "", "multiline": False,
                                            "tooltip": "Free-form tag for traceability — e.g. the Studio project id. Logged at run start and surfaced in /history so canvas runs can be matched to their originating project."}),
                "audio_chunk_seconds": ("FLOAT", {"default": 30.0, "min": 10.0, "max": 30.0, "step": 1.0,
                                                    "tooltip": "Audio window per GGUF call. Capped at 30s by the encoder. Leave at 30."}),
                # Sampling tuned per 2026 research (Qwen3 official + community):
                # avoid greedy decoding on MoE (loops). 0.55–0.70 is the
                # sweet spot for structured-creative tasks. JSONSchemaBench
                # + Few-Shot Dilemma confirm temps <0.5 cause Qwen3 MoE to
                # repeat field-name patterns and shorten freeform fields.
                "treatment_temperature": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "scenes_temperature": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                # Doubled to give headroom for: (1) the richer markdown
                # treatment with all 7 sections, (2) the 60-100 + 70-120 word
                # paragraphs per shot, (3) the appended `## What's been
                # generated` bullet on every chunk, (4) any <think>...</think>
                # scratchpad the model may emit before the JSON.
                "max_tokens_treatment": ("INT", {"default": 2400, "min": 300, "max": 8000}),
                "max_tokens_scenes": ("INT", {"default": 8000, "min": 1000, "max": 32000}),
                "ctx_size": ("INT", {"default": 16384, "min": 4096, "max": 65536, "step": 1024}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("scenes_json", "treatment")
    FUNCTION = "direct"
    CATEGORY = "Omni Audio"
    OUTPUT_NODE = True

    def _run_treatment_pass(
        self,
        prompt_treatment,
        analysis_block,
        style_hint_text,
        audio,
        gguf_path,
        mmproj_path,
        max_tokens_treatment,
        treatment_temperature,
        ctx_size,
    ):
        """Pass A: one GGUF call that writes the initial treatment paragraph.

        The wav lifecycle is contained here (try/finally). Returns the
        treatment string (never empty — falls back to a sentinel string).
        """
        treatment_wav = _audio_to_temp_wav(audio)
        try:
            treatment_prompt_text = _safe_format(prompt_treatment, {
                "analysis_block": analysis_block,
                "style_hint": style_hint_text,
            })
            print("[Video Scenes] Pass A — initial treatment...", flush=True)
            treatment = _run_captioner_gguf(
                gguf_path, mmproj_path, treatment_wav, treatment_prompt_text,
                max_tokens=max_tokens_treatment,
                temperature=treatment_temperature,
                ctx_size=ctx_size,
            ).strip()
            _free_vram()
        finally:
            try:
                if treatment_wav and os.path.exists(treatment_wav):
                    os.unlink(treatment_wav)
            except OSError:
                pass

        if not treatment:
            treatment = "(empty treatment — model returned nothing)"
        return treatment

    def _run_one_chunk(
        self,
        chunk_entry,
        treatment,
        analysis_json,
        style_hint_text,
        gguf_path,
        mmproj_path,
        mono,
        sr,
        prompt_chunk_scenes,
        scenes_temperature,
        max_tokens_scenes,
        ctx_size,
        total_chunks,
        duration,
    ):
        """Pass B, single chunk: build prompt, write chunk WAV, call GGUF, validate,
        retry on failure, return (model_shots_list, updated_treatment_str).

        The chunk wav lifecycle (try/finally cleanup) lives inside this helper.
        """
        chunk_idx = chunk_entry["chunk_idx"]
        chunk_start = chunk_entry["chunk_start"]
        chunk_end = chunk_entry["chunk_end"]
        windows = chunk_entry["windows"]
        num_shots = len(windows)

        # Pre-rendered slot block for the prompt.
        slot_block = "\n".join(
            f"  Shot {sid}: {s:.2f}–{e:.2f}"
            for sid, s, e in windows
        )

        if chunk_idx == 0:
            position_clause = (
                "🟢 THIS IS THE OPENING CHUNK. Establish the look, the character(s), "
                "the palette, and the central visual idea here."
            )
        else:
            position_clause = (
                f"🔴 THIS IS NOT THE START. The video has been running for {chunk_start:.1f}s "
                f"of {duration:.1f}s. Continue from the treatment — same character(s), "
                f"same palette, evolving camera language. DO NOT write 'the video opens…' "
                f"or 'the song begins…'."
            )

        prompt_text = _safe_format(prompt_chunk_scenes, {
            "treatment": treatment,
            "analysis_json": analysis_json,
            "style_hint": style_hint_text,
            "chunk_idx_label": str(chunk_idx + 1),
            "num_chunks": str(total_chunks),
            "chunk_start": f"{chunk_start:.2f}",
            "chunk_end": f"{chunk_end:.2f}",
            "num_shots": str(num_shots),
            "slot_block": slot_block,
            "position_clause": position_clause,
        })

        chunk_wav = _write_audio_chunk(mono, sr, chunk_start, chunk_end)
        try:
            print(
                f"[Video Scenes]   chunk {chunk_idx + 1}/{total_chunks} "
                f"({chunk_start:.1f}-{chunk_end:.1f}s) — {num_shots} shots...",
                flush=True,
            )
            # First-pass call.
            raw_out = _run_captioner_gguf(
                gguf_path, mmproj_path, chunk_wav, prompt_text,
                max_tokens=max_tokens_scenes,
                temperature=scenes_temperature,
                ctx_size=ctx_size,
            )
            parsed = _safe_parse_chunk_object(raw_out)

            # Tweak B — validate the parsed shots and, if any hard rule
            # failed, retry ONCE with a tight feedback prompt naming the
            # exact failures. The wav stays on disk for the retry; both
            # calls happen inside the try block before the finally deletes.
            if parsed:
                first_shots = parsed.get("shots") or []
                first_shots = first_shots if isinstance(first_shots, list) else []
                failures = _validate_shots(first_shots, num_shots)
                if failures:
                    print(
                        f"[Video Scenes]     chunk {chunk_idx + 1}: "
                        f"{len(failures)} validation failure(s); retrying once...",
                        flush=True,
                    )
                    for f in failures[:4]:
                        print(f"        - {f}", flush=True)
                    if len(failures) > 4:
                        print(f"        - (and {len(failures) - 4} more)", flush=True)

                    retry_prompt = (
                        prompt_text
                        + "\n\n══════════════════════════════════════════════\n"
                        + "PREVIOUS OUTPUT FAILED VALIDATION — REVISION REQUIRED\n"
                        + "══════════════════════════════════════════════\n\n"
                        + "Problems found:\n"
                        + "\n".join(f"  - {f}" for f in failures)
                        + "\n\nOutput the corrected JSON below. Fix ONLY the listed "
                        + "problems — keep every other shot and the updated_treatment "
                        + "exactly as in the previous output. Do NOT invent new shots. "
                        + "Do NOT change scene order or scene_id assignments. "
                        + "Expand ONLY the indicated fields.\n\n"
                        + "PREVIOUS OUTPUT (to be corrected):\n"
                        + _strip_think_blocks(raw_out).strip()
                        + "\n\nCORRECTED JSON OBJECT:\n"
                    )
                    retry_out = _run_captioner_gguf(
                        gguf_path, mmproj_path, chunk_wav, retry_prompt,
                        max_tokens=max_tokens_scenes,
                        temperature=max(0.05, scenes_temperature * 0.7),
                        ctx_size=ctx_size,
                    )
                    retry_parsed = _safe_parse_chunk_object(retry_out)
                    if retry_parsed:
                        retry_shots = retry_parsed.get("shots") or []
                        retry_shots = retry_shots if isinstance(retry_shots, list) else []
                        retry_failures = _validate_shots(retry_shots, num_shots)
                        if len(retry_failures) < len(failures):
                            print(
                                f"[Video Scenes]     chunk {chunk_idx + 1}: retry improved "
                                f"failures {len(failures)} → {len(retry_failures)}; using retry.",
                                flush=True,
                            )
                            parsed = retry_parsed
                        else:
                            print(
                                f"[Video Scenes]     chunk {chunk_idx + 1}: retry didn't improve "
                                f"({len(retry_failures)} failures); keeping first attempt.",
                                flush=True,
                            )
                    else:
                        print(
                            f"[Video Scenes]     chunk {chunk_idx + 1}: retry response unparseable; "
                            f"keeping first attempt.",
                            flush=True,
                        )
        finally:
            try:
                if chunk_wav and os.path.exists(chunk_wav):
                    os.unlink(chunk_wav)
            except OSError:
                pass

        if not parsed:
            head = _strip_think_blocks(raw_out or "").strip().replace("\n", " ")[:400]
            print(
                f"[Video Scenes]     chunk {chunk_idx + 1}: model returned non-parseable output. "
                f"Filling with placeholders. Raw head: {head!r}",
                flush=True,
            )
            model_shots = []
            model_treatment = ""
        else:
            model_shots = parsed.get("shots") or []
            if not isinstance(model_shots, list):
                model_shots = []
            model_treatment = _coerce_str(parsed.get("updated_treatment"))

        return model_shots, model_treatment

    def direct(
        self,
        audio,
        analysis_json,
        captioner_gguf,
        captioner_mmproj,
        shot_seconds,
        style_hint,
        prompt_treatment,
        prompt_chunk_scenes,
        identifier="",
        audio_chunk_seconds=30.0,
        treatment_temperature=0.65,
        scenes_temperature=0.65,
        max_tokens_treatment=2400,
        max_tokens_scenes=8000,
        ctx_size=16384,
    ):
        if not captioner_gguf or captioner_gguf == NO_MODEL_SENTINEL \
                or not captioner_mmproj or captioner_mmproj == NO_MODEL_SENTINEL:
            raise RuntimeError(
                "Captioner GGUF + mmproj must be installed in ComfyUI/models/llm_gguf/."
            )

        # Identifier is just a tracing tag — log it loud so canvas users can
        # match the run to its originating project. Studio also reads it back
        # from /history.identifier[0] as a sanity check on which project a
        # prompt belonged to (the in-memory jobTracker is the source of truth).
        identifier_str = (identifier or "").strip()
        if identifier_str:
            print(f"[Video Scenes] identifier='{identifier_str}'", flush=True)

        # External callers (e.g. Studio backend) submit the workflow via /api/prompt,
        # where the widget `default=` is NOT auto-filled — every required input must
        # carry a value. To avoid duplicating the 80-line prompt templates outside
        # this file, callers may pass '' and we fall back to the built-in defaults.
        if not prompt_treatment or not prompt_treatment.strip():
            prompt_treatment = _DEFAULT_TREATMENT_PROMPT
        if not prompt_chunk_scenes or not prompt_chunk_scenes.strip():
            prompt_chunk_scenes = _DEFAULT_CHUNK_SCENES_PROMPT

        try:
            analysis = json.loads(analysis_json) if analysis_json.strip() else {}
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"analysis_json is not valid JSON: {exc}") from exc

        duration = float(analysis.get("duration") or 0.0)
        if duration <= 0:
            raise RuntimeError("analysis_json has no positive 'duration' — re-run Music Analyze first.")

        chunk_seconds = min(float(audio_chunk_seconds or _AUDIO_CHUNK_SECONDS_MAX), _AUDIO_CHUNK_SECONDS_MAX)
        shot_target = max(1.0, float(shot_seconds))

        # ---- Deterministic time grid ---------------------------------------
        grid = _compute_shot_grid(duration, chunk_seconds, shot_target)
        if not grid:
            return ("[]", "")
        total_shots = sum(len(c["windows"]) for c in grid)
        avg_shot_len = sum((e - s) for c in grid for _, s, e in c["windows"]) / total_shots
        per_chunk_counts = ", ".join(str(len(c["windows"])) for c in grid)
        print(
            f"[Video Scenes] Plan: {len(grid)} chunks, {total_shots} shots total, "
            f"avg {avg_shot_len:.1f}s each. Per-chunk: [{per_chunk_counts}].",
            flush=True,
        )

        gguf_root = os.path.join(folder_paths.models_dir, "llm_gguf")
        gguf_path = os.path.join(gguf_root, captioner_gguf)
        mmproj_path = os.path.join(gguf_root, captioner_mmproj)

        analysis_block = _format_analysis_for_prompt(analysis)
        style_hint_text = style_hint.strip() or "(none — director's discretion)"

        # ---- Pass A: initial treatment -------------------------------------
        mono, sr = _audio_to_mono_numpy(audio)
        treatment = self._run_treatment_pass(
            prompt_treatment, analysis_block, style_hint_text, audio,
            gguf_path, mmproj_path, max_tokens_treatment, treatment_temperature, ctx_size,
        )

        # ---- Pass B: per-chunk shot writer + rolling treatment update ------
        print(f"[Video Scenes] Pass B — {len(grid)} chunks (one call each)...", flush=True)
        all_shots: list = []

        for entry in grid:
            chunk_idx = entry["chunk_idx"]
            windows = entry["windows"]

            # treatment_at_chunk_start = the treatment the model SAW this chunk
            # (the rolling memory before this chunk's update). Stamp each shot
            # with it so the JSON shows treatment evolution per chunk.
            treatment_at_chunk_start = treatment

            model_shots, model_treatment = self._run_one_chunk(
                entry, treatment, analysis_json, style_hint_text,
                gguf_path, mmproj_path, mono, sr,
                prompt_chunk_scenes, scenes_temperature, max_tokens_scenes, ctx_size,
                total_chunks=len(grid), duration=duration,
            )

            # Pair each pre-computed slot with the model's content (or a
            # placeholder if the model didn't produce enough shots).
            for slot_idx, (sid, s, e) in enumerate(windows):
                content = model_shots[slot_idx] if slot_idx < len(model_shots) and isinstance(model_shots[slot_idx], dict) else {}
                all_shots.append({
                    "scene_id": sid,
                    "start": s,
                    "end": e,
                    "chunk_idx": chunk_idx,
                    "description": _coerce_str(content.get("description")) or "(model did not write a description)",
                    "image_prompt": _coerce_str(content.get("image_prompt")),
                    "video_prompt": _coerce_str(content.get("video_prompt")),
                    "key_visual": _coerce_str(content.get("key_visual")),
                    # Snapshot of the treatment AS THE MODEL SAW IT for this
                    # chunk. Lets you read the JSON top-to-bottom and watch
                    # the directorial brief evolve chunk by chunk.
                    "treatment_snapshot": treatment_at_chunk_start,
                })

            written = sum(1 for c in model_shots if isinstance(c, dict))
            num_shots = len(windows)
            print(
                f"[Video Scenes]     chunk {chunk_idx + 1}: model wrote {written}/{num_shots} shots; "
                f"treatment update {'OK' if model_treatment else 'MISSING'}.",
                flush=True,
            )

            # Roll the treatment forward. Empty model_treatment → keep prior.
            if model_treatment:
                treatment = model_treatment

            _free_vram()

        scenes_json_out = json.dumps(all_shots, indent=2, ensure_ascii=False)
        # Single wire output + ui mirror so Studio can read it from /history.
        # Same pattern as OMNI_AUDIO_Analyze; without this wrap the tuple flows
        # to downstream wires but never appears in ComfyUI's history endpoint.
        # The `identifier` ui key is informational (mainly for canvas users) —
        # Studio doesn't depend on it. Empty string when no tag was passed.
        return {
            "ui": {
                "scenes": [scenes_json_out],
                "treatment": [treatment],
                "identifier": [identifier_str],
            },
            "result": (scenes_json_out, treatment),
        }
