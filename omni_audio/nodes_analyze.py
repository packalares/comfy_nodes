"""Omni Audio Music Analyze node.

Single audio-in → structured JSON-out:
  - duration, BPM (+ range + tempo tag), time signature, key (librosa)
  - lyrics (one Transcriber pass, section labels preserved)
  - container/codec metadata (ffprobe, when audio_source_path is provided)
  - director-grade JSON caption (Qwen3-Omni-30B Q4_K_M via llama-mtmd-cli)

The Captioner GGUF can only see ~30s of audio at a time, so we deliberately
do NOT use it for per-word alignment. Per-line timestamps and song-structure
sections are intentionally not produced — the Transcriber model doesn't emit
real word timestamps and we don't want to ship fake ones.
"""
import gc
import json
import os
import re
import subprocess
import tarfile
import tempfile
import urllib.request
import uuid

import folder_paths
import numpy as np
import soundfile as sf
import torch

from .nodes_captioner import NO_MODEL_SENTINEL
from .nodes_prompts import get_prompt
from .nodes_transcriber import OMNI_AUDIO_TRANSCRIBER, get_omni_transcriber_models


# ---------------------------------------------------------------------------
# llama-mtmd-cli binary management (lazy first-run install)
# ---------------------------------------------------------------------------

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_BIN_ROOT = os.path.join(_NODE_DIR, "bin")
_CUDA_BIN_DIR = os.path.join(_BIN_ROOT, "cuda-12.8")
_MTMD_CLI = os.path.join(_CUDA_BIN_DIR, "llama-mtmd-cli")
_LLAMA_TARBALL_URL = (
    "https://github.com/ai-dock/llama.cpp-cuda/releases/download/"
    "b9371/llama.cpp-b9371-cuda-12.8-amd64.tar.gz"
)


def _safe_tarball_members(tar: tarfile.TarFile, dest: str):
    """Generator that yields only tar members whose extraction target stays
    inside `dest`. Defends against Zip-Slip / path traversal — Python 3.12+
    deprecates bare tar.extractall() for this reason."""
    dest_abs = os.path.realpath(dest)
    for m in tar.getmembers():
        # Reject absolute paths and any '..' segments.
        if os.path.isabs(m.name) or ".." in m.name.split("/"):
            print(f"[Omni Analyze] tar member rejected (unsafe path): {m.name!r}", flush=True)
            continue
        target = os.path.realpath(os.path.join(dest_abs, m.name))
        if not (target == dest_abs or target.startswith(dest_abs + os.sep)):
            print(f"[Omni Analyze] tar member rejected (escapes dest): {m.name!r}", flush=True)
            continue
        yield m


def _ensure_mtmd_cli() -> str:
    if os.path.exists(_MTMD_CLI) and os.access(_MTMD_CLI, os.X_OK):
        return _MTMD_CLI
    os.makedirs(_BIN_ROOT, exist_ok=True)
    print("[Omni Analyze] First run — fetching llama.cpp tarball (~150 MB)...", flush=True)
    archive = os.path.join(_BIN_ROOT, "llama.cpp-cuda.tar.gz")
    urllib.request.urlretrieve(_LLAMA_TARBALL_URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(_BIN_ROOT, members=_safe_tarball_members(tar, _BIN_ROOT))
    os.unlink(archive)
    if not os.path.exists(_MTMD_CLI):
        raise RuntimeError(f"Extracted tarball but llama-mtmd-cli not at {_MTMD_CLI}")
    os.chmod(_MTMD_CLI, 0o755)
    print(f"[Omni Analyze] Binary ready at {_MTMD_CLI}", flush=True)
    return _MTMD_CLI


def _gguf_ld_library_path() -> str:
    nvidia_root = "/usr/local/lib/python3.12/site-packages/nvidia"
    nvidia_libs = []
    if os.path.exists(nvidia_root):
        for sub in sorted(os.listdir(nvidia_root)):
            lib = os.path.join(nvidia_root, sub, "lib")
            if os.path.isdir(lib):
                nvidia_libs.append(lib)
    inherited = os.environ.get("LD_LIBRARY_PATH", "")
    return ":".join([_CUDA_BIN_DIR] + nvidia_libs + ([inherited] if inherited else []))


def _run_captioner_gguf(
    model_gguf: str,
    mmproj_gguf: str,
    audio_path: str,
    prompt: str,
    max_tokens: int = 2500,
    temperature: float = 0.2,
    ctx_size: int = 16384,
    timeout_sec: int = 900,
) -> str:
    binary = _ensure_mtmd_cli()
    env = {**os.environ, "LD_LIBRARY_PATH": _gguf_ld_library_path()}
    args = [
        binary,
        "-m", model_gguf,
        "--mmproj", mmproj_gguf,
        "--audio", audio_path,
        "-p", prompt,
        "-ngl", "999",
        "-c", str(ctx_size),
        "-fa", "on",
        "-n", str(max_tokens),
        "--temp", str(temperature),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, env=env, timeout=timeout_sec)
    if proc.returncode != 0:
        tail = proc.stderr[-1500:] if proc.stderr else ""
        raise RuntimeError(f"llama-mtmd-cli failed (rc={proc.returncode}):\n{tail}")
    return proc.stdout


# ---------------------------------------------------------------------------
# GGUF model discovery (the captioner side — lives in models/llm_gguf/)
# ---------------------------------------------------------------------------

def _list_gguf_models() -> list:
    root = os.path.join(folder_paths.models_dir, "llm_gguf")
    if not os.path.isdir(root):
        return [NO_MODEL_SENTINEL]
    files = sorted(
        f for f in os.listdir(root)
        if f.lower().endswith(".gguf")
        and not f.lower().startswith("mmproj")
        and "omni" in f.lower()
    )
    return files if files else [NO_MODEL_SENTINEL]


def _list_mmproj_files() -> list:
    root = os.path.join(folder_paths.models_dir, "llm_gguf")
    if not os.path.isdir(root):
        return [NO_MODEL_SENTINEL]
    files = sorted(
        f for f in os.listdir(root)
        if f.lower().startswith("mmproj") and f.lower().endswith(".gguf")
    )
    return files if files else [NO_MODEL_SENTINEL]


# ---------------------------------------------------------------------------
# librosa DSP — duration / BPM / range / tempo tag / time signature / key
# ---------------------------------------------------------------------------

def _octave_cap_bpm(raw: float) -> int:
    bpm = float(raw)
    while bpm > 140 and bpm / 2 >= 60:
        bpm = bpm / 2
    return int(round(bpm))


def _tempo_tag_from_bpm(bpm: int):
    if not bpm or bpm <= 0:
        return None
    if bpm < 76:
        return "Slow"
    if bpm < 100:
        return "Mid"
    if bpm < 120:
        return "Upbeat"
    return "Fast"


def _compute_bpm_range(librosa, y, sr, bpm_main):
    tempos = None
    for fn_path in (("feature", "rhythm", "tempo"), ("beat", "tempo")):
        obj = librosa
        try:
            for part in fn_path:
                obj = getattr(obj, part)
        except AttributeError:
            continue
        try:
            tempos = obj(y=y, sr=sr, aggregate=None)
            break
        except (TypeError, ValueError):
            continue
    if tempos is None:
        return None, None
    arr = np.asarray(tempos).flatten()
    if arr.size == 0:
        return None, None
    capped = [_octave_cap_bpm(float(t)) for t in arr if float(t) > 0]
    if not capped:
        return None, None
    bpm_min = min(capped)
    bpm_max = max(capped)
    if bpm_main:
        bpm_min = min(bpm_min, bpm_main)
        bpm_max = max(bpm_max, bpm_main)
    return bpm_min, bpm_max


def _detect_time_signature(librosa, y, sr, beats):
    if beats is None or len(beats) < 16:
        return None
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    except Exception:
        return None
    if onset_env is None or len(onset_env) == 0:
        return None
    try:
        beat_strengths = librosa.util.sync(onset_env, beats, aggregate=np.median)
    except Exception:
        return None
    beat_strengths = np.asarray(beat_strengths).flatten()
    if beat_strengths.size < 8:
        return None
    try:
        acf = librosa.autocorrelate(beat_strengths, max_size=8)
    except Exception:
        bs = beat_strengths - beat_strengths.mean()
        acf = np.correlate(bs, bs, mode="full")
        acf = acf[acf.size // 2:][:8]
    acf = np.asarray(acf).flatten()
    if acf.size <= 6 or float(acf[0]) <= 0:
        return None
    candidates = {2: "2/4", 3: "3/4", 4: "4/4", 6: "6/8"}
    scores = {label: float(acf[lag] / acf[0]) for lag, label in candidates.items() if lag < acf.size}
    if not scores:
        return None
    best_label = max(scores, key=scores.get)
    if scores[best_label] < 0.25:
        return None
    return best_label


def _analyze_dsp(waveform: np.ndarray, sample_rate: int) -> dict:
    import librosa

    if waveform.ndim > 1 and waveform.shape[0] > 1:
        mono = np.mean(waveform, axis=0)
    elif waveform.ndim > 1:
        mono = waveform[0]
    else:
        mono = waveform
    mono = mono.astype(np.float32)

    duration = round(float(len(mono) / sample_rate), 2)

    bpm = 0
    beats = None
    try:
        tempo, beats = librosa.beat.beat_track(y=mono, sr=sample_rate)
        raw_bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        bpm = _octave_cap_bpm(raw_bpm)
    except Exception as exc:
        print(f"[Omni Analyze] BPM detection failed: {exc}", flush=True)

    bpm_min, bpm_max = _compute_bpm_range(librosa, mono, sample_rate, bpm)
    tempo_tag = _tempo_tag_from_bpm(bpm)
    time_signature = _detect_time_signature(librosa, mono, sample_rate, beats)

    try:
        keyscale = _detect_key_krumhansl(mono, sample_rate)
    except Exception as exc:
        print(f"[Omni Analyze] Key detection failed: {exc}", flush=True)
        keyscale = "unknown"

    return {
        "duration": duration,
        "bpm": bpm,
        "bpm_min": bpm_min,
        "bpm_max": bpm_max,
        "tempo_tag": tempo_tag,
        "time_signature": time_signature,
        "keyscale": keyscale,
    }


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Schmuckler perceived-key profiles
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _detect_key_krumhansl(y: np.ndarray, sr: int) -> str:
    import librosa
    chroma = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr), axis=1)
    best_score = -np.inf
    best_label = "C major"
    chroma_norm = np.linalg.norm(chroma) + 1e-9
    for shift in range(12):
        for profile, mode in ((_KS_MAJOR, "major"), (_KS_MINOR, "minor")):
            rotated = np.roll(profile, shift)
            score = float(np.dot(chroma, rotated) / (chroma_norm * (np.linalg.norm(rotated) + 1e-9)))
            if score > best_score:
                best_score = score
                best_label = f"{_NOTE_NAMES[shift]} {mode}"
    return best_label


# ---------------------------------------------------------------------------
# ffprobe — container / codec metadata
# ---------------------------------------------------------------------------

def _audio_meta_from_tensor(waveform_np: np.ndarray, sr: int) -> dict:
    channels = int(waveform_np.shape[0]) if waveform_np.ndim > 1 else 1
    return {
        "format": None,
        "size_bytes": None,
        "bitrate_kbps": None,
        "channels": channels,
        "sample_rate": int(sr),
    }


def _audio_meta_via_ffprobe(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", path],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0:
            return None
        info = json.loads(proc.stdout or "{}")
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
        print(f"[Omni Analyze] ffprobe failed: {exc}", flush=True)
        return None
    fmt = info.get("format") or {}
    streams = info.get("streams") or []
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    s = audio_streams[0] if audio_streams else {}

    def _int_or_none(value, divisor: int = 1):
        try:
            return int(int(value) / divisor) if value is not None else None
        except (TypeError, ValueError):
            return None

    fmt_name = (fmt.get("format_name") or "").split(",")[0] or None
    return {
        "format": fmt_name,
        "size_bytes": _int_or_none(fmt.get("size")),
        "bitrate_kbps": _int_or_none(fmt.get("bit_rate") or s.get("bit_rate"), divisor=1000),
        "channels": _int_or_none(s.get("channels")),
        "sample_rate": _int_or_none(s.get("sample_rate")),
    }


# ---------------------------------------------------------------------------
# Audio → temp WAV (mtmd-cli takes a file path)
# ---------------------------------------------------------------------------

def _audio_to_temp_wav(audio: dict) -> str:
    waveform = audio["waveform"]
    sr = audio["sample_rate"]
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = np.mean(waveform, axis=0)
    elif waveform.ndim > 1:
        waveform = waveform[0]
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="omni_analyze_")
    os.close(fd)
    sf.write(path, waveform.astype(np.float32), sr)
    return path


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _free_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    try:
        import comfy.model_management as mm
        mm.free_memory(1.0, mm.get_torch_device())
        mm.soft_empty_cache()
    except Exception:
        pass


_LANG_CODE_RE = re.compile(r"^[a-z]{2,3}(-[A-Z]{2})?$")


def _strip_lyrics_header(raw_lyrics: str):
    """Strip the `# Languages\\n<code>\\n# Lyrics` markdown header that the
    Transcriber prepends. Returns (clean_lyrics_body, lang_code_or_None).
    Section labels and parentheticals in the body are preserved verbatim.
    """
    if not raw_lyrics:
        return "", None
    lines = raw_lyrics.splitlines()
    lang_code = None
    body_start = len(lines)
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        low = stripped.lower()
        if low.startswith("# language"):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                cand = lines[j].strip()
                if _LANG_CODE_RE.match(cand):
                    lang_code = cand
                    i = j + 1
                    continue
            i += 1
            continue
        if low.startswith("# lyrics"):
            body_start = i + 1
            break
        # Non-header content: the body begins here, no markdown wrapper present.
        body_start = i
        break
    body = "\n".join(lines[body_start:]).strip()
    return body, lang_code


def _safe_format(template: str, variables: dict) -> str:
    out = template
    for key, value in variables.items():
        out = out.replace("{" + key + "}", str(value))
    return out


_JSON_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")


def _safe_parse_json(text: str) -> dict:
    text = _JSON_FENCE_RE.sub("", text.strip())
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"raw": text.strip()}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"raw": text.strip()}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_DEFAULT_TRANSCRIBE_PROMPT = (
    "Transcribe the sung lyrics of this audio. Include section labels "
    "(e.g. [Intro], [Verse 1], [Pre-Chorus], [Chorus], [Bridge], [Outro]) "
    "on their own line before each section. Detect the sung language "
    "automatically. Do not add commentary or descriptions outside the lyrics."
)


_DEFAULT_CAPTION_PROMPT = """Listen to this audio carefully and return ONLY a JSON object with EVERY
field below populated. Use double quotes. Be specific — name actual
instruments and concrete imagery, never vague labels like 'percussion'.

{
  "language": "<full language name in English, e.g. 'Romanian', 'English', 'Brazilian Portuguese'>",
  "genre": "<primary genre + sub-genre>",
  "style": "<production aesthetic / era / regional flavor>",
  "short_description": "<one sentence summary>",
  "full_description": "<a detailed 4-6 sentence music-director-style paragraph covering: tempo + key + structure (with rough [mm:ss] section markers), each instrument and how it's played, lead + backing vocals, production / mix character, lyrical themes (quote 2-3 memorable lines verbatim), and the mood arc from intro to outro>",
  "mood": "<single descriptor word or short phrase, e.g. 'melancholic', 'euphoric', 'tense'>",
  "keywords": ["<5 to 8 short keywords describing the song's vibe — single words or short phrases>"],
  "instruments": ["<each distinct instrument you can hear, named specifically>"],
  "vocals": "<vocalist description: register + timbre + delivery, e.g. 'single female alto, intimate close-mic, breathy'>",
  "era_feel": "<implied era / decade vibe, e.g. 'late-2000s indie', '1970s funk', 'contemporary'>",
  "narrative_arc": "<how the song's intensity evolves, e.g. 'slow build, release in final chorus'>",
  "subject": "<what the song is about, one short clause>",
  "color_palette": ["<3 to 5 visual colors the music evokes>"],
  "setting_hint": "<implied physical setting/location the song calls to mind>"
}

Do NOT include anything outside the JSON. Every field is REQUIRED."""


# Override literals with prompts.md content (single source of truth). The
# literals above are kept as fallback in case prompts.md is missing.
_DEFAULT_TRANSCRIBE_PROMPT = get_prompt("analyze.transcribe", _DEFAULT_TRANSCRIBE_PROMPT)
_DEFAULT_CAPTION_PROMPT = get_prompt("analyze.caption", _DEFAULT_CAPTION_PROMPT)


# ---------------------------------------------------------------------------
# Internal Transcriber call (reuses our existing node)
# ---------------------------------------------------------------------------

def _run_transcriber(audio, model_id, device, dtype, chunk_length_s, temperature, custom_prompt):
    node = OMNI_AUDIO_TRANSCRIBER()
    return node.transcribe(
        audio=audio,
        model_id=model_id,
        device=device,
        dtype=dtype,
        language="auto",
        chunk_length_s=chunk_length_s,
        return_timestamps="false",
        custom_prompt=(custom_prompt or "").strip(),
        max_new_tokens=4096,
        temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.1,
        num_beams=1,
        seed=0,
    )[0]


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

class OMNI_AUDIO_ANALYZE:
    """Single audio-in → enriched JSON-out (duration / bpm + range / tempo
    tag / time-sig / key / section-labeled lyrics / ffprobe metadata /
    director-grade caption). VRAM-bounded per phase."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio to analyze."}),
                "identifier": ("STRING", {"default": "", "multiline": False,
                                            "tooltip": "Free-form identifier (project id, slug, song name). Stored verbatim under 'identifier' in the analysis JSON so downstream consumers can match this run to a record. Leave empty to auto-generate an 8-char hex slug."}),
                "transcriber_model": (get_omni_transcriber_models(), {
                    "default": get_omni_transcriber_models()[0],
                    "tooltip": "Local ACE-Step Transcriber (Qwen2.5-Omni-7B). Used for lyrics text. Only acestep-transcriber dirs are listed.",
                }),
                "captioner_gguf": (_list_gguf_models(), {
                    "default": _list_gguf_models()[0],
                    "tooltip": "Qwen3-Omni-30B Instruct GGUF in models/llm_gguf/. Only Omni audio GGUFs are listed (vision-only GGUFs like Qwen2.5-VL are filtered out).",
                }),
                "captioner_mmproj": (_list_mmproj_files(), {
                    "default": _list_mmproj_files()[0],
                    "tooltip": "Matching mmproj-*.gguf for the captioner.",
                }),
                "lyrics_enabled": ("BOOLEAN", {"default": True, "tooltip": "Run Transcriber to extract lyrics."}),
                "caption_enabled": ("BOOLEAN", {"default": True, "tooltip": "Run Captioner GGUF to produce the rich director-grade JSON caption."}),
                "chunk_length_s": ("FLOAT", {"default": 450.0, "min": 30.0, "max": 1800.0, "step": 10.0,
                                              "tooltip": "Transcriber chunking window. 450s covers ~7.5 min songs in one shot."}),
                "prompt_transcribe": ("STRING", {
                    "default": _DEFAULT_TRANSCRIBE_PROMPT,
                    "multiline": True,
                    "tooltip": "Prompt sent to the ACE-Step Transcriber. The model emits lyrics with [Section] labels inline.",
                }),
                "prompt_caption": ("STRING", {
                    "default": _DEFAULT_CAPTION_PROMPT,
                    "multiline": True,
                    "tooltip": "Prompt sent to the Captioner GGUF for the rich JSON caption. No template variables.",
                }),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "dtype": (["auto", "float16", "float32"], {"default": "auto"}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens_caption": ("INT", {"default": 2500, "min": 500, "max": 8000}),
                "ctx_size": ("INT", {"default": 16384, "min": 4096, "max": 65536, "step": 1024}),
                # audio_source_path stays LAST so adding a future optional input
                # never shifts the widget positions of fields already saved in
                # existing workflow JSONs. Reordering optional fields breaks the
                # positional widget binding.
                "audio_source_path": ("STRING", {"default": "", "multiline": False,
                                                  "tooltip": "Absolute path to the original audio file (for ffprobe metadata). Leave empty to skip — channel/sample-rate are still derived from the loaded tensor."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("analysis_json",)
    FUNCTION = "analyze"
    CATEGORY = "Omni Audio"
    OUTPUT_NODE = True

    def _phase_ffprobe(self, audio_source_path):
        """Phase 0: Run ffprobe on the source file if a path was given. Returns dict or None."""
        return _audio_meta_via_ffprobe(audio_source_path.strip()) if audio_source_path else None

    def _phase_dsp(self, waveform_np, sr):
        """Phase 1: DSP — duration / BPM + range / tempo tag / time-sig / key.
        Returns a dict with dsp fields, duration_ms, and audio_meta_from_tensor."""
        print("[Omni Analyze] DSP (duration / BPM + range / tempo tag / time-sig / key)...", flush=True)
        dsp = _analyze_dsp(waveform_np, sr)
        duration_ms = int(round(dsp["duration"] * 1000))
        from_tensor = _audio_meta_from_tensor(waveform_np, sr)
        return {"dsp": dsp, "duration_ms": duration_ms, "from_tensor": from_tensor}

    def _phase_lyrics(self, audio, lyrics_enabled, transcriber_model, device, dtype,
                      chunk_length_s, temperature, prompt_transcribe):
        """Phase 2: Lyrics via Transcriber. Returns (plain_lines, section_markers, raw_lyrics_text, lang_code)."""
        if not lyrics_enabled:
            return None, None
        if not transcriber_model or transcriber_model == NO_MODEL_SENTINEL:
            print("[Omni Analyze] lyrics_enabled but no transcriber installed — skipping.", flush=True)
            return None, None
        print("[Omni Analyze] Running Transcriber for lyrics...", flush=True)
        raw_lyrics = _run_transcriber(
            audio, transcriber_model, device, dtype, chunk_length_s, temperature,
            prompt_transcribe,
        ) or ""
        _free_vram()
        body, lang_code = _strip_lyrics_header(raw_lyrics)
        return body or None, lang_code

    def _phase_caption(self, captioner_gguf, captioner_mmproj, wav_path, prompt_caption,
                       max_tokens_caption, temperature, ctx_size, caption_enabled):
        """Phase 3: Rich caption via GGUF subprocess. Returns parsed caption dict or None."""
        if not (caption_enabled and captioner_gguf and captioner_mmproj):
            return None
        print("[Omni Analyze] Running Captioner GGUF for caption...", flush=True)
        cap_out = _run_captioner_gguf(
            captioner_gguf, captioner_mmproj, wav_path, prompt_caption,
            max_tokens=max_tokens_caption,
            temperature=temperature,
            ctx_size=ctx_size,
        )
        cap = _safe_parse_json(cap_out)
        _free_vram()
        return cap

    def _compose_result(self, identifier, dsp_out, audio_meta, lyrics, lang_code, cap):
        """Phase 4: Build the final result dict from all phase outputs."""
        dsp = dsp_out["dsp"]
        result = {
            "identifier": identifier,
            "duration": dsp["duration"],
            "duration_ms": dsp_out["duration_ms"],
            "bpm": dsp["bpm"],
            "bpm_min": dsp.get("bpm_min"),
            "bpm_max": dsp.get("bpm_max"),
            "tempo_tag": dsp.get("tempo_tag"),
            "time_signature": dsp.get("time_signature"),
            "keyscale": dsp["keyscale"],
            # language pair: long-form name (from caption) + short ISO code (from transcriber header)
            "language": None,
            "lang_code": lang_code,
            "audio_meta": audio_meta,
            "lyrics": lyrics,
            # Flattened caption fields — populated when caption_enabled and parse OK.
            "genre": None,
            "style": None,
            "short_description": None,
            "full_description": None,
            "mood": None,
            "keywords": None,
            "instruments": None,
            "vocals": None,
            "era_feel": None,
            "narrative_arc": None,
            "subject": None,
            "color_palette": None,
            "setting_hint": None,
            # Set only when the captioner returned non-JSON we couldn't parse.
            "caption_raw": None,
        }
        if isinstance(cap, dict):
            # Parse failed → keep the raw text; don't try to flatten { raw: ... }.
            if set(cap.keys()) == {"raw"}:
                result["caption_raw"] = cap["raw"]
            else:
                _SCALAR_KEYS = (
                    "language", "genre", "style", "short_description",
                    "full_description", "mood", "vocals", "era_feel",
                    "narrative_arc", "subject", "setting_hint",
                )
                _LIST_KEYS = ("keywords", "instruments", "color_palette")
                for k in _SCALAR_KEYS:
                    v = cap.get(k)
                    if isinstance(v, str) and v.strip():
                        result[k] = v.strip()
                for k in _LIST_KEYS:
                    v = cap.get(k)
                    if isinstance(v, list) and v:
                        result[k] = v
        return result

    def analyze(
        self,
        audio,
        identifier,
        transcriber_model,
        captioner_gguf,
        captioner_mmproj,
        lyrics_enabled,
        caption_enabled,
        chunk_length_s,
        prompt_transcribe,
        prompt_caption,
        audio_source_path="",
        device="auto",
        dtype="auto",
        temperature=0.2,
        max_tokens_caption=2500,
        ctx_size=16384,
    ):
        # --- Phase 0: ffprobe metadata (optional) ---------------------------
        meta_from_ffprobe = self._phase_ffprobe(audio_source_path)

        # --- Phase 1: DSP (always) -----------------------------------------
        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.cpu().numpy()
        else:
            waveform_np = np.asarray(waveform)
        if waveform_np.ndim == 3:
            waveform_np = waveform_np[0]

        dsp_out = self._phase_dsp(waveform_np, sr)
        audio_meta = {**dsp_out["from_tensor"],
                      **{k: v for k, v in (meta_from_ffprobe or {}).items() if v is not None}}
        identifier_resolved = (identifier or "").strip() or uuid.uuid4().hex[:8]

        # --- Phase 2: Lyrics (Transcriber) ---------------------------------
        lyrics, lang_code = self._phase_lyrics(
            audio, lyrics_enabled, transcriber_model, device, dtype,
            chunk_length_s, temperature, prompt_transcribe,
        )

        # --- Pre-flight for the Captioner GGUF -----------------------------
        gguf_root = os.path.join(folder_paths.models_dir, "llm_gguf")
        gguf_path = None
        mmproj_path = None
        if (captioner_gguf and captioner_gguf != NO_MODEL_SENTINEL
                and captioner_mmproj and captioner_mmproj != NO_MODEL_SENTINEL):
            gguf_path = os.path.join(gguf_root, captioner_gguf)
            mmproj_path = os.path.join(gguf_root, captioner_mmproj)

        will_run_gguf = caption_enabled and gguf_path and mmproj_path
        wav_path = _audio_to_temp_wav(audio) if will_run_gguf else None

        try:
            # --- Phase 3: Rich caption (GGUF subprocess) -------------------
            cap = self._phase_caption(
                gguf_path, mmproj_path, wav_path, prompt_caption,
                max_tokens_caption, temperature, ctx_size, caption_enabled,
            )
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

        # --- Phase 4: Compose return value ---------------------------------
        # Single wire output (analysis_json) — downstream consumers parse the
        # JSON for whichever fields they need. The same string is also
        # published via `ui.analysis` so Studio can read it from /history.
        result = self._compose_result(identifier_resolved, dsp_out, audio_meta, lyrics, lang_code, cap)
        analysis_json = json.dumps(result, indent=2, ensure_ascii=False)
        return {
            "ui": {"analysis": [analysis_json]},
            "result": (analysis_json,),
        }
