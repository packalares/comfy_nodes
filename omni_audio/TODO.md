# Omni Audio Studio — TODO / Deferred Work

Living document of work that's been scoped but not landed. Edit freely.

---

## 1. Persistent `llama-server` migration

**One-line:** Replace the per-call `llama-mtmd-cli` subprocess pattern with a long-lived `llama-server` HTTP process so the GGUF stays resident in VRAM between calls.

### Why we'd do it

| Win | Real-world value |
|---|---|
| Cold-start elimination | Small — verified ~10-20 s/song savings (mmap + OS page cache already hide most of the disk I/O cost). NOT the 2-5 min I originally estimated. |
| JSON schema enforcement via `response_format: {"type": "json_schema", ...}` | Real — zero parse failures, can drop our `_safe_parse_chunk_object` defensive code |
| DSPy / Outlines / Guidance / instructor compatibility | Future-proofing — any structured-output toolchain that speaks OpenAI API works |
| Streaming output for UI progress bars | Nice — can show "writing shot 2 of 3..." |
| Cleaner architecture (HTTP > subprocess + stdout parsing) | Modest |

### Verdict

**Hold off until we actually want one of these benefits.** Tweak B (our Python validator + retry) already covers structured-output compliance for our current failure modes. The speed win is too small to justify the engineering cost on its own.

### What we'd build (~half day)

```
custom_nodes/comfyui-studio-omni-audio/
  nodes_llama_server.py    ← NEW: server lifecycle + HTTP client
```

Approximate API:
```python
def ensure_server_running(model_gguf, mmproj_gguf, ctx_size) -> int:
    """Idempotent — starts server if not running, returns port. Health-checks before returning."""

def call_completion(port, prompt, audio_path, *, max_tokens, temperature, json_schema=None) -> str:
    """One HTTP request to /v1/chat/completions. Returns model text."""

def shutdown_server() -> None:
    """Called from ComfyUI's model_management hooks at unload time."""
```

Then `nodes_analyze.py::_run_captioner_gguf` becomes a thin wrapper that picks subprocess vs HTTP based on a config flag (or always HTTP after migration).

### Risks to think about

- Port collisions (hardcode 8080, or pick a free port via socket?)
- Server died mid-run (detect via timeout + restart)
- ComfyUI worker thread blocked while server starts (~15 s first time)
- Multiple ComfyUI instances on same pod (unlikely but possible)
- Audio multimodal request payload encoding (base64 in JSON vs file path — verify llama-server's `/chat/completions` supports both)

### Already-installed prerequisites

`llama-server` is part of the same tarball we already extract for `llama-mtmd-cli`. Zero new downloads.

---

## 2. Studio Shot Settings → Director wiring

**One-line:** The Analyze page's "Shot Settings" card writes to `project.settings.{shotMode, fixedShotSeconds, bpmBarsPerShot, snapToLyrics, snapToSections}` but nothing reads them yet.

### Three settings, three difficulty levels

#### 2a. BPM-derived shot length 🟢 IMPLEMENTABLE TODAY

User intent: shot length = N bars at the song's tempo. Feels more musical than fixed seconds.

Math: `shot_seconds = bars_per_shot × 4 / (bpm / 60)` (for 4/4 time signatures)
Example: 4 bars × 4 beats × (60/87 BPM) = 11.0 s/shot

What we need:
- `bpm` field in `analysis_json` — already present
- New widget on Video Scenes node: `shot_mode: "fixed" | "bpm"` + `bars_per_shot: int` (default 4)
- In `direct()`:
  ```python
  if shot_mode == "bpm":
      bpm = float(analysis.get("bpm") or 120)
      shot_seconds = bars_per_shot * 4 * (60.0 / bpm)
  else:
      shot_seconds = shot_seconds  # widget value
  ```

**Effort:** ~30 lines in `nodes_video_scenes.py`. Land any time.

#### 2b. Don't break mid-word (snap to lyric word end) 🔴 BLOCKED — needs Whisper

User intent: if a shot ends in the middle of a sung word, shift the boundary so the word completes.

What we need: per-word lyric timestamps `[(start_ms, end_ms, word), ...]`. We don't have them today — the ACE-Step Transcriber refuses to emit timestamps even when prompted.

Three paths:
- **Add Whisper-large-v3 to Music Analyze** (recommended). ~3 GB model, ~30 s extra analyze time, gives reliable word-level alignment. Append `lyrics_word_times: [...]` to `analysis_json`.
- **Onset detection only** (no text alignment). `librosa.onset.onset_detect` gives note/syllable onset times but no words. Could approximate but quality is worse.
- **Skip the feature.** Acceptable for v1 — shot boundaries already align to lyric *chunks* via the Director's 30 s audio chunks, just not individual words.

#### 2c. Prefer verse/chorus boundaries 🟡 PARTIAL — librosa works, Whisper better

User intent: if a `[Chorus]` starts at 1:08 and our shot grid would put a cut at 1:05, shift to 1:08 so the chorus opens its own shot.

What we need: section timestamps `[{start_ms, end_ms, label}, ...]`. We have section LABELS in the `lyrics` string but no times.

Two paths:
- **librosa structural segmentation** (`librosa.segment.agglomerative` on MFCC features). Free, fast, but labels are generic ("Segment A / B / C"), not "Verse" / "Chorus". Could approximate by matching the repeating segment (= Chorus heuristic).
- **Whisper word-timestamps → derive section boundaries** from where `[Verse 2]`-style labels appear in the timed transcription. Same dependency as 2b. Best quality.

### Recommended order

1. Land 2a (BPM mode) now — 30 lines, no new model
2. Build the Whisper integration as a separate task (unblocks 2b + 2c simultaneously)
3. Wire the Studio→Director bridge (also a separate task — Studio backend doesn't call our new Director node yet at all)

### Studio→Director bridge (prerequisite for any of 2a/2b/2c to be user-visible)

Currently the Video Director node only runs in ComfyUI canvas; Studio's videoboard backend doesn't call it. To plumb the user's Shot Settings through:

1. New `server/src/services/videoboard/buildScenesWorkflow.ts` — mirrors `buildAnalyzeWorkflow.ts` pattern
2. New `server/src/services/videoboard/comfyScenes.ts` — submit + wait + parse
3. Update `POST /api/videoboard/projects/:id/storyboard/generate` to call `scenesViaComfyUI(...)` instead of the current `setTimeout` stub
4. Pass `project.settings.{shotMode, fixedShotSeconds, bpmBarsPerShot, snapToLyrics, snapToSections}` as workflow widget values

---

## 3. Misc deferred items from the omni-audio audit

### Optimization (low priority)

- [ ] Remove no-op `_free_vram()` calls after GGUF subprocess (subprocess VRAM isn't visible to parent). 3-line deletion, ~50 ms per call saved.
- [ ] Avoid double-decode of audio in `direct()` — reuse the mono numpy from `_audio_to_mono_numpy` instead of calling `_audio_to_temp_wav` separately.
- [ ] Replace hardcoded `/usr/local/lib/python3.12/site-packages/nvidia/` with `sysconfig.get_paths()["purelib"]` so we work on Python 3.10/3.11/3.13.

### Defense-in-depth (low priority)

- [ ] Add SHA256 verification on the `llama.cpp-cuda.tar.gz` download in `_ensure_mtmd_cli()`. Belt-and-suspenders on top of HTTPS + Zip-Slip filter.
- [ ] Document `nodes_loader.py`'s arbitrary-path risk for any future multi-tenant deployment.

### Whisper / per-word timestamps integration (medium project)

- [ ] Add `transformers.AutoModelForSpeechSeq2Seq` loader for `openai/whisper-large-v3`
- [ ] Add a new phase in Music Analyze that runs AFTER the current transcriber and produces `lyrics_word_times: [(start_ms, end_ms, word), ...]`
- [ ] Derive `section_times: [(start_ms, end_ms, label), ...]` by aligning the ACE-Step section labels to Whisper's timed words
- [ ] Extend `Analysis` contract + the videoboard.repo.ts JSON blob shape accordingly
- [ ] Update UI's Lyrics card to highlight currently-singing lyric line based on audio playback time
