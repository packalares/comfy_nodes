# Omni Audio Studio — Prompt Library

All prompts used by the Omni Audio custom nodes live here. Edit a section
and **restart ComfyUI** to refresh the defaults. (Existing nodes with the
prompt baked into their widget will keep the baked value — drop a fresh
node to pick up your edits.)

## Conventions
- Sections are marked with `## prompt:<id>` on a line by itself.
- Everything between two markers is the prompt body, verbatim.
- Template variables look like `{analysis_block}` and are substituted
  at runtime via `_safe_format`. Unknown placeholders are left as-is.
- Literal `{` and `}` in JSON examples are kept as single braces — the
  loader doesn't use Python `.format()`, so no escaping is needed.

---

## prompt:analyze.transcribe

Transcribe the sung lyrics of this audio. Include section labels (e.g. [Intro], [Verse 1], [Pre-Chorus], [Chorus], [Bridge], [Outro]) on their own line before each section. Detect the sung language automatically. Do not add commentary or descriptions outside the lyrics.

---

## prompt:analyze.caption

Listen to this audio carefully and return ONLY a JSON object with EVERY
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

Do NOT include anything outside the JSON. Every field is REQUIRED.

---

## prompt:director.treatment

You are a senior music-video director writing the PROJECT BIBLE for a new
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
not a list of clichés.

---

## prompt:director.chunk_scenes

You are a cinematographer + art director writing shot briefs for a
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
Chunk {chunk_idx_label} of {num_chunks}, covering song seconds {chunk_start}–{chunk_end}.

{position_clause}

# SHOTS TO FILL — write exactly {num_shots} shot objects, in this order
{slot_block}

═════════════════════════════════════════════════════════════════════
HOW TO WRITE image_prompt  —  60–100 words, sentences (NOT tag lists)
═════════════════════════════════════════════════════════════════════
Fill these SIX SLOTS in order. Skipping any one makes the diffusion model
improvise unpredictably.

  (1) FRAME + SUBJECT   — shot size + subject appearance (age, clothing item by item, hair, expression, position in frame)
  (2) ENVIRONMENT       — specific location + time of day + named props the viewer should see
  (3) LIGHTING          — single key source + direction + quality (hard / soft / practical), fill ratio
  (4) COLOR + GRADE     — 2-3 named dominant colors + grade reference (teal-amber, bleach-bypass, golden halation, etc.)
  (5) LENS + DOF        — focal length + aperture / depth-of-field commitment (35mm anamorphic at T2, shallow DOF)
  (6) MOOD / ATMOSPHERE — fog / rain / dust / haze / silence-before-motion type cue

❌ NEVER USE in image_prompt:
  - Narrative verbs: "the chorus erupts", "the music swells", "feels lonely"
  - Quality boosters: "masterpiece, 8K, ultra detailed, best quality, award-winning"
  - Comma-tag lists (modern T5/Qwen-based diffusion ignores them — write sentences)
  - Contradictions: "warm cool-toned", "sharp soft focus"
  - (weight:1.3) syntax — these models ignore it

✅ GOOD image_prompt EXAMPLE (~85 words):
"Medium close-up of a man in his late 20s, sharp black blazer over a white tee, slicked-back dark hair, tired but defiant eyes, center frame. Standing under a flickering pink neon sign in a fog-filled nightclub alcove, velvet drapes faintly visible behind. Single hard practical neon from screen left, deep ambient blue fill stage-right, no top light. Saturated magenta highlights, deep teal shadows, bleach-bypass grade. 35mm anamorphic at T2, shallow DOF blurring the drapes. Charged stillness, faint cigarette smoke drifting through the frame."

═════════════════════════════════════════════════════════════════════
HOW TO WRITE video_prompt  —  70–120 words, ALL 5 REQUIREMENTS BELOW
═════════════════════════════════════════════════════════════════════
If your draft is under 70 words, EXPAND with more atmospheric/in-frame
motion detail before moving on. All five elements below are MANDATORY.

  ① KEYFRAME-CHANGE OPENER (don't re-describe the still)
      The image_prompt already describes frame 1. video_prompt must START
      with what HAPPENS over time.
      Good:  "Over 4 seconds, the protagonist slowly raises his chin..."
      Bad:   "The protagonist stands under neon lights..."  ← that's the image

  ② EXACTLY ONE CAMERA MOVE — pick ONE from this list verbatim:
        dolly in / dolly out / slow push / handheld push-in / tracking shot
        arc left / arc right / crane up / crane down / tilt up / tilt down
        pan left / pan right / locked-off static / rack focus / whip pan

  ③ SUBJECT + IN-FRAME MOTION
      Physical verbs + rough timing for what the subject does.
      Atmospheric motion in frame: smoke drifts, rain streaks, neon flickers,
      drapes sway, dust catches light, lens flare stretches.

  ④ REQUIRED TECHNICAL TAG (verbatim — Wan/LTX reliability flag):
        "180-degree shutter, natural motion blur"
      Add "fine film grain" if the look is gritty.

  ⑤ REQUIRED END-STATE CLAUSE (literally start with "End frame:"):
        End frame: <one sentence describing what the viewer sees at the
        final frame, AFTER the motion completes>

✅ GOOD video_prompt EXAMPLE A (~95 words, all 5 elements present — energetic):
"Over 4 seconds, the man slowly tilts his chin upward and exhales smoke that catches the magenta key light. Camera handheld push-in from medium-close to tight close-up at a steady pace. Velvet drapes behind him sway 5cm in ambient air, the neon sign flickers once mid-shot, smoke drifts diagonally left-to-right across the frame, anamorphic lens flare stretches as the camera nears. 180-degree shutter, natural motion blur, fine film grain. End frame: tight close-up of his face, smoke fully framing his head, neon at full intensity."

✅ GOOD video_prompt EXAMPLE B (~85 words, all 5 elements — quiet/still):
"Over 3 seconds, the singer holds completely still under the spotlight, only her chest rising as she draws breath, then her eyes drift down toward the microphone. Locked-off static wide shot, no camera movement. A single dust mote drifts through the cone of light, the spotlight intensity holds steady, far stage curtains hang motionless. 180-degree shutter, natural motion blur on her subtle head turn. End frame: her gaze fixed downward on the microphone, spotlight unchanged, audience pitch-black around her."

<!-- ============================================================ -->
<!-- INCORRECT video_prompt EXAMPLE — DO NOT FOLLOW THIS FORMAT  -->
<!-- ============================================================ -->
❌ WRONG video_prompt (too short, missing camera move from the list, no technical tag, no "End frame:" clause):
"The man looks at the camera. The lights flash and he turns away."
❌ Why this is BAD:
  - Only 13 words. Required minimum: 70 words.
  - "looks" is not a named camera move from the list.
  - Missing "180-degree shutter, natural motion blur" technical tag.
  - Missing "End frame:" clause.
  - No atmospheric or in-frame motion detail.
DO NOT EVER produce video_prompts shaped like this. The example above is shown ONLY so you know what to avoid.

═════════════════════════════════════════════════════════════════════
OUTPUT — emit ONLY this JSON object (no prose, no markdown fence)
═════════════════════════════════════════════════════════════════════
{
  "shots": [
    {
      "description":  "<one short human-readable sentence — what happens visually>",
      "image_prompt": "<60–100 words, six slots in order, sentences not tags>",
      "video_prompt": "<70–120 words, change-from-keyframe, ONE camera move from the list, atmospheric motion, 180-degree shutter natural motion blur, End frame: clause>",
      "key_visual":   "<one short noun phrase — the anchor tying this shot to neighbors (e.g. 'pink neon sign', 'man in black blazer')>"
    }
    /* exactly {num_shots} shot objects total, in the order given */
  ],
  "updated_treatment": "<the FULL treatment markdown, preserving every section from the input verbatim. APPEND ONE concise bullet under '## What's been generated' summarizing chunk {chunk_idx_label} ({chunk_start}–{chunk_end}s): what setting was shown, what camera language, what continuity hooks were planted. Describe what was SHOWN — never use narrative/musical verbs like 'the chorus erupts' or 'the music swells' in the bullet. Future chunks will read THIS treatment, NOT the original.>"
}

CONSTRAINTS:
- Exactly {num_shots} shot objects, in the slot order given.
- Every image_prompt obeys the 6 slots + word range + bans above.
- Every video_prompt has ALL 5 elements (opener, camera move, motion, technical tag, End frame clause) and is 70-120 words.
- updated_treatment preserves the markdown structure; only append the new bullet under '## What's been generated'.
- Continuity from the treatment is non-negotiable: same character clothing, same palette, same grade.

Begin. JSON object only.
