"""Prompt library loader.

All prompt templates used by Omni Audio nodes live in `prompts.md` next
to this file. This module reads that file once on import, splits it on
`## prompt:<id>` markers, and exposes the sections as a dict keyed by id.

If `prompts.md` is missing or a key is absent, the relevant nodes fall
back to hardcoded built-ins (defined inside each node file) so the user
never sees an empty default.
"""
import os
import re

_PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.md")
_SECTION_RE = re.compile(r"^##\s+prompt:([\w.-]+)\s*$", re.MULTILINE)


def _load_prompts() -> dict:
    if not os.path.exists(_PROMPTS_FILE):
        print(f"[Omni Audio] prompts.md not found at {_PROMPTS_FILE} — nodes will use built-in fallbacks.", flush=True)
        return {}
    try:
        text = open(_PROMPTS_FILE, encoding="utf-8").read()
    except OSError as exc:
        print(f"[Omni Audio] Could not read {_PROMPTS_FILE}: {exc}", flush=True)
        return {}
    sections = {}
    matches = list(_SECTION_RE.finditer(text))
    for i, m in enumerate(matches):
        key = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        # Strip the trailing horizontal rule and surrounding whitespace.
        body = text[start:end].strip()
        # If the body ends with a stand-alone "---" rule, drop it.
        body = re.sub(r"\n\s*-{3,}\s*$", "", body).strip()
        sections[key] = body
    print(f"[Omni Audio] Loaded {len(sections)} prompt sections from prompts.md: {sorted(sections.keys())}", flush=True)
    return sections


PROMPTS = _load_prompts()


def get_prompt(key: str, fallback: str = "") -> str:
    """Look up a prompt by `## prompt:<key>` id, with an explicit fallback."""
    return PROMPTS.get(key, fallback)
