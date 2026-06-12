# packalares-custom-nodes

Monorepo of ComfyUI custom nodes used by packalares / Studio.

## Layout

```
.
├── __init__.py          # auto-discovers sub-packages
├── docres/              # DocRes — document restoration (CamScanner-style)
└── omni_audio/          # Qwen-Omni audio transcription + captioning
```

Each sub-folder is a self-contained Python package exporting
`NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`. The root `__init__.py`
walks `pkgutil.iter_modules(__path__)` and merges all of them.

## Install in ComfyUI

```bash
cd ComfyUI/custom_nodes
git clone <this-repo>.git packalares-custom-nodes
```

Restart ComfyUI. All sub-nodes register automatically.

## Update

```bash
cd ComfyUI/custom_nodes/packalares-custom-nodes
git pull
```

Restart ComfyUI. (ComfyUI-Manager's "Try update" also calls `git pull`.)

## Add a new sub-node

1. Create a new folder with a valid Python identifier name (no hyphens).
2. Inside it, an `__init__.py` exporting `NODE_CLASS_MAPPINGS` and
   `NODE_DISPLAY_NAME_MAPPINGS`.
3. Commit + push. The root will pick it up next ComfyUI restart — no root
   edits needed.

## Per-sub-node dependencies

Sub-nodes carry their own Python deps. See each sub-folder's `README.md`.
If you add a sub-node with new pip deps, append them to its README and
optionally to a top-level `requirements.txt` (ComfyUI-Manager auto-installs
that on update).
