"""Packalares custom-node monorepo for ComfyUI.

Each sub-package (folder with its own __init__.py) is auto-discovered and its
NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS are merged. To add a node:
drop a new folder, commit, push. A broken sub-package logs a warning and is
skipped — the rest still load.
"""

import importlib
import pkgutil

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for mod_info in pkgutil.iter_modules(__path__):
    if mod_info.name.startswith('_') or not mod_info.ispkg:
        continue
    try:
        mod = importlib.import_module(f'.{mod_info.name}', __name__)
    except Exception as e:
        print(f'[packalares-custom-nodes] skip {mod_info.name}: {e}', flush=True)
        continue
    NODE_CLASS_MAPPINGS.update(getattr(mod, 'NODE_CLASS_MAPPINGS', {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod, 'NODE_DISPLAY_NAME_MAPPINGS', {}))

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
