"""ComfyUI custom node: omni-audio.

Audio understanding nodes (transcription + captioning) backed by Qwen2.5-Omni
*and* Qwen3-Omni model families. The right loader is picked from the model's
config.json automatically — same workflow, just point `model_id` at whichever
checkpoint is on disk or on HuggingFace.

Origin: forked and trimmed from `kana112233/ComfyUI-kaola-ace-step`
(MIT license). Music-generation nodes from upstream were dropped because they
require the `acestep` PyPI package that is not installable in our environment;
this fork keeps only the Transcriber, Captioner, and ClearVRAM nodes plus
adds Qwen3-Omni dispatch.
"""
import ctypes
import glob
import os
import sys


# Critical TLS fix: force-load libgomp before any other library does it,
# else ComfyUI under conda hits "Inconsistency detected by ld.so".
def _force_load_libgomp():
    try:
        ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
        return
    except OSError:
        pass
    try:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if not conda_prefix and "envs" in sys.executable:
            conda_prefix = sys.executable.split("/bin/python")[0]
        if conda_prefix:
            lib_paths = glob.glob(os.path.join(conda_prefix, "lib", "libgomp.so.1"))
            if lib_paths:
                ctypes.CDLL(lib_paths[0], mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass


_force_load_libgomp()
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# Transformers v5 removed `layer_type_validation` from configuration_utils, but
# Qwen2.5-Omni's remote code still references it. Shim it back as a passthrough.
def _patch_transformers_v5():
    try:
        import transformers.configuration_utils as cu
        if not hasattr(cu, "layer_type_validation"):
            cu.layer_type_validation = lambda *args, **kwargs: args[0] if args else None
    except Exception:
        pass


_patch_transformers_v5()


from .nodes_transcriber import OMNI_AUDIO_TRANSCRIBER  # noqa: E402
from .nodes_captioner import OMNI_AUDIO_CAPTIONER  # noqa: E402
from .nodes_vram import OMNI_AUDIO_CLEAR_VRAM  # noqa: E402
from .nodes_analyze import OMNI_AUDIO_ANALYZE  # noqa: E402
from .nodes_video_scenes import OMNI_AUDIO_VIDEO_SCENES  # noqa: E402
from .nodes_loader import OMNI_AUDIO_LOAD_AUDIO_PATH  # noqa: E402


NODE_CLASS_MAPPINGS = {
    "OMNI_AUDIO_Transcriber": OMNI_AUDIO_TRANSCRIBER,
    "OMNI_AUDIO_Captioner": OMNI_AUDIO_CAPTIONER,
    "OMNI_AUDIO_ClearVRAM": OMNI_AUDIO_CLEAR_VRAM,
    "OMNI_AUDIO_Analyze": OMNI_AUDIO_ANALYZE,
    "OMNI_AUDIO_VideoScenes": OMNI_AUDIO_VIDEO_SCENES,
    "OMNI_AUDIO_LoadAudioPath": OMNI_AUDIO_LOAD_AUDIO_PATH,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OMNI_AUDIO_Transcriber": "Omni Audio — Transcriber",
    "OMNI_AUDIO_Captioner": "Omni Audio — Captioner",
    "OMNI_AUDIO_ClearVRAM": "Omni Audio — Clear VRAM",
    "OMNI_AUDIO_Analyze": "Omni Audio — Music Analyze",
    "OMNI_AUDIO_VideoScenes": "Omni Audio — Video Scenes (Director)",
    "OMNI_AUDIO_LoadAudioPath": "Omni Audio — Load Audio (path)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
