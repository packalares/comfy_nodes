"""Omni Audio Captioner Node.

Music captioning via Qwen2.5-Omni-7B and Qwen3-Omni family models.
The upstream HF repo for the small-model variant is
https://huggingface.co/ACE-Step/acestep-captioner (the directory name
`acestep-captioner` is preserved verbatim because it's the actual on-disk
path under ComfyUI/models/).

Recommended prompt: "*Task* Describe this audio in detail".
"""

import os
import json
import inspect
import re
import torch
import numpy as np
import librosa
import folder_paths
import comfy.utils
import comfy.model_management
from transformers.generation.streamers import BaseStreamer
from typing import Dict, Any, Tuple

# Streamer for ComfyUI progress bar and interruption
class ComfyStreamer(BaseStreamer):
    def __init__(self, pbar):
        self.pbar = pbar

    def put(self, value):
        self.pbar.update(1)
        comfy.model_management.throw_exception_if_processing_interrupted()

    def end(self):
        pass


# Constants
ACESTEP_MODEL_NAME = "Ace-Step1.5"
NO_MODEL_SENTINEL = "[no model installed — see node tooltip]"

INSTALL_HINT = (
    "Place an ACE-Step Captioner (Qwen2.5-Omni-7B) model directory at "
    "ComfyUI/models/acestep-captioner/. You can download it with:\n"
    "    huggingface-cli download ACE-Step/acestep-captioner "
    "--local-dir ComfyUI/models/acestep-captioner"
)


def detect_omni_family(load_path: str) -> str:
    """Return 'qwen3_omni' or 'qwen2_5_omni' based on config.json or path hints.
    Falls back to qwen2_5_omni (the original behavior) when uncertain."""
    cfg_path = os.path.join(load_path, "config.json") if os.path.isdir(load_path) else None
    if cfg_path and os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            arch = (cfg.get("architectures") or [""])[0]
            mt = (cfg.get("model_type") or "").lower()
            if "Qwen3Omni" in arch or "qwen3_omni" in mt or "qwen3-omni" in mt:
                return "qwen3_omni"
            if "Qwen2_5Omni" in arch or "qwen2_5_omni" in mt:
                return "qwen2_5_omni"
        except Exception:
            pass
    low = str(load_path).lower()
    if "qwen3-omni" in low or "qwen3_omni" in low:
        return "qwen3_omni"
    return "qwen2_5_omni"


def _load_qwen2_5_omni(load_path: str, device: str, torch_dtype, log_prefix: str = "OMNI"):
    """Existing Qwen2.5-Omni loader path: pipeline ASR + OOM guards (talker disable + token2wav stub)."""
    from transformers import pipeline, AutoProcessor

    print(f"{log_prefix}: Loading Qwen2.5-Omni pipeline from {load_path}...")
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=load_path,
        device=device,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model = asr_pipeline.model

    # OOM guards: Qwen2.5-Omni tries to generate audio (DiT) by default; we only need text.
    if hasattr(model.config, "disable_audio_generation"):
        model.config.disable_audio_generation = True
    if hasattr(model, "generation_config") and hasattr(model.generation_config, "disable_audio_generation"):
        model.generation_config.disable_audio_generation = True

    if hasattr(model, "token2wav"):
        print(f"{log_prefix}: Monkeypatching token2wav to prevent audio generation and OOM.")

        class _DummyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("dummy_param", torch.tensor(0.0))

            def forward(self, *args, **kwargs):
                return torch.tensor([], device=self.dummy_param.device)

            @property
            def dtype(self):
                return self.dummy_param.dtype

        model.token2wav = _DummyModule().to(device)

    processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
    return model, processor


def _load_qwen3_omni(load_path: str, device: str, torch_dtype, log_prefix: str = "OMNI"):
    """Qwen3-Omni-30B-A3B loader.

    Loads via Qwen3OmniMoeForConditionalGeneration directly (no ASR pipeline registered).
    AWQ-quantized checkpoints self-describe via their config.json `quantization_config`,
    so no explicit AwqConfig is needed — from_pretrained handles it.

    Captioner variant of Qwen3-Omni is thinker-only (no talker / no token2wav) so no
    audio-generation OOM guards are required. Instruct/Thinking variants have a talker
    but we skip its initialization via the standard model knob below where supported.
    """
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3OmniMoeForConditionalGeneration
    except ImportError as e:
        raise RuntimeError(
            f"{log_prefix}: transformers in this environment does not expose "
            f"Qwen3OmniMoeForConditionalGeneration. Upgrade transformers >= 4.51 "
            f"and ensure trust_remote_code support is current. Original: {e}"
        )

    print(f"{log_prefix}: Loading Qwen3-Omni from {load_path}...")

    kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    try:
        from transformers.utils import is_flash_attn_2_available
        if is_flash_attn_2_available():
            kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(load_path, **kwargs)

    # If this is an Instruct/Thinking variant that ships with a talker, free its weights:
    # the Captioner variant simply lacks this attr so the branch is a no-op.
    for attr in ("talker", "talker_model", "token2wav"):
        if hasattr(model, attr):
            try:
                setattr(model, attr, None)
                print(f"{log_prefix}: Released model.{attr} to save VRAM (text-only inference).")
            except Exception:
                pass

    processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
    return model, processor


_HF_REPO_RE = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")


def _resolve_local_path(model_id: str) -> str:
    """Resolve a dropdown value to an absolute model directory.

    Accepts: absolute paths, paths relative to ComfyUI/models/, and short
    names that match a directory under ComfyUI/models/<short_name>/.
    Refuses bare HuggingFace repo ids (owner/repo) — auto-download is
    intentionally off so users decide explicitly when and where to install.
    """
    if os.path.isabs(model_id):
        if not os.path.exists(os.path.join(model_id, "config.json")):
            raise RuntimeError(f"No config.json at {model_id}.\n{INSTALL_HINT}")
        return model_id

    candidate_paths = [
        os.path.join(folder_paths.models_dir, model_id),
        os.path.join(folder_paths.models_dir, ACESTEP_MODEL_NAME, model_id),
    ]
    for p in candidate_paths:
        if os.path.exists(os.path.join(p, "config.json")):
            return p

    if _HF_REPO_RE.match(model_id):
        raise RuntimeError(
            f"'{model_id}' looks like a HuggingFace repo id. Auto-download is "
            f"disabled in this node so installs are explicit. Run:\n"
            f"    huggingface-cli download {model_id} "
            f"--local-dir ComfyUI/models/<dirname>"
        )
    raise RuntimeError(
        f"Model '{model_id}' was not found locally.\n{INSTALL_HINT}"
    )


def _is_omni_dir(path: str) -> bool:
    """A directory is an Omni-audio model if its config.json's `architectures`
    or `model_type` references Qwen2.5-Omni or Qwen3-Omni."""
    cfg_path = os.path.join(path, "config.json")
    if not os.path.exists(cfg_path):
        return False
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception:
        return False
    arch = (cfg.get("architectures") or [""])[0]
    mt = (cfg.get("model_type") or "").lower()
    return (
        "Qwen2_5Omni" in arch
        or "Qwen3Omni" in arch
        or "qwen2_5_omni" in mt
        or "qwen3_omni" in mt
        or "qwen3-omni" in mt
    )


def _scan_omni_models() -> list:
    """Return all Omni-audio model directories under ComfyUI/models/.

    Auto-detects by reading each directory's config.json — naming the
    directory whatever you like still works. Scans one level deep so legacy
    layouts like `Ace-Step1.5/<variant>/` are picked up too.
    """
    root = folder_paths.models_dir
    if not os.path.exists(root):
        return []
    found = []
    for name in sorted(os.listdir(root)):
        candidate = os.path.join(root, name)
        if not os.path.isdir(candidate):
            continue
        if _is_omni_dir(candidate):
            found.append(name)
            continue
        # One level deeper (e.g. Ace-Step1.5/captioner)
        try:
            for sub in sorted(os.listdir(candidate)):
                inner = os.path.join(candidate, sub)
                if os.path.isdir(inner) and _is_omni_dir(inner):
                    found.append(os.path.join(name, sub))
        except (PermissionError, FileNotFoundError):
            continue
    return found


def get_omni_captioner_models():
    """Return the dropdown choices for the Omni Audio Captioner node.

    Restricted to ACE-Step Captioner Qwen2.5-Omni-7B finetune variants — looks
    only under ComfyUI/models/acestep-captioner/ (top level + one level deep)
    plus the legacy ComfyUI/models/Ace-Step1.5/captioner/ location. Auto-download
    from HF is disabled — install manually with `huggingface-cli download`.
    """
    models = []

    # Primary location: models/acestep-captioner/[<variant>/]
    root = os.path.join(folder_paths.models_dir, "acestep-captioner")
    if os.path.exists(root):
        if os.path.exists(os.path.join(root, "config.json")):
            models.append("acestep-captioner")
        else:
            for name in sorted(os.listdir(root)):
                sub = os.path.join(root, name)
                if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "config.json")):
                    models.append(os.path.join("acestep-captioner", name))

    # Legacy: models/Ace-Step1.5/captioner/
    legacy = os.path.join(folder_paths.models_dir, ACESTEP_MODEL_NAME, "captioner")
    if os.path.exists(os.path.join(legacy, "config.json")):
        models.append(os.path.join(ACESTEP_MODEL_NAME, "captioner"))

    if not models:
        return [NO_MODEL_SENTINEL]
    return sorted(set(models))


class OMNI_AUDIO_CAPTIONER:
    """Omni Audio Captioner Node — professional-grade music captioning via
    Qwen-Omni audio-multimodal models. Generates structured descriptions
    of musical style, instruments, structure, and timbre. Loader is shared
    with the Transcriber node; auto-detects Qwen2.5-Omni vs Qwen3-Omni
    based on the model directory's config.json."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio to caption/describe."}),
                "model_id": (get_omni_captioner_models(), {
                    "default": get_omni_captioner_models()[0],
                    "tooltip": (
                        "Locally-installed ACE-Step Captioner (Qwen2.5-Omni-7B finetune). "
                        "No auto-download. Install manually:\n"
                        "    huggingface-cli download ACE-Step/acestep-captioner "
                        "--local-dir ComfyUI/models/acestep-captioner"
                    ),
                }),
                "device": (["auto", "cuda", "cpu", "mps"], {
                    "default": "auto",
                    "tooltip": "Inference device. Use 'auto' or 'mps' for Mac."
                }),
                "dtype": (["auto", "float16", "float32"], {
                    "default": "auto",
                    "tooltip": "Model precision. 'auto' uses float16 for CUDA and float32 for CPU/MPS."
                }),
            },
            "optional": {
                "custom_prompt": ("STRING", {
                    "default": "*Task* Describe this audio in detail",
                    "multiline": True,
                    "tooltip": "Custom prompt for captioning. Default is the recommended prompt from ACE-Step."
                }),
                "max_new_tokens": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "tooltip": "Maximum number of tokens to generate. Increase for longer descriptions."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Sampling temperature. Lower values (0.1-0.3) are more deterministic and accurate."
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Nucleus sampling: cumulative probability threshold."
                }),
                "top_k": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 1000,
                    "tooltip": "Top-K sampling. 0 = disabled. Lower values can improve accuracy."
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.1,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Penalty for repeating tokens. Increase if output gets stuck in loops."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffff,
                    "tooltip": "Random seed for reproducible results. 0 for random."
                }),
                "chunk_length_s": ("FLOAT", {
                    "default": 450.0,
                    "min": 0.0,
                    "max": 1800.0,
                    "step": 10.0,
                    "tooltip": "Audio chunk length in seconds. 450s covers ~7.5-min songs in one shot. Drop lower for >7.5-min audio."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("caption", "style_tags", "full_description")
    FUNCTION = "caption"
    CATEGORY = "Omni Audio"
    OUTPUT_NODE = True

    def caption(
        self,
        audio: Dict[str, Any],
        model_id: str,
        device: str,
        dtype: str,
        custom_prompt: str = "*Task* Describe this audio in detail",
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        seed: int = 0,
        chunk_length_s: float = 450.0,
    ) -> Tuple[str, str, str]:
        """
        Generate a detailed caption/description of the input audio.

        Returns:
            caption: A concise summary caption
            style_tags: Comma-separated style/instrument tags
            full_description: The complete detailed description
        """
        print(f"OMNI_AUDIO_CAPTIONER: Captioning with model {model_id} on {device} ({dtype})")

        # Set random seed for reproducibility
        if seed != 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            print(f"OMNI_AUDIO_CAPTIONER: Using seed {seed}")

        # 1. Device Setup
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        print(f"OMNI_AUDIO_CAPTIONER: Using device: {device}")

        # 2. Path Resolution — local-only, no HF auto-download.
        if model_id == NO_MODEL_SENTINEL:
            raise RuntimeError(
                "OMNI_AUDIO_CAPTIONER: no captioner model is installed.\n" + INSTALL_HINT
            )
        load_path = _resolve_local_path(model_id)
        print(f"OMNI_AUDIO_CAPTIONER: Resolved model path: {load_path}")

        # 2.5 Memory Management
        if device == "cuda":
            torch.cuda.empty_cache()

        # 3. Model Loading — dispatch on detected model family
        # Determine dtype
        torch_dtype = torch.float32  # default
        if dtype == "auto":
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "float32":
            torch_dtype = torch.float32

        # This node is restricted to ACE-Step Captioner (Qwen2.5-Omni-7B finetune)
        # — always use the Qwen2.5-Omni loader. The Qwen3-Omni dispatch lives in
        # the shared helpers (and is still importable by the Transcriber node)
        # but is intentionally not invoked here.
        try:
            model, processor = _load_qwen2_5_omni(load_path, device, torch_dtype, log_prefix="OMNI_AUDIO_CAPTIONER")
            # Record whether this model's generate() accepts return_audio
            try:
                gen_sig = inspect.signature(model.generate)
                model._supports_return_audio = "return_audio" in gen_sig.parameters
            except (TypeError, ValueError):
                model._supports_return_audio = False
            print("OMNI_AUDIO_CAPTIONER: Model and Processor loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load Captioner model {model_id}: {e}")

        # 4. Audio Preparation
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']

        # Convert to numpy and mono
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()

        if waveform.ndim == 3:
            waveform = waveform[0]  # [channels, samples]

        # Mix to mono if stereo
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = np.mean(waveform, axis=0)
        elif waveform.ndim > 1:
            waveform = waveform[0]

        waveform = waveform.astype(np.float32)

        # Resample to 16kHz if needed
        TARGET_SR = 16000
        if sample_rate != TARGET_SR:
            print(f"OMNI_AUDIO_CAPTIONER: Resampling audio from {sample_rate}Hz to {TARGET_SR}Hz")
            try:
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=TARGET_SR)
                sample_rate = TARGET_SR
            except Exception as e:
                print(f"OMNI_AUDIO_CAPTIONER: Resampling failed: {e}")
                raise e

        # Calculate audio duration
        audio_duration = len(waveform) / sample_rate
        print(f"OMNI_AUDIO_CAPTIONER: Audio duration: {audio_duration:.1f}s")

        # Determine chunking strategy
        OVERLAP_S = 5
        chunk_samples = int(chunk_length_s * sample_rate)
        overlap_samples = int(OVERLAP_S * sample_rate)
        needs_chunking = audio_duration > chunk_length_s

        if needs_chunking:
            step_samples = chunk_samples - overlap_samples
            num_chunks = int(np.ceil((len(waveform) - overlap_samples) / step_samples))
            print(f"OMNI_AUDIO_CAPTIONER: Splitting into {num_chunks} chunks (chunk={chunk_length_s}s, overlap={OVERLAP_S}s)")
        else:
            num_chunks = 1
            print(f"OMNI_AUDIO_CAPTIONER: Processing as single chunk")

        # 5. Inference
        try:
            # Build prompt
            instruction = custom_prompt.strip() if custom_prompt.strip() else "*Task* Describe this audio in detail"
            print(f"OMNI_AUDIO_CAPTIONER: Instruction: {instruction}")

            # Build text prompt with chat template
            if hasattr(processor, "apply_chat_template"):
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": "placeholder"},
                            {"type": "text", "text": instruction}
                        ]
                    }
                ]
                text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                text_prompt = f"<|im_start|>system\nYou are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

            print(f"OMNI_AUDIO_CAPTIONER: Generation params: temp={temperature}, top_p={top_p}, top_k={top_k}")

            # Helper function to caption a single chunk
            def caption_chunk(audio_chunk, chunk_idx=None):
                inputs = processor(
                    text=[text_prompt],
                    audio=audio_chunk,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                )

                inputs = {k: v.to(device) for k, v in inputs.items()}
                if torch_dtype == torch.float16 and "input_features" in inputs:
                    inputs["input_features"] = inputs["input_features"].to(dtype=torch.float16)

                pbar = comfy.utils.ProgressBar(max_new_tokens)
                streamer = ComfyStreamer(pbar)

                extra_gen = {"return_audio": False} if getattr(model, "_supports_return_audio", False) else {}

                with torch.no_grad():
                    generation_output = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        streamer=streamer,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k if top_k > 0 else None,
                        repetition_penalty=repetition_penalty,
                        do_sample=True if temperature > 0 else False,
                        **extra_gen,
                    )

                generated_ids = generation_output
                full_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # Extract assistant's response
                caption = full_output
                assistant_marker = "assistant"
                if assistant_marker in full_output:
                    parts = full_output.split(assistant_marker)
                    if len(parts) > 1:
                        caption = parts[-1].strip()

                for marker in ["system", "user"]:
                    if f"\n{marker}" in caption:
                        caption = caption.split(f"\n{marker}")[0].strip()

                return caption

            # Process chunks
            all_captions = []

            if needs_chunking:
                step_samples = chunk_samples - overlap_samples
                for i in range(num_chunks):
                    start = i * step_samples
                    end = min(start + chunk_samples, len(waveform))
                    chunk = waveform[start:end]

                    chunk_duration = len(chunk) / sample_rate
                    print(f"OMNI_AUDIO_CAPTIONER: Processing chunk {i+1}/{num_chunks} ({chunk_duration:.1f}s)")

                    chunk_result = caption_chunk(chunk, chunk_idx=i)
                    all_captions.append(chunk_result)
                    print(f"OMNI_AUDIO_CAPTIONER: Chunk {i+1} result: {chunk_result[:80]}...")

                # Merge captions
                # For captioning, we take the most detailed description
                # Usually the first chunk has the most complete analysis
                full_description = all_captions[0]
                if len(all_captions) > 1:
                    # Append additional details from other chunks
                    for i, cap in enumerate(all_captions[1:], 1):
                        # Only add unique content
                        if cap not in full_description:
                            full_description += f"\n\n[Section {i+1}]: {cap}"
            else:
                print("OMNI_AUDIO_CAPTIONER: Running inference...")
                full_description = caption_chunk(waveform)

            # Extract style tags from description
            style_tags = self._extract_style_tags(full_description)

            # Generate concise caption (first sentence or summary)
            caption = self._generate_concise_caption(full_description)

            print(f"OMNI_AUDIO_CAPTIONER: Final caption: {caption[:100]}...")
            return (caption, style_tags, full_description)

        except Exception as e:
            print(f"OMNI_AUDIO_CAPTIONER: Inference failed: {e}")
            raise e

    def _extract_style_tags(self, description: str) -> str:
        """Extract style/instrument tags from the description."""
        tags = []

        # Common music style keywords
        style_keywords = [
            # Genres
            "ambient", "techno", "house", "drum and bass", "synthwave", "downtempo",
            "rock", "alternative", "indie", "post-rock", "progressive", "psychedelic",
            "pop", "synth-pop", "electropop", "dream pop", "art pop",
            "classical", "orchestral", "chamber", "minimalist", "cinematic",
            "jazz", "fusion", "smooth", "bebop", "modal",
            "hip-hop", "trap", "boom bap", "lo-fi", "cloud rap",
            "folk", "indie folk", "acoustic", "singer-songwriter",
            "electronic", "edm", "idm", "breakbeat",
            "r&b", "soul", "funk", "disco",
            "metal", "heavy metal", "death metal", "black metal",
            "punk", "post-punk", "new wave",
            "reggae", "dub", "dancehall",
            "country", "blues", "gospel",
            # Instruments
            "piano", "guitar", "acoustic guitar", "electric guitar", "bass",
            "drums", "percussion", "synthesizer", "synth", "strings", "violin",
            "cello", "saxophone", "trumpet", "flute", "keyboard", "organ",
            "harp", "mandolin", "banjo", "ukulele", "accordion",
            # Vocal styles
            "male vocals", "female vocals", "choir", "harmonies", "backing vocals",
            # Mood/Character
            "melancholic", "upbeat", "energetic", "calm", "peaceful", "dark",
            "bright", "warm", "cold", "ethereal", "atmospheric", "groovy",
            # Tempo
            "slow", "mid-tempo", "fast", "upbeat",
        ]

        desc_lower = description.lower()
        for keyword in style_keywords:
            if keyword in desc_lower and keyword not in [t.lower() for t in tags]:
                tags.append(keyword.title())

        return ", ".join(tags[:10]) if tags else "Music"

    def _generate_concise_caption(self, description: str) -> str:
        """Generate a concise caption from the full description."""
        # Take the first sentence or up to 200 characters
        sentences = description.split('. ')
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 200:
                return first_sentence[:200] + "..."
            return first_sentence + ("." if not first_sentence.endswith('.') else "")
        return description[:200] + "..." if len(description) > 200 else description
