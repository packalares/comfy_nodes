
import os
import torch
import numpy as np
import librosa
import folder_paths # Ensure this is accessible. It is provided by ComfyUI context.

import comfy.utils
import comfy.model_management

# Share family detection, loaders, the "no auto-download" path resolver, the
# legacy model directory constant, and the progress streamer with the
# captioner node. Relative import so this works when ComfyUI loads us as a
# package (the only supported entry).
from .nodes_captioner import (
    detect_omni_family,
    _load_qwen2_5_omni,
    _load_qwen3_omni,
    _resolve_local_path,
    ACESTEP_MODEL_NAME,
    ComfyStreamer,
    NO_MODEL_SENTINEL,
    INSTALL_HINT,
)
import inspect


def get_omni_transcriber_models():
    """Return the dropdown choices for the Omni Audio Transcriber node.

    Restricted to ACE-Step Transcriber Qwen2.5-Omni-7B finetune variants —
    looks only under ComfyUI/models/acestep-transcriber/ (top level + one
    level deep) plus the legacy ComfyUI/models/Ace-Step1.5/transcriber/
    location. Auto-download from HF is disabled.
    """
    models = []

    # Primary location
    root = os.path.join(folder_paths.models_dir, "acestep-transcriber")
    if os.path.exists(root):
        if os.path.exists(os.path.join(root, "config.json")):
            models.append("acestep-transcriber")
        else:
            for name in sorted(os.listdir(root)):
                sub = os.path.join(root, name)
                if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "config.json")):
                    models.append(os.path.join("acestep-transcriber", name))

    # Legacy layout
    legacy = os.path.join(folder_paths.models_dir, ACESTEP_MODEL_NAME, "transcriber")
    if os.path.exists(os.path.join(legacy, "config.json")):
        models.append(os.path.join(ACESTEP_MODEL_NAME, "transcriber"))

    if not models:
        return [NO_MODEL_SENTINEL]
    return sorted(set(models))

class OMNI_AUDIO_TRANSCRIBER:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio to transcribe."}),
                "model_id": (get_omni_transcriber_models(), {"default": get_omni_transcriber_models()[0], "tooltip": "Locally-installed Omni audio model. No auto-download. Install manually, e.g.:\n    huggingface-cli download ACE-Step/acestep-transcriber --local-dir ComfyUI/models/acestep-transcriber"}),
                "device": (["cuda", "cpu", "mps", "auto"], {"default": "auto", "tooltip": "Inference device. Use 'auto' or 'mps' for Mac."}),
                "dtype": (["auto", "float16", "float32"], {"default": "auto", "tooltip": "Model precision. 'auto' uses float16 for CUDA and float32 for CPU/MPS."}),
                "language": (["auto", "en", "zh", "ja", "ko", "fr", "de", "es", "it", "ru", "pt"], {"default": "auto", "tooltip": "Target language for transcription. 'auto' uses default prompt."}),
                "chunk_length_s": ("FLOAT", {"default": 450.0, "min": 0.0, "max": 1800.0, "step": 10.0, "tooltip": "Audio chunk length in seconds. 450s covers ~7.5-min songs in one shot (no chunking). Drop lower for >7.5-min audio."}),
                "return_timestamps": (["true", "false", "word"], {"default": "false", "tooltip": "Whether to return timestamps. 'word' for word-level timestamps, 'true' for segment-level."}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Custom prompt to override built-in language prompts. e.g. 'Transcribe the audio to Chinese:'"}),
                "max_new_tokens": ("INT", {"default": 4096, "min": 64, "max": 8192, "tooltip": "Maximum number of tokens to generate. Increase for longer lyrics."}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Sampling temperature. Lower values are more deterministic."}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Nucleus sampling: cumulative probability threshold."}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.1, "tooltip": "Penalty for repeating tokens. Increase if output gets stuck in loops."}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": "Number of beams for beam search. 1 = no beam search."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff, "tooltip": "Random seed for reproducible results. 0 for random."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcription",)
    FUNCTION = "transcribe"
    CATEGORY = "Omni Audio"

    def transcribe(self, audio, model_id, device, dtype, language, chunk_length_s, return_timestamps, custom_prompt="", max_new_tokens=4096, temperature=0.2, top_p=0.95, repetition_penalty=1.1, num_beams=1, seed=0):
        print(f"OMNI_AUDIO_TRANSCRIBER: Transcribing with model {model_id} on {device} ({dtype})")

        # Set random seed for reproducibility
        if seed != 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            print(f"OMNI_AUDIO_TRANSCRIBER: Using seed {seed}")
        
        # 1. Device Setup
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        print(f"OMNI_AUDIO_TRANSCRIBER: Using device: {device}")

        # 2. Path Resolution — local-only, no HF auto-download.
        if model_id == NO_MODEL_SENTINEL:
            raise RuntimeError(
                "OMNI_AUDIO_TRANSCRIBER: no transcriber model is installed.\n" + INSTALL_HINT
            )
        load_path = _resolve_local_path(model_id)
        print(f"OMNI_AUDIO_TRANSCRIBER: Resolved model path: {load_path}")

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

        family = detect_omni_family(load_path)
        print(f"OMNI_AUDIO_TRANSCRIBER: Detected model family: {family}")

        try:
            if family == "qwen3_omni":
                model, processor = _load_qwen3_omni(load_path, device, torch_dtype, log_prefix="OMNI_AUDIO_TRANSCRIBER")
            else:
                model, processor = _load_qwen2_5_omni(load_path, device, torch_dtype, log_prefix="OMNI_AUDIO_TRANSCRIBER")
            try:
                gen_sig = inspect.signature(model.generate)
                model._supports_return_audio = "return_audio" in gen_sig.parameters
            except (TypeError, ValueError):
                model._supports_return_audio = False
            print("OMNI_AUDIO_TRANSCRIBER: Model and Processor loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load ASR model {model_id}: {e}")

        # 4. Audio Preparation
        # ComfyUI audio: {'waveform': [1, channels, samples], 'sample_rate': int}
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']
        
        # Convert to numpy and mono
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        
        # waveform shape is [batch, channels, samples]. 
        if waveform.ndim == 3:
            waveform = waveform[0] # [channels, samples]
        
        # Mix to mono if stereo
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = np.mean(waveform, axis=0)
        elif waveform.ndim > 1:
             waveform = waveform[0]
            
        # Ensure float32
        waveform = waveform.astype(np.float32)

        # Resample to 16kHz if needed (WhisperFeatureExtractor requires 16000Hz)
        TARGET_SR = 16000
        if sample_rate != TARGET_SR:
            print(f"OMNI_AUDIO_TRANSCRIBER: Resampling audio from {sample_rate}Hz to {TARGET_SR}Hz")
            try:
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=TARGET_SR)
                sample_rate = TARGET_SR
            except Exception as e:
                print(f"OMNI_AUDIO_TRANSCRIBER: Resampling failed: {e}")
                raise e

        # Calculate audio duration
        audio_duration = len(waveform) / sample_rate
        print(f"OMNI_AUDIO_TRANSCRIBER: Audio duration: {audio_duration:.1f}s")

        # Determine chunking strategy
        OVERLAP_S = 5  # 5 second overlap between chunks
        chunk_samples = int(chunk_length_s * sample_rate)
        overlap_samples = int(OVERLAP_S * sample_rate)

        # Only chunk if audio is longer than chunk_length
        needs_chunking = audio_duration > chunk_length_s

        if needs_chunking:
            # Calculate number of chunks
            step_samples = chunk_samples - overlap_samples
            num_chunks = int(np.ceil((len(waveform) - overlap_samples) / step_samples))
            print(f"OMNI_AUDIO_TRANSCRIBER: Splitting into {num_chunks} chunks (chunk={chunk_length_s}s, overlap={OVERLAP_S}s)")
        else:
            num_chunks = 1
            print(f"OMNI_AUDIO_TRANSCRIBER: Processing as single chunk")

        # 5. Inference
        try:
            # Construct Prompt with Chat Template
            # Official recommended prompt from ACE-Step: "Transcribe this audio in detail"
            # See: https://huggingface.co/ACE-Step/acestep-transcriber

            # Add timestamp instruction based on return_timestamps parameter
            if return_timestamps == "word":
                instruction = "Transcribe the sung lyrics of this audio with word-level timestamps. Format each line as: [MM:SS.ms] word. Skip section labels like [Verse]/[Chorus]."
            elif return_timestamps == "true":
                instruction = "Transcribe the sung lyrics of this audio with timestamps. Add [MM:SS] at the start of each section."
            else:
                instruction = "Transcribe ONLY the sung lyrics of this song, one lyric line per output line. Skip section labels (no [Verse]/[Chorus]/[Intro]). Do not add commentary or descriptions."

            lang_map = {
                "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
                "fr": "French", "de": "German", "es": "Spanish", "it": "Italian",
                "ru": "Russian", "pt": "Portuguese"
            }

            if custom_prompt.strip():
                instruction = custom_prompt.strip()
            elif language in lang_map:
                # Use official prompt format with language specification
                if return_timestamps == "word":
                    instruction = f"Transcribe this audio into {lang_map[language]} with word-level timestamps. Format each line as: [MM:SS.ms] word"
                elif return_timestamps == "true":
                    instruction = f"Transcribe this audio into {lang_map[language]} with timestamps. Add [MM:SS] at the beginning of each section."
                else:
                    instruction = f"Transcribe this audio in detail into {lang_map[language]}."

            print(f"OMNI_AUDIO_TRANSCRIBER: Instruction: {instruction}")

            # Build text prompt
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

            print(f"OMNI_AUDIO_TRANSCRIBER: Using prompt: '{text_prompt[:80]}...'")
            print(f"OMNI_AUDIO_TRANSCRIBER: Generation params: temp={temperature}, rep_penalty={repetition_penalty}")

            # Helper function to transcribe a single chunk
            def transcribe_chunk(audio_chunk, chunk_idx=None):
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

                # Beam search doesn't support streaming, but we still show progress
                if num_beams > 1:
                    print(f"OMNI_AUDIO_TRANSCRIBER: Running beam search (num_beams={num_beams}), progress bar will update on completion...")

                streamer = ComfyStreamer(pbar)

                extra_gen = {"return_audio": False} if getattr(model, "_supports_return_audio", False) else {}

                with torch.no_grad():
                    generation_output = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        streamer=streamer if num_beams == 1 else None,  # beam search doesn't support streaming
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        num_beams=num_beams,
                        do_sample=True if temperature > 0 and num_beams == 1 else False,
                        **extra_gen,
                    )

                # Update progress bar to completion for beam search
                if num_beams > 1:
                    pbar.update(max_new_tokens)

                generated_ids = generation_output
                full_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # Extract assistant's response
                transcription = full_output
                assistant_marker = "assistant"
                if assistant_marker in full_output:
                    parts = full_output.split(assistant_marker)
                    if len(parts) > 1:
                        transcription = parts[-1].strip()

                for marker in ["system", "user"]:
                    if f"\n{marker}" in transcription:
                        transcription = transcription.split(f"\n{marker}")[0].strip()

                return transcription

            # Process chunks
            all_transcriptions = []

            if needs_chunking:
                step_samples = chunk_samples - overlap_samples
                for i in range(num_chunks):
                    start = i * step_samples
                    end = min(start + chunk_samples, len(waveform))
                    chunk = waveform[start:end]

                    chunk_duration = len(chunk) / sample_rate
                    print(f"OMNI_AUDIO_TRANSCRIBER: Processing chunk {i+1}/{num_chunks} ({chunk_duration:.1f}s)")

                    chunk_result = transcribe_chunk(chunk, chunk_idx=i)
                    all_transcriptions.append(chunk_result)
                    print(f"OMNI_AUDIO_TRANSCRIBER: Chunk {i+1} result: {chunk_result[:50]}...")

                # Merge transcriptions
                # Simple merge: concatenate with newlines, remove duplicate headers
                merged = []
                seen_header = False
                for trans in all_transcriptions:
                    lines = trans.split('\n')
                    for line in lines:
                        # Skip duplicate language headers after first chunk
                        if line.strip().startswith('# Languages') or line.strip().startswith('# Lyrics'):
                            if not seen_header or line.strip() != '# Lyrics':
                                merged.append(line)
                                seen_header = True
                        else:
                            merged.append(line)

                transcription = '\n'.join(merged)
            else:
                print("OMNI_AUDIO_TRANSCRIBER: Running inference...")
                transcription = transcribe_chunk(waveform)

            print(f"OMNI_AUDIO_TRANSCRIBER: Final result: {transcription[:100]}...")
            return (transcription,)

        except Exception as e:
            print(f"OMNI_AUDIO_TRANSCRIBER: Inference failed: {e}")
            raise e
