"""Load an audio file from an absolute filesystem path → ComfyUI AUDIO dict.

ComfyUI's built-in LoadAudio only accepts filenames inside its `input/`
directory. This node accepts any absolute path the ComfyUI process can read,
so a backend that drops audio in its own runtime state directory can wire
it directly into Omni Audio Analyze without copying.

Decoding uses PyAV (libavformat) — the exact same path ComfyUI's own audio
loaders use. Whatever formats ComfyUI's built-in LoadAudio supports for
input/ files (mp3/wav/flac/m4a/ogg/...), we support too.
"""
import os

import av
import torch


def _f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert a PCM tensor (int16/int32/float) to float32 in [-1, 1]."""
    if wav.dtype.is_floating_point:
        return wav
    if wav.dtype == torch.int16:
        return wav.float() / (2 ** 15)
    if wav.dtype == torch.int32:
        return wav.float() / (2 ** 31)
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def _decode_via_av(path: str) -> tuple[torch.Tensor, int]:
    with av.open(path) as af:
        if not af.streams.audio:
            raise RuntimeError("File has no audio stream")
        stream = af.streams.audio[0]
        sr = stream.codec_context.sample_rate
        n_channels = stream.channels

        frames: list[torch.Tensor] = []
        for frame in af.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())
            # PyAV may emit interleaved or planar; normalize to (channels, samples).
            if buf.shape[0] != n_channels:
                buf = buf.view(-1, n_channels).t()
            frames.append(buf)

        if not frames:
            raise RuntimeError("No audio frames decoded")

        wav = torch.cat(frames, dim=1)
        return _f32_pcm(wav), int(sr)


class OMNI_AUDIO_LOAD_AUDIO_PATH:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Absolute path to an audio file (mp3/wav/flac/m4a/ogg/...). Must be readable by the ComfyUI process.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load"
    CATEGORY = "Omni Audio"

    def load(self, path: str):
        if not path or not path.strip():
            raise RuntimeError("OMNI_AUDIO_LoadAudioPath: 'path' is required")
        resolved = path.strip()
        if not os.path.exists(resolved):
            raise RuntimeError(f"OMNI_AUDIO_LoadAudioPath: file not found — {resolved}")
        try:
            waveform, sr = _decode_via_av(resolved)
        except av.AVError as exc:
            raise RuntimeError(
                f"OMNI_AUDIO_LoadAudioPath: PyAV could not decode {resolved} ({exc})",
            ) from exc

        # ComfyUI AUDIO convention: (batch=1, channels, samples) float32 torch tensor.
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sr},)
