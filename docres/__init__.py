"""ComfyUI wrapper for DocRes (CVPR 2024 — document restoration).

Exposes a single node `DocRes (Document Restoration)` that runs one of the
five DocRes tasks on an IMAGE:

    - appearance       — "CamScanner" cleanup (background normalisation, contrast)
    - deshadowing      — remove cast shadows from paper
    - dewarping        — flatten a curled/folded page (needs MBD checkpoint)
    - deblurring       — sharpen mild motion / focus blur
    - binarization     — pure black-on-white (returned as 3-channel grayscale)
    - end2end          — dewarp → deshadow → appearance, sequentially

Repo + checkpoints are fetched automatically on first use. To prefill
manually, see the URLs in `_REPO_URL`, `_DOCRES_CKPT_URL`, `_MBD_CKPT_URL`.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request

import cv2
import numpy as np
import torch


_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_DOCRES_DIR = os.path.join(_NODE_DIR, 'DocRes')

_REPO_URL = 'https://github.com/ZZZHANG-jx/DocRes.git'
_DOCRES_CKPT_URL = 'https://huggingface.co/DaVinciCode/doctra-docres-main/resolve/main/docres.pkl'
_MBD_CKPT_URL = 'https://huggingface.co/DaVinciCode/doctra-docres-mbd/resolve/main/mbd.pkl'

_TASKS = ['appearance', 'deshadowing', 'dewarping', 'deblurring', 'binarization', 'end2end']
# Tasks whose prompt-builder needs the MBD checkpoint to be present.
_MBD_REQUIRED_TASKS = {'dewarping', 'end2end'}

_model = None
_inference_mod = None


class _ChdirTo:
    def __init__(self, target):
        self.target = target
        self._prev = None

    def __enter__(self):
        self._prev = os.getcwd()
        os.chdir(self.target)

    def __exit__(self, *exc):
        os.chdir(self._prev)


def _log(msg):
    print(f'[ComfyUI-DocRes] {msg}', flush=True)


def _download(url, dest):
    _log(f'downloading {url}')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + '.partial'
    last_pct = [-1]

    def hook(block_n, block_sz, total_sz):
        if total_sz <= 0:
            return
        pct = int(min(100, (block_n * block_sz) * 100 / total_sz))
        # Log every 10% to keep the console readable.
        if pct >= last_pct[0] + 10:
            last_pct[0] = pct
            _log(f'  … {pct}%')

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    os.replace(tmp, dest)
    _log(f'saved → {dest}')


def _ensure_repo():
    if os.path.isfile(os.path.join(_DOCRES_DIR, 'inference.py')):
        return
    if not shutil.which('git'):
        raise RuntimeError(
            f"DocRes repo missing at {_DOCRES_DIR} and `git` is not on PATH. "
            f"Manually clone {_REPO_URL} there."
        )
    _log(f'cloning {_REPO_URL} → {_DOCRES_DIR}')
    subprocess.run(
        ['git', 'clone', '--depth', '1', _REPO_URL, _DOCRES_DIR],
        check=True,
    )


def _ensure_main_ckpt():
    p = os.path.join(_DOCRES_DIR, 'checkpoints', 'docres.pkl')
    if not os.path.isfile(p):
        _download(_DOCRES_CKPT_URL, p)
    return p


def _ensure_mbd_ckpt():
    p = os.path.join(_DOCRES_DIR, 'data', 'MBD', 'checkpoint', 'mbd.pkl')
    if not os.path.isfile(p):
        _download(_MBD_CKPT_URL, p)
    return p


def _load_inference_module():
    """Import the DocRes inference module once. Must run with CWD == DocRes
    root because the module body appends './data/MBD/' to sys.path at import
    time."""
    global _inference_mod
    if _inference_mod is not None:
        return _inference_mod
    _ensure_repo()
    if _DOCRES_DIR not in sys.path:
        sys.path.insert(0, _DOCRES_DIR)
    with _ChdirTo(_DOCRES_DIR):
        try:
            import inference as inf  # noqa: WPS433
        except ImportError as e:
            raise RuntimeError(
                f"DocRes import failed ({e}). Install its Python deps inside "
                f"your ComfyUI env: pip install -r {_DOCRES_DIR}/requirements.txt"
            ) from e
    _inference_mod = inf
    return inf


def _get_model(device):
    global _model
    if _model is not None:
        _model.to(device)
        return _model

    inf = _load_inference_module()
    ckpt = _ensure_main_ckpt()

    # The inference module re-uses a module-level DEVICE global set by its
    # own __main__. We replicate that.
    inf.DEVICE = device

    with _ChdirTo(_DOCRES_DIR):
        from models import restormer_arch
        from utils import convert_state_dict
        model = restormer_arch.Restormer(
            inp_channels=6, out_channels=3, dim=48,
            num_blocks=[2, 3, 3, 4], num_refinement_blocks=4,
            heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
            bias=False, LayerNorm_type='WithBias', dual_pixel_task=True,
        )
        map_loc = 'cpu' if device.type == 'cpu' else 'cuda:0'
        state = convert_state_dict(torch.load(ckpt, map_location=map_loc)['model_state'])
        model.load_state_dict(state)
        model.eval().to(device)

    _model = model
    return model


def _tensor_to_bgr_uint8(image):
    """ComfyUI IMAGE is (B, H, W, 3) float32 RGB in [0,1]. We use the first
    item; multi-batch goes through one item at a time at the caller."""
    arr = image.detach().cpu().numpy()
    arr = (arr.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _bgr_uint8_to_tensor(img):
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _run_one(model, inf, device, bgr, task):
    fd, tmp_path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    try:
        cv2.imwrite(tmp_path, bgr)
        with _ChdirTo(_DOCRES_DIR):
            if task == 'end2end':
                os.makedirs('restorted', exist_ok=True)
            inf.DEVICE = device
            _, _, _, restored = inf.inference_one_im(model, tmp_path, task)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return restored


class DocResProcess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'task': (_TASKS, {'default': 'appearance'}),
            },
        }

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('image',)
    FUNCTION = 'run'
    CATEGORY = 'image/document'

    def run(self, image, task):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inf = _load_inference_module()
        model = _get_model(device)
        if task in _MBD_REQUIRED_TASKS:
            _ensure_mbd_ckpt()

        # Batch loop — one item at a time, then stack back. Most document
        # photos come in batches of 1 but keep the contract clean.
        outs = []
        for i in range(image.shape[0]):
            bgr = _tensor_to_bgr_uint8(image[i])
            restored = _run_one(model, inf, device, bgr, task)
            outs.append(_bgr_uint8_to_tensor(restored))

        # Each item may have a different output H/W (dewarping in particular
        # preserves source size). Pad to the largest H/W with zeros so we can
        # stack into a single tensor — ComfyUI requires uniform batch shapes.
        max_h = max(t.shape[1] for t in outs)
        max_w = max(t.shape[2] for t in outs)
        padded = []
        for t in outs:
            _, h, w, _ = t.shape
            if h == max_h and w == max_w:
                padded.append(t)
            else:
                pad = torch.zeros((1, max_h, max_w, 3), dtype=t.dtype)
                pad[0, :h, :w, :] = t[0]
                padded.append(pad)
        return (torch.cat(padded, dim=0),)


NODE_CLASS_MAPPINGS = {
    'DocResProcess': DocResProcess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'DocResProcess': 'DocRes (Document Restoration)',
}
