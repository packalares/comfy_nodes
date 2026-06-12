# ComfyUI-DocRes

Wraps [DocRes](https://github.com/ZZZHANG-jx/DocRes) (CVPR 2024) as a single
ComfyUI node. One model, five document-restoration tasks:

| task            | what it does                                            |
| --------------- | ------------------------------------------------------- |
| `appearance`    | CamScanner-style cleanup — flatten bg, boost contrast   |
| `deshadowing`   | remove cast shadows from paper                          |
| `dewarping`     | flatten a curled or folded page                         |
| `deblurring`    | sharpen mild motion / focus blur                        |
| `binarization`  | pure B/W — output stretched to 3-channel grayscale      |
| `end2end`       | dewarp → deshadow → appearance, in sequence             |

DocRes is trained on document datasets, so unlike SD-based upscalers it
will **not hallucinate text** — text stays faithful.

## Setup

Everything is automatic on first node run:

1. Repo is `git clone`d into `./DocRes/`
2. `docres.pkl` (~175 MB) is pulled from Hugging Face into `DocRes/checkpoints/`
3. `mbd.pkl` (~680 MB) is pulled into `DocRes/data/MBD/checkpoint/` — only the
   first time you run `dewarping` or `end2end`

Watch the ComfyUI console for `[ComfyUI-DocRes]` log lines.

Sources used (URLs hard-coded in `__init__.py`):

- repo: <https://github.com/ZZZHANG-jx/DocRes.git>
- `docres.pkl`: <https://huggingface.co/DaVinciCode/doctra-docres-main>
- `mbd.pkl`: <https://huggingface.co/DaVinciCode/doctra-docres-mbd>

Manual install of Python deps in your ComfyUI env may still be needed if any
DocRes dependency isn't already there:

```bash
pip install -r DocRes/requirements.txt
```

GPU is strongly recommended. CPU works but the model runs in fp16 for most
tasks, which is slow on CPU.

## Node

After ComfyUI restart you'll find **DocRes (Document Restoration)** under
`image / document`. IMAGE in, IMAGE out, one `task` dropdown.

## CamScanner-like pipeline

Single node: `task = end2end`. That's it.

If you only need cleanup (no curl), use `appearance` — it's the fastest.
For best invoice quality, follow `end2end` with a text-aware upscaler
(`4x-UltraSharp` or `Real-ESRGAN-x4plus`).
