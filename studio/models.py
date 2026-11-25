"""Model download + ONNX Runtime session helpers."""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

from .config import FACE_MODELS, MODELS_DIR

_preloaded = False


def download(url: str, dst: Path, retries: int = 4) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ai-video-studio"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.rename(dst)
            return dst
        except Exception as e:  # network hiccup -> retry with backoff
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Не удалось скачать модель {url}: {last}. Проверьте, что в Kaggle включён Интернет.")


def face_model_path(name: str) -> Path:
    spec = FACE_MODELS[name]
    return download(spec["url"], MODELS_DIR / "faces" / f"{name}.onnx")


def providers() -> list:
    import onnxruntime as ort

    global _preloaded
    avail = ort.get_available_providers()
    if "CUDAExecutionProvider" in avail:
        if not _preloaded and hasattr(ort, "preload_dlls"):
            try:  # loads CUDA/cuDNN from the pip "nvidia-*" wheels that come with torch
                import torch  # noqa: F401
                ort.preload_dlls()
            except Exception:
                pass
            _preloaded = True
        return [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def session(path: Path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    if os.environ.get("STUDIO_THREADS"):
        so.intra_op_num_threads = int(os.environ["STUDIO_THREADS"])
    return ort.InferenceSession(str(path), sess_options=so, providers=providers())


def using_gpu() -> bool:
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False
