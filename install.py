"""One-shot installer for Kaggle (GPU T4) or any Linux machine with an NVIDIA GPU.

    python install.py            # install everything, download models, self-check

Versions are pinned to a combination that was tested together (see docs/ARCHITECTURE.md).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

TORCH = ["torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
PACKAGES = [
    "numpy>=2,<2.3", "transformers==4.46.3", "librosa>=0.10.2", "munch==4.0.0", "einops>=0.8.0",
    "scipy>=1.13", "soundfile>=0.12.1", "huggingface_hub>=0.28.1", "matplotlib", "pydub",
    "insightface==2.1", "audio-separator==0.48.0", "silero-vad==6.3", "gradio==5.50.0",
    "scikit-learn", "opencv-python", "pillow-heif", "onnx",
]
ORT_GPU = "onnxruntime-gpu==1.22.0"


def sh(cmd: list[str] | str, check=True):
    print("$", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    r = subprocess.run(cmd, shell=isinstance(cmd, str))
    if check and r.returncode != 0:
        raise SystemExit(f"Ошибка установки на шаге: {cmd}")
    return r.returncode


def pip(*args):
    sh([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", *args])


def main():
    print("=== 1/5 Проверка GPU и ffmpeg ===")
    if shutil.which("nvidia-smi"):
        sh("nvidia-smi --query-gpu=name,memory.total --format=csv", check=False)
    else:
        print("⚠️  GPU не найден. В Kaggle: Settings → Accelerator → GPU T4 x2.")
    if not shutil.which("ffmpeg"):
        sh("apt-get update -qq && apt-get install -y -qq ffmpeg", check=False)

    print("=== 2/5 Установка библиотек (5–8 минут) ===")
    import importlib.metadata as md

    try:
        have = md.version("torch")
    except md.PackageNotFoundError:
        have = ""
    if not have.startswith("2.5.1"):
        pip(*TORCH, "--index-url", TORCH_INDEX)
    pip(*PACKAGES)
    # insightface/audio-separator may pull the CPU onnxruntime; it shadows the GPU build.
    sh([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "onnxruntime"], check=False)
    pip("coloredlogs", "flatbuffers", "sympy", "protobuf")
    pip("--force-reinstall", "--no-deps", ORT_GPU)  # CUDA 12 + cuDNN 9 (same libs as torch cu124)

    print("=== 3/5 Seed-VC (преобразование голоса) ===")
    from studio.config import HOME, SEED_VC_COMMIT, SEED_VC_DIR, SEED_VC_REPO

    HOME.mkdir(parents=True, exist_ok=True)
    if not (SEED_VC_DIR / "inference.py").exists():
        sh(["git", "clone", "-q", SEED_VC_REPO, str(SEED_VC_DIR)])
    sh(["git", "-C", str(SEED_VC_DIR), "checkout", "-q", SEED_VC_COMMIT])

    print("=== 4/5 Загрузка моделей лиц ===")
    from studio.config import FACE_MODELS
    from studio.models import face_model_path

    for name in ["hyperswap_1a_256", "gfpgan_1.4", "xseg_1"]:
        print("  ", name, "—", FACE_MODELS[name]["license"])
        face_model_path(name)

    print("=== 5/5 Самопроверка ===")
    check = (
        "import torch, onnxruntime as ort, gradio, insightface;"
        "from studio.models import providers;"
        "print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available());"
        "providers();"
        "print('onnxruntime', ort.__version__, ort.get_available_providers());"
        "print('gradio', gradio.__version__)"
    )
    sh([sys.executable, "-c", check], check=False)
    print("\n✅ Установка завершена. Запустите следующую ячейку.")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
