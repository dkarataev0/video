"""Paths, model registry and default settings."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Where models and per-project data live. On Kaggle /kaggle/temp is fast local
# disk that is wiped when the session ends (good for privacy).
_default_home = "/kaggle/temp/studio" if Path("/kaggle").exists() else str(Path.home() / ".cache" / "ai_studio")
HOME = Path(os.environ.get("STUDIO_HOME", _default_home))
MODELS_DIR = HOME / "models"
PROJECTS_DIR = HOME / "projects"

SEED_VC_DIR = Path(os.environ.get("SEED_VC_DIR", str(HOME / "seed-vc")))
SEED_VC_REPO = "https://github.com/Plachtaa/seed-vc.git"
# Commit verified against this code (inference.py CLI + load_models/main API).
SEED_VC_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"
DAC_SHIM_DIR = ROOT / "third_party" / "dac_shim"

FF = "https://github.com/facefusion/facefusion-assets/releases/download"

# Face models (ONNX). Licenses are listed so the user knows the terms.
FACE_MODELS = {
    # swappers
    "hyperswap_1a_256": {"url": f"{FF}/models-3.3.0/hyperswap_1a_256.onnx", "type": "hyperswap", "size": 256,
                         "license": "ResearchRAIL (FaceFusion)"},
    "hyperswap_1b_256": {"url": f"{FF}/models-3.3.0/hyperswap_1b_256.onnx", "type": "hyperswap", "size": 256,
                         "license": "ResearchRAIL (FaceFusion)"},
    "hyperswap_1c_256": {"url": f"{FF}/models-3.3.0/hyperswap_1c_256.onnx", "type": "hyperswap", "size": 256,
                         "license": "ResearchRAIL (FaceFusion)"},
    "inswapper_128_fp16": {"url": f"{FF}/models-3.0.0/inswapper_128_fp16.onnx", "type": "inswapper", "size": 128,
                           "license": "Non-Commercial (InsightFace)"},
    # helpers
    "gfpgan_1.4": {"url": f"{FF}/models-3.0.0/gfpgan_1.4.onnx", "license": "Apache-2.0"},
    "xseg_1": {"url": f"{FF}/models-3.1.0/xseg_1.onnx", "license": "GPL-3.0 (DeepFaceLab XSeg)"},
}
SWAPPER_CHOICES = ["hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256", "inswapper_128_fp16"]

# Aligned-face templates (normalised 5-point landmarks), same values as FaceFusion.
WARP_TEMPLATES = {
    "arcface_128": [[0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219],
                    [0.38710391, 0.72160547], [0.61507734, 0.72034453]],
    "ffhq_512": [[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                 [0.39308822, 0.72541100], [0.61150205, 0.72490465]],
}

# Vocal separation model for python-audio-separator (BS-RoFormer, vocals SDR 12.97).
SEPARATOR_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

MAX_CHARACTERS = 8
MAX_SPEAKERS = 6
SEGMENT_SECONDS = 10          # render granularity (used for re-generation of fragments)
TEST_SECONDS = 5
MAX_SHORT_SIDE = 1080         # "up to 1080p"

DEFAULT_SETTINGS = {
    # faces
    "swapper": "hyperswap_1a_256",
    "pixel_boost": 512,         # 256 (fast) / 512 (sharper) / 768
    "face_weight": 0.5,         # 0..1 (FaceFusion semantics): >0.5 pushes identity further to the new face, <0.5 keeps more of the original
    "enhancer_blend": 40,       # 0..100 GFPGAN blend; 0 = off
    "occlusion_mask": True,     # XSeg: keeps hands/hair/glasses in front of the face
    "mask_blur": 0.3,
    "deflicker": 0.5,           # 0..1 temporal stabilisation strength
    "landmark_smoothing": 2,    # Savitzky-Golay half-window (frames); 0 = off
    "min_face_px": 40,
    # voices
    "vc_steps": 30,
    "vc_cfg_rate": 0.7,
    "vc_keep_intonation": True,  # f0-conditioned 44.1 kHz model + auto pitch adjust
    "vc_semitones": 0,
    "clean_reference": True,     # run separation on voice samples (remove music/noise)
    # output
    "crf": 17,
    "label_text": "Создано с помощью ИИ",
    "label_position": "bottom-right",
}
