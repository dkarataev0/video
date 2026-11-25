"""End-to-end smoke test on synthetic data (works on CPU, slowly; ~1 min on a T4).

    python tests/smoke_test.py

Builds a 14 s video from InsightFace's sample group photo (6 faces, camera pan/zoom)
with two speakers (Seed-VC examples) over a synthetic music bed, then runs:
analysis -> assign 1 face + 1 voice -> 5 s test -> full render -> fragment re-generation.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.gettempdir()) / "studio_smoke"))

from studio.config import SEED_VC_DIR  # noqa: E402
from studio.media import probe  # noqa: E402
from studio.pipeline import Project  # noqa: E402


def make_inputs(d: Path):
    import insightface
    import librosa

    imgs = Path(insightface.__file__).parent / "data" / "images"
    img = cv2.imread(str(imgs / "t1.jpg"))
    H, W = img.shape[:2]
    fps, n = 12, 12 * 14
    vw = cv2.VideoWriter(str(d / "raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 360))
    for t in range(n):
        s = 1.0 + 0.1 * np.sin(t / 15)
        cx, cy = W / 2 + 80 * np.sin(t / 20), H / 2
        w, h = W / s * 0.9, W / s * 0.9 * 360 / 640
        m = cv2.getAffineTransform(np.float32([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx - w / 2, cy + h / 2]]),
                                   np.float32([[0, 0], [640, 0], [0, 360]]))
        vw.write(cv2.warpAffine(img, m, (640, 360), borderMode=cv2.BORDER_REFLECT))
    vw.release()
    ex = SEED_VC_DIR / "examples"
    a, _ = librosa.load(ex / "source" / "source_s1.wav", sr=44100)
    b, _ = librosa.load(ex / "source" / "source_s4.wav", sr=44100)
    sp = np.concatenate([a[:6 * 44100], np.zeros(int(0.7 * 44100)), b[:6 * 44100]])
    tt = np.arange(14 * 44100) / 44100
    mix = 0.05 * (np.sin(2 * np.pi * 220 * tt) + 0.5 * np.sin(2 * np.pi * 330 * tt))
    mix[:len(sp)] += sp * 0.8
    sf.write(d / "mix.wav", np.stack([mix, mix], 1), 44100)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", d / "raw.mp4", "-i", d / "mix.wav", "-c:v", "libx264",
                    "-c:a", "aac", "-shortest", d / "in.mp4"], check=True)
    return d / "in.mp4", imgs / "Tom_Hanks_54745.png", ex / "reference" / "s3p1.wav"


def run(p: Project, name, fn, *args):
    t = time.time()
    p.start(name, fn, *args)
    while p.running:
        time.sleep(1)
    assert p.state["status"] == "done", p.state["error"]
    print(f"[ok] {name}: {time.time() - t:.0f} s")


def main():
    d = Path(tempfile.mkdtemp())
    video, face, voice = make_inputs(d)
    p = Project()
    p.wipe()
    p.set_video(str(video))
    run(p, "analyze", p.analyze_job)
    chars, spk = p.state["characters"], p.state["speakers"]
    print(f"characters={len(chars)} speakers={len(spk)} segments={p.state['segments']}")
    assert len(chars) >= 4 and len(spk) == 2
    p.set_face_photos(0, [str(face)])
    p.set_voice_files(1, [str(voice)])
    if os.environ.get("SMOKE_FAST"):
        p.set_settings(pixel_boost=256, enhancer_blend=0, vc_steps=10)
    run(p, "test", p.test_job, 4.0)
    info = probe(p.state["test"])
    assert info["audio"] is not None and info["video"]["r_frame_rate"] == "12/1", info["video"]
    run(p, "full", p.full_job)
    info = probe(p.state["result"])
    dur = float(info["format"]["duration"])
    assert abs(dur - 14.0) < 0.2, dur
    run(p, "regen", p.full_job, (2.0, 7.0), "all")
    print("result:", p.state["result"])


if __name__ == "__main__":
    main()
