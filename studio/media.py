"""ffmpeg helpers: probing, normalisation, frame I/O, AI label, muxing."""
from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

from .config import MAX_SHORT_SIDE


class MediaError(RuntimeError):
    pass


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        tail = "\n".join(p.stderr.strip().splitlines()[-15:])
        raise MediaError(f"Команда завершилась с ошибкой: {' '.join(cmd[:3])} ...\n{tail}")
    return p.stdout


def probe(path: str | Path) -> dict:
    out = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)])
    info = json.loads(out)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if v is None:
        raise MediaError("В файле нет видеодорожки.")
    return {"video": v, "audio": a, "format": info.get("format", {})}


def _fps_of(v: dict) -> Fraction:
    for key in ("r_frame_rate", "avg_frame_rate"):
        try:
            f = Fraction(v.get(key, "0/0"))
        except (ZeroDivisionError, ValueError):
            continue
        if 1 <= f <= 240:
            return f.limit_denominator(1001)
    return Fraction(30, 1)


def has_filter(name: str) -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return any(line.split()[1:2] == [name] for line in out.splitlines() if len(line.split()) > 1)


def normalize_video(src: str | Path, dst: Path, audio_dst: Path) -> dict:
    """Convert any phone video into constant-fps SDR H.264 (short side <= 1080)
    plus a separate 44.1 kHz stereo WAV. Rotation metadata is applied by ffmpeg."""
    info = probe(src)
    v = info["video"]
    fps = _fps_of(v)
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = abs(int(float(sd["rotation"]))) % 180
    if rot == 90:
        w, h = h, w
    scale = min(1.0, MAX_SHORT_SIDE / min(w, h))
    tw, th = int(round(w * scale / 2) * 2), int(round(h * scale / 2) * 2)

    filters = []
    transfer = (v.get("color_transfer") or "").lower()
    hdr = transfer in ("smpte2084", "arib-std-b67")
    if hdr and has_filter("zscale") and has_filter("tonemap"):
        filters.append("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
                       "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv")
    filters.append(f"scale={tw}:{th}:flags=lanczos")
    filters.append("format=yuv420p")
    gop = max(1, round(float(fps)))
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-map", "0:v:0", "-vf", ",".join(filters),
          "-r", str(fps), "-fps_mode", "cfr", "-c:v", "libx264", "-preset", "veryfast", "-crf", "12",
          "-g", str(gop), "-bf", "0", "-color_primaries", "bt709", "-color_trc", "bt709",
          "-colorspace", "bt709", "-an", str(dst)])
    has_audio = info["audio"] is not None
    if has_audio:
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-map", "0:a:0", "-ac", "2", "-ar", "44100",
              "-c:a", "pcm_f32le", str(audio_dst)])
    nb = count_frames(dst)
    return {"fps_num": fps.numerator, "fps_den": fps.denominator, "width": tw, "height": th,
            "frames": nb, "has_audio": has_audio, "hdr_tonemapped": hdr,
            "duration": nb * fps.denominator / fps.numerator}


def count_frames(path: Path) -> int:
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)])
    return int(out.strip().split(",")[0])


class FrameReader:
    """Reads BGR frames [start, start+count) from the normalised video."""

    def __init__(self, path: Path, width: int, height: int, fps: Fraction, start: int = 0, count: int | None = None):
        self.w, self.h = width, height
        cmd = ["ffmpeg", "-v", "error"]
        if start > 0:
            # seek half a frame before the wanted frame: ffmpeg's accurate seek then drops
            # every earlier frame and keeps exactly frame `start`.
            cmd += ["-ss", f"{(start - 0.5) / float(fps):.6f}"]
        cmd += ["-i", str(path)]
        if count is not None:
            cmd += ["-frames:v", str(count)]
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 8)
        self.frame_bytes = width * height * 3

    def __iter__(self):
        while True:
            buf = self.proc.stdout.read(self.frame_bytes)
            if len(buf) < self.frame_bytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3).copy()
        self.close()

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


class FrameWriter:
    def __init__(self, path: Path, width: int, height: int, fps: Fraction, crf: int = 17):
        gop = max(1, round(float(fps)))
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-g", str(gop), "-color_primaries", "bt709", "-color_trc", "bt709",
               "-colorspace", "bt709", "-movflags", "+faststart", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="ignore")
        if self.proc.wait() != 0:
            raise MediaError("Ошибка кодирования видео: " + err[-800:])


# ---------------------------------------------------------------- AI label
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
]


def make_label_overlay(width: int, height: int, text: str, position: str = "bottom-right"):
    """Returns (bgr, alpha, x, y) for a semi-transparent 'AI-generated' badge."""
    from PIL import Image, ImageDraw, ImageFont

    size = max(14, int(min(width, height) * 0.032))
    font = None
    for f in _FONT_CANDIDATES:
        if Path(f).exists():
            font = ImageFont.truetype(f, size)
            break
    if font is None:  # no Cyrillic font available -> English label
        font = ImageFont.load_default(size=size)
        text = "AI-generated video"
    pad = size // 2
    tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    l, t, r, b = tmp.textbbox((0, 0), text, font=font)
    bw, bh = r - l + 2 * pad, b - t + 2 * pad
    img = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, bw - 1, bh - 1], radius=pad, fill=(0, 0, 0, 120))
    d.text((pad - l, pad - t), text, font=font, fill=(255, 255, 255, 235))
    arr = np.array(img)
    bgr = arr[:, :, 2::-1].astype(np.float32)
    alpha = (arr[:, :, 3:4].astype(np.float32)) / 255.0
    m = int(min(width, height) * 0.03)
    x = width - bw - m if "right" in position else m
    y = height - bh - m if "bottom" in position else m
    return bgr, alpha, x, y


def apply_label(frame: np.ndarray, overlay) -> np.ndarray:
    bgr, alpha, x, y = overlay
    h, w = bgr.shape[:2]
    roi = frame[y:y + h, x:x + w].astype(np.float32)
    frame[y:y + h, x:x + w] = (roi * (1 - alpha) + bgr * alpha).astype(np.uint8)
    return frame


# ---------------------------------------------------------------- final assembly
def concat_and_mux(segment_files: list[Path], audio_wav: Path | None, out_path: Path, label_text: str):
    lst = out_path.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in segment_files))
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst)]
    if audio_wav is not None and Path(audio_wav).exists():
        cmd += ["-i", str(audio_wav), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "copy", "-movflags", "+faststart",
            "-metadata", f"comment={label_text}. AI-modified video (face swap / voice conversion).",
            "-metadata", "description=AI-generated / modified with AI", str(out_path)]
    _run(cmd)
    lst.unlink(missing_ok=True)


def cut_audio(src: Path, dst: Path, start: float, duration: float):
    _run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(src),
          "-c:a", "pcm_f32le", str(dst)])


def to_wav(src: str | Path, dst: Path, sr: int = 44100, mono: bool = False):
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ar", str(sr)]
    if mono:
        cmd += ["-ac", "1"]
    _run(cmd + ["-c:a", "pcm_f32le", str(dst)])


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
