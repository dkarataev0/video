"""Audio side: vocal/background separation, speaker diarisation, reference preparation,
voice conversion (Seed-VC, in a subprocess) and final mixing.

Voice conversion (not TTS) is used on purpose: it keeps the original words, emotion,
intonation, pauses and tempo, and the output has exactly the input length, so the
converted speech stays in sync with the original lip movements.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import DAC_SHIM_DIR, MODELS_DIR, SEED_VC_DIR, SEPARATOR_MODEL
from .media import to_wav

SR = 44100


def load_mono(path: Path, sr: int = SR) -> np.ndarray:
    import librosa

    y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32)


def load_stereo(path: Path, sr: int = SR) -> np.ndarray:
    import librosa

    y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, 1)
    y = y[:, :2]
    if file_sr != sr:
        y = np.stack([librosa.resample(y[:, c], orig_sr=file_sr, target_sr=sr) for c in range(2)], 1)
    return y.astype(np.float32)


# ------------------------------------------------------------------ separation
_separator = None


def separate(src_wav: Path, out_dir: Path) -> tuple[Path, Path]:
    """BS-RoFormer via python-audio-separator -> (vocals.wav, background.wav)."""
    global _separator
    from audio_separator.separator import Separator

    out_dir.mkdir(parents=True, exist_ok=True)
    if _separator is None:
        _separator = Separator(model_file_dir=str(MODELS_DIR / "separator"), output_dir=str(out_dir),
                               output_format="WAV", sample_rate=SR)
        _separator.load_model(model_filename=SEPARATOR_MODEL)
    _separator.output_dir = str(out_dir)
    if getattr(_separator, "model_instance", None) is not None:
        _separator.model_instance.output_dir = str(out_dir)
    stem = src_wav.stem
    names = {"Vocals": f"{stem}_vocals", "Instrumental": f"{stem}_background"}
    files = _separator.separate(str(src_wav), custom_output_names=names)
    found = {}
    for f in files:
        p = Path(f) if Path(f).is_absolute() else out_dir / f
        if p.name.startswith(f"{stem}_vocals"):
            found["v"] = p
        elif p.name.startswith(f"{stem}_background"):
            found["b"] = p
    if len(found) != 2:
        raise RuntimeError(f"Разделение аудио не дало ожидаемых дорожек: {files}")
    return found["v"], found["b"]


# ------------------------------------------------------------------ diarisation
_vad = None
_spk = None


def _speaker_model():
    """CAM++ speaker-embedding model (same checkpoint Seed-VC uses for timbre)."""
    global _spk
    if _spk is None:
        import torch
        from huggingface_hub import hf_hub_download

        sys.path.insert(0, str(SEED_VC_DIR))
        from modules.campplus.DTDNN import CAMPPlus  # type: ignore

        ckpt = hf_hub_download("funasr/campplus", "campplus_cn_common.bin", cache_dir=str(MODELS_DIR / "hf"))
        m = CAMPPlus(feat_dim=80, embedding_size=192)
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _spk = (m.eval().to(dev), dev)
    return _spk


def _embed(wave16: np.ndarray) -> np.ndarray:
    import torch
    import torchaudio

    model, dev = _speaker_model()
    x = torch.from_numpy(wave16).float()[None]
    feat = torchaudio.compliance.kaldi.fbank(x, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(0, keepdim=True)
    with torch.no_grad():
        e = model(feat[None].to(dev))[0].cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9)


def speech_regions(wave16: np.ndarray) -> list[tuple[float, float]]:
    global _vad
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    if _vad is None:
        _vad = load_silero_vad()
    ts = get_speech_timestamps(torch.from_numpy(wave16), _vad, sampling_rate=16000, return_seconds=True,
                               min_silence_duration_ms=200, speech_pad_ms=80, min_speech_duration_ms=200)
    return [(float(t["start"]), float(t["end"])) for t in ts]


def diarize(vocals_wav: Path, n_speakers: int | None = None, threshold: float = 0.6, progress=None) -> list[dict]:
    """Returns [{start, end, speaker}] sorted by start; speakers numbered by talk time."""
    wave = load_mono(vocals_wav, 16000)
    regions = speech_regions(wave)
    if not regions:
        return []
    win, hop = 1.5, 0.5
    centers, embs = [], []
    for i, (a, b) in enumerate(regions):
        if progress:
            progress(i / len(regions), f"Анализ голосов: фрагмент {i + 1}/{len(regions)}")
        starts = [a] if b - a <= win else list(np.arange(a, b - win + 1e-6, hop))
        for s in starts:
            e = min(b, s + win)
            seg = wave[int(s * 16000):int(e * 16000)]
            if len(seg) < 8000:  # < 0.5 s: pad by reflection so the model gets enough context
                seg = np.pad(seg, (0, 8000 - len(seg)), mode="wrap")
            embs.append(_embed(seg))
            centers.append((s + e) / 2)
    embs = np.stack(embs)
    if len(embs) == 1:
        labels = np.array([0])
    else:
        from sklearn.cluster import AgglomerativeClustering

        if n_speakers:
            cl = AgglomerativeClustering(n_clusters=min(n_speakers, len(embs)), metric="cosine", linkage="average")
        else:
            cl = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                         distance_threshold=threshold)
        labels = cl.fit_predict(embs)
    centers = np.asarray(centers)
    # label every 50 ms inside speech regions by the nearest window centre, then median-filter
    segs = []
    for a, b in regions:
        grid = np.arange(a, b, 0.05)
        if len(grid) == 0:
            continue
        idx = np.abs(centers[None, :] - grid[:, None]).argmin(1)
        lab = labels[idx]
        if len(lab) >= 7:
            from scipy.ndimage import median_filter
            lab = median_filter(lab, size=7, mode="nearest")
        start = 0
        for i in range(1, len(lab) + 1):
            if i == len(lab) or lab[i] != lab[start]:
                segs.append({"start": float(grid[start]), "end": float(min(b, grid[i - 1] + 0.05)), "speaker": int(lab[start])})
                start = i
    # absorb very short turns (< 0.3 s) into the previous segment of the same region
    clean = []
    for s in segs:
        if clean and s["end"] - s["start"] < 0.3 and abs(clean[-1]["end"] - s["start"]) < 0.06:
            clean[-1]["end"] = s["end"]
        else:
            clean.append(s)
    # renumber speakers by total talk time
    tot = {}
    for s in clean:
        tot[s["speaker"]] = tot.get(s["speaker"], 0) + s["end"] - s["start"]
    order = {spk: i for i, spk in enumerate(sorted(tot, key=lambda k: -tot[k]))}
    for s in clean:
        s["speaker"] = order[s["speaker"]]
        s["start"], s["end"] = round(s["start"], 2), round(s["end"], 2)
    return clean


def speaker_preview(vocals_wav: Path, segments: list[dict], speaker: int, dst: Path, max_sec: float = 8.0):
    y = load_mono(vocals_wav)
    parts, total = [], 0.0
    for s in sorted((s for s in segments if s["speaker"] == speaker), key=lambda s: -(s["end"] - s["start"])):
        a, b = int(s["start"] * SR), int(s["end"] * SR)
        parts.append(y[a:b])
        parts.append(np.zeros(int(0.25 * SR), np.float32))
        total += s["end"] - s["start"]
        if total >= max_sec:
            break
    if parts:
        sf.write(str(dst), np.concatenate(parts), SR)
        return dst
    return None


# ------------------------------------------------------------------ references
def prepare_reference(paths: list[str], dst: Path, work: Path, clean: bool) -> float:
    """Concatenates the speech of the provided samples (any audio/video format), optionally
    removes music/noise, trims silences, loudness-normalises; Seed-VC uses up to 25 s."""
    work.mkdir(parents=True, exist_ok=True)
    chunks = []
    for i, p in enumerate(paths):
        wav = work / f"ref_{i}.wav"
        to_wav(p, wav)
        if clean:
            try:
                wav, _ = separate(wav, work)
            except Exception as e:  # separation is a nicety; keep the raw sample if it fails
                print("reference cleaning skipped:", e)
        y = load_mono(wav)
        y16 = load_mono(wav, 16000)
        regs = speech_regions(y16) or [(0.0, len(y) / SR)]
        for a, b in regs:
            chunks.append(y[int(a * SR):int(b * SR)])
            chunks.append(np.zeros(int(0.15 * SR), np.float32))
    if not chunks:
        raise RuntimeError("В образце голоса не найдена речь.")
    y = np.concatenate(chunks)[: int(25 * SR)]
    peak = np.abs(y).max() + 1e-9
    rms = np.sqrt((y ** 2).mean()) + 1e-9
    y = y * min(0.1 / rms, 0.95 / peak)
    sf.write(str(dst), y, SR)
    return len(y) / SR


# ------------------------------------------------------------------ conversion
PAD = 0.25  # seconds of context around each segment


def extract_piece(vocals: np.ndarray, seg: dict, dst: Path) -> dict:
    """Cuts one diarised segment (+context) for conversion; returns its placement info."""
    n = len(vocals)
    a = max(0, int((seg["start"] - PAD) * SR))
    b = min(n, int((seg["end"] + PAD) * SR))
    sf.write(str(dst), vocals[a:b], SR)
    return {"src_a": a, "src_b": b, "seg_a": int(seg["start"] * SR), "seg_b": int(seg["end"] * SR)}


def run_seed_vc(jobs: list[dict], log_path: Path, progress=None):
    """Runs Seed-VC in a separate process (keeps GPU memory of the UI process clean)."""
    if not jobs:
        return
    jf = log_path.with_suffix(".jobs.json")
    jf.write_text(json.dumps(jobs))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(DAC_SHIM_DIR), str(SEED_VC_DIR), env.get("PYTHONPATH", "")])
    env["HF_HUB_CACHE"] = str(MODELS_DIR / "hf")
    cmd = [sys.executable, "-m", "studio.vc_worker", str(jf)]
    root = Path(__file__).resolve().parent.parent
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=str(root), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        done = 0
        for line in p.stdout:
            log.write(line)
            log.flush()
            if line.startswith("JOB_DONE"):
                done += 1
                if progress:
                    progress(done / len(jobs), f"Преобразование голоса: {done}/{len(jobs)}")
        if p.wait() != 0:
            tail = "".join(open(log_path).readlines()[-20:])
            raise RuntimeError("Ошибка Seed-VC:\n" + tail)


def place_converted(pieces: list[tuple[Path, dict]], original: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Puts converted pieces back at their original time; returns (weighted track, mask)."""
    track = np.zeros(length, np.float32)
    mask = np.zeros(length, np.float32)
    fade = int(0.03 * SR)
    for path, m in pieces:
        need = m["src_b"] - m["src_a"]
        y = load_mono(path)
        if abs(len(y) - need) > 0.02 * SR:  # should not happen: Seed-VC keeps the length
            import librosa
            y = librosa.resample(y, orig_sr=len(y), target_sr=need)
        piece = np.pad(y[:need], (0, max(0, need - len(y))))
        sa, sb = m["seg_a"] - m["src_a"], m["seg_b"] - m["src_a"]
        # loudness match to the original speech of this segment
        o_rms = np.sqrt((original[m["seg_a"]:m["seg_b"]] ** 2).mean() + 1e-12)
        c_rms = np.sqrt((piece[sa:sb] ** 2).mean() + 1e-12)
        piece = piece * float(np.clip(o_rms / c_rms, 0.1, 10))
        w = np.zeros(need, np.float32)
        lo, hi = max(0, sa - fade), min(need, sb + fade)
        w[lo:hi] = 1.0
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        w[lo:lo + fade] = ramp[: len(w[lo:lo + fade])]
        w[max(lo, hi - fade):hi] = ramp[::-1][-len(w[max(lo, hi - fade):hi]):] if hi > lo else w[lo:hi]
        sl = slice(m["src_a"], m["src_b"])
        track[sl] = track[sl] * (1 - w) + piece * w
        mask[sl] = np.maximum(mask[sl], w)
    return track, mask


def mix(background: Path, vocals: Path, converted: list[tuple[np.ndarray, np.ndarray]], dst: Path):
    bg = load_stereo(background)
    vo = load_stereo(vocals)
    n = min(len(bg), len(vo))
    bg, vo = bg[:n], vo[:n]
    keep = np.ones(n, np.float32)
    new = np.zeros(n, np.float32)
    for track, mask in converted:
        keep = np.minimum(keep, 1 - mask[:n])
        new += track[:n]  # already weighted by its fade mask
    out = bg + vo * keep[:, None] + new[:, None]
    peak = np.abs(out).max()
    if peak > 0.99:
        out = out / peak * 0.99
    sf.write(str(dst), out, SR, subtype="FLOAT")
