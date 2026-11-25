"""Face analysis (detection, tracking, character clustering) and face swapping.

Pipeline per frame of a target face:
  smoothed 5-point landmarks -> aligned crop (arcface_128 template, "pixel boost" size)
  -> swapper (HyperSwap 256 / InSwapper 128, ONNX) -> temporal deflicker in aligned space
  -> box + XSeg occlusion mask -> paste back -> optional GFPGAN blend.
Model I/O conventions follow FaceFusion (OpenRAIL-AS), re-implemented here so that
identity is decided per *track* (stable through profile views) instead of per frame.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import FACE_MODELS, MODELS_DIR, WARP_TEMPLATES
from .models import face_model_path, providers, session


# ------------------------------------------------------------------ geometry
def estimate_matrix(kps: np.ndarray, template: str, size: int) -> np.ndarray:
    dst = np.array(WARP_TEMPLATES[template], np.float32) * size
    return cv2.estimateAffinePartial2D(kps.astype(np.float32), dst, method=cv2.RANSAC, ransacReprojThreshold=100)[0]


def warp(frame: np.ndarray, kps: np.ndarray, template: str, size: int):
    m = estimate_matrix(kps, template, size)
    crop = cv2.warpAffine(frame, m, (size, size), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
    return crop, m


def paste_back(frame: np.ndarray, crop: np.ndarray, mask: np.ndarray, m: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    ch, cw = crop.shape[:2]
    inv = cv2.invertAffineTransform(m)
    pts = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float32)
    pts = cv2.transform(pts[None], inv)[0]
    x1, y1 = np.clip(np.floor(pts.min(0)).astype(int), 0, [w, h])
    x2, y2 = np.clip(np.ceil(pts.max(0)).astype(int), 0, [w, h])
    if x2 - x1 < 2 or y2 - y1 < 2:
        return frame
    pm = inv.copy()
    pm[0, 2] -= x1
    pm[1, 2] -= y1
    inv_mask = cv2.warpAffine(mask, pm, (x2 - x1, y2 - y1)).clip(0, 1)[..., None]
    inv_crop = cv2.warpAffine(crop, pm, (x2 - x1, y2 - y1), borderMode=cv2.BORDER_REPLICATE)
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = (roi * (1 - inv_mask) + inv_crop.astype(np.float32) * inv_mask).round().clip(0, 255).astype(np.uint8)
    return frame


def box_mask(size: int, blur: float, padding=(0, 0, 0, 0)) -> np.ndarray:
    blur_amount = int(size * 0.5 * blur)
    area = max(blur_amount // 2, 1)
    m = np.ones((size, size), np.float32)
    m[:max(area, int(size * padding[0] / 100)), :] = 0
    m[-max(area, int(size * padding[2] / 100)):, :] = 0
    m[:, :max(area, int(size * padding[3] / 100))] = 0
    m[:, -max(area, int(size * padding[1] / 100)):] = 0
    if blur_amount > 0:
        m = cv2.GaussianBlur(m, (0, 0), blur_amount * 0.25)
    return m


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ------------------------------------------------------------------ detection
class FaceDetector:
    """InsightFace buffalo_l: SCRFD-10G detector (5 landmarks) + ArcFace w600k_r50."""

    def __init__(self, det_size=640, det_thresh=0.5):
        from insightface.app import FaceAnalysis

        root = MODELS_DIR / "insightface"
        self.app = FaceAnalysis(name="buffalo_l", root=str(root), allowed_modules=["detection", "recognition"],
                                providers=providers())
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=det_thresh)

    def detect(self, img: np.ndarray, min_px: int = 0) -> list[dict]:
        out = []
        for f in self.app.get(img):
            bw = float(f.bbox[2] - f.bbox[0])
            if bw < min_px or f.kps is None or getattr(f, "embedding", None) is None:
                continue
            emb = f.embedding.astype(np.float32)
            out.append({"bbox": f.bbox.astype(np.float32), "kps": f.kps.astype(np.float32),
                        "score": float(f.det_score), "emb": emb, "nemb": emb / (np.linalg.norm(emb) + 1e-9)})
        return out

    def largest(self, img: np.ndarray):
        faces = self.detect(img)
        if not faces:
            return None
        return max(faces, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))


def frontalness(kps: np.ndarray) -> float:
    """1.0 for frontal faces, ->0 for profiles (nose relative to eye centre)."""
    le, re, nose = kps[0], kps[1], kps[2]
    eye_d = np.linalg.norm(re - le) + 1e-6
    off = abs(nose[0] - (le[0] + re[0]) / 2) / eye_d
    return float(np.clip(1.0 - off, 0.05, 1.0))


# ------------------------------------------------------------------ tracking + clustering
@dataclass
class Track:
    tid: int
    frames: list = field(default_factory=list)
    bboxes: list = field(default_factory=list)
    kps: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    embs: list = field(default_factory=list)
    best: tuple = (0.0, -1)  # (quality, frame index) for thumbnail
    character: int = -1

    def mean_emb(self) -> np.ndarray:
        w = np.asarray(self.weights, np.float32)[:, None]
        e = (np.asarray(self.embs, np.float32) * w).sum(0)
        return e / (np.linalg.norm(e) + 1e-9)

    def recent_emb(self, n=10) -> np.ndarray:
        e = np.asarray(self.embs[-n:], np.float32).mean(0)
        return e / (np.linalg.norm(e) + 1e-9)


def build_tracks(dets_per_frame: list[list[dict]], fps: float) -> list[Track]:
    from scipy.optimize import linear_sum_assignment

    max_gap = max(2, int(round(fps * 0.5)))
    tracks: list[Track] = []
    active: list[Track] = []
    for t, dets in enumerate(dets_per_frame):
        active = [tr for tr in active if t - tr.frames[-1] <= max_gap]
        if dets and active:
            cost = np.full((len(active), len(dets)), 10.0, np.float32)
            for i, tr in enumerate(active):
                re = tr.recent_emb()
                for j, d in enumerate(dets):
                    ov = iou(tr.bboxes[-1], d["bbox"])
                    sim = float(re @ d["nemb"])
                    if (ov > 0.15 and sim > 0.15) or sim > 0.5:
                        cost[i, j] = -(ov + sim)
            rows, cols = linear_sum_assignment(cost)
            matched = set()
            for i, j in zip(rows, cols):
                if cost[i, j] < 0:
                    _append(active[i], t, dets[j])
                    matched.add(j)
            new = [d for j, d in enumerate(dets) if j not in matched]
        else:
            new = dets
        for d in new:
            tr = Track(tid=len(tracks))
            _append(tr, t, d)
            tracks.append(tr)
            active.append(tr)
    return tracks


def _append(tr: Track, t: int, d: dict):
    size = float(d["bbox"][2] - d["bbox"][0])
    fr = frontalness(d["kps"])
    w = d["score"] * fr * min(size, 256) / 256
    tr.frames.append(t)
    tr.bboxes.append(d["bbox"])
    tr.kps.append(d["kps"])
    tr.embs.append(d["nemb"].astype(np.float16))
    tr.weights.append(max(w, 1e-3))
    q = d["score"] * fr * size
    if q > tr.best[0]:
        tr.best = (q, t)


def cluster_tracks(tracks: list[Track], threshold: float = 0.55, fps: float = 30.0) -> list[dict]:
    """Groups tracks into characters. threshold = max cosine distance inside a character."""
    if not tracks:
        return []
    embs = np.stack([tr.mean_emb() for tr in tracks])
    if len(tracks) == 1:
        labels = np.array([0])
    else:
        from sklearn.cluster import AgglomerativeClustering

        labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                         distance_threshold=threshold).fit_predict(embs)
    chars = {}
    for tr, lab in zip(tracks, labels):
        c = chars.setdefault(int(lab), {"tracks": [], "frames": 0, "best": (0.0, -1, -1)})
        c["tracks"].append(tr.tid)
        c["frames"] += len(tr.frames)
        if tr.best[0] > c["best"][0]:
            c["best"] = (tr.best[0], tr.best[1], tr.tid)
    ordered = sorted(chars.values(), key=lambda c: -c["frames"])
    out = []
    for cid, c in enumerate(ordered):
        for tid in c["tracks"]:
            tracks[tid].character = cid
        out.append({"id": cid, "tracks": c["tracks"], "frames": c["frames"],
                    "seconds": round(c["frames"] / fps, 1), "best_frame": c["best"][1], "best_track": c["best"][2]})
    return out


def build_render_plan(tracks: list[Track], n_frames: int, fps: float, smooth: int = 2):
    """frame -> list of (track_id, character, kps(5x2), target_nemb). Fills short detection gaps
    by interpolation and smooths landmarks with a zero-lag Savitzky-Golay filter."""
    from scipy.signal import savgol_filter

    max_gap = max(2, int(round(fps * 0.25)))
    plan: list[list] = [[] for _ in range(n_frames)]
    for tr in tracks:
        if tr.character < 0:
            continue
        temb = tr.mean_emb()
        frames = np.asarray(tr.frames)
        kps = np.asarray(tr.kps, np.float32)
        # split into contiguous runs, interpolating gaps <= max_gap
        runs, start = [], 0
        for i in range(1, len(frames) + 1):
            if i == len(frames) or frames[i] - frames[i - 1] > max_gap:
                runs.append((start, i))
                start = i
        for a, b in runs:
            f, k = frames[a:b], kps[a:b]
            full = np.arange(f[0], f[-1] + 1)
            flat = k.reshape(len(f), -1)
            interp = np.stack([np.interp(full, f, flat[:, j]) for j in range(flat.shape[1])], 1)
            win = 2 * smooth + 1
            if smooth > 0 and len(full) >= win + 2:
                interp = savgol_filter(interp, win, 2, axis=0, mode="interp")
            for t, row in zip(full, interp):
                if t < n_frames:
                    plan[t].append((tr.tid, tr.character, row.reshape(5, 2).astype(np.float32), temb))
    return plan


# ------------------------------------------------------------------ analysis entry point
def analyze(reader, n_frames: int, fps: float, min_px: int, progress=None, det_thresh: float = 0.35) -> dict:
    det = FaceDetector(det_thresh=det_thresh)  # low threshold keeps faces half out of frame (zooms)
    dets_per_frame = []
    for t, frame in enumerate(reader):
        dets_per_frame.append(det.detect(frame, min_px=min_px))
        if progress and t % 10 == 0:
            progress(t / max(1, n_frames), f"Поиск лиц: кадр {t}/{n_frames}")
    tracks = build_tracks(dets_per_frame, fps)
    return {"tracks": tracks, "n_frames": len(dets_per_frame)}


def save_tracks(path: Path, data: dict):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_tracks(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def crop_thumb(frame: np.ndarray, bbox, size=192) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 0.8
    h, w = frame.shape[:2]
    a, b = int(max(0, cx - s)), int(max(0, cy - s))
    c, d = int(min(w, cx + s)), int(min(h, cy + s))
    crop = frame[b:d, a:c]
    if crop.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------ source identities
def source_identity(det: FaceDetector, image_paths: list[str]) -> dict | None:
    embs, nembs, used = [], [], 0
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            try:  # HEIC from iPhone and other formats via Pillow
                from PIL import Image
                try:
                    import pillow_heif  # type: ignore
                    pillow_heif.register_heif_opener()
                except Exception:
                    pass
                img = cv2.cvtColor(np.array(Image.open(p).convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception:
                continue
        s = 1600 / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        f = det.largest(img)
        if f is None:  # very tight crops/close-up selfies: give the detector some context
            pad = max(img.shape[:2]) // 2
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(127, 127, 127))
            if max(img.shape[:2]) < 640:
                img = cv2.resize(img, None, fx=640 / max(img.shape[:2]), fy=640 / max(img.shape[:2]))
            f = det.largest(img)
        if f is None:
            continue
        embs.append(f["emb"])
        nembs.append(f["nemb"])
        used += 1
    if not embs:
        return None
    return {"emb": np.mean(embs, 0).astype(np.float32), "nemb": np.mean(nembs, 0).astype(np.float32), "photos": used}


# ------------------------------------------------------------------ swapping
class FaceSwapper:
    def __init__(self, settings: dict):
        self.s = settings
        name = settings["swapper"]
        spec = FACE_MODELS[name]
        self.type = spec["type"]
        self.size = spec["size"]
        path = face_model_path(name)
        self.sess = session(path)
        self.inputs = [i.name for i in self.sess.get_inputs()]
        if self.type == "inswapper":
            import onnx
            from onnx import numpy_helper

            self.emap = numpy_helper.to_array(onnx.load(str(path)).graph.initializer[-1]).astype(np.float32)
        self.boost = max(self.size, int(settings.get("pixel_boost", 512)) // self.size * self.size)
        self.k = self.boost // self.size
        self.occluder = session(face_model_path("xseg_1")) if settings.get("occlusion_mask", True) else None
        blend = float(settings.get("enhancer_blend", 0))
        self.enhancer = session(face_model_path("gfpgan_1.4")) if blend > 0 else None
        self.box = box_mask(self.boost, float(settings.get("mask_blur", 0.3)))
        self.box512 = box_mask(512, float(settings.get("mask_blur", 0.3)))
        self.prev: dict[int, tuple[int, np.ndarray]] = {}

    # --- embeddings
    def source_vector(self, ident: dict, target_nemb: np.ndarray) -> np.ndarray:
        if self.type == "hyperswap":
            src = ident["nemb"].reshape(1, -1)
        else:  # inswapper latent
            e = ident["emb"].reshape(1, -1)
            src = (e @ self.emap) / np.linalg.norm(e)
        w = float(np.interp(float(self.s.get("face_weight", 0.5)), [0, 1], [0.35, -0.35]))
        t = (target_nemb / (np.linalg.norm(target_nemb) + 1e-9)).reshape(1, -1)
        return (src * (1 - w) + t * w).astype(np.float32)

    # --- models
    def _run_swap(self, tile: np.ndarray, src_vec: np.ndarray) -> np.ndarray:
        x = tile[:, :, ::-1].astype(np.float32) / 255.0
        if self.type == "hyperswap":
            x = (x - 0.5) / 0.5
        x = x.transpose(2, 0, 1)[None]
        feeds = {}
        for n in self.inputs:
            feeds[n] = src_vec if n == "source" else x
        y = self.sess.run(None, feeds)[0][0].transpose(1, 2, 0)
        if self.type == "hyperswap":
            y = y * 0.5 + 0.5
        return (y.clip(0, 1)[:, :, ::-1] * 255).astype(np.float32)

    def _occlusion(self, crop: np.ndarray) -> np.ndarray:
        x = cv2.resize(crop, (256, 256))[None].astype(np.float32) / 255.0
        m = self.occluder.run(None, {self.occluder.get_inputs()[0].name: x})[0][0]
        m = cv2.resize(m.clip(0, 1).astype(np.float32), crop.shape[:2][::-1])
        if m.ndim == 3:
            m = m[..., 0]
        return (cv2.GaussianBlur(m.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2

    def _enhance(self, frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
        crop, m = warp(frame, kps, "ffhq_512", 512)
        x = ((crop[:, :, ::-1].astype(np.float32) / 255.0) - 0.5) / 0.5
        feeds = {}
        for i in self.enhancer.get_inputs():
            feeds[i.name] = x.transpose(2, 0, 1)[None] if i.name == "input" else np.array([1.0], np.float64)
        y = self.enhancer.run(None, feeds)[0][0]
        y = ((np.clip(y, -1, 1) + 1) / 2).transpose(1, 2, 0)[:, :, ::-1] * 255
        mask = self.box512.copy()
        if self.occluder is not None:
            mask = np.minimum(mask, self._occlusion(crop))
        mask *= float(self.s.get("enhancer_blend", 40)) / 100.0
        return paste_back(frame, y.round().astype(np.uint8), mask, m)

    # --- public
    def reset(self):
        self.prev.clear()

    def swap(self, frame: np.ndarray, t: int, track_id: int, kps: np.ndarray, src_vec: np.ndarray) -> np.ndarray:
        crop, m = warp(frame, kps, "arcface_128", self.boost)
        k, n = self.k, self.size
        tiles = crop.reshape(n, k, n, k, 3).transpose(1, 3, 0, 2, 4).reshape(k * k, n, n, 3)
        outs = [self._run_swap(tile, src_vec) for tile in tiles]
        out = np.stack(outs).reshape(k, k, n, n, 3).transpose(2, 0, 3, 1, 4).reshape(self.boost, self.boost, 3)

        # temporal deflicker: blend with previous output where the change is small (no mouth ghosting)
        strength = float(self.s.get("deflicker", 0.5))
        prev = self.prev.get(track_id)
        if strength > 0 and prev is not None and t - prev[0] == 1:
            diff = cv2.GaussianBlur(np.abs(out - prev[1]).mean(2), (0, 0), 2)
            a = (strength * 0.7 * np.exp(-(diff / 10.0) ** 2))[..., None]
            out = out * (1 - a) + prev[1] * a
        self.prev[track_id] = (t, out)

        mask = self.box
        if self.occluder is not None:
            mask = np.minimum(mask, self._occlusion(crop))
        frame = paste_back(frame, out.round().clip(0, 255).astype(np.uint8), mask, m)
        if self.enhancer is not None:
            frame = self._enhance(frame, kps)
        return frame
