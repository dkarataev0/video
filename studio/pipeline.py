"""Project state + background jobs (analyse, 5-second test, full render, fragment re-generation)."""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
import traceback
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from . import audio as A
from . import faces as F
from .config import DEFAULT_SETTINGS, PROJECTS_DIR, SEGMENT_SECONDS, TEST_SECONDS
from .media import FrameReader, FrameWriter, apply_label, concat_and_mux, cut_audio, make_label_overlay, normalize_video

WARMUP = 8  # frames decoded before a segment so temporal filters start "warm"


def _hash_files(paths) -> str:
    h = hashlib.sha1()
    for p in sorted(map(str, paths)):
        h.update(p.encode())
        try:
            with open(p, "rb") as f:
                h.update(hashlib.sha1(f.read()).digest())
        except OSError:
            pass
    return h.hexdigest()[:12]


def _sig(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


class Busy(RuntimeError):
    pass


class Project:
    FACE_KEYS = ["swapper", "pixel_boost", "face_weight", "enhancer_blend", "occlusion_mask", "mask_blur",
                 "deflicker", "landmark_smoothing", "label_text", "label_position", "crf", "hair_color", "hair_strength"]
    VOICE_KEYS = ["vc_steps", "vc_cfg_rate", "vc_keep_intonation", "vc_semitones"]

    def __init__(self, root: Path | None = None):
        self.dir = Path(root or PROJECTS_DIR / "current")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self._tracks_cache = None
        self.state = self._load_state()
        if self.state.get("status") == "running":  # process restarted mid-job
            self.state.update(status="error", message="Задача была прервана (перезапуск). Запустите её снова.")
            self.save()

    # ---------------------------------------------------------------- state
    def _load_state(self) -> dict:
        p = self.dir / "state.json"
        base = {"status": "idle", "progress": 0.0, "message": "Загрузите видео.", "error": "", "job": "",
                "meta": None, "characters": [], "speakers": [], "segments": [], "settings": dict(DEFAULT_SETTINGS),
                "face_photos": {}, "voice_files": {}, "result": None, "test": None, "log": []}
        if p.exists():
            try:
                base.update(json.loads(p.read_text()))
            except Exception:
                pass
        base["settings"] = {**DEFAULT_SETTINGS, **base.get("settings", {})}
        return base

    def save(self):
        tmp = self.dir / "state.json.tmp"
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1, default=str))
        tmp.replace(self.dir / "state.json")

    def update(self, progress: float | None = None, message: str | None = None):
        if progress is not None:
            self.state["progress"] = float(max(0.0, min(1.0, progress)))
        if message:
            self.state["message"] = message
        self.save()

    def sub(self, a: float, b: float):
        """progress callback mapping 0..1 into [a, b] of the whole job"""
        return lambda p, msg=None: self.update(a + (b - a) * p, msg)

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, job: str, fn, *args):
        with self.lock:
            if self.running:
                raise Busy("Уже выполняется задача: " + self.state.get("job", ""))
            self.state.update(status="running", job=job, progress=0.0, error="", message="Запуск…")
            self.save()

            def run():
                t0 = time.time()
                try:
                    fn(*args)
                    self.state.update(status="done", progress=1.0,
                                      message=f"Готово за {int(time.time() - t0)} с.")
                except Exception as e:  # shown in the UI; full traceback goes to job.log
                    (self.dir / "job.log").write_text(traceback.format_exc())
                    self.state.update(status="error", error=str(e), message="Ошибка: " + str(e)[:500])
                self.save()

            self.thread = threading.Thread(target=run, daemon=True)
            self.thread.start()

    # ---------------------------------------------------------------- paths
    @property
    def meta(self):
        return self.state["meta"]

    @property
    def fps(self) -> Fraction:
        return Fraction(self.meta["fps_num"], self.meta["fps_den"])

    def p(self, *parts) -> Path:
        path = self.dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # ---------------------------------------------------------------- 1. analysis
    def set_video(self, src: str):
        if self.running:
            raise Busy("Дождитесь окончания текущей задачи.")
        self.wipe(keep_settings=True)
        ext = Path(src).suffix.lower() or ".mp4"
        dst = self.p("input", "original" + ext)
        shutil.copy(src, dst)
        self.state["input"] = str(dst)
        self.save()

    def analyze_job(self):
        st = self.state
        self.update(0.01, "Подготовка видео (поворот, HDR→SDR, до 1080p, постоянная частота кадров)…")
        meta = normalize_video(st["input"], self.p("norm.mp4"), self.p("audio.wav"))
        st["meta"] = meta
        self.save()
        fps = float(self.fps)
        reader = FrameReader(self.p("norm.mp4"), meta["width"], meta["height"], self.fps)
        data = F.analyze(reader, meta["frames"], fps, int(st["settings"]["min_face_px"]), self.sub(0.05, 0.55),
                         float(st["settings"].get("det_thresh", 0.35)))
        F.save_tracks(self.p("tracks.pkl"), data)
        self._tracks_cache = data
        self.recluster(0.55, save_only=True)
        if meta["has_audio"]:
            self.update(0.56, "Отделение голосов от музыки и фона (BS-RoFormer)…")
            v, b = A.separate(self.p("audio.wav"), self.dir / "stems")
            st["vocals"], st["background"] = str(v), str(b)
            self.save()
            self.update(0.75, "Определение говорящих…")
            self.rediarize(None, 0.6, progress=self.sub(0.75, 0.98))
        self.update(1.0, "Анализ завершён. Назначьте лица и голоса.")

    def tracks(self):
        if self._tracks_cache is None:
            self._tracks_cache = F.load_tracks(self.p("tracks.pkl"))
        return self._tracks_cache

    def recluster(self, threshold: float, save_only=False):
        data = self.tracks()
        chars = F.cluster_tracks(data["tracks"], threshold, float(self.fps))
        F.save_tracks(self.p("tracks.pkl"), data)
        # thumbnails
        thumbs = self.dir / "thumbs"
        shutil.rmtree(thumbs, ignore_errors=True)
        thumbs.mkdir()
        for c in chars:
            tr = data["tracks"][c["best_track"]]
            idx = tr.frames.index(c["best_frame"])
            frame = next(iter(FrameReader(self.p("norm.mp4"), self.meta["width"], self.meta["height"], self.fps,
                                          start=c["best_frame"], count=1)), None)
            if frame is not None:
                path = thumbs / f"char_{c['id']}.jpg"
                cv2.imwrite(str(path), F.crop_thumb(frame, tr.bboxes[idx]))
                c["thumb"] = str(path)
        self.state["characters"] = chars
        self.state["face_photos"] = {}
        self.state["cluster_threshold"] = threshold
        self.save()
        return chars

    def rediarize(self, n_speakers: int | None, threshold: float = 0.6, progress=None):
        segs = A.diarize(Path(self.state["vocals"]), n_speakers, threshold, progress)
        self.set_segments(segs)

    def set_segments(self, segs: list[dict]):
        segs = sorted(({"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2),
                        "speaker": int(s["speaker"])} for s in segs if float(s["end"]) > float(s["start"])),
                      key=lambda s: s["start"])
        self.state["segments"] = segs
        spk = sorted({s["speaker"] for s in segs})
        speakers = []
        for i in spk:
            dur = sum(s["end"] - s["start"] for s in segs if s["speaker"] == i)
            prev = A.speaker_preview(Path(self.state["vocals"]), segs, i, self.p("speakers", f"spk_{i}.wav"))
            speakers.append({"id": i, "seconds": round(dur, 1), "preview": str(prev) if prev else None})
        self.state["speakers"] = speakers
        self.save()

    # ---------------------------------------------------------------- 2. assignments
    def set_face_photos(self, char_id: int, files: list[str] | None):
        key = str(char_id)
        d = self.dir / "faces_src" / f"char_{char_id}"
        shutil.rmtree(d, ignore_errors=True)
        if not files:
            self.state["face_photos"].pop(key, None)
        else:
            d.mkdir(parents=True)
            saved = []
            for i, f in enumerate(files):
                dst = d / f"{i}{Path(f).suffix.lower() or '.jpg'}"
                shutil.copy(f, dst)
                saved.append(str(dst))
            self.state["face_photos"][key] = saved
        self.save()

    def set_voice_files(self, spk_id: int, files: list[str] | None):
        key = str(spk_id)
        d = self.dir / "voices_src" / f"spk_{spk_id}"
        shutil.rmtree(d, ignore_errors=True)
        if not files:
            self.state["voice_files"].pop(key, None)
        else:
            d.mkdir(parents=True)
            saved = []
            for i, f in enumerate(files):
                dst = d / f"{i}{Path(f).suffix.lower() or '.wav'}"
                shutil.copy(f, dst)
                saved.append(str(dst))
            self.state["voice_files"][key] = saved
        self.save()

    def set_settings(self, **kw):
        self.state["settings"].update({k: v for k, v in kw.items() if v is not None})
        self.save()

    # ---------------------------------------------------------------- 3. rendering helpers
    def _identities(self, det) -> dict[int, dict]:
        out = {}
        cache = self.dir / "faces_src" / "identities.json"
        known = json.loads(cache.read_text()) if cache.exists() else {}
        for key, files in self.state["face_photos"].items():
            h = _hash_files(files)
            if known.get(key, {}).get("hash") == h:
                ident = {k: np.asarray(v, np.float32) for k, v in known[key]["ident"].items() if k != "photos"}
            else:
                ident = F.source_identity(det, files)
                if ident is None:
                    raise RuntimeError(f"На фотографиях для персонажа {int(key) + 1} не найдено лицо.")
                known[key] = {"hash": h, "ident": {"emb": ident["emb"].tolist(), "nemb": ident["nemb"].tolist()}}
            out[int(key)] = ident
        cache.write_text(json.dumps(known))
        return out

    def face_signature(self) -> str:
        s = self.state["settings"]
        photos = {k: _hash_files(v) for k, v in self.state["face_photos"].items()}
        return _sig([{k: s[k] for k in self.FACE_KEYS}, photos, self.state.get("cluster_threshold"),
                     s.get("per_character")])

    def _render_frames(self, start: int, end: int, out_path: Path, ctx: dict, progress=None):
        meta = self.meta
        w0 = max(0, start - WARMUP)
        reader = FrameReader(self.p("norm.mp4"), meta["width"], meta["height"], self.fps, start=w0, count=end - w0)
        writer = FrameWriter(out_path, meta["width"], meta["height"], self.fps, int(self.state["settings"]["crf"]))
        swappers, plan, vecs = ctx["swappers"], ctx["plan"], ctx["vecs"]
        for sw in ctx["pool"]:
            sw.reset()
        n = 0
        try:
            for i, frame in enumerate(reader):
                t = w0 + i
                if t >= end:
                    break
                for tid, char, kps, _ in plan[t] if t < len(plan) else []:
                    if tid in vecs:
                        frame = swappers[tid].swap(frame, t, tid, kps, vecs[tid])
                if t >= start:
                    writer.write(apply_label(frame, ctx["label"]))
                    n += 1
                    if progress and n % 5 == 0:
                        progress(n / max(1, end - start))
        finally:
            writer.close()
        if n != end - start:
            raise RuntimeError(f"Прочитано {n} кадров вместо {end - start} (ошибка декодирования).")

    def _video_context(self) -> dict:
        """Per-character overrides (settings["per_character"] = {"0": {"swapper": ..., "face_weight": ...}})
        let e.g. one person use InSwapper with stronger identity while others keep HyperSwap."""
        s = self.state["settings"]
        data = self.tracks()
        plan = F.build_render_plan(data["tracks"], self.meta["frames"], float(self.fps), int(s["landmark_smoothing"]))
        overrides = s.get("per_character") or {}
        pool: dict[str, F.FaceSwapper] = {}

        def swapper_for(char: int) -> F.FaceSwapper:
            cfg = {**s, **overrides.get(str(char), {})}
            key = json.dumps({k: cfg[k] for k in self.FACE_KEYS if k in cfg}, sort_keys=True, default=str)
            if key not in pool:
                pool[key] = F.FaceSwapper(cfg)
            return pool[key]

        vecs, swappers = {}, {}
        if self.state["face_photos"]:
            idents = self._identities(F.FaceDetector())
            for tr in data["tracks"]:
                if tr.character in idents:
                    sw = swapper_for(tr.character)
                    swappers[tr.tid] = sw
                    vecs[tr.tid] = sw.source_vector(idents[tr.character], tr.mean_emb())
        label = make_label_overlay(self.meta["width"], self.meta["height"], s["label_text"], s["label_position"])
        return {"swappers": swappers, "pool": list(pool.values()), "plan": plan, "vecs": vecs, "label": label}

    def _segments(self):
        n, per = self.meta["frames"], max(1, round(SEGMENT_SECONDS * float(self.fps)))
        return [(i, a, min(n, a + per)) for i, a in enumerate(range(0, n, per))]

    # ---------------------------------------------------------------- voices
    def _references(self, progress=None) -> dict[int, Path]:
        s = self.state["settings"]
        refs = {}
        for key, files in self.state["voice_files"].items():
            h = _hash_files(files) + ("c" if s["clean_reference"] else "")
            dst = self.p("voices_ref", f"ref_{key}_{h}.wav")
            if not dst.exists():
                if progress:
                    progress(0, f"Подготовка образца голоса {int(key) + 1}…")
                A.prepare_reference(files, dst, self.dir / "voices_ref" / f"work_{key}", s["clean_reference"])
            refs[int(key)] = dst
        return refs

    def voice_signature(self, spk: int, ref: Path) -> str:
        s = self.state["settings"]
        return _sig([{k: s[k] for k in self.VOICE_KEYS}, ref.name])

    def _convert_voices(self, t0: float, t1: float, force: bool, progress=None) -> Path | None:
        """Converts all diarised segments overlapping [t0, t1] of speakers that have a voice
        assigned (cached per segment), then mixes the full soundtrack."""
        if not self.meta["has_audio"]:
            return None
        s = self.state["settings"]
        vocals = A.load_mono(Path(self.state["vocals"]))
        refs = self._references(progress)
        jobs, pieces = [], {}
        for seg in self.state["segments"]:
            spk = seg["speaker"]
            if spk not in refs:
                continue
            key = f"{spk}_{seg['start']:.2f}_{seg['end']:.2f}"
            out = self.p("conv", key + ".wav")
            side = out.with_suffix(".json")
            in_range = seg["end"] > t0 and seg["start"] < t1
            sig = self.voice_signature(spk, refs[spk])
            m = A.extract_piece(vocals, seg, self.p("conv_src", key + ".wav"))
            have = out.exists() and side.exists()
            stale = have and json.loads(side.read_text()).get("sig") != sig
            if in_range and (force or not have or stale):
                jobs.append({"source": str(self.p("conv_src", key + ".wav")), "target": str(refs[spk]),
                             "output": str(out), "steps": s["vc_steps"], "cfg_rate": s["vc_cfg_rate"],
                             "f0_condition": s["vc_keep_intonation"], "auto_f0_adjust": s["vc_keep_intonation"],
                             "semitones": s["vc_semitones"], "_sig": sig})
            if in_range or have:
                pieces.setdefault(spk, []).append((out, m))
        if jobs:
            A.run_seed_vc(jobs, self.p("logs", "seed_vc.log"), progress)
            for j in jobs:
                Path(j["output"]).with_suffix(".json").write_text(json.dumps({"sig": j["_sig"]}))
        converted = []
        for spk, items in pieces.items():
            items = [(p, m) for p, m in items if p.exists()]
            if items:
                converted.append(A.place_converted(items, vocals, len(vocals)))
        dst = self.p("final_audio.wav")
        A.mix(Path(self.state["background"]), Path(self.state["vocals"]), converted, dst)
        return dst

    # ---------------------------------------------------------------- jobs
    def test_job(self, start_sec: float):
        fps = float(self.fps)
        a = int(max(0.0, min(start_sec, self.meta["duration"] - 0.5)) * fps)
        b = min(self.meta["frames"], a + int(TEST_SECONDS * fps))
        self.update(0.02, "Загрузка моделей лиц…")
        ctx = self._video_context()
        vid = self.p("output", "test_video.mp4")
        self.update(0.05, f"Замена лиц в тестовом фрагменте ({a / fps:.1f}–{b / fps:.1f} с)…")
        self._render_frames(a, b, vid, ctx, self.sub(0.05, 0.6))
        self.update(0.6, "Голоса…")
        aud = self._convert_voices(a / fps, b / fps, force=False, progress=self.sub(0.6, 0.95))
        part = None
        if aud is not None:
            part = self.p("output", "test_audio.wav")
            cut_audio(aud, part, a / fps, (b - a) / fps)
        out = self.p("output", f"test_{int(time.time())}.mp4")
        for old in (self.dir / "output").glob("test_*.mp4"):
            if old.name != "test_video.mp4":
                old.unlink()
        concat_and_mux([vid], part, out, self.state["settings"]["label_text"])
        self.state["test"] = str(out)

    def full_job(self, regen_range: tuple[float, float] | None = None, what: str = "all"):
        fps = float(self.fps)
        segs = self._segments()
        sig = self.face_signature()
        todo = []
        for i, a, b in segs:
            path = self.p("render", f"seg_{i:04d}.mp4")
            side = path.with_suffix(".json")
            ok = path.exists() and side.exists() and json.loads(side.read_text()).get("sig") == sig
            if regen_range is not None:
                overlap = b / fps > regen_range[0] and a / fps < regen_range[1]
                if overlap and what in ("all", "faces"):
                    todo.append((i, a, b))
                elif not path.exists():
                    todo.append((i, a, b))
            elif not ok:
                todo.append((i, a, b))
        if todo:
            self.update(0.02, "Загрузка моделей лиц…")
            ctx = self._video_context()
            for k, (i, a, b) in enumerate(todo):
                lo, hi = 0.03 + 0.67 * k / len(todo), 0.03 + 0.67 * (k + 1) / len(todo)
                self.update(lo, f"Замена лиц: фрагмент {k + 1}/{len(todo)} ({a / fps:.0f}–{b / fps:.0f} с)")
                path = self.p("render", f"seg_{i:04d}.mp4")
                self._render_frames(a, b, path, ctx, lambda p, lo=lo, hi=hi: self.update(lo + (hi - lo) * p))
                path.with_suffix(".json").write_text(json.dumps({"sig": sig}))
        self.update(0.7, "Голоса…")
        if regen_range is not None:
            force = what in ("all", "voices")
            t0, t1 = regen_range if force else (0.0, 0.0)
            aud = self._convert_voices(t0, t1, force=force, progress=self.sub(0.7, 0.95))
        else:
            aud = self._convert_voices(0.0, 1e9, force=False, progress=self.sub(0.7, 0.95))
        if aud is None and self.meta["has_audio"]:
            aud = self.p("audio.wav")
        self.update(0.96, "Сборка итогового MP4…")
        out = self.p("output", f"result_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        for old in (self.dir / "output").glob("result_*.mp4"):
            old.unlink()
        concat_and_mux([self.p("render", f"seg_{i:04d}.mp4") for i, _, _ in segs], aud, out,
                       self.state["settings"]["label_text"])
        self.state["result"] = str(out)

    # ---------------------------------------------------------------- privacy
    def wipe(self, keep_settings=False):
        settings = self.state.get("settings") if keep_settings else None
        if self.running:
            raise Busy("Дождитесь окончания задачи, затем удалите данные.")
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._tracks_cache = None
        self.state = self._load_state()
        if settings:
            self.state["settings"] = settings
        self.save()
