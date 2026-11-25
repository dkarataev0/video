"""Seed-VC batch worker. Run as: python -m studio.vc_worker jobs.json

Uses the official Seed-VC ``inference.main`` unchanged, but
  * loads each model variant once for all jobs (patched ``load_models``),
  * saves with soundfile (newer torchaudio builds need torchcodec for ``save``).
Each job: {source, target, output, steps, cfg_rate, f0_condition, auto_f0_adjust, semitones}
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def main():
    jobs = json.loads(Path(sys.argv[1]).read_text())
    from studio.config import SEED_VC_DIR

    os.chdir(SEED_VC_DIR)
    sys.path.insert(0, str(SEED_VC_DIR))

    import soundfile as sf
    import torch
    import torchaudio

    def _save(path, wav, sr, *a, **k):
        w = wav.detach().cpu().float().numpy()
        sf.write(str(path), w.T if w.ndim == 2 else w, sr)

    torchaudio.save = _save
    import inference  # Seed-VC

    cache = {}
    original_load = inference.load_models
    use_fp16 = torch.cuda.is_available()

    def cached_load(args):
        key = bool(args.f0_condition)
        if key not in cache:
            cache[key] = original_load(args)
        inference.fp16 = args.fp16
        return cache[key]

    inference.load_models = cached_load

    for job in jobs:
        out_dir = Path(tempfile.mkdtemp(prefix="vc_"))
        ns = argparse.Namespace(
            source=job["source"], target=job["target"], output=str(out_dir),
            diffusion_steps=int(job.get("steps", 30)), length_adjust=1.0,
            inference_cfg_rate=float(job.get("cfg_rate", 0.7)),
            f0_condition=bool(job.get("f0_condition", True)),
            auto_f0_adjust=bool(job.get("auto_f0_adjust", True)),
            semi_tone_shift=int(job.get("semitones", 0)),
            checkpoint=None, config=None, fp16=use_fp16,
        )
        inference.main(ns)
        produced = sorted(out_dir.glob("*.wav"))
        if not produced:
            raise RuntimeError(f"Seed-VC did not produce output for {job['source']}")
        Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced[0]), job["output"])
        shutil.rmtree(out_dir, ignore_errors=True)
        print("JOB_DONE", job["output"], flush=True)


if __name__ == "__main__":
    main()
