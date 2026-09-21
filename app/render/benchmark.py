"""Automatic pipeline selection by measurement.

There is no universally best swapper: a model that nails one face can be
mediocre on another, and a restorer that sharpens beautifully may quietly erase
the identity. So AUTO mode does not pick by reputation -- it renders candidate
pipelines on frames sampled from *this* video, measures them, and picks a
winner from the numbers.

Frame selection matters as much as the metrics. Scoring only easy frontal
frames rewards the wrong pipeline, so we deliberately sample the hard cases:
profiles, motion blur, small faces, close-ups, dark and bright frames, and
frames with more than one person.

Every metric is kept individually in the report -- collapsing to a single score
early would hide *why* something won, and the trade-offs are the whole point.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from app.render import (detection, metrics, recognition, sessions, video)
from app.render.pipeline import FrameRenderer, RenderOptions
from app.render.registry import ENHANCERS, SWAPPERS
from app.render.types import (Face, PipelineConfig, RenderError, SourceIdentity)

BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "benchmarks"
CACHE_DIR = BENCH_DIR / "cache"

# Bump when the candidate matrix, the metrics or the weights change, so a
# cached verdict from an older algorithm is never silently reused.
ALGORITHM_VERSION = 3

# Cosine below which two consecutive rendered faces are treated as different
# people. Sampled frames are far apart in time, so genuine pose/lighting drift
# is expected; only a real identity change falls this far.
IDENTITY_JUMP = 0.45


# ---------------------------------------------------------------- sampling
def _frame_traits(frame: np.ndarray, faces: list[Face]) -> dict[str, Any]:
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    big = max(faces, key=lambda f: f.area) if faces else None
    crop_blur = 0.0
    yaw = 0.0
    if big is not None:
        x1, y1, x2, y2 = [int(max(0, v)) for v in big.box]
        c = grey[y1:y2, x1:x2]
        if c.size:
            crop_blur = float(cv2.Laplacian(c, cv2.CV_64F).var())
        from app.render.alignment import pose_from_kps
        yaw, _ = pose_from_kps(big.kps)
    return {
        "n_faces": len(faces),
        "face_size": float(big.size) if big is not None else 0.0,
        "blur": crop_blur,
        "yaw": abs(float(yaw)),
        "brightness": float(grey.mean()),
        "score": float(big.score) if big is not None else 0.0,
    }


def select_frames(target_path: str, info: video.VideoInfo,
                  want: int = 14, scan: int = 60) -> list[tuple[int, np.ndarray, dict]]:
    """Pick representative frames, biased toward difficulty.

    We scan evenly across the video, then choose a spread that covers the
    categories a real clip throws at the renderer rather than N easy frames.
    """
    total = info.total_frames or 0
    if total <= 0:
        raise RenderError("could not determine frame count for benchmarking")

    scan = min(scan, total)
    idxs = sorted(set(int(round(i)) for i in np.linspace(0, max(total - 1, 0), scan)))
    grabbed = video.sample_frames(target_path, info, idxs)

    catalogued = []
    for i in sorted(grabbed):
        frame = grabbed[i]
        faces = detection.detect_robust(frame, 0.4, "quality")
        if not faces:
            continue
        catalogued.append((i, frame, _frame_traits(frame, faces)))

    if not catalogued:
        raise RenderError(
            "no face was detected anywhere in the target video. "
            "The face may be too small, too blurred, or never visible.")

    # Cover the hard categories explicitly; each picks its most extreme frame.
    picks: dict[int, tuple[int, np.ndarray, dict]] = {}

    def take(key: Callable[[dict], float], n: int = 2, reverse: bool = True) -> None:
        for item in sorted(catalogued, key=lambda c: key(c[2]), reverse=reverse)[:n]:
            picks[item[0]] = item

    take(lambda t: t["yaw"])                       # profile
    take(lambda t: -t["blur"], reverse=True)       # motion blur (lowest sharpness)
    take(lambda t: t["face_size"])                 # close-up
    take(lambda t: -t["face_size"], reverse=True)  # small face
    take(lambda t: -t["brightness"], reverse=True) # dark
    take(lambda t: t["brightness"])                # bright
    take(lambda t: t["n_faces"])                   # multiple people

    # Fill the remainder with an even spread so easy frames are represented too.
    for item in catalogued:
        if len(picks) >= want:
            break
        picks.setdefault(item[0], item)

    return [picks[k] for k in sorted(picks)][:want]


# ---------------------------------------------------------------- candidates
def candidate_configs(available: Optional[list[str]] = None,
                      thorough: bool = True) -> list[PipelineConfig]:
    """Build the candidate matrix: swappers x enhancers x blends.

    Deliberately not a full cross product -- that would take far too long. We
    benchmark every swapper bare (which isolates identity), then explore
    enhancer/blend combinations, and let the second stage refine the winner.
    """
    from app.render.sessions import MODELS_DIR
    from app.render.registry import ALL

    def present(name: str) -> bool:
        return (MODELS_DIR / ALL[name].filename).exists()

    swappers = [s for s in (available or SWAPPERS) if present(s)]
    if not swappers:
        raise RenderError("no swap model is installed. Run: python -m app.render.download")

    # Stage 1: every installed swapper, bare. Bare isolates identity -- an
    # enhancer can mask a weak swapper, so the families must be compared
    # without one before any restoration is layered on.
    configs = [PipelineConfig(swapper=s, enhancer=None, enhancer_blend=0.0,
                              mask="model", label=f"{s} bare") for s in swappers]
    if not thorough:
        return configs

    # Stage 2: restoration options, explored on the swapper most likely to
    # win. A full cross product of 9 swappers x 4 enhancers x 2 blends would
    # be 72 candidates -- minutes of GPU time for combinations that cannot
    # win, since a weak swapper plus a restorer is still a weak identity.
    enhancers = [e for e in ENHANCERS if present(e)]
    lead = swappers[0]
    for e in enhancers:
        for blend in (0.4, 0.7):
            configs.append(PipelineConfig(
                swapper=lead, enhancer=e, enhancer_blend=blend,
                mask="model", label=f"{lead} + {e}@{int(blend*100)}"))
    return configs


def refine_with_enhancers(winner_swapper: str, samples, identity, opts,
                          exclude: set[str]) -> list[dict[str, Any]]:
    """Second pass: re-explore restoration on whichever swapper actually won.

    Stage 2 above guesses which swapper to pair enhancers with. If a different
    family wins stage 1, that guess was wrong, so the enhancer sweep is redone
    against the real winner rather than reporting a combination that was never
    measured.
    """
    from app.render.sessions import MODELS_DIR
    from app.render.registry import ALL

    out: list[dict[str, Any]] = []
    for e in ENHANCERS:
        if not (MODELS_DIR / ALL[e].filename).exists():
            continue
        for blend in (0.4, 0.7):
            label = f"{winner_swapper} + {e}@{int(blend*100)}"
            if label in exclude:
                continue
            cfg = PipelineConfig(swapper=winner_swapper, enhancer=e,
                                 enhancer_blend=blend, mask="model", label=label)
            try:
                out.append(score_config(cfg, samples, identity, opts))
            except RenderError as ex:
                out.append({"config": asdict(cfg), "label": label,
                            "error": str(ex)[:300], "score": -1.0})
    return out


# ---------------------------------------------------------------- measuring
def score_config(cfg: PipelineConfig, samples: list[tuple[int, np.ndarray, dict]],
                 identity: SourceIdentity, opts: RenderOptions) -> dict[str, Any]:
    """Render the sample frames with one config and measure the result."""
    diag = float(np.hypot(samples[0][1].shape[1], samples[0][1].shape[0]))
    renderer = FrameRenderer(cfg, opts, identity, diag)

    per_frame: list[dict[str, Any]] = []
    switches = 0
    prev_out: Optional[np.ndarray] = None
    prev_emb: Optional[np.ndarray] = None
    t0 = time.time()
    vram_peak = 0

    for idx, frame, traits in samples:
        faces = detection.detect_robust(frame, opts.detect_threshold, "quality")
        if not faces:
            continue
        face = max(faces, key=lambda f: f.area)

        try:
            t = time.time()
            out, full_mask = renderer.render_face(frame, face)
            ms = (time.time() - t) * 1000.0
        except RenderError as e:
            per_frame.append({"frame": idx, "error": str(e)[:160]})
            continue

        # Re-detect in the OUTPUT: identity must be measured on the face that
        # actually ended up in the frame, not where the input face was.
        out_faces = detection.detect_robust(out, 0.3, "quality")
        out_face = max(out_faces, key=lambda f: f.area) if out_faces else None

        ident = ident_stab = None
        emb = None
        if out_face is not None:
            try:
                emb = recognition.embed(out, out_face.kps)
                ident = recognition.similarity(emb, identity.embedding)
                ident_stab = metrics.identity_stability(prev_emb, emb)
            except RenderError:
                pass

        if (prev_emb is not None and emb is not None
                and recognition.similarity(prev_emb, emb) < IDENTITY_JUMP):
            switches += 1

        row = {
            "frame": idx,
            "traits": {k: round(v, 3) if isinstance(v, float) else v
                       for k, v in traits.items()},
            "identity": None if ident is None else round(ident, 4),
            "identity_stability": None if ident_stab is None else round(ident_stab, 4),
            "expression_delta": round(metrics.expression_delta(
                face.kps, out_face.kps), 5) if out_face is not None else None,
            "landmark_delta": round(metrics.landmark_delta(
                face.kps, out_face.kps, face.size), 5) if out_face is not None else None,
            "seam": round(metrics.seam_score(out, full_mask), 4),
            "color_discontinuity": round(metrics.color_discontinuity(out, full_mask), 3),
            "sharpness": round(metrics.sharpness(out, face.box), 2),
            "texture_retention": round(metrics.texture_retention(frame, out, face.box), 4),
            "mask_jitter": round(renderer.last_mask_jitter, 5),
            "flow_flicker": round(metrics.flow_warped_difference(
                prev_out, out, face.box), 3) if prev_out is not None else None,
            "ms": round(ms, 1),
        }
        per_frame.append(row)
        prev_out, prev_emb = out, emb

    # VRAM is sampled ONCE per candidate, not once per frame. gpu_info()
    # shells out to nvidia-smi, which costs hundreds of milliseconds to
    # launch; calling it per frame meant ~240 process spawns per benchmark and
    # left the GPU idling at 3% while the run waited on subprocess startup.
    # It was the single largest cost in Auto Max.
    vram_peak = int(sessions.gpu_info().get("vram_used_mb") or 0)

    def col(key: str) -> list[Optional[float]]:
        return [r.get(key) for r in per_frame if "error" not in r]

    agg = {
        "identity_mean": metrics.summarise(col("identity"))["mean"],
        "identity_min": metrics.summarise(col("identity"))["min"],
        "identity_stability_mean": metrics.summarise(col("identity_stability"))["mean"],
        "expression_delta_mean": metrics.summarise(col("expression_delta"))["mean"],
        "landmark_delta_mean": metrics.summarise(col("landmark_delta"))["mean"],
        "seam_mean": metrics.summarise(col("seam"))["mean"],
        "color_discontinuity_mean": metrics.summarise(col("color_discontinuity"))["mean"],
        "sharpness_mean": metrics.summarise(col("sharpness"))["mean"],
        "texture_retention_mean": metrics.summarise(col("texture_retention"))["mean"],
        "mask_jitter_mean": metrics.summarise(col("mask_jitter"))["mean"],
        "flow_flicker_mean": metrics.summarise(col("flow_flicker"))["mean"],
        "ms_per_frame": metrics.summarise(col("ms"))["mean"],
        "identity_switches": switches,
        "frames_measured": len([r for r in per_frame if "error" not in r]),
        "frames_failed": len([r for r in per_frame if "error" in r]),
        "vram_peak_mb": vram_peak,
        "wall_s": round(time.time() - t0, 2),
    }
    score, contrib = metrics.composite_score(agg)
    return {"config": asdict(cfg), "label": cfg.label or cfg.describe(),
            "aggregate": agg, "score": score, "contributions": contrib,
            "per_frame": per_frame}


def _identity_guard(bare: dict, enhanced: dict, max_loss: float = 0.04) -> bool:
    """Reject an enhancer that buys sharpness by giving up identity.

    A restorer that sharpens while pulling the face away from the person is not
    an improvement, however good the frame looks in isolation.
    """
    b = bare["aggregate"].get("identity_mean")
    e = enhanced["aggregate"].get("identity_mean")
    if b is None or e is None:
        return True
    return (b - e) <= max_loss


def free_mb() -> Optional[int]:
    """Free VRAM, or None when it cannot be measured."""
    g = sessions.gpu_info()
    if g.get("vram_total_mb") is None:
        return None
    return int(g["vram_total_mb"]) - int(g["vram_used_mb"])


# ---------------------------------------------------------------- caching
def _file_fingerprint(path: Path, sample_bytes: int = 1 << 20) -> str:
    """Cheap content fingerprint: size + head + tail.

    A full hash of a multi-GB video would cost more than the benchmark it is
    meant to save. Size plus both ends catches re-encodes, trims and swaps;
    it would miss a same-length edit confined to the middle, which is not a
    realistic way to arrive at a different video with an identical name.
    """
    h = hashlib.sha256()
    st = path.stat()
    h.update(str(st.st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(sample_bytes))
        if st.st_size > sample_bytes * 2:
            f.seek(-sample_bytes, 2)
            h.update(f.read(sample_bytes))
    return h.hexdigest()


def cache_key(target_path: str, identity: SourceIdentity,
              opts: RenderOptions) -> str:
    """Deterministic key over everything that could change the verdict."""
    from app.render.registry import ALL
    from app.render.sessions import MODELS_DIR

    h = hashlib.sha256()
    h.update(f"v{ALGORITHM_VERSION}|".encode())
    h.update(_file_fingerprint(Path(target_path)).encode())
    # The fused identity vector itself -- different photos, different verdict.
    h.update(np.asarray(identity.embedding, np.float32).tobytes())
    h.update(f"|{opts.swap_all_faces}|{opts.detect_threshold}".encode())

    # Model set AND their recorded hashes: replacing a weights file must
    # invalidate, not silently reuse a verdict measured on the old one.
    try:
        recorded = json.loads((MODELS_DIR / "hashes.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        recorded = {}
    for name in sorted(ALL):
        if (MODELS_DIR / ALL[name].filename).exists():
            entry = recorded.get(name) or {}
            h.update(f"{name}:{entry.get('sha256', '?')}|".encode())
    return h.hexdigest()[:32]


def load_cached(key: str) -> Optional[dict[str, Any]]:
    p = CACHE_DIR / f"{key}.json"
    if not p.is_file():
        return None
    try:
        rep = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if rep.get("algorithm_version") != ALGORITHM_VERSION:
        return None
    return rep


def save_cached(key: str, report: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    report = dict(report)
    report["algorithm_version"] = ALGORITHM_VERSION
    (CACHE_DIR / f"{key}.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------- driver
def run(target_path: str, identity: SourceIdentity, opts: RenderOptions,
        job_id: str, on_progress: Optional[Callable[[str, float, dict], None]] = None,
        thorough: bool = True, use_cache: bool = True
        ) -> tuple[PipelineConfig, dict[str, Any]]:
    """Benchmark candidate pipelines and return (winner, full report)."""
    def emit(stage: str, pct: float, **extra: Any) -> None:
        if on_progress:
            on_progress(stage, pct, extra)

    key = cache_key(target_path, identity, opts) if use_cache else None
    if key:
        cached = load_cached(key)
        if cached:
            emit("selecting best pipeline", 100.0,
                 winner=f"{cached['winner']['label']} (cached)")
            BENCH_DIR.mkdir(parents=True, exist_ok=True)
            cached["job_id"] = job_id
            cached["from_cache"] = True
            (BENCH_DIR / f"{job_id}.json").write_text(
                json.dumps(cached, indent=2, default=str), encoding="utf-8")
            return PipelineConfig(**cached["winner"]["config"]), cached

    info = video.probe(target_path)
    emit("analysing video", 5.0)
    samples = select_frames(target_path, info)
    emit("benchmarking swap models", 12.0, frames=len(samples))

    configs = candidate_configs(thorough=thorough)
    results: list[dict[str, Any]] = []
    for i, cfg in enumerate(configs):
        label = cfg.label or cfg.describe()
        stage = "testing restoration" if cfg.enhancer else "benchmarking swap models"
        emit(stage, 12.0 + 70.0 * i / max(len(configs), 1), candidate=label,
             index=i + 1, total=len(configs))
        try:
            results.append(score_config(cfg, samples, identity, opts))
        except RenderError as e:
            results.append({"config": asdict(cfg), "label": label,
                            "error": str(e)[:300], "score": -1.0})
        finally:
            # Evict the swapper we just finished with, but ONLY when the next
            # candidate needs a different one -- and never the shared
            # detector / recogniser / parser / enhancer sessions.
            #
            # An earlier version also released everything whenever free VRAM
            # looked low. That deadlocked into a thrash: releasing a Python
            # session does not hand VRAM straight back (ORT's CUDA allocator
            # keeps its arena), so "free" stayed low, so it released again
            # after every candidate and rebuilt ~1.5 GB of models each time.
            # One clip went from minutes to over an hour. Free VRAM is not a
            # usable trigger here; the next-candidate check is.
            nxt = configs[i + 1].swapper if i + 1 < len(configs) else None
            if cfg.swapper != nxt:
                sessions.release(cfg.swapper)

    ok = [r for r in results if r.get("score", -1) >= 0]
    if not ok:
        raise RenderError("every candidate pipeline failed during benchmarking; "
                          f"first error: {results[0].get('error') if results else 'unknown'}")

    # If a swapper OTHER than the one we paired enhancers with won stage 1,
    # the enhancer sweep was run against the wrong base. Redo it on the actual
    # winner so we never report a combination that was never measured.
    bare_now = [r for r in ok if not r["config"].get("enhancer")]
    if thorough and bare_now:
        top_bare = max(bare_now, key=lambda r: r["score"])
        top_name = top_bare["config"]["swapper"]
        paired = {r["config"]["swapper"] for r in ok if r["config"].get("enhancer")}
        if top_name not in paired:
            emit("testing restoration", 85.0, candidate=f"refining on {top_name}")
            extra = refine_with_enhancers(
                top_name, samples, identity, opts,
                exclude={r.get("label") for r in results})
            results.extend(extra)
            ok = [r for r in results if r.get("score", -1) >= 0]

    emit("selecting best pipeline", 90.0)

    bare = [r for r in ok if not r["config"].get("enhancer")]
    best_bare = max(bare, key=lambda r: r["score"]) if bare else None

    # An enhanced config only wins if it also keeps the identity.
    eligible = []
    for r in ok:
        if r["config"].get("enhancer") and best_bare is not None:
            if not _identity_guard(best_bare, r):
                r["rejected"] = "identity loss exceeded threshold vs bare swapper"
                continue
        eligible.append(r)

    winner = max(eligible or ok, key=lambda r: r["score"])
    cfg = PipelineConfig(**winner["config"])

    report = {
        "job_id": job_id,
        "created_at": time.time(),
        "video": info.as_dict(),
        "source_identity": {"n_images": identity.n_images,
                            "per_image": identity.per_image},
        "sample_frames": [{"index": i, "traits": t} for i, _, t in samples],
        "weights": metrics.WEIGHTS,
        "candidates": sorted(results, key=lambda r: -r.get("score", -1)),
        "winner": {"label": winner["label"], "score": winner["score"],
                   "config": winner["config"],
                   "contributions": winner.get("contributions"),
                   "aggregate": winner.get("aggregate")},
        "gpu": sessions.gpu_info(),
    }

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    report["from_cache"] = False
    (BENCH_DIR / f"{job_id}.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    if key:
        save_cached(key, report)

    emit("selecting best pipeline", 100.0, winner=winner["label"])
    return cfg, report
