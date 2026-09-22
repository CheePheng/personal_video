"""Build the strongest 512-d identity vector from 1..5 analysed photos.

AlphaFace accepts one 512-d vector, so the whole opportunity is deciding how
much each photo contributes to it. The decisions, in order:

  1. reject only genuine outliers -- a different person, not a hard photo
  2. weight by real defects: size, sharpness, exposure, and -- the term
     that actually pays -- how much of the face is VISIBLE

DUPLICATE SUPPRESSION WAS BUILT, MEASURED AND REMOVED. Halving a
near-duplicate's weight made the fusion WORSE (0.03055 mean drift against
0.02034), because duplicates act as stabilising ballast: the more
independent evidence agrees, the less any single bad photo can move the
result. Duplicates are still detected and reported, but no longer penalised.

BEST-SUBSET SELECTION WAS BUILT, MEASURED AND REMOVED. Any criterion for
"which photos to keep" scores subsets by agreement with the other references,
which is circular: the most distinctive view -- a genuine profile -- always
looks like the odd one out. On a five-photo set containing +72 and -76 degree
profiles, switching subsets moved the identity by 0.063 cosine, larger than
most of the contaminations it was supposed to defend against. Selection was
adding more instability than it removed, so all accepted references are used
and the weighting decides influence.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from app.render.source_analysis import (PhotoAnalysis, consensus_scores,
                                        mark_duplicates, quality_weight)
from app.render.types import RenderError, SourceIdentity

# Below this a reference is a DIFFERENT PERSON, not merely a bad photo.
#
# Set low on purpose, and the reason is measured. Rejecting a reference
# removes its evidence entirely, which moves the fused vector further than
# keeping it at low weight does. A blurred or badly exposed photo of the
# right person drops to 0.07-0.16 consensus -- it looks like an outlier
# without being one -- and rejecting it caused MORE identity drift than the
# blur itself. Two strangers score ~0.02, so 0.05 still catches the case
# this exists for: someone else's face in the upload.
#
# Degraded photos are handled by quality_weight, which already collapses
# for blur, exposure and occlusion. Rejection is for wrong people only.
OUTLIER_FLOOR = 0.05

def _fuse(items: list[PhotoAnalysis]) -> np.ndarray:
    w = np.array([quality_weight(a) for a in items], np.float32)
    stack = np.stack([a.embedding for a in items])
    v = (stack * w[:, None]).sum(0) / max(float(w.sum()), 1e-6)
    return v / max(float(np.linalg.norm(v)), 1e-6)


def build(items: list[PhotoAnalysis],
          allow_subset: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """Fuse analysed photos into one identity vector, with a report."""
    if not items:
        raise RenderError("no usable source faces")

    for a in items:
        a.weight = 0.0
        a.rejected = None

    mark_duplicates(items)
    consensus_scores(items)

    # Outlier rejection, only with enough references to have a real consensus.
    usable = list(items)
    if len(items) >= 3:
        keep = [a for a in items if a.consensus >= OUTLIER_FLOOR]
        for a in items:
            if a not in keep:
                a.rejected = "identity outlier (consensus %.3f)" % a.consensus
        if keep:
            usable = keep
            consensus_scores(usable)

    if not usable:
        raise RenderError("no usable source faces after outlier rejection")

    chosen = usable
    subset_note = "all %d references" % len(chosen)

    for a in chosen:
        a.weight = quality_weight(a)
    total = sum(a.weight for a in chosen) or 1.0
    for a in chosen:
        a.weight /= total

    fused = _fuse(chosen)
    report = {
        "n_input": len(items),
        "n_used": len(chosen),
        "subset": subset_note,
        "rejected": [a.name for a in items if a.rejected],
        "duplicates": [a.name for a in items if a.duplicate_of],
        "photos": [a.as_dict() for a in items],
    }
    return fused.astype(np.float32), report


def to_identity(fused: np.ndarray, report: dict[str, Any]) -> SourceIdentity:
    return SourceIdentity(embedding=fused.reshape(1, -1).astype(np.float32),
                          per_image=report["photos"],
                          n_images=report["n_used"])


def summary(report: dict[str, Any]) -> str:
    """One short line for the UI. No tuning knobs, no jargon."""
    n = report["n_input"]
    strong = sum(1 for p in report["photos"]
                 if not p["rejected"] and not p["occluded_regions"])
    obstructed = sum(1 for p in report["photos"]
                     if not p["rejected"] and p["occluded_regions"])
    bits = ["%d photo%s analysed" % (n, "" if n == 1 else "s")]
    if strong:
        bits.append("%d strong" % strong)
    if obstructed:
        bits.append("%d partially obstructed" % obstructed)
    if report["rejected"]:
        bits.append("%d rejected" % len(report["rejected"]))
    return ", ".join(bits)
