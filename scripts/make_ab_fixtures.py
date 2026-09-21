"""Build an A -> B quality-validation set: source and subject are DIFFERENT people.

The existing synthetic suite pastes ``face_a`` into every clip and then uses
that same ``_face_a.png`` as the source identity. Rendering it swaps a face
onto itself, so its 0.96-0.97 identity scores measure "the pipeline did not
destroy a face that already matched" -- not how well identity transfers. They
are fine as an engineering regression (does it still run, does it still track,
does audio survive) and useless as evidence of face-swap quality.

This builds a second suite where:

    identity A  the SOURCE photo            -- the face we want to see
    identity B  the SUBJECT in every clip   -- the person being replaced
    identity C  the DISTRACTOR              -- a third person, for tracking

and refuses to emit anything unless A, B and C are confirmed distinct by BOTH
recognisers: ArcFace (which the swap is conditioned on, so it cannot be the
only witness) and SFace (independently trained, and never used for selection).

Source images are public-domain NASA official portraits, so the whole fixture
set is rights-cleared and anyone can regenerate it.

Run:  python scripts/make_ab_fixtures.py [--portraits DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AB = ROOT / "data" / "testclips" / "ab"

# Public-domain NASA official portraits (works of the US government).
PORTRAITS = {
    "a": ("person_a_src.jpg", "Kalpana Chawla, NASA official portrait (public domain)"),
    "b": ("person_b_src.jpg", "Jack Hathaway, NASA official portrait (public domain)"),
    "c": ("person_c_src.jpg", "Laurel Clark, NASA official portrait (public domain)"),
}

# Two different people should sit far below this under either recogniser.
# ArcFace same-person pairs run >0.5 and the tracker's own identity floor is
# 0.28, so 0.25 is a deliberately conservative ceiling for "definitely not
# the same person".
MAX_CROSS_SIMILARITY = 0.25


def crop_face(img: np.ndarray, size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Square crop around the largest face, and its landmarks in that crop."""
    from app.render import detection

    faces = detection.detect_robust(img, 0.3)
    if not faces:
        raise SystemExit("no face found in a portrait -- cannot build fixtures")
    face = max(faces, key=lambda f: f.area)
    cx, cy = face.centre
    s = int(face.size * 2.0)
    x1, y1 = max(0, int(cx - s // 2)), max(0, int(cy - s // 2))
    x2, y2 = min(img.shape[1], x1 + s), min(img.shape[0], y1 + s)
    crop = cv2.resize(img[y1:y2, x1:x2], (size, size),
                      interpolation=cv2.INTER_LANCZOS4)
    sub = detection.detect_robust(crop, 0.3)
    if not sub:
        raise SystemExit("face lost after cropping -- portrait unusable")
    return crop, max(sub, key=lambda f: f.area).kps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--portraits", default=str(
        Path("C:/Users/PC/AppData/Local/Temp/claude/"
             "c--Users-PC-Downloads-myproject-idea-personal-video/"
             "72122950-1114-48f6-8f32-96b2b1e68289/scratchpad/fixtures")))
    args = ap.parse_args()
    src_dir = Path(args.portraits)

    from app.render import judges, recognition

    AB.mkdir(parents=True, exist_ok=True)
    crops: dict[str, np.ndarray] = {}
    arc: dict[str, np.ndarray] = {}
    sfc: dict[str, np.ndarray] = {}

    print("Cropping identities")
    for key, (fname, credit) in PORTRAITS.items():
        p = src_dir / fname
        img = cv2.imread(str(p))
        if img is None:
            raise SystemExit("missing portrait: %s" % p)
        crop, kps = crop_face(img)
        crops[key] = crop
        arc[key] = recognition.embed(crop, kps)
        sfc[key] = judges.embed(crop, kps) if judges.available() else None
        out = AB / ("_ident_%s.png" % key)
        cv2.imwrite(str(out), crop)
        print("  %s  %-70s -> %s" % (key.upper(), credit, out.name))

    print("\nCross-identity check (must be well below %.2f)" % MAX_CROSS_SIMILARITY)
    worst = -1.0
    for x, y in (("a", "b"), ("a", "c"), ("b", "c")):
        sa = recognition.similarity(arc[x], arc[y])
        ss = (judges.similarity(sfc[x], sfc[y])
              if sfc[x] is not None and sfc[y] is not None else float("nan"))
        worst = max(worst, sa, ss if np.isfinite(ss) else -1.0)
        flag = "ok " if max(sa, ss if np.isfinite(ss) else -1) < MAX_CROSS_SIMILARITY else "FAIL"
        print("  %s %s vs %s   ArcFace %+.4f   SFace %+.4f" % (flag, x.upper(), y.upper(), sa, ss))

    if worst >= MAX_CROSS_SIMILARITY:
        print("\nREFUSING to build: two identities are too similar (max %.4f). "
              "A benchmark whose source and subject might be the same person "
              "cannot measure identity transfer." % worst)
        return 1
    print("  all pairs distinct (worst %+.4f)" % worst)

    # The source photo the renders will use.
    cv2.imwrite(str(AB / "_source_a.png"), crops["a"])

    print("\nBuilding clips with B as the subject and C as the distractor")
    import scripts.make_testclips as m1
    import scripts.make_testclips2 as m2

    m1.OUT = AB
    m2.OUT = AB
    made = {}
    made.update(m1.build(crops["b"], crops["c"]))
    made.update(m2.build_synth(crops["b"]))
    for name in sorted(made):
        print("  %s" % name)

    print("\n%d clips in %s" % (len(made), AB))
    print("Source identity: %s" % (AB / "_source_a.png"))
    print("\nThese are A->B: the source is never the person in the clip.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
