"""End-to-end smoke test against the RUNNING server, over real HTTP.

Not a unit test: it drives the same endpoints the browser drives, in the same
order, so it catches things unit tests structurally cannot -- an auth cookie
that does not stick, a chunked upload that reassembles wrong, a library row
whose file is missing, a video that will not seek in a player.

Checks, in order:
    login -> 5 source photos uploaded -> target video uploaded
    -> Max job created -> progress observed -> render completes
    -> audio present in output -> library lists it
    -> thumbnail 200 -> HTTP Range 206 -> full download intact

Run:  python scripts/smoke_app.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE = "http://127.0.0.1:8765"
CHUNK = 4 * 1024 * 1024

SRC_DIR = Path(
    "C:/Users/PC/AppData/Local/Temp/claude/"
    "c--Users-PC-Downloads-myproject-idea-personal-video/"
    "72122950-1114-48f6-8f32-96b2b1e68289/scratchpad/fixtures/wsrc")
TARGET = ROOT / "data" / "testclips" / "ab" / "G_talking.mp4"

OK: list[str] = []
BAD: list[str] = []


def check(label: str, cond: bool, note: str = "") -> bool:
    (OK if cond else BAD).append(label)
    print("  [%s] %-40s %s" % ("PASS" if cond else "FAIL", label, note), flush=True)
    return cond


_opener = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(CookieJar()),
    urllib.request.HTTPRedirectHandler())


def req(path: str, data=None, method=None, headers=None, raw=False):
    """Returns (status, body). Never raises on an HTTP error status."""
    url = path if path.startswith("http") else BASE + path
    r = urllib.request.Request(url, data=data, method=method,
                               headers=headers or {})
    try:
        with _opener.open(r, timeout=120) as resp:
            body = resp.read()
            return resp.status, (body if raw else body.decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        body = e.read()
        return e.code, (body if raw else body.decode("utf-8", "replace"))


def jreq(path: str, obj, method="POST"):
    s, b = req(path, data=json.dumps(obj).encode(),
               method=method, headers={"Content-Type": "application/json"})
    try:
        return s, json.loads(b)
    except (ValueError, TypeError):
        return s, {}


def upload(path: Path) -> str | None:
    """Chunked upload, exactly as the browser does it."""
    size = path.stat().st_size
    s, j = jreq("/api/upload/init", {"filename": path.name, "size": size})
    if s != 200 or "upload_id" not in j:
        return None
    uid = j["upload_id"]
    n = 0
    with open(path, "rb") as f:
        while True:
            blob = f.read(CHUNK)
            if not blob:
                break
            st, _ = req("/api/upload/chunk/%s/%d" % (uid, n), data=blob,
                        method="POST",
                        headers={"Content-Type": "application/octet-stream"})
            if st != 200:
                return None
            n += 1
    s, j = jreq("/api/upload/complete", {"upload_id": uid, "chunks": n})
    if s != 200:
        return None
    return j.get("media_id") or j.get("id") or uid


def has_audio(p: Path) -> bool:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(p)],
            capture_output=True, text=True, timeout=30)
        return bool(r.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def main() -> int:
    print("SMOKE TEST -- real HTTP against %s\n" % BASE)

    pw_file = ROOT / "data" / ".password"
    if not pw_file.is_file():
        print("no data/.password; cannot log in")
        return 1
    pw = pw_file.read_text(encoding="utf-8").strip()

    s, _ = req("/api/library")
    check("unauthenticated request is refused", s in (401, 403, 302, 303),
          "got %s" % s)

    s, _ = req("/login", data=("password=%s" % pw).encode(), method="POST",
               headers={"Content-Type": "application/x-www-form-urlencoded"})
    check("login", s == 200, "got %s" % s)

    s, _ = req("/")
    check("authenticated UI", s == 200, "got %s" % s)

    photos = sorted(SRC_DIR.glob("w_photo*.png"))[:5]
    if len(photos) < 5:
        check("5 source photos available", False, "found %d" % len(photos))
        return 1
    src_ids = []
    for p in photos:
        mid = upload(p)
        if mid:
            src_ids.append(mid)
    check("upload 5 source photos", len(src_ids) == 5,
          "%d/5 accepted" % len(src_ids))

    tgt = upload(TARGET)
    check("upload target video", bool(tgt), TARGET.name)
    if not tgt or len(src_ids) < 5:
        return 1

    s, j = jreq("/api/jobs", {"source_ids": src_ids, "target_id": tgt,
                              "quality": "max", "face_mode": "single"})
    jid = j.get("job_id")
    check("create Max job", s == 200 and bool(jid), "id %s" % jid)
    if not jid:
        return 1

    saw_progress = False
    status = ""
    deadline = time.time() + 900
    while time.time() < deadline:
        s, j = req("/api/jobs/%s" % jid)
        try:
            job = json.loads(j)
        except (ValueError, TypeError):
            time.sleep(2)
            continue
        status = job.get("status", "")
        if job.get("progress"):
            saw_progress = True
        if status in ("done", "error", "cancelled"):
            break
        time.sleep(3)

    check("progress reported", saw_progress, "status=%s" % status)
    if not check("render completed", status == "done", "status=%s" % status):
        print("     error: %s" % str(job.get("error"))[:300])
        return 1

    # Verify the stored options really recorded Max, not a silent fallback.
    try:
        opts = json.loads(job.get("options") or "{}")
    except (ValueError, TypeError):
        opts = {}
    check("job recorded quality=max", opts.get("quality") == "max",
          "quality=%s" % opts.get("quality"))

    out = job.get("output_path")
    op = Path(out) if out else None
    check("output file exists", bool(op and op.exists()),
          op.name if op else "none")
    if op and op.exists():
        check("output has audio", has_audio(op), "")

    s, b = req("/api/library")
    try:
        items = json.loads(b)
    except (ValueError, TypeError):
        items = []
    ids = [i.get("id") for i in items]
    check("library lists the render", jid in ids,
          "%d item(s)" % len(items))

    s, _ = req("/api/library/%s/thumb" % jid)
    check("thumbnail", s == 200, "got %s" % s)

    s, body = req("/api/library/%s/video" % jid,
                  headers={"Range": "bytes=0-1023"}, raw=True)
    check("HTTP Range", s == 206, "got %s, %d bytes" % (s, len(body)))

    s, full = req("/api/library/%s/video" % jid, raw=True)
    disk = op.stat().st_size if (op and op.exists()) else -1
    check("full download intact", s == 200 and len(full) == disk,
          "%d bytes served, %d on disk" % (len(full), disk))

    print("\n" + "=" * 62)
    print("  %d passed, %d failed" % (len(OK), len(BAD)))
    for b in BAD:
        print("    FAILED: %s" % b)
    print("=" * 62)
    return 1 if BAD else 0


if __name__ == "__main__":
    raise SystemExit(main())
