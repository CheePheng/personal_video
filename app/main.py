"""FastAPI backend for the local face swap tool.

Endpoints are shaped by two Cloudflare limits (measured, see docs/FINDINGS.md):
  * large single-POST uploads are rejected at the edge -> chunked upload
  * 125s proxy read timeout, and no SSE on Quick Tunnels -> async jobs + polling

The render itself runs in-process: ONNX Runtime (CUDA) + OpenCV + ffmpeg, in
app/swapper.py. Progress is a real frame count, not parsed console output.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import jobs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
OUTPUTS = DATA / "outputs"
PARTS = DATA / "jobs" / "parts"

for d in (UPLOADS, OUTPUTS, PARTS):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Face Swap")
app.mount("/static", StaticFiles(directory=str(ROOT / "app" / "static")), name="static")

# ---------------------------------------------------------------- auth
# The URL is public, so without this anyone who finds it can queue renders on
# this machine's GPU. One password, no username -- a login page rather than HTTP
# Basic, because Basic always shows a username box the user does not want.
PASSWORD_FILE = DATA / ".password"
SESSION_SECRET_FILE = DATA / ".session_secret"
COOKIE = "fsw_session"


def _load_password() -> str:
    env = os.environ.get("FSW_PASSWORD", "").strip()
    if env:
        return env
    if PASSWORD_FILE.exists():
        existing = PASSWORD_FILE.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    generated = secrets.token_urlsafe(9)
    PASSWORD_FILE.write_text(generated, encoding="utf-8")
    return generated


def _session_token() -> str:
    """Stateless session value: an HMAC over the current password.

    Stateless so a server restart does not log you out, and derived from the
    password so changing the password invalidates every existing cookie.
    """
    if SESSION_SECRET_FILE.exists():
        secret = SESSION_SECRET_FILE.read_text(encoding="utf-8").strip()
    else:
        secret = secrets.token_urlsafe(32)
        SESSION_SECRET_FILE.write_text(secret, encoding="utf-8")
    return hmac.new(secret.encode(), PASSWORD.encode(), hashlib.sha256).hexdigest()


PASSWORD = _load_password()
SESSION_VALUE = _session_token()

LOGIN_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Face Swap</title>
<style>
:root{--bg:#0e1116;--panel:#161b22;--line:#2a323d;--text:#e6edf3;--muted:#8b949e;--accent:#3b82f6;--err:#f85149}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--text);
 font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;padding:24px}
form{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:28px;max-width:340px;width:100%}
h1{font-size:18px;margin:0 0 4px}
p{color:var(--muted);font-size:13px;margin:0 0 18px}
input{width:100%;background:#0b0f14;color:var(--text);border:1px solid var(--line);border-radius:9px;
 padding:12px 13px;font-size:16px}
input:focus{outline:none;border-color:var(--accent)}
button{width:100%;margin-top:12px;background:var(--accent);color:#fff;border:0;border-radius:9px;
 padding:12px;font-size:15px;font-weight:600;cursor:pointer}
.err{color:var(--err);font-size:13px;margin-top:10px}
</style></head>
<body><form method="post" action="/login">
<h1>Face Swap</h1>
<p>Enter your password to continue.</p>
<input type="password" name="password" placeholder="Password" autofocus autocomplete="current-password">
<button type="submit">Unlock</button>
__ERR__
</form></body></html>"""

# Paths reachable without a session.
OPEN_PATHS = {"/healthz", "/login"}


@app.middleware("http")
async def require_password(request: Request, call_next):
    path = request.url.path
    if path in OPEN_PATHS or path.startswith("/static/"):
        return await call_next(request)

    cookie = request.cookies.get(COOKIE, "")
    if cookie and secrets.compare_digest(cookie, SESSION_VALUE):
        return await call_next(request)

    # XHR/fetch calls get a clean 401 so the front end can react; browsers
    # navigating to a page get the login screen.
    if path.startswith("/api/"):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return HTMLResponse(LOGIN_HTML.replace("__ERR__", ""), status_code=401)


@app.get("/login", response_class=HTMLResponse)
def login_page() -> HTMLResponse:
    return HTMLResponse(LOGIN_HTML.replace("__ERR__", ""))


@app.post("/login")
async def login_submit(request: Request) -> Response:
    form = await request.form()
    supplied = str(form.get("password", ""))
    if not secrets.compare_digest(supplied, PASSWORD):
        return HTMLResponse(
            LOGIN_HTML.replace("__ERR__", '<div class="err">Wrong password.</div>'),
            status_code=401,
        )
    # secure=True would make the cookie unusable over plain http on localhost,
    # so mirror however this request actually arrived. Through the Cloudflare
    # Worker that is always https; direct on 127.0.0.1 it is not.
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    resp = Response(status_code=303, headers={"location": "/"})
    resp.set_cookie(
        COOKIE, SESSION_VALUE,
        max_age=60 * 60 * 24 * 365,  # stay logged in; this is a personal tool
        httponly=True, samesite="lax", secure=(proto == "https"), path="/",
    )
    return resp

_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_name(name: str) -> str:
    """Windows-safe basename; upload ids keep files unique so collisions are fine."""
    return _SAFE.sub("_", os.path.basename(name or "file"))[-80:] or "file"


@app.on_event("startup")
def _startup() -> None:
    jobs.init_db()


@app.get("/healthz")
def healthz() -> Response:
    # start.ps1 polls this through the tunnel before printing the URL, because
    # cloudflared prints the hostname before it is actually routable.
    return JSONResponse({"ok": True, "active_jobs": jobs.active_count()})


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((ROOT / "app" / "templates" / "index.html").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- uploads
class InitReq(BaseModel):
    filename: str
    size: int = 0


@app.post("/api/upload/init")
def upload_init(req: InitReq) -> dict:
    uid = os.urandom(8).hex()
    d = PARTS / uid
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"filename": _safe_name(req.filename), "size": req.size}), encoding="utf-8")
    return {"upload_id": uid}


@app.post("/api/upload/chunk/{uid}/{index}")
async def upload_chunk(uid: str, index: int, request: Request) -> dict:
    d = PARTS / _SAFE.sub("", uid)
    if not d.is_dir():
        raise HTTPException(404, "unknown upload id")
    # Stream straight to disk. Starlette's UploadFile would spool the whole body
    # to a temp file first, doubling I/O on multi-GB videos.
    written = 0
    with open(d / f"{index:08d}.part", "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
            written += len(chunk)
    return {"ok": True, "bytes": written}


class CompleteReq(BaseModel):
    upload_id: str
    chunks: int


@app.post("/api/upload/complete")
def upload_complete(req: CompleteReq) -> dict:
    d = PARTS / _SAFE.sub("", req.upload_id)
    if not d.is_dir():
        raise HTTPException(404, "unknown upload id")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    final = UPLOADS / f"{req.upload_id}_{meta['filename']}"

    missing = [i for i in range(req.chunks) if not (d / f"{i:08d}.part").exists()]
    if missing:
        raise HTTPException(400, f"missing chunks: {missing[:10]}")

    with open(final, "wb") as out:
        for i in range(req.chunks):
            p = d / f"{i:08d}.part"
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out, 1024 * 1024)
    shutil.rmtree(d, ignore_errors=True)
    return {"path": str(final), "size": final.stat().st_size}


# ---------------------------------------------------------------- jobs
# ---------------------------------------------------------------- jobs
# The presets themselves live in app/swapper.py (QUALITY_PRESETS), next to the
# code that acts on them -- one source of truth, so a preset cannot be accepted
# here and then silently ignored by the pipeline.
QUALITY = ("fast", "balanced", "best")


class JobReq(BaseModel):
    source_path: str
    target_path: str
    quality: str = "balanced"
    face_mode: str = "reference"


@app.post("/api/jobs")
def create_job(req: JobReq) -> dict:
    for p in (req.source_path, req.target_path):
        if not os.path.exists(p):
            raise HTTPException(400, f"missing file: {p}")

    jid_name = os.urandom(6).hex()
    output = str(OUTPUTS / f"{jid_name}.mp4")
    opts = {
        "quality": req.quality if req.quality in QUALITY else "balanced",
        "face_mode": req.face_mode,
    }

    jid = jobs.create_job(req.source_path, req.target_path, output, json.dumps(opts))
    jobs.start(jid, req.source_path, req.target_path, output, opts)
    return {"job_id": jid}

@app.get("/api/jobs/{jid}")
def job_status(jid: str) -> dict:
    j = jobs.get_job(jid)
    if not j:
        raise HTTPException(404, "no such job")
    j.pop("source_path", None)
    j.pop("target_path", None)
    return j


@app.post("/api/jobs/{jid}/cancel")
def job_cancel(jid: str) -> dict:
    return {"cancelled": jobs.cancel(jid)}


@app.get("/api/library/{jid}/video")
@app.get("/api/jobs/{jid}/video")
def job_video(jid: str, request: Request, download: int = 0) -> Response:
    j = jobs.get_job(jid)
    if not j or j["status"] != "done":
        raise HTTPException(404, "not ready")
    path = Path(j["output_path"])
    if not path.exists():
        raise HTTPException(404, "output missing")

    size = path.stat().st_size
    rng = request.headers.get("range")
    headers = {"accept-ranges": "bytes"}
    if download:
        headers["content-disposition"] = f'attachment; filename="faceswap-{jid}.mp4"'

    # Range support so the browser player can seek rather than only play start-to-end.
    if rng and rng.startswith("bytes="):
        try:
            s, _, e = rng[6:].partition("-")
            start = int(s) if s else 0
            end = int(e) if e else size - 1
            end = min(end, size - 1)
            if start > end:
                raise ValueError
        except ValueError:
            return Response(status_code=416, headers={"content-range": f"bytes */{size}"})

        def it():
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    b = f.read(min(1024 * 512, left))
                    if not b:
                        break
                    left -= len(b)
                    yield b

        headers |= {"content-range": f"bytes {start}-{end}/{size}", "content-length": str(end - start + 1)}
        return StreamingResponse(it(), status_code=206, media_type="video/mp4", headers=headers)

    return FileResponse(path, media_type="video/mp4", headers=headers)


@app.get("/api/jobs")
def job_list() -> list[dict]:
    out = []
    for j in jobs.list_jobs(30):
        j.pop("source_path", None)
        j.pop("target_path", None)
        out.append(j)
    return out


# ---------------------------------------------------------------- library
# Every finished render is kept and listed here. The job rows are the source of
# truth rather than a directory scan, because they carry the quality/face-mode
# options and timestamps that make old renders identifiable months later.

THUMBS = DATA / "thumbs"
THUMBS.mkdir(parents=True, exist_ok=True)


def _ffprobe_duration(path: Path) -> Optional[float]:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=20,
        )
        return round(float(r.stdout.strip()), 1)
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


@app.get("/api/library")
def library_list() -> list[dict]:
    """Finished renders, newest first, with the file actually on disk.

    A row whose output has been deleted from disk is skipped rather than shown
    as a broken tile -- files can be removed outside the app.
    """
    items = []
    for j in jobs.list_jobs(500):
        if j["status"] != "done" or not j.get("output_path"):
            continue
        p = Path(j["output_path"])
        if not p.exists():
            continue
        try:
            opts = json.loads(j.get("options") or "{}")
        except (ValueError, TypeError):
            opts = {}
        items.append({
            "id": j["id"],
            "filename": p.name,
            "size": p.stat().st_size,
            "created_at": j.get("finished_at") or j.get("created_at"),
            "quality": opts.get("quality"),
            "face_mode": opts.get("face_mode"),
            "duration": _ffprobe_duration(p),
        })
    return items


@app.get("/api/library/{jid}/thumb")
def library_thumb(jid: str) -> Response:
    """A poster frame, generated once and then cached on disk.

    Grabbed at 1s rather than 0s: the very first frame is often a fade-in or a
    black lead-in, which makes every tile look identical.
    """
    j = jobs.get_job(jid)
    if not j or j["status"] != "done" or not j.get("output_path"):
        raise HTTPException(404, "not ready")
    src = Path(j["output_path"])
    if not src.exists():
        raise HTTPException(404, "output missing")

    thumb = THUMBS / f"{jid}.jpg"
    if not thumb.exists() or thumb.stat().st_size == 0:
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-ss", "1", "-i", str(src), "-frames:v", "1",
                 "-vf", "scale=480:-2", "-q:v", "4", str(thumb)],
                capture_output=True, timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        if not thumb.exists() or thumb.stat().st_size == 0:
            raise HTTPException(404, "no thumbnail")
    return FileResponse(thumb, media_type="image/jpeg",
                        headers={"cache-control": "public, max-age=86400"})


@app.delete("/api/library/{jid}")
def library_delete(jid: str) -> dict:
    """Remove a render: the video, its thumbnail, and the job row."""
    j = jobs.get_job(jid)
    if not j:
        raise HTTPException(404, "no such job")
    for p in (Path(j["output_path"]) if j.get("output_path") else None, THUMBS / f"{jid}.jpg"):
        if p:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
    jobs.delete_job(jid)
    return {"deleted": jid}
