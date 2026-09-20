"""Model downloader: resumable, hash-verified, explicit on failure.

Models are large (up to 400 MB) and the registry is the single source of truth
for where each one comes from. A partial file is resumed via HTTP Range rather
than restarted, and a file whose hash does not match a recorded SHA256 is
treated as corrupt -- never silently used.

First download of a model has no recorded hash (we are the ones recording it),
so the hash is computed and stored; subsequent runs verify against it.

Usage:
    python -m app.render.download                 # everything registered
    python -m app.render.download hyperswap_1b_256 codeformer
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

from app.render.registry import ALL, REGISTRY_JSON, get_model
from app.render.sessions import MODELS_DIR
from app.render.types import RenderError

CHUNK = 1 << 20
HASHES = MODELS_DIR / "hashes.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _load_hashes() -> dict[str, str]:
    if HASHES.exists():
        try:
            return json.loads(HASHES.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}
    return {}


def _save_hash(name: str, digest: str, size: int) -> None:
    data = _load_hashes()
    data[name] = {"sha256": digest, "bytes": size}
    HASHES.parent.mkdir(parents=True, exist_ok=True)
    HASHES.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def fetch(name: str, force: bool = False, verbose: bool = True) -> Path:
    """Download one model if absent, verify it, and return its path."""
    spec = get_model(name)
    dest = MODELS_DIR / spec.filename
    part = dest.with_suffix(dest.suffix + ".part")
    known = _load_hashes().get(name)

    if dest.exists() and not force:
        if known:
            actual = _sha256(dest)
            if actual != known["sha256"]:
                raise RenderError(
                    f"model '{name}' failed integrity check.\n"
                    f"  expected sha256 {known['sha256']}\n"
                    f"  actual   sha256 {actual}\n"
                    f"Delete {dest} and re-download.")
        else:
            _save_hash(name, _sha256(dest), dest.stat().st_size)
        return dest

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    offset = part.stat().st_size if part.exists() else 0

    req = urllib.request.Request(spec.source_url, headers={"User-Agent": "faceswap-v2"})
    if offset:
        req.add_header("Range", f"bytes={offset}-")

    if verbose:
        print(f"  {name}: downloading{' (resume)' if offset else ''} ...", flush=True)

    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            # A server that ignores Range restarts the file; do not append.
            if offset and r.status != 206:
                offset = 0
            mode = "ab" if offset else "wb"
            total = int(r.headers.get("Content-Length") or 0) + offset
            done = offset
            with open(part, mode) as f:
                while chunk := r.read(CHUNK):
                    f.write(chunk)
                    done += len(chunk)
                    if verbose and total:
                        pct = done / total * 100
                        print(f"\r    {pct:5.1f}%  {done/1e6:7.1f} / {total/1e6:.1f} MB",
                              end="", flush=True)
    except urllib.error.HTTPError as e:
        raise RenderError(f"download failed for '{name}': HTTP {e.code} from {spec.source_url}") from e
    except (urllib.error.URLError, OSError) as e:
        raise RenderError(f"download failed for '{name}': {type(e).__name__}: {e}") from e

    if verbose:
        print(flush=True)

    size = part.stat().st_size
    if size < 1024:
        part.unlink(missing_ok=True)
        raise RenderError(f"download for '{name}' was empty or truncated ({size} bytes)")

    digest = _sha256(part)
    if spec.sha256 and digest != spec.sha256:
        part.unlink(missing_ok=True)
        raise RenderError(
            f"model '{name}' SHA mismatch: expected {spec.sha256}, got {digest}")

    part.replace(dest)
    _save_hash(name, digest, size)
    if verbose:
        print(f"    ok  {size/1e6:.1f} MB  sha256 {digest[:16]}...", flush=True)
    return dest


def ensure(names: Iterable[str], verbose: bool = True) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for n in names:
        out[n] = fetch(n, verbose=verbose)
    return out


def missing(names: Optional[Iterable[str]] = None) -> list[str]:
    return [n for n in (names or ALL) if not (MODELS_DIR / ALL[n].filename).exists()]


def main(argv: list[str]) -> int:
    from app.render.registry import export_registry

    names = argv[1:] or sorted(ALL)
    unknown = [n for n in names if n not in ALL]
    if unknown:
        print(f"unknown model(s): {unknown}\nregistered: {sorted(ALL)}")
        return 2

    print(f"Fetching {len(names)} model(s) into {MODELS_DIR}")
    failed: list[tuple[str, str]] = []
    for n in names:
        try:
            fetch(n)
        except RenderError as e:
            failed.append((n, str(e)))
            print(f"  {n}: FAILED - {e}", flush=True)

    export_registry()
    print(f"\nregistry written to {REGISTRY_JSON}")
    if failed:
        print(f"{len(failed)} model(s) failed:")
        for n, e in failed:
            print(f"  - {n}: {e[:160]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
