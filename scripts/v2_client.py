"""Test client: exercises the real HTTP API exactly as the browser does.

Used by the V2 test suite so the endpoints are verified end to end (auth,
chunked upload, media ids, job polling, range playback) rather than by calling
the pipeline directly in-process.
"""

from __future__ import annotations

import http.cookiejar
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8765"
CHUNK = 8 * 1024 * 1024


class Client:
    def __init__(self, base: str = BASE):
        self.base = base
        self.cj = http.cookiejar.CookieJar()
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cj))

    def login(self, password: Optional[str] = None) -> None:
        if password is None:
            password = (ROOT / "data" / ".password").read_text(encoding="utf-8").strip()
        self.op.open(f"{self.base}/login", data=f"password={password}".encode())

    def _json(self, path: str, body: Optional[dict] = None, method: str = "GET") -> Any:
        url = f"{self.base}{path}"
        if body is None:
            req = urllib.request.Request(url, method=method)
        else:
            req = urllib.request.Request(
                url, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
        with self.op.open(req) as r:
            return json.load(r)

    def upload(self, path: str) -> str:
        """Chunked upload; returns the server-issued media id."""
        p = Path(path)
        size = p.stat().st_size
        uid = self._json("/api/upload/init", {"filename": p.name, "size": size})["upload_id"]
        n = max(1, (size + CHUNK - 1) // CHUNK)
        with open(p, "rb") as f:
            for i in range(n):
                blob = f.read(CHUNK)
                req = urllib.request.Request(
                    f"{self.base}/api/upload/chunk/{uid}/{i}", data=blob,
                    headers={"Content-Type": "application/octet-stream"}, method="POST")
                self.op.open(req)
        return self._json("/api/upload/complete", {"upload_id": uid, "chunks": n})["media_id"]

    def start_job(self, source_ids: list[str], target_id: str,
                  quality: str = "quality", face_mode: str = "reference",
                  engine: str = "v2") -> str:
        return self._json("/api/jobs", {
            "source_ids": source_ids, "target_id": target_id,
            "quality": quality, "face_mode": face_mode, "engine": engine,
        })["job_id"]

    def wait(self, jid: str, timeout: float = 3600, quiet: bool = False) -> dict:
        t0 = time.time()
        last = ""
        while time.time() - t0 < timeout:
            j = self._json(f"/api/jobs/{jid}")
            line = f"{j['status']:<9} {j['progress']:6.2f}%  {j.get('phase') or '':<26} {str(j.get('message'))[:52]}"
            if not quiet and line != last:
                print(f"    {line}", flush=True)
                last = line
            if j["status"] not in ("running", "queued"):
                return j
            time.sleep(2)
        raise TimeoutError(f"job {jid} did not finish in {timeout}s")

    def library(self) -> list[dict]:
        return self._json("/api/library")

    def range_status(self, jid: str, rng: str = "bytes=0-999") -> int:
        req = urllib.request.Request(f"{self.base}/api/library/{jid}/video",
                                     headers={"Range": rng})
        try:
            with self.op.open(req) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    def delete(self, jid: str) -> None:
        try:
            self._json(f"/api/library/{jid}", method="DELETE")
        except urllib.error.HTTPError:
            pass


if __name__ == "__main__":
    c = Client()
    c.login()
    print(json.dumps(c.library(), indent=2)[:2000])
