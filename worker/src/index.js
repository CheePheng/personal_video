/**
 * Permanent public front door for the home face-swap box.
 *
 *   browser -> faceswap.<acct>.workers.dev  (this Worker, always up)
 *                      |
 *                      +-- PC online  -> proxy to the current tunnel URL
 *                      +-- PC offline -> serve an Offline page
 *
 * The PC's tunnel URL changes on every restart, so the PC registers its current
 * URL here and re-registers periodically as a heartbeat.
 *
 * State lives in a Durable Object, NOT Workers KV: KV caches reads for ~60s, so
 * right after a PC restart the Worker would keep proxying to a dead hostname for
 * a minute -- precisely when it is supposed to recover. A DO is single-threaded
 * and strongly consistent, so a read always sees the heartbeat just written.
 */

const STALE_MS = 3 * 60 * 1000; // no heartbeat for 3 min => treat as offline

export class TunnelState {
  constructor(state) {
    this.state = state;
  }

  async fetch(request) {
    const url = new URL(request.url);

    if (url.pathname === "/set" && request.method === "POST") {
      const body = await request.json();
      await this.state.storage.put("tunnel", { url: body.url, ts: Date.now() });
      return Response.json({ ok: true });
    }

    if (url.pathname === "/clear" && request.method === "POST") {
      await this.state.storage.delete("tunnel");
      return Response.json({ ok: true });
    }

    const rec = await this.state.storage.get("tunnel");
    if (!rec) return Response.json({ online: false, reason: "never registered" });
    const age = Date.now() - rec.ts;
    return Response.json({
      online: age < STALE_MS,
      url: rec.url,
      age_ms: age,
      reason: age < STALE_MS ? "ok" : "heartbeat stale",
    });
  }
}

function stub(env) {
  return env.TUNNEL_STATE.get(env.TUNNEL_STATE.idFromName("singleton"));
}

/**
 * Hitting the Durable Object on every proxied request would cost ~1.7M DO
 * requests/month under 1.5s status polling. Cache the lookup per-isolate for a
 * few seconds instead. Strong consistency is still preserved where it actually
 * matters: the cache is dropped immediately whenever a proxy attempt fails,
 * which is exactly the symptom of a stale tunnel URL after a PC restart.
 */
const CACHE_MS = 3000;
let _cache = null; // { value, at }

async function readState(env, { fresh = false } = {}) {
  const now = Date.now();
  if (!fresh && _cache && now - _cache.at < CACHE_MS) return _cache.value;
  const value = await (await stub(env).fetch("https://do/get")).json();
  _cache = { value, at: now };
  return value;
}

function dropCache() {
  _cache = null;
}

const OFFLINE_HTML = `<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Offline</title>
<style>
:root{--bg:#0e1116;--panel:#161b22;--line:#2a323d;--text:#e6edf3;--muted:#8b949e;--warn:#d29922}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--text);
 font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;padding:24px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:30px;max-width:420px;width:100%;text-align:center}
.dot{width:11px;height:11px;border-radius:50%;background:var(--warn);display:inline-block;margin-right:8px;
 animation:p 1.8s ease-in-out infinite}
@keyframes p{0%,100%{opacity:1}50%{opacity:.35}}
h1{font-size:19px;margin:0 0 6px}
p{color:var(--muted);margin:8px 0 0;font-size:14px}
.hint{margin-top:20px;padding-top:16px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}
code{background:#0b0f14;border:1px solid var(--line);border-radius:6px;padding:2px 7px;font-size:12.5px}
</style></head>
<body><div class="card">
<h1><span class="dot"></span>Face Swap is offline</h1>
<p>The render PC is switched off, so there is nothing to connect to right now.</p>
<div class="hint">Turn the PC on and run <code>start.bat</code>.<br>This page reconnects on its own.</div>
</div>
<script>setTimeout(()=>location.reload(),20000)</script>
</body></html>`;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // --- PC registers / heartbeats its current tunnel URL -----------------
    if (url.pathname === "/__register" && request.method === "POST") {
      const secret = request.headers.get("x-register-secret") || "";
      if (!env.REGISTER_SECRET || secret !== env.REGISTER_SECRET) {
        return new Response("forbidden", { status: 403 });
      }
      let body;
      try {
        body = await request.json();
      } catch {
        return new Response("bad json", { status: 400 });
      }
      if (!body || typeof body.url !== "string" || !/^https:\/\/[a-z0-9.-]+$/i.test(body.url)) {
        return new Response("bad url", { status: 400 });
      }
      const r = await stub(env).fetch("https://do/set", {
        method: "POST",
        body: JSON.stringify({ url: body.url }),
      });
      dropCache(); // a re-register means the URL may have just changed
      return new Response(await r.text(), { status: r.status, headers: { "content-type": "application/json" } });
    }

    // --- PC signals a clean shutdown --------------------------------------
    if (url.pathname === "/__offline" && request.method === "POST") {
      const secret = request.headers.get("x-register-secret") || "";
      if (!env.REGISTER_SECRET || secret !== env.REGISTER_SECRET) {
        return new Response("forbidden", { status: 403 });
      }
      await stub(env).fetch("https://do/clear", { method: "POST" });
      dropCache();
      return Response.json({ ok: true });
    }

    // --- machine-readable status (no auth; leaks nothing sensitive) -------
    // /__status reads through the cache so a monitor cannot be fooled by it.
    const state = await readState(env, { fresh: url.pathname === "/__status" });
    if (url.pathname === "/__status") {
      return Response.json({ online: !!state.online, reason: state.reason, age_ms: state.age_ms ?? null });
    }

    if (!state.online) {
      return new Response(OFFLINE_HTML, {
        status: 503,
        headers: { "content-type": "text/html; charset=utf-8", "retry-after": "30", "cache-control": "no-store" },
      });
    }

    // --- proxy everything else to the PC ----------------------------------
    // Build the subrequest FROM the original so Range, Authorization and the
    // streamed body all pass through untouched.
    const target = new URL(url.pathname + url.search, state.url);
    const proxied = new Request(target.toString(), request);
    proxied.headers.set("host", new URL(state.url).host);

    try {
      const resp = await fetch(proxied);
      // Return the body as a stream; never buffer (a video may be 500 MB).
      const out = new Response(resp.body, resp);
      out.headers.set("cache-control", "no-store");
      return out;
    } catch (e) {
      // A failed proxy is the signature of a stale tunnel hostname (the PC
      // restarted and got a new one). Drop the cache so the very next request
      // re-reads the Durable Object rather than failing for another 3 seconds.
      dropCache();
      return new Response(OFFLINE_HTML, {
        status: 502,
        headers: { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" },
      });
    }
  },
};
