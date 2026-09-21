/* Face Swap front end.
 *
 * Two constraints from Cloudflare shape all of this:
 *   1. Large single-POST uploads fail at the edge, so the video is sliced and
 *      sent as a sequence of raw-body chunk requests (resumable on failure).
 *   2. Quick Tunnels do not support Server-Sent Events, and the proxy read
 *      timeout is 125s -- so progress is short JSON polls, never a held-open
 *      connection.
 */
const CHUNK = 8 * 1024 * 1024;   // comfortably under any edge body limit
const POLL_MS = 1500;
const LS_KEY = "fsw_job";

const $ = (id) => document.getElementById(id);
const el = {
  dropV: $("dropV"), dropF: $("dropF"), fileV: $("fileV"), fileF: $("fileF"),
  pickV: $("pickV"), pickF: $("pickF"), go: $("go"),
  prog: $("prog"), stage: $("stage"), pill: $("pill"), fill: $("fill"),
  pct: $("pct"), eta: $("eta"), log: $("log"), cancel: $("cancel"),
  result: $("result"), vid: $("vid"), dl: $("dl"), again: $("again"),
  failed: $("failed"), errmsg: $("errmsg"), retry: $("retry"),
  pick: $("pick"), quality: $("quality"), whichface: $("whichface"),
  libGrid: $("libGrid"), libEmpty: $("libEmpty"), libCount: $("libCount"),
  modal: $("modal"), modalVid: $("modalVid"), modalClose: $("modalClose"),
  statline: $("statline"), opts: $("opts"),
};

let videoFile = null, faceFiles = [], jobId = null, polling = null, cancelled = false;

/* ---------- file pickers ---------- */
function wireDrop(drop, input, onPick) {
  drop.onclick = () => input.click();
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => {
    e.preventDefault(); drop.classList.remove("over");
    if (e.dataTransfer.files[0]) { input.files = e.dataTransfer.files; onPick(e.dataTransfer.files[0], e.dataTransfer.files); }
  };
  input.onchange = () => input.files[0] && onPick(input.files[0], input.files);
}
const fmt = (b) => b > 1e9 ? (b/1e9).toFixed(1)+" GB" : b > 1e6 ? (b/1e6).toFixed(0)+" MB" : (b/1e3).toFixed(0)+" KB";

wireDrop(el.dropV, el.fileV, (f) => {
  videoFile = f;
  el.pickV.classList.remove("hide");
  el.pickV.querySelector(".nm").textContent = `${f.name} (${fmt(f.size)})`;
  refresh();
});
wireDrop(el.dropF, el.fileF, (f, all) => {
  // Accept several photos of the same person; the renderer fuses them into
  // one identity, weighted by how usable each face is.
  faceFiles = (all && all.length ? Array.from(all) : [f]).slice(0, 5);
  el.pickF.classList.remove("hide");
  const names = faceFiles.map((x) => x.name).join(", ");
  el.pickF.querySelector(".nm").textContent =
    faceFiles.length > 1 ? `${faceFiles.length} photos - ${names}` : `${names} (${fmt(f.size)})`;
  const img = el.pickF.querySelector("img");
  img.src = URL.createObjectURL(faceFiles[0]);
  refresh();
});
function refresh() { el.go.disabled = !(videoFile && faceFiles.length); }

/* ---------- chunked upload ---------- */
async function postJSON(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`${url} -> ${r.status} ${await r.text()}`);
  return r.json();
}

async function uploadFile(file, onPct) {
  const { upload_id } = await postJSON("/api/upload/init", { filename: file.name, size: file.size });
  const total = Math.ceil(file.size / CHUNK);
  for (let i = 0; i < total; i++) {
    const blob = file.slice(i * CHUNK, Math.min((i + 1) * CHUNK, file.size));
    let ok = false, lastErr = null;
    // Retry each chunk; a flaky link should not restart the whole upload.
    for (let attempt = 0; attempt < 5 && !ok; attempt++) {
      if (cancelled) throw new Error("cancelled");
      try {
        const r = await fetch(`/api/upload/chunk/${upload_id}/${i}`, {
          method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: blob,
        });
        if (!r.ok) throw new Error(`chunk ${i} -> ${r.status}`);
        ok = true;
      } catch (e) {
        lastErr = e;
        await new Promise((s) => setTimeout(s, 400 * Math.pow(2, attempt)));
      }
    }
    if (!ok) throw lastErr || new Error(`chunk ${i} failed`);
    onPct(((i + 1) / total) * 100);
  }
  // The server returns an opaque media id, never a filesystem path.
  const { media_id } = await postJSON("/api/upload/complete", { upload_id, chunks: total });
  return media_id;
}

/* ---------- progress UI ---------- */
function showStage(name, pct, pill, pillCls) {
  el.stage.textContent = name;
  el.fill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  el.pct.textContent = `${pct.toFixed(0)}%`;
  if (pill) { el.pill.textContent = pill; el.pill.className = `pill ${pillCls || "run"}`; }
}
function toPanel(which) {
  for (const p of ["pick", "prog", "result", "failed"]) el[p].classList.add("hide");
  // Address the options card by id. It used to be querySelectorAll(".card")[1],
  // which silently binds to whatever card happens to sit second in the markup.
  el.opts.classList.toggle("hide", which !== "pick");
  el[which].classList.remove("hide");
  if (which === "pick") el.pick.classList.remove("hide");
}

/* ---------- run ---------- */
el.go.onclick = async () => {
  cancelled = false;
  toPanel("prog");
  showStage("Uploading video", 0, "uploading");
  try {
    const targetId = await uploadFile(videoFile, (p) => showStage("Uploading video", p * 0.9, "uploading"));
    showStage("Uploading photos", 92, "uploading");
    // 1..5 photos of the same person: more angles give a steadier identity.
    const sourceIds = [];
    for (const f of faceFiles) sourceIds.push(await uploadFile(f, () => {}));
    showStage("Starting render", 97, "starting");

    const res = await postJSON("/api/jobs", {
      source_ids: sourceIds,
      target_id: targetId,
      quality: el.quality.value,
      face_mode: el.whichface.value,
    });
    jobId = res.job_id;
    localStorage.setItem(LS_KEY, jobId);
    startPolling();
  } catch (e) {
    if (String(e.message) === "cancelled") { toPanel("pick"); return; }
    fail(String(e.message || e));
  }
};

function startPolling() {
  clearInterval(polling);
  polling = setInterval(poll, POLL_MS);
  poll();
}

async function poll() {
  if (!jobId) return;
  let j;
  try {
    const r = await fetch(`/api/jobs/${jobId}`);
    if (r.status === 404) { clearInterval(polling); localStorage.removeItem(LS_KEY); toPanel("pick"); return; }
    j = await r.json();
  } catch { return; }   // transient network blip: keep polling

  if (j.log_tail) { el.log.classList.remove("hide"); el.log.textContent = j.log_tail; }

  if (j.status === "running" || j.status === "queued") {
    const label = j.phase ? j.phase[0].toUpperCase() + j.phase.slice(1) : "Rendering";
    showStage(label, j.progress || 0, "rendering");
    showStats(j);
    if (j.started_at && j.progress > 3) {
      const elapsed = Date.now() / 1000 - j.started_at;
      const left = elapsed * (100 - j.progress) / j.progress;
      el.eta.textContent = left > 1 ? `~${Math.ceil(left / 60)} min left` : "";
    }
  } else if (j.status === "done") {
    clearInterval(polling);
    localStorage.removeItem(LS_KEY);
    el.vid.src = `/api/jobs/${jobId}/video`;
    el.dl.href = `/api/jobs/${jobId}/video?download=1`;
    el.dl.setAttribute("download", `faceswap-${jobId}.mp4`);
    toPanel("result");
    loadLibrary();   // the render that just finished belongs in the library now
  } else {
    clearInterval(polling);
    localStorage.removeItem(LS_KEY);
    fail(j.error || `job ${j.status}`);
  }
}

/* Live stat chips. The server already formats frame/fps/ETA into `message`;
 * these add the pipeline facts that do not change per frame. */
function showStats(j) {
  const bits = [];
  if (j.phase) bits.push(j.phase);
  const m = String(j.message || "");
  const frames = m.match(/frame (\d+) of (\d+)/);
  if (frames) bits.push(`${frames[1]} / ${frames[2]} frames`);
  const fps = m.match(/([\d.]+) fps/);
  if (fps) bits.push(`${fps[1]} fps`);
  const cand = m.match(/^(\S+ \+ \S+|\S+ bare)/);
  if (cand && /benchmark|restor/i.test(j.phase || "")) bits.push(cand[1]);
  if (!bits.length) { el.statline.classList.add("hide"); return; }
  el.statline.replaceChildren();
  for (const b of bits) {
    const s = document.createElement("span");
    s.textContent = b;
    el.statline.append(s);
  }
  el.statline.classList.remove("hide");
}

function fail(msg) {
  el.errmsg.textContent = msg;
  toPanel("failed");
}

el.cancel.onclick = async () => {
  cancelled = true;
  if (jobId) { try { await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" }); } catch {} }
  clearInterval(polling);
  localStorage.removeItem(LS_KEY);
  toPanel("pick");
};
el.again.onclick = el.retry.onclick = () => { jobId = null; toPanel("pick"); };

/* Resume a render that was still running when the page was closed. */
(function resume() {
  const saved = localStorage.getItem(LS_KEY);
  if (!saved) return;
  jobId = saved;
  toPanel("prog");
  showStage("Reconnecting", 0, "resuming");
  startPolling();
})();

/* ---------- library ----------
 * Every completed render is listed here so a finished video is never lost
 * behind a closed tab. Refreshed on load and whenever a job finishes.
 */
const QUALITY_LABEL = { fast: "Fast", quality: "Quality", auto: "Auto Max", balanced: "Balanced", best: "Best" };
const FACE_LABEL = { reference: "One face", many: "All faces" };

function relTime(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  const mins = Math.round((Date.now() - d.getTime()) / 60000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins} min ago`;
  if (mins < 1440) return `${Math.round(mins / 60)} h ago`;
  if (mins < 10080) return `${Math.round(mins / 1440)} d ago`;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
function clock(sec) {
  if (!sec && sec !== 0) return "";
  const m = Math.floor(sec / 60), s = Math.round(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

async function loadLibrary() {
  let items = [];
  try {
    const r = await fetch("/api/library");
    if (!r.ok) return;
    items = await r.json();
  } catch { return; }   // offline/transient: leave whatever is on screen

  el.libCount.textContent = items.length ? `${items.length} video${items.length > 1 ? "s" : ""}` : "";
  el.libEmpty.classList.toggle("hide", items.length > 0);
  el.libGrid.replaceChildren();

  for (const it of items) {
    const tile = document.createElement("div");
    tile.className = "tile";

    const tags = [
      QUALITY_LABEL[it.quality] || it.quality,
      FACE_LABEL[it.face_mode] || it.face_mode,
      fmt(it.size),
      it.engine ? it.engine.toUpperCase() : null,
      it.swapper || null,
      it.enhancer ? `${it.enhancer} ${Math.round((it.enhancer_blend || 0) * 100)}%` : "no restore",
      it.benchmark_winner ? `auto-picked from ${it.benchmark_candidates}` : null,
      (it.identity_switches === 0) ? "0 id switches" : null,
    ].filter(Boolean);

    // textContent everywhere below (no innerHTML with data) so a filename can
    // never inject markup.
    const wrap = document.createElement("div");
    wrap.className = "thumbwrap";
    wrap.title = "Play";
    const img = document.createElement("img");
    img.loading = "lazy";
    img.alt = "";
    img.src = `/api/library/${it.id}/thumb`;
    img.onerror = () => {
      const ph = document.createElement("div");
      ph.className = "noimg";
      ph.textContent = "No preview";
      img.replaceWith(ph);
    };
    wrap.append(img);

    const play = document.createElement("div");
    play.className = "play";
    play.innerHTML = "<span>&#9654;</span>";
    wrap.append(play);

    if (it.duration) {
      const d = document.createElement("div");
      d.className = "dur";
      d.textContent = clock(it.duration);
      wrap.append(d);
    }
    wrap.onclick = () => openModal(it.id);

    const meta = document.createElement("div");
    meta.className = "meta";
    const when = document.createElement("div");
    when.className = "when";
    when.textContent = relTime(it.created_at);
    meta.append(when);

    const tagRow = document.createElement("div");
    tagRow.className = "tags";
    for (const t of tags) {
      const s = document.createElement("span");
      s.className = "tag";
      s.textContent = t;
      tagRow.append(s);
    }
    meta.append(tagRow);

    const acts = document.createElement("div");
    acts.className = "acts";
    const dl = document.createElement("a");
    dl.href = `/api/library/${it.id}/video?download=1`;
    dl.setAttribute("download", `faceswap-${it.id}.mp4`);
    dl.textContent = "Download";
    const del = document.createElement("button");
    del.textContent = "Delete";
    del.onclick = () => removeItem(it.id, when.textContent);
    acts.append(dl, del);
    meta.append(acts);

    tile.append(wrap, meta);
    el.libGrid.append(tile);
  }
}

async function removeItem(id, label) {
  if (!confirm(`Delete this video (${label})? This cannot be undone.`)) return;
  try {
    await fetch(`/api/library/${id}`, { method: "DELETE" });
  } catch {}
  loadLibrary();
}

function openModal(id) {
  el.modalVid.src = `/api/library/${id}/video`;
  el.modal.classList.remove("hide");
  // Called from a tap/click handler, so this counts as a user gesture and
  // satisfies iOS/iPadOS autoplay rules. Rejects harmlessly if the browser
  // still declines -- the controls are right there.
  el.modalVid.play?.().catch(() => {});
}
function closeModal() {
  el.modalVid.pause();
  el.modalVid.removeAttribute("src");
  el.modalVid.load();
  el.modal.classList.add("hide");
}
el.modalClose.onclick = closeModal;
el.modal.onclick = (e) => { if (e.target === el.modal) closeModal(); };
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !el.modal.classList.contains("hide")) closeModal();
});

loadLibrary();
