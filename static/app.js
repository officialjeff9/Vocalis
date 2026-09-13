"use strict";

console.log("[app] Vocalis UI loaded — app.js build 20260913.1");

const SCENARIOS = {
  job_interview: {
    icon: "\u{1F468}\u200D\u{1F4BB}",
    title: "Job Interview",
    blurb: "Ace the behavioral round.",
    primary: true,
  },
  startup_pitch: {
    icon: "\u{1F680}",
    title: "Startup Pitch",
    blurb: "Win the investor's yes.",
    primary: false,
  },
  town_hall: {
    icon: "\u{1F4E2}",
    title: "Leadership Q&A",
    blurb: "Keep the whole room with you.",
    primary: false,
  },
};

const $ = (id) => document.getElementById(id);

/* Update the record status line. Guarded so a missing element can never
   crash the WebSocket message handler chain. */
function setStatus(text) {
  const el = $("rec-status");
  if (!el) {
    console.warn("[ui] #rec-status not found in DOM — status not updated:", text);
    return;
  }
  el.textContent = text;
}

function fmtClock(ms) {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/* Show a live elapsed-time counter in the status line while the slow local
   AI is drafting a question or analyzing an attempt, so the page never
   looks frozen during the multi-minute LLM round trips. */
function startStatusClock(line2) {
  const t0 = Date.now();
  return setInterval(() => {
    setStatus(`${line2}  (elapsed ${fmtClock(Date.now() - t0)})`);
  }, 1000);
}

function stopStatusClock(id) {
  if (id) clearInterval(id);
  return null;
}

/* Keep the record button ALWAYS visible. It is disabled (never hidden)
   while we have no question, so the page never looks broken while the
   coach is connecting or drafting the next question. */
function setRecordEnabled(enabled) {
  const btn = $("record-btn");
  if (!btn) return;
  btn.disabled = !enabled;
  btn.classList.toggle("disabled", !enabled);
  btn.setAttribute("aria-disabled", String(!enabled));
}

function showDrafting(text) {
  const el = $("drafting");
  const label = $("drafting-text");
  if (!el || !label) return;
  label.textContent = text;
  el.classList.remove("hidden");
}

function hideDrafting() {
  const el = $("drafting");
  if (el) el.classList.add("hidden");
}

/* Single source of truth for the loading state: pick the right drafting
   message (or hide it) from connection + session progress. Never touches
   the record button's visibility. */
function refreshLoadingUI() {
  const sessionActive = $("session-view").classList.contains("active");
  if (!sessionActive) {
    hideDrafting();
    return;
  }
  if (app.question) {
    hideDrafting();
    return;
  }
  const connected = socket && socket.readyState === WebSocket.OPEN;
  if (!connected) showDrafting("Connecting to coach…");
  else if (app.title) showDrafting("The AI is drafting your question…");
  else showDrafting("Waiting for your question…");
}

let socket = null;
let reconnectTimerId = null;
let mic = { ctx: null, resampler: null, feeding: false, stream: null, tracks: [], frames: 0 };
let timerId = null;
let startTs = 0;
let wsUpdates = 0;
let sawMetrics = false;
let draftTimerId = null;
let analyzeTimerId = null;

const app = {
  scenario: null,
  title: null,
  question: null,
  attempt: 0,
  phase: null,
  recording: false,
  analyzing: false,
  hasAttempt1: false,
  beforeSummary: null,
};

/* ---------------- websocket ---------------- */

function connect() {
  if (reconnectTimerId) {
    clearTimeout(reconnectTimerId);
    reconnectTimerId = null;
  }
  if (
    socket &&
    (socket.readyState === WebSocket.OPEN ||
      socket.readyState === WebSocket.CONNECTING)
  ) {
    console.log("[ws] already connected — reusing existing socket");
    return;
  }
  if (socket) {
    const stale = socket;
    stale.onopen = stale.onmessage = stale.onerror = stale.onclose = null;
    try {
      stale.close();
    } catch (err) {
      console.warn("[ws] failed to discard stale socket:", err);
    }
    socket = null;
  }
  const suffix = location.protocol === "https:" ? "wss" : "ws";
  const url = `${suffix}://${location.host}/ws/stream`;
  console.log("[ws] connecting to " + url);
  const ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    console.log("[ws] connected to /ws/stream");
    $("conn-banner").classList.add("hidden");
    refreshLoadingUI();
  };
  ws.onerror = (err) => {
    console.error("[ws] connection error:", err);
  };
  ws.onclose = (e) => {
    if (ws !== socket) {
      console.warn("[ws] stale socket closed — ignoring");
      return;
    }
    console.warn(
      `[ws] closed (code=${e.code}, reason="${e.reason}") — reconnecting in 2s`
    );
    socket = null;
    $("conn-banner").classList.remove("hidden");
    setRecordEnabled(false);
    showDrafting("Connection lost — reconnecting…");
    reconnectTimerId = setTimeout(connect, 2000);
  };
  ws.onmessage = (e) => {
    let msg;
    try {
      msg = e.data instanceof ArrayBuffer
        ? { type: "binary", data: e.data }
        : JSON.parse(e.data);
    } catch {
      return;
    }
    onMessage(msg);
  };
  socket = ws;
}

function sendJson(obj) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(obj));
    console.log("[ws] sent:", obj.type);
  } else {
    console.warn("[ws] socket not open — dropped control:", obj.type);
  }
}

function sendAudio(int16) {
  if (!(socket && socket.readyState === WebSocket.OPEN)) {
    console.warn("[ws] audio dropped — socket not open");
    return;
  }
  const bytes = new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength);
  socket.send(bytes.buffer);
}

function onMessage(msg) {
  if (msg.type === "update") {
    wsUpdates += 1;
    if (wsUpdates % 20 === 1) console.log(`[ws] received live update frame #${wsUpdates}`);
  } else {
    console.log("[ws] received:", msg.type);
  }

  const sessionActive = () => $("session-view").classList.contains("active");
  if ((msg.type === "question" || msg.type === "attempt_started" || msg.type === "attempt_result") && !sessionActive()) {
    console.log("[ws] ignoring", msg.type, "— session view not active");
    return;
  }

  switch (msg.type) {
    case "session_started":
      app.title = msg.title;
      $("scenario-title").textContent = msg.title;
      draftTimerId = startStatusClock("The AI is drafting your question…");
      setRecordEnabled(false);
      refreshLoadingUI();
      break;
    case "question":
      app.question = msg.question;
      draftTimerId = stopStatusClock(draftTimerId);
      $("question-text").textContent = msg.question;
      $("question-card").classList.remove("hidden");
      setRecordEnabled(true);
      refreshLoadingUI();
      $("record-btn").classList.remove("recording");
      $("record-label").textContent = "Start record";
      setStatus("Question loaded — press Start record and answer aloud.");
      $("elapsed").classList.add("hidden");
      $("analyzing").classList.add("hidden");
      break;
    case "attempt_started":
      app.attempt = msg.attempt;
      app.phase = msg.phase;
      app.recording = true;
      $("attempt-badge").textContent =
        app.phase === "after" ? `RETAKE — ATTEMPT ${app.attempt}` : `ATTEMPT ${app.attempt}`;
      $("record-btn").classList.add("recording");
      $("record-label").textContent = "Stop";
      setStatus("Listening/Analyzing… answer until you're done.");
      setRecordEnabled(true);
      refreshLoadingUI();
      $("analyzing").classList.add("hidden");
      $("result-block").classList.add("hidden");
      $("result-block").innerHTML = "";
      startTimer();
      startFeeding();
      break;
    case "update":
      renderMetrics(msg.metrics);
      break;
    case "coach_nudge":
      if (msg.coaching && msg.coaching.feedback) {
        const tip = $("tip-text");
        tip.textContent = msg.coaching.feedback;
        tip.classList.add("hot");
      }
      break;
    case "attempt_result":
      app.analyzing = false;
      analyzeTimerId = stopStatusClock(analyzeTimerId);
      $("analyzing").classList.add("hidden");
      $("record-btn").classList.remove("recording");
      setRecordEnabled(true);
      refreshLoadingUI();
      if (msg.short) {
        toast(msg.message);
        $("record-label").textContent = "Start record";
        setStatus("Try again — press Start record.");
      } else {
        renderResult(msg);
      }
      break;
    case "error":
      toast(msg.message);
      break;
    default:
      break;
  }
}

/* ---------------- home / scenarios ---------------- */

function showView(name) {
  $("home-view").classList.toggle("active", name === "home");
  $("session-view").classList.toggle("active", name === "session");
}

function renderScenarios() {
  const grid = $("scenario-cards");
  grid.innerHTML = "";
  Object.entries(SCENARIOS).forEach(([id, s]) => {
    const card = document.createElement("div");
    card.className = "scenario-card" + (s.primary ? " primary" : "");
    card.innerHTML = `
      <div class="sc-icon">${s.icon}</div>
      <h3>${s.title}</h3>
      <p>${s.blurb}</p>
      <span class="go">Start practicing &rarr;</span>
    `;
    card.addEventListener("click", () => startSession(id));
    grid.appendChild(card);
  });
}

async function startSession(id) {
  app.scenario = id;
  app.title = null;
  app.question = null;
  app.attempt = 0;
  app.phase = null;
  app.hasAttempt1 = false;
  app.beforeSummary = null;

  resetTiles();
  $("result-block").classList.add("hidden");
  $("result-block").innerHTML = "";
  $("analyzing").classList.add("hidden");
  $("attempt-badge").textContent = "";
  $("question-card").classList.add("hidden");

  draftTimerId = stopStatusClock(draftTimerId);
  analyzeTimerId = stopStatusClock(analyzeTimerId);

  stopFeeding();
  stopTimer();

  showView("session");
  setRecordEnabled(false);
  refreshLoadingUI();
  try {
    await ensureMic();
  } catch (err) {
    toast("Microphone access is needed. Allow mic permission and try again: " + err.message);
  }
  sendJson({ type: "start_session", scenario: id });
}

/* ---------------- mic ---------------- */

async function ensureMic() {
  if (mic.ctx) {
    console.log("[mic] pipeline already initialized — reusing it");
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    console.error(
      "[mic] navigator.mediaDevices.getUserMedia unavailable — " +
        "page must be served over https or localhost"
    );
    throw new Error("getUserMedia is not available (need https or localhost)");
  }

  console.log("[mic] requesting microphone access (getUserMedia)…");
  mic.stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      sampleRate: 16000,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });
  mic.tracks = mic.stream.getTracks();
  mic.frames = 0;
  console.log(
    "[mic] microphone access granted — track:",
    (mic.stream.getAudioTracks()[0] || {}).label || "unknown"
  );

  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  mic.ctx = new AudioCtx();
  console.log(`[mic] AudioContext ready — context sampleRate = ${mic.ctx.sampleRate} Hz`);
  const source = mic.ctx.createMediaStreamSource(mic.stream);
  const processor = mic.ctx.createScriptProcessor(4096, 1, 1);
  const mute = mic.ctx.createGain();
  mute.gain.value = 0;
  source.connect(processor);
  processor.connect(mute);
  mute.connect(mic.ctx.destination);
  mic.resampler = new Resampler(16000);
  processor.onaudioprocess = (e) => {
    if (!mic.feeding) return;
    const input = e.inputBuffer.getChannelData(0);
    const samples = mic.resampler.process(input, mic.ctx.sampleRate);
    const out = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const v = Math.max(-1, Math.min(1, samples[i]));
      out[i] = (v < 0 ? v * 0x8000 : v * 0x7fff) | 0;
    }
    if (out.length) {
      sendAudio(out);
      mic.frames += 1;
      if (mic.frames === 1) {
        console.log("[mic] streaming int16 PCM to /ws/stream — first chunk sent");
      } else if (mic.frames % 100 === 0) {
        console.log(`[mic] streamed ${mic.frames} audio chunks`);
      }
    }
  };
  await mic.ctx.resume();
  console.log("[mic] audio pipeline ready — feeding:", mic.feeding);
}

function startFeeding() {
  setTimeout(() => {
    if (app.recording) {
      mic.feeding = true;
      console.log("[mic] feeding ON — streaming int16 PCM to /ws/stream");
    }
  }, 80);
}

function stopFeeding() {
  if (mic.feeding) console.log("[mic] feeding OFF");
  mic.feeding = false;
}

function releaseMic() {
  stopFeeding();
  mic.tracks.forEach((t) => t.stop());
  mic.tracks = [];
  mic.stream = null;
  if (mic.ctx) {
    mic.ctx.close().catch(() => {});
    mic.ctx = null;
    mic.resampler = null;
  }
  mic.frames = 0;
}

/* ---------------- recording controls ---------------- */

async function toggleRecord() {
  if (app.analyzing) return;
  if (!app.question) return;
  if (!app.recording) {
    try {
      await ensureMic();
    } catch (err) {
      console.error("[mic] could not start microphone:", err);
      toast("Microphone access is needed. Allow mic permission and try again: " + err.message);
      return;
    }
    sendJson({ type: "start_attempt" });
  } else {
    app.recording = false;
    app.analyzing = true;
    stopFeeding();
    stopTimer();
    $("record-btn").classList.remove("recording");
    $("record-label").textContent = "…";
    analyzeTimerId = startStatusClock("Analyzing your attempt… the AI is reviewing your delivery.");
    $("analyzing").classList.remove("hidden");
    sendJson({ type: "end_attempt" });
  }
}

function startTimer() {
  startTs = Date.now();
  $("elapsed").classList.remove("hidden");
  clearInterval(timerId);
  timerId = setInterval(() => {
    const s = Math.floor((Date.now() - startTs) / 1000);
    $("elapsed").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }, 250);
}

function stopTimer() {
  clearInterval(timerId);
  timerId = null;
  const s = $("elapsed");
  s.classList.remove("hidden");
  s.textContent = "";
}

/* ---------------- live metrics ---------------- */

function resetTiles() {
  ["pace", "fillers", "energy", "pauses", "presence"].forEach((k) => {
    $(`m-${k}`).textContent = "—";
    $(`b-${k}`).style.width = "0%";
  });
  ["d-pace", "d-energy", "d-pauses", "d-presence"].forEach((id) => {
    const el = $(id);
    if (el) {
      el.textContent = "";
      el.classList.remove("up");
    }
  });
  const pauseSub = $("m-pauses-sub");
  if (pauseSub) pauseSub.textContent = "";
  const tip = $("tip-text");
  tip.textContent = "Listening — I'll nudge you live.";
  tip.classList.remove("hot");
}

/* Compare one live metric frame against the previous attempt's final summary.
   Returns mapped per-metric chip text; absent keys mean "not improved". */
function liveDeltas(m) {
  const b = app.beforeSummary;
  const out = {};
  if (m.pace_wpm > 0 && Number.isFinite(b.avg_pace_wpm)) {
    const live = {
      avg_pace_wpm: m.pace_wpm,
      baseline_pace_wpm: m.baseline_pace_wpm || b.baseline_pace_wpm,
    };
    if (paceClosingIn(b, live)) out.pace = "▲";
  }
  if (Number.isFinite(b.avg_energy) && m.energy > b.avg_energy) {
    out.energy = `▲+${Math.round(m.energy - b.avg_energy)}`;
  }
  if (
    m.pause_ratio > 0 &&
    Number.isFinite(b.pause_ratio) &&
    m.pause_ratio < b.pause_ratio
  ) {
    out.pauses = `▼${Math.round((b.pause_ratio - m.pause_ratio) * 100)}%`;
  }
  if (Number.isFinite(b.avg_presence) && m.presence > b.avg_presence) {
    out.presence = `▲+${Math.round(m.presence - b.avg_presence)}`;
  }
  return out;
}

/* Show the growth chips on the live tiles only while re-recording the same
   question on Attempt 2+, i.e. when a real previous summary exists. */
function applyLiveDeltas(m) {
  const deltas =
    app.attempt >= 2 && app.recording && app.beforeSummary
      ? liveDeltas(m)
      : null;
  const map = [
    ["d-pace", "pace"],
    ["d-energy", "energy"],
    ["d-pauses", "pauses"],
    ["d-presence", "presence"],
  ];
  for (const [id, key] of map) {
    const el = $(id);
    if (!el) continue;
    const text = deltas ? (deltas[key] || "") : "";
    el.textContent = text;
    el.classList.toggle("up", Boolean(text));
  }
}

function renderMetrics(m) {
  if (!m) return;
  applyLiveDeltas(m);

  if (!sawMetrics) {
    sawMetrics = true;
    console.log(
      "[metrics] first live frame — pace:", m.pace_wpm,
      "energy:", m.energy, "fillers/min:", m.fillers_per_minute,
      "pause_ratio:", m.pause_ratio, "presence:", m.presence
    );
  }

  if (m.pace_wpm > 0) {
    $("m-pace").textContent = m.pace_wpm;
    $("b-pace").style.width = Math.min(100, (m.pace_wpm / 220) * 100) + "%";
  }
  $("m-fillers").textContent = m.fillers_per_minute || 0;
  $("b-fillers").style.width = Math.min(100, (m.fillers_per_minute / 6) * 100) + "%";

  $("m-energy").textContent = m.energy ?? "—";
  $("b-energy").style.width = (m.energy ?? 0) + "%";

  if (m.pause_ratio > 0) {
    $("m-pauses").textContent = Math.round(m.pause_ratio * 100) + "%";
    $("b-pauses").style.width = Math.min(100, (m.pause_ratio / 0.5) * 100) + "%";
    $("m-pauses").title = m.pause_quality_label || "";
    const pauseSub = $("m-pauses-sub");
    if (pauseSub) pauseSub.textContent = m.pause_quality_label || "";
  }

  if (m.presence > 0) {
    $("m-presence").textContent = m.presence;
    $("b-presence").style.width = m.presence + "%";
  }
}

/* ---------------- attempt result ---------------- */

function pctDelta(before, after, key) {
  const b = before ? before[key] : null;
  const a = after ? after[key] : null;
  if (
    Number.isFinite(b) && Number.isFinite(a) && b !== 0 && Math.abs(b) > 1e-6
  ) {
    return ((a - b) / Math.abs(b)) * 100;
  }
  return null;
}

/* Pace is "improving" only when the new pace sits closer to the speaker's
   baseline target than the previous attempt (rushing slower OR sluggish
   faster). Mirrors the backend's baseline-aware logic — never assumed. */
function paceClosingIn(before, after) {
  const b = before ? before.avg_pace_wpm : null;
  const a = after ? after.avg_pace_wpm : null;
  const base =
    (after && after.baseline_pace_wpm) ||
    (before && before.baseline_pace_wpm) ||
    150;
  if (!Number.isFinite(b) || !Number.isFinite(a)) return false;
  return Math.abs(a - base) < Math.abs(b - base);
}

function fmtDelta(key, before, after) {
  const pct = pctDelta(before, after, key);
  if (pct === null) return "";
  let improving;
  if (key === "pause_ratio" || key === "fillers_per_minute") improving = pct < 0;
  else if (key === "avg_pace_wpm") improving = paceClosingIn(before, after);
  else improving = pct > 0;
  const cls = improving ? "up" : pct === 0 ? "flat" : "down";
  const arrow = improving ? "▲" : pct === 0 ? "•" : "▼";
  return `<span class="delta ${cls}">${arrow} ${Math.abs(pct).toFixed(0)}%</span>`;
}

function summaryRows(summary, extraDelta) {
  const pauseLabel = summary.pause_quality_label
    ? ` · ${summary.pause_quality_label}`
    : "";
  const rows = [
    { k: "Pace", v: `${summary.avg_pace_wpm} wpm`, delta: extraDelta ? extraDelta("avg_pace_wpm") : "" },
    { k: "Energy", v: `${summary.avg_energy}/100`, delta: extraDelta ? extraDelta("avg_energy") : "" },
    { k: "Fillers", v: `${summary.fillers_per_minute}/min`, delta: extraDelta ? extraDelta("fillers_per_minute") : "" },
    { k: "Time paused", v: `${Math.round(summary.pause_ratio * 100)}%${pauseLabel}`, delta: extraDelta ? extraDelta("pause_ratio") : "" },
    { k: "Delivery Presence", v: `${summary.avg_presence}/100`, delta: extraDelta ? extraDelta("avg_presence") : "" },
  ];
  return rows
    .map((r) => `<div class="sum-row"><span class="k">${r.k}</span><span class="v">${r.v}${r.delta}</span></div>`)
    .join("");
}

function coachCard(coaching, prefix) {
  const pill = coaching.focus_label || "FOCUS";
  const head = prefix ? `${prefix} · ${pill}` : pill;
  const cue = coaching.cue || "Keep practicing.";
  const exp = coaching.explanation ? `<p class="coach-exp">${coaching.explanation}</p>` : "";
  let banner = "";
  if (coaching.improvement && coaching.improvement.recognized) {
    const note = coaching.improvement.note || "Your second attempt shows real progress.";
    banner = `<div class="improve-banner">${
      note.startsWith("Your second attempt") ? note : "Progress detected — " + note
    }</div>`;
  }
  return `
    <div class="coach-card">
      <span class="focus-pill">${head}</span>
      <p class="coach-cue">${cue}</p>
      ${exp}
      ${banner}
    </div>
  `;
}

function absDelta(before, after, key) {
  const b = before ? before[key] : null;
  const a = after ? after[key] : null;
  if (Number.isFinite(b) && Number.isFinite(a)) return a - b;
  return null;
}

/* Real deltas between the two measured attempts -> green improvement chips.
   Only gates WHICH deltas qualify as meaningful improvement; every number
   rendered is a true before/after measurement, never fabricated. */
function improvementChips(before, after) {
  if (!before || !after) return "";
  const chips = [];

  const energy = absDelta(before, after, "avg_energy");
  if (energy !== null && energy >= 3) chips.push({ k: "Energy", v: `▲ +${Math.round(energy)} pts` });
  const presence = absDelta(before, after, "avg_presence");
  if (presence !== null && presence >= 3) chips.push({ k: "Delivery presence", v: `▲ +${Math.round(presence)} pts` });
  const fillers = absDelta(before, after, "fillers_per_minute");
  if (fillers !== null && fillers <= -0.5) chips.push({ k: "Fillers", v: `▼ ${fillers.toFixed(1)}/min` });
  const pausePct = pctDelta(before, after, "pause_ratio");
  if (pausePct !== null && pausePct <= -5) chips.push({ k: "Pausing", v: `▼ ${Math.abs(pausePct).toFixed(0)}%` });
  const pace = absDelta(before, after, "avg_pace_wpm");
  if (pace !== null && paceClosingIn(before, after) && Math.abs(pace) >= 3) {
    chips.push({ k: "Pace", v: `${Math.abs(pace).toFixed(0)} wpm toward target` });
  }

  if (!chips.length) return "";
  return `
    <div class="improve-block">
      <div class="improve-title">Measurable improvement</div>
      <div class="improve-chips">
        ${chips.map((c) => `<span class="improve-chip"><span class="ic-k">${c.k}</span><span class="ic-v">${c.v}</span></span>`).join("")}
      </div>
    </div>
  `;
}

function renderResult(msg) {
  const block = $("result-block");
  block.classList.remove("hidden");
  setStatus("Nice work. Review your feedback below.");
  $("record-label").textContent = "Start record";

  const after = msg.after_summary;
  const before = msg.before_summary;
  app.afterSummary = after;
  app.beforeSummary = after;

  if (msg.after_phase && before) {
    app.hasAttempt1 = true;
    const deltaFor = (key) => fmtDelta(key, before, after);
    block.innerHTML = `
      <div class="result-panel">
        <div class="result-stage">ATTEMPT 2</div>
        <h2 class="result-headline">Your second attempt is in — here's what changed, and what's next.</h2>
        ${coachCard(msg.coaching || {}, "NEXT FOCUS")}
        ${improvementChips(before, after)}
        <div class="compare-cols">
          <div>
            <div class="col-head">Try 1</div>
            <div class="summary-list">${summaryRows(before)}</div>
          </div>
          <div class="vs-badge">VS</div>
          <div>
            <div class="col-head good">Try 2</div>
            <div class="summary-list">${summaryRows(after, deltaFor)}</div>
          </div>
        </div>
        <div class="result-actions">
          <button class="primary-btn" id="btn-next-q">Next question</button>
          <button class="ghost-btn" id="btn-home">Change scenario</button>
        </div>
      </div>
    `;
    $("btn-next-q").addEventListener("click", requestNextQuestion);
    $("btn-home").addEventListener("click", goHome);
  } else {
    app.hasAttempt1 = true;
    block.innerHTML = `
      <div class="result-panel">
        <div class="result-stage">ATTEMPT 1</div>
        <h2 class="result-headline">Your first attempt is in. Here's what your coach noticed.</h2>
        ${coachCard(msg.coaching || {}, "FOCUS")}
        <div class="summary-list">${summaryRows(after)}</div>
        <div class="result-actions">
          <button class="primary-btn" id="btn-retry">Try again — improve</button>
          <button class="ghost-btn" id="btn-home">Change scenario</button>
        </div>
      </div>
    `;
    $("btn-retry").addEventListener("click", () => {
      $("result-block").classList.add("hidden");
      $("result-block").innerHTML = "";
      $("attempt-badge").textContent = "RETAKE";
      setStatus("This is your retake — implement the focus and try again.");
      sendJson({ type: "start_attempt" });
    });
    $("btn-home").addEventListener("click", goHome);
  }
  block.scrollIntoView({ behavior: "smooth", block: "start" });
}

function requestNextQuestion() {
  app.question = null;
  app.attempt = 0;
  app.hasAttempt1 = false;
  app.beforeSummary = null;
  $("result-block").classList.add("hidden");
  $("result-block").innerHTML = "";
  $("attempt-badge").textContent = "";
  $("question-card").classList.add("hidden");
  resetTiles();
  draftTimerId = startStatusClock("The AI is drafting a new question…");
  setRecordEnabled(false);
  refreshLoadingUI();
  sendJson({ type: "next_question" });
}

function goHome() {
  releaseMic();
  stopTimer();
  draftTimerId = stopStatusClock(draftTimerId);
  analyzeTimerId = stopStatusClock(analyzeTimerId);
  resetTiles();
  showView("home");
  setRecordEnabled(false);
  refreshLoadingUI();
  $("record-btn").classList.remove("recording");
  $("record-label").textContent = "Start record";
}

/* ---------------- helpers ---------------- */

function toast(text) {
  const el = $("toast");
  el.textContent = text;
  el.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add("hidden"), 4200);
}

/* Resample any input rate down to a target rate (linear interpolation). */
class Resampler {
  constructor(outRate) {
    this.outRate = outRate;
    this.buffer = [];
    this.phase = 0;
  }
  process(input, inRate) {
    this.buffer.push(...input);
    const step = inRate / this.outRate;
    const n = this.buffer.length;
    const out = [];
    let idx = Math.floor(this.phase);
    while (idx < n - 1) {
      idx = Math.floor(this.phase);
      const frac = this.phase - idx;
      const a = this.buffer[idx];
      const b = this.buffer[idx + 1];
      out.push(a + (b - a) * frac);
      this.phase += step;
    }
    const consumed = Math.min(n, Math.max(0, Math.floor(this.phase)));
    this.buffer = this.buffer.slice(consumed);
    this.phase -= consumed;
    return out;
  }
}

/* ---------------- init ---------------- */

$("home-btn").addEventListener("click", goHome);
$("record-btn").addEventListener("click", toggleRecord);

renderScenarios();
connect();