/**
 * Local Mama — browser test console.
 *
 * Audio here uses the browser's own Web Speech API for both recognition and
 * synthesis, so the whole multilingual workflow is testable with zero STT/TTS
 * keys. The server sends a BCP-47 locale each turn and we retune both the
 * recogniser and the voice to it, which is what makes the language switch
 * audible. The LiveKit path (backend/app/agent.py) drives the same backend
 * workflow with real telephony-grade providers.
 *
 * Three Web Speech API realities this file works around:
 *
 *   1. Chrome's SpeechRecognition is a *cloud* service. Restarting it in a
 *      tight loop gets throttled and surfaces as `error: network`. So we keep
 *      one continuous stream open and restart lazily, with backoff.
 *   2. getVoices() returns voices in arbitrary order and most systems default
 *      to a low-quality compact voice. We rank instead of taking the first
 *      match, and let the user override.
 *   3. speechSynthesis does not reliably fire `onend` — notably when no voice
 *      is installed for the language. Anything that waits on it needs a
 *      watchdog, or the mic deadlocks behind a "still speaking" flag.
 *
 * Recognition state is therefore never derived from captured snapshots. There
 * is one intent flag (`listening`) and one reconciler (`applyRecognitionState`)
 * that decides what the engine should be doing right now.
 */

const $ = (id) => document.getElementById(id);

const els = {
  connect: $("btn-connect"),
  mic: $("btn-mic"),
  hangup: $("btn-hangup"),
  send: $("btn-send"),
  form: $("text-form"),
  input: $("text-input"),
  transcript: $("transcript"),
  badge: $("conn-badge"),
  states: $("state-list"),
  language: $("f-language"),
  name: $("f-name"),
  service: $("f-service"),
  location: $("f-location"),
  lead: $("lead-json"),
  runtime: $("runtime"),
  toast: $("whatsapp-toast"),
  toastBody: $("toast-body"),
  voiceSelect: $("voice-select"),
  voiceNote: $("voice-note"),
  rate: $("rate"),
  rateOut: $("rate-out"),
  micStatus: $("mic-status"),
  micLevel: $("mic-level"),
  micDiag: $("mic-diag"),
  testMic: $("btn-test-mic"),
};

// Guarded lookups: this file is also run headless by tests/frontend/.
const nav = typeof navigator !== "undefined" ? navigator : null;
const AudioCtx =
  typeof window !== "undefined"
    ? window.AudioContext || window.webkitAudioContext
    : null;

// --- recognition state ---------------------------------------------------
let recognition = null;
let listening = false;    // user intent: should the mic be on at all?
let recognising = false;  // engine reports it is running
let starting = false;     // start() called, waiting for onstart
let syncTimer = null;
let networkFailures = 0;

// --- synthesis state -----------------------------------------------------
let speaking = false;
let speechGen = 0;        // invalidates callbacks from cancelled utterances
let speechWatchdog = null;

// --- microphone diagnostics ----------------------------------------------
// "It says listening but nothing happens" is unfalsifiable without these.
// The Web Speech API exposes a ladder — audiostart → soundstart → speechstart
// → result — and where it stops tells you which layer is broken.
let mediaStream = null;
let audioCtx = null;
let analyser = null;
let levelTimer = null;
let peakLevel = 0;
let sawSound = false;
let silenceHintTimer = null;
let noResultHintTimer = null;
const diagLog = [];

// Chrome's `continuous` recognition mode is the more convenient one but is
// also the one that most often yields zero results on some setups. If we hear
// speech and get nothing back, we fall back to single-utterance mode, which is
// the most widely supported configuration.
let continuousMode = true;
let triedSingleUtterance = false;
let activityTimer = null;

// --- session state -------------------------------------------------------
let socket = null;
let currentLocale = "en-IN";
let callOver = false;
let states = [];
let voiceOverride = "";

const MAX_NETWORK_FAILURES = 3;

// ---------------------------------------------------------------- UI ----

function addMessage(role, text) {
  const row = document.createElement("div");
  row.className = `msg ${role}`;
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = role === "agent" ? "Mami" : role.startsWith("user") ? "You" : "";
  const body = document.createElement("p");
  body.textContent = text;
  if (who.textContent) row.appendChild(who);
  row.appendChild(body);
  els.transcript.appendChild(row);
  els.transcript.scrollTop = els.transcript.scrollHeight;
  return body;
}

function setBadge(state, label) {
  els.badge.className = `badge ${state}`;
  els.badge.textContent = label;
}

/** Surfaces the real engine state, so a dead mic is visible rather than silent. */
function updateMicStatus() {
  els.mic.textContent = listening ? "🔴 Stop mic" : "🎤 Start mic";
  els.mic.classList.toggle("recording", listening);
  if (!els.micStatus) return;

  let text;
  let cls;
  if (!listening) {
    text = "Mic off";
    cls = "";
  } else if (speaking) {
    text = "Paused — Mami is speaking";
    cls = "paused";
  } else if (recognising) {
    text = "Listening…";
    cls = "live";
  } else if (starting) {
    text = "Starting mic…";
    cls = "paused";
  } else {
    text = "Reconnecting mic…";
    cls = "paused";
  }
  els.micStatus.textContent = text;
  els.micStatus.className = `mic-status ${cls}`;
}

// --------------------------------------------------- Mic diagnostics ----

function diag(message) {
  const stamp = new Date().toLocaleTimeString([], {
    hour12: false,
    minute: "2-digit",
    second: "2-digit",
  });
  diagLog.push(`${stamp}  ${message}`);
  while (diagLog.length > 14) diagLog.shift();
  if (els.micDiag) els.micDiag.textContent = diagLog.join("\n");
}

function setLevel(fraction) {
  if (!els.micLevel) return;
  els.micLevel.style.width = `${Math.min(100, Math.round(fraction * 100))}%`;
}

/**
 * Live input level from a getUserMedia stream.
 *
 * This is the decisive diagnostic: if the bar never moves while you speak, the
 * browser is receiving silence and no amount of application code will help —
 * it is an OS permission or input-device problem. If the bar moves but no
 * transcript appears, the capture is fine and the speech service is at fault.
 */
function startLevelMeter() {
  if (!AudioCtx || !mediaStream || analyser) return;
  try {
    audioCtx = audioCtx || new AudioCtx();
    if (audioCtx.state === "suspended") audioCtx.resume();
    const source = audioCtx.createMediaStreamSource(mediaStream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
    const buffer = new Uint8Array(analyser.fftSize);

    clearInterval(levelTimer);
    levelTimer = setInterval(() => {
      analyser.getByteTimeDomainData(buffer);
      let sum = 0;
      for (let i = 0; i < buffer.length; i += 1) {
        const v = (buffer[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / buffer.length);
      peakLevel = Math.max(peakLevel * 0.92, rms);
      setLevel(Math.min(peakLevel * 6, 1));
      if (rms > 0.02) sawSound = true;
    }, 100);
  } catch (err) {
    diag(`level meter unavailable: ${err.name || err}`);
  }
}

function stopLevelMeter() {
  clearInterval(levelTimer);
  levelTimer = null;
  analyser = null;
  peakLevel = 0;
  setLevel(0);
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
}

/**
 * Show recogniser-reported activity on the meter during a call.
 *
 * We deliberately do NOT hold a getUserMedia stream while SpeechRecognition is
 * running — two simultaneous consumers of the microphone can starve the
 * recogniser on some setups, which is exactly the "level bar moves but nothing
 * is transcribed" failure. During a call the meter is driven by the
 * recogniser's own soundstart/speechstart events instead; use "Test mic" for a
 * true level reading while the mic is off.
 */
function pulseActivity(strength) {
  setLevel(strength);
  clearTimeout(activityTimer);
  activityTimer = setTimeout(() => setLevel(0), 700);
}

/**
 * Explicitly acquire the microphone before starting recognition.
 *
 * SpeechRecognition acquires the mic itself, but it reports permission and
 * device failures inconsistently — sometimes as a generic error, sometimes as
 * silence. getUserMedia gives a precise, named error we can act on. The stream
 * is released immediately so it cannot compete with the recogniser.
 */
async function ensureMicAccess({ keepStream = false } = {}) {
  if (!nav || !nav.mediaDevices || !nav.mediaDevices.getUserMedia) return true;
  if (mediaStream) return true;
  try {
    mediaStream = await nav.mediaDevices.getUserMedia({ audio: true });
    const label = mediaStream.getAudioTracks()[0]?.label || "default device";
    diag(`mic acquired: ${label}`);
    if (keepStream) startLevelMeter();
    else stopLevelMeter(); // release before recognition starts
    return true;
  } catch (err) {
    const name = err && err.name;
    let hint;
    if (name === "NotAllowedError" || name === "SecurityError") {
      hint =
        "Microphone permission denied. Click the padlock in the address bar → " +
        "Microphone → Allow, then reload. On macOS also check System Settings → " +
        "Privacy & Security → Microphone and enable your browser.";
    } else if (name === "NotFoundError" || name === "OverconstrainedError") {
      hint = "No microphone was found. Check your input device, then try again.";
    } else if (name === "NotReadableError") {
      hint =
        "The microphone is in use by another application. Close it (Zoom, Meet, " +
        "Teams…) and try again.";
    } else {
      hint = `Could not open the microphone (${name || err}).`;
    }
    diag(`getUserMedia failed: ${name || err}`);
    addMessage("system", `${hint} The text box works regardless.`);
    return false;
  }
}

function clearMicHints() {
  clearTimeout(silenceHintTimer);
  clearTimeout(noResultHintTimer);
}

/** If audio is flowing but nothing is heard, the capture layer is the problem. */
function armSilenceHint() {
  clearTimeout(silenceHintTimer);
  silenceHintTimer = setTimeout(() => {
    if (!listening || sawSound) return;
    addMessage(
      "system",
      "The microphone is open but no sound is reaching it — the level bar has " +
        "not moved. Check that the right input device is selected and that your " +
        "browser has OS-level microphone permission (macOS: System Settings → " +
        "Privacy & Security → Microphone). Use the text box meanwhile."
    );
    diag("no sound detected after 8s");
  }, 8000);
}

/**
 * Sound arrived but the speech service returned nothing.
 *
 * First occurrence: retry in single-utterance mode, which is the most
 * compatible configuration and fixes this on many setups. Only if that also
 * fails do we tell the user to fall back to text.
 */
function armNoResultHint() {
  clearTimeout(noResultHintTimer);
  noResultHintTimer = setTimeout(() => {
    if (!listening) return;

    if (continuousMode && !triedSingleUtterance) {
      triedSingleUtterance = true;
      continuousMode = false;
      diag("no result in continuous mode — retrying single-utterance mode");
      addMessage(
        "system",
        "Speech was heard but produced no text. Switching the recogniser to " +
          "single-utterance mode, which is more widely supported. Say your " +
          "answer again — pause briefly when you finish."
      );
      rebuildRecognition();
      syncRecognition(300);
      return;
    }

    addMessage(
      "system",
      `Speech was detected but the recogniser returned no text for ` +
        `${currentLocale}. Chrome's speech service may not support this ` +
        `language, or it could not be reached. Try speaking English, or use ` +
        `the text box — it drives the identical workflow.`
    );
    diag(`speech detected but no result for ${currentLocale}`);
  }, 7000);
}

/** Tear down the recogniser so the next start picks up new options. */
function rebuildRecognition() {
  if (recognition) {
    try {
      recognition.onend = null; // suppress the auto-restart for this teardown
      if (recognition.abort) recognition.abort();
      else recognition.stop();
    } catch (_) {
      /* already down */
    }
  }
  recognition = null;
  recognising = false;
  starting = false;
}

/** Standalone level check. Safe only while recognition is NOT running. */
async function testMic() {
  if (listening) {
    addMessage("system", "Stop the mic first, then run the microphone test.");
    return;
  }
  diag("microphone test started");
  if (!(await ensureMicAccess({ keepStream: true }))) return;
  addMessage(
    "system",
    "Microphone test running for 6 seconds — speak now and watch the level bar."
  );
  setTimeout(() => {
    const heard = sawSound;
    stopLevelMeter();
    addMessage(
      "system",
      heard
        ? "Microphone test passed: audio is reaching the browser. If speech " +
            "still produces no text, the problem is Chrome's speech service, " +
            "not your mic."
        : "Microphone test heard nothing. Check your input device and that " +
            "your browser has OS-level microphone permission."
    );
    diag(`microphone test finished: ${heard ? "sound detected" : "silent"}`);
  }, 6000);
  sawSound = false;
}

function renderStates(active) {
  els.states.innerHTML = "";
  states.forEach((name) => {
    const li = document.createElement("li");
    li.textContent = name.replace(/_/g, " ").toLowerCase();
    if (name === active) li.className = "active";
    els.states.appendChild(li);
  });
}

function renderCaptured(captured) {
  els.language.textContent = captured.selected_language || "—";
  els.name.textContent = captured.user_name || "—";
  els.service.textContent = captured.requested_service || "—";
  els.location.textContent = captured.city_or_area || "—";
}

function showWhatsAppToast(lead) {
  els.toastBody.textContent =
    `Sending best ${lead.requested_service} matches in ${lead.city_or_area} ` +
    `to ${lead.user_name}'s WhatsApp.`;
  els.toast.classList.remove("hidden");
  setTimeout(() => els.toast.classList.add("hidden"), 8000);
}

// ------------------------------------------------------- Voice picking ----

const norm = (lang) => (lang || "").replace("_", "-").toLowerCase();

/**
 * Rank a voice for a locale. Higher is better; -1 means wrong language.
 *
 * These heuristics matter more than they look: on macOS the default pick is
 * usually a "compact" system voice, which is the robotic one. Chrome's
 * "Google …" voices and Edge's "… Natural" voices are network-backed and
 * dramatically better, but they do not sort first in getVoices().
 */
function scoreVoice(voice, locale) {
  const want = norm(locale);
  const wantLang = want.split("-")[0];
  const have = norm(voice.lang);
  if (!have) return -1;

  let score;
  if (have === want) score = 100;
  else if (have.split("-")[0] === wantLang) score = 50;
  else return -1;

  const name = (voice.name || "").toLowerCase();
  if (name.includes("natural") || name.includes("neural")) score += 45;
  if (name.includes("google")) score += 40;
  if (name.includes("premium") || name.includes("enhanced")) score += 30;
  if (voice.localService === false) score += 20;
  if (name.includes("compact")) score -= 40;
  if (name.includes("espeak")) score -= 50;
  return score;
}

function voicesFor(locale) {
  if (!("speechSynthesis" in window)) return [];
  return window.speechSynthesis
    .getVoices()
    .map((v) => ({ voice: v, score: scoreVoice(v, locale) }))
    .filter((entry) => entry.score >= 0)
    .sort((a, b) => b.score - a.score)
    .map((entry) => entry.voice);
}

function pickVoice(locale) {
  if (voiceOverride) {
    const chosen = window.speechSynthesis
      .getVoices()
      .find((v) => v.voiceURI === voiceOverride);
    if (chosen) return chosen;
  }
  return voicesFor(locale)[0] || null;
}

function renderVoicePicker() {
  if (!els.voiceSelect) return;
  const ranked = voicesFor(currentLocale);
  const previous = els.voiceSelect.value;

  els.voiceSelect.innerHTML = "";
  const auto = document.createElement("option");
  auto.value = "";
  auto.textContent = ranked.length
    ? `Auto — ${ranked[0].name}`
    : "Auto — no matching voice";
  els.voiceSelect.appendChild(auto);

  ranked.forEach((voice) => {
    const option = document.createElement("option");
    option.value = voice.voiceURI;
    option.textContent = `${voice.name} (${voice.lang})${
      voice.localService === false ? " · network" : ""
    }`;
    els.voiceSelect.appendChild(option);
  });

  els.voiceSelect.value = ranked.some((v) => v.voiceURI === previous) ? previous : "";
  voiceOverride = els.voiceSelect.value;

  if (!ranked.length) {
    els.voiceNote.textContent =
      `No voice installed for ${currentLocale}. Speech falls back to the system ` +
      `default. On macOS add one in System Settings → Accessibility → Spoken ` +
      `Content → System Voice → Manage Voices.`;
    els.voiceNote.classList.remove("hidden");
  } else if (ranked[0].localService !== false && /compact/i.test(ranked[0].name)) {
    els.voiceNote.textContent =
      `Only a low-quality "compact" voice is available for ${currentLocale}. ` +
      `Installing the enhanced/premium variant will sound much better.`;
    els.voiceNote.classList.remove("hidden");
  } else {
    els.voiceNote.classList.add("hidden");
  }
}

// --------------------------------------------------------- Speech out ----

/** Rough upper bound on utterance duration, used only for the watchdog. */
function estimateSpeechMs(text, rate) {
  const words = Math.max(text.split(/\s+/).length, 1);
  return Math.min(Math.max((words / rate) * 500 + 4000, 4000), 40000);
}

function speak(text, locale) {
  if (!("speechSynthesis" in window)) return;

  // Invalidate any in-flight utterance's callbacks before cancelling it,
  // otherwise its onend lands later and clears `speaking` for the NEW one.
  const gen = ++speechGen;
  window.speechSynthesis.cancel();
  clearTimeout(speechWatchdog);

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = locale;
  const voice = pickVoice(locale);
  if (voice) utterance.voice = voice;
  const rate = parseFloat(els.rate?.value || "1");
  utterance.rate = rate;
  utterance.pitch = 1;

  speaking = true;
  applyRecognitionState(); // mute the mic immediately, do not wait on a timer

  const finish = () => {
    if (gen !== speechGen) return; // a newer utterance owns the state now
    clearTimeout(speechWatchdog);
    speaking = false;
    // Resume purely from current intent — never from a captured snapshot.
    syncRecognition(250);
    updateMicStatus();
  };

  utterance.onend = finish;
  utterance.onerror = finish;
  // speechSynthesis silently drops onend in several cases (no voice for the
  // language, tab backgrounded). Without this the mic would never come back.
  speechWatchdog = setTimeout(finish, estimateSpeechMs(text, rate));

  window.speechSynthesis.speak(utterance);
  updateMicStatus();
}

// ---------------------------------------------------------- Speech in ----

function supportsRecognition() {
  return "SpeechRecognition" in window || "webkitSpeechRecognition" in window;
}

/**
 * The single reconciler. Decides what the engine should be doing from the
 * variables as they are *now*. Safe to call at any time, from anywhere.
 */
function applyRecognitionState() {
  const shouldRun = listening && !callOver && !speaking;
  // rebuildRecognition() clears the instance; recreate it lazily so a mode
  // switch (continuous -> single-utterance) can restart cleanly.
  if (shouldRun && !recognition) recognition = buildRecognition();
  if (!recognition) return;

  if (shouldRun && !recognising && !starting) {
    try {
      recognition.lang = currentLocale;
      starting = true;
      recognition.start();
    } catch (_) {
      starting = false; // already running; onstart will correct the flags
    }
  } else if (!shouldRun && recognising) {
    try {
      recognition.stop();
    } catch (_) {
      /* already stopping */
    }
  }
  updateMicStatus();
}

function syncRecognition(delay = 0) {
  clearTimeout(syncTimer);
  if (delay <= 0) {
    applyRecognitionState();
    return;
  }
  syncTimer = setTimeout(applyRecognitionState, delay);
}

function buildRecognition() {
  const Impl = window.SpeechRecognition || window.webkitSpeechRecognition;
  const rec = new Impl();
  // continuous keeps ONE cloud stream open, which avoids the restart churn
  // that triggers Chrome's throttling and `network` errors. Some setups return
  // no results in this mode, so armNoResultHint() can flip us to
  // single-utterance mode and rebuild.
  rec.continuous = continuousMode;
  rec.interimResults = true;
  rec.maxAlternatives = 1;
  rec.lang = currentLocale;

  let interimNode = null;

  const clearInterim = () => {
    if (interimNode && interimNode.parentElement) interimNode.parentElement.remove();
    interimNode = null;
  };

  rec.onstart = () => {
    starting = false;
    recognising = true;
    networkFailures = 0; // a successful start clears the backoff
    diag(
      `recognition started (${rec.lang}, ` +
        `${rec.continuous ? "continuous" : "single-utterance"})`
    );
    updateMicStatus();
  };

  // The diagnostic ladder. Where it stops identifies the broken layer:
  //   no audiostart          -> mic never opened
  //   audiostart, no sound   -> capturing silence (OS permission / wrong device)
  //   sound, no speechstart  -> input too quiet, or noise only
  //   speechstart, no result -> speech service or unsupported language
  rec.onaudiostart = () => {
    diag("audiostart — mic open");
    sawSound = false;
    armSilenceHint();
  };
  rec.onsoundstart = () => {
    diag("soundstart — sound detected");
    sawSound = true;
    clearTimeout(silenceHintTimer);
    pulseActivity(0.45);
  };
  rec.onspeechstart = () => {
    diag("speechstart — speech detected");
    pulseActivity(0.85);
    armNoResultHint();
  };
  rec.onspeechend = () => diag("speechend");
  rec.onsoundend = () => diag("soundend");
  rec.onaudioend = () => diag("audioend");

  rec.onresult = (event) => {
    let interim = "";
    let final = "";
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const chunk = event.results[i];
      if (chunk.isFinal) final += chunk[0].transcript;
      else interim += chunk[0].transcript;
    }

    if (interim && !final) {
      if (!interimNode) interimNode = addMessage("user interim", interim);
      else interimNode.textContent = interim;
    }

    if (interim || final) clearTimeout(noResultHintTimer);

    if (final.trim()) {
      diag(`result: "${final.trim().slice(0, 40)}"`);
      if (interimNode) {
        interimNode.parentElement.classList.remove("interim");
        interimNode.textContent = final.trim();
        interimNode = null;
      } else {
        addMessage("user", final.trim());
      }
      send({ type: "user_text", text: final.trim() });
    }
  };

  rec.onerror = (event) => {
    starting = false;
    if (event.error === "no-speech" || event.error === "aborted") return;
    clearInterim();

    if (event.error === "network") {
      networkFailures += 1;
      if (networkFailures >= MAX_NETWORK_FAILURES) {
        listening = false;
        updateMicStatus();
        setBadge("on", "connected");
        addMessage(
          "system",
          "Speech recognition could not reach Chrome's speech service " +
            "(error: network). This browser API sends audio to Google's " +
            "servers, so a VPN, firewall, ad-blocker, or offline network will " +
            "block it. Use the text box below — it drives the exact same " +
            "workflow. For production audio, run the LiveKit path instead."
        );
        return;
      }
      syncRecognition(1000 * 2 ** (networkFailures - 1)); // 1s, 2s, 4s
      return;
    }

    if (event.error === "not-allowed" || event.error === "service-not-allowed") {
      listening = false;
      updateMicStatus();
      addMessage(
        "system",
        "Microphone permission was denied. Click the padlock in the address " +
          "bar, allow the microphone for this site, then press Start mic again."
      );
      return;
    }

    if (event.error === "audio-capture") {
      listening = false;
      updateMicStatus();
      addMessage("system", "No microphone was found. Use the text box instead.");
      return;
    }

    addMessage("system", `Mic error: ${event.error}`);
    updateMicStatus();
  };

  rec.onend = () => {
    starting = false;
    recognising = false;
    clearInterim();
    // Never restart synchronously here — that is the throttling trap.
    if (listening && !callOver) syncRecognition(400);
    else updateMicStatus();
  };

  return rec;
}

async function startMic() {
  if (!supportsRecognition()) {
    addMessage(
      "system",
      "This browser has no speech recognition (Chrome and Edge support it; " +
        "Firefox does not). Use the text box — same workflow, no mic."
    );
    return;
  }
  if (!window.isSecureContext) {
    addMessage(
      "system",
      "Speech recognition needs a secure context. Open the app on " +
        "http://127.0.0.1:8000 or http://localhost:8000, not a LAN IP."
    );
    return;
  }
  // Acquire the mic explicitly first — it yields a precise, named error where
  // SpeechRecognition would just deliver silence.
  if (!(await ensureMicAccess())) return;

  if (!recognition) recognition = buildRecognition();
  listening = true;
  networkFailures = 0;
  sawSound = false;
  diag("mic requested by user");
  // If Mami is mid-greeting, the reconciler simply waits; the utterance's
  // finish handler calls syncRecognition() and the mic starts then.
  syncRecognition(0);
}

function stopMic() {
  listening = false;
  clearTimeout(syncTimer);
  clearMicHints();
  stopLevelMeter();
  syncRecognition(0);
}

// ------------------------------------------------------------ Socket ----

function send(payload) {
  if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(payload));
}

function connect() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${scheme}://${location.host}/ws/chat`);
  setBadge("wait", "connecting…");

  socket.onopen = () => {
    callOver = false;
    setBadge("on", "connected");
    els.connect.disabled = true;
    els.mic.disabled = false;
    els.hangup.disabled = false;
    els.input.disabled = false;
    els.send.disabled = false;
    els.transcript.innerHTML = "";
    els.lead.textContent = "Not captured yet.";
    send({ type: "start" });
  };

  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "error") {
      addMessage("system", data.error);
      return;
    }
    addMessage("agent", data.reply);

    if (data.locale && data.locale !== currentLocale) {
      currentLocale = data.locale;
      renderVoicePicker();
      if (recognition) recognition.lang = currentLocale;
    }

    renderStates(data.state);
    renderCaptured(data.captured);
    speak(data.reply, currentLocale);

    if (data.lead) {
      els.lead.textContent = JSON.stringify(data.lead, null, 2);
      showWhatsAppToast(data.lead);
    }
    if (data.is_final) {
      callOver = true;
      stopMic();
      setBadge("done", "call complete");
      els.input.disabled = true;
      els.send.disabled = true;
      els.mic.disabled = true;
      els.connect.disabled = false;
      els.connect.textContent = "Start a new call";
    }
  };

  socket.onclose = () => {
    setBadge("off", "disconnected");
    stopMic();
    els.connect.disabled = false;
    els.mic.disabled = true;
    els.hangup.disabled = true;
    els.input.disabled = true;
    els.send.disabled = true;
  };

  socket.onerror = () => setBadge("off", "connection error");
}

// ------------------------------------------------------------ Wiring ----

els.connect.addEventListener("click", connect);
els.hangup.addEventListener("click", () => {
  stopMic();
  if (socket) socket.close();
});
els.mic.addEventListener("click", () => (listening ? stopMic() : startMic()));
if (els.testMic) els.testMic.addEventListener("click", testMic);

els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = els.input.value.trim();
  if (!text) return;
  addMessage("user", text);
  send({ type: "user_text", text });
  els.input.value = "";
});

if (els.voiceSelect) {
  els.voiceSelect.addEventListener("change", () => {
    voiceOverride = els.voiceSelect.value;
    const samples = {
      "en-IN": "Hello Mama, this is Mami.",
      "hi-IN": "नमस्ते मामा, मैं मामी हूँ।",
      "bn-IN": "নমস্কার মামা, আমি মামি।",
      "te-IN": "నమస్తే మామా, నేను మామిని.",
      "ta-IN": "வணக்கம் மாமா, நான் மாமி.",
      "kn-IN": "ನಮಸ್ಕಾರ ಮಾಮಾ, ನಾನು ಮಾಮಿ.",
    };
    speak(samples[currentLocale] || samples["en-IN"], currentLocale);
  });
}

if (els.rate) {
  els.rate.addEventListener("input", () => {
    els.rateOut.textContent = `${parseFloat(els.rate.value).toFixed(2)}×`;
  });
}

if ("speechSynthesis" in window) {
  window.speechSynthesis.getVoices();
  window.speechSynthesis.addEventListener("voiceschanged", renderVoicePicker);
  setTimeout(renderVoicePicker, 250);
}

updateMicStatus();

fetch("/api/config")
  .then((r) => r.json())
  .then((cfg) => {
    states = cfg.states;
    renderStates(null);
    els.runtime.textContent = JSON.stringify(
      {
        languages: cfg.languages.map((l) => l.value),
        llm_extraction: cfg.llm_extraction,
        llm_model: cfg.llm_model,
        stt: cfg.providers.stt,
        tts: cfg.providers.tts,
        livekit_configured: cfg.livekit_configured,
        browser_speech_recognition: supportsRecognition(),
        secure_context: window.isSecureContext,
      },
      null,
      2
    );
  })
  .catch(() => {
    els.runtime.textContent = "could not load /api/config";
  });
