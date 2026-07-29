/**
 * Headless tests for the microphone diagnostic ladder.
 *
 * These cover the "it says listening but nothing happens" class of failure,
 * where the recognition engine is genuinely running but no transcript arrives.
 * The app must distinguish, and say out loud, which layer is broken:
 *
 *   permission denied      -> precise, actionable message; mic stays off
 *   mic open but silent    -> OS permission / wrong input device
 *   speech heard, no text  -> speech service or unsupported language
 *
 * Run:  node tests/frontend/mic_diagnostics.test.js     (or: make test-ui)
 */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

function makeEl() {
  return {
    textContent: "", innerHTML: "", value: "", disabled: false,
    className: "", style: {},
    classList: { add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    addEventListener(){}, appendChild(){}, remove(){},
    scrollTop: 0, scrollHeight: 0, parentElement: null,
  };
}

function buildSandbox({ getUserMedia }) {
  const elements = {};
  const systemMessages = [];
  const getEl = (id) => (elements[id] ||= makeEl());

  let recInstance = null;
  class FakeRecognition {
    constructor(){ this.started = false; recInstance = this; }
    start(){
      if (this.started) throw new Error("InvalidStateError");
      this.started = true;
      setTimeout(() => this.onstart && this.onstart(), 0);
    }
    stop(){
      if (!this.started) return;
      this.started = false;
      setTimeout(() => this.onend && this.onend(), 0);
    }
  }

  const sandbox = {
    console, setTimeout, clearTimeout, setInterval, clearInterval, Date, Math, JSON,
    document: { getElementById: getEl, createElement: () => makeEl() },
    window: {
      isSecureContext: true,
      SpeechRecognition: FakeRecognition,
      speechSynthesis: {
        cancel(){}, getVoices(){ return []; }, speak(){}, addEventListener(){},
      },
      addEventListener(){},
    },
    navigator: { mediaDevices: { getUserMedia } },
    location: { protocol: "http:", host: "127.0.0.1:8000" },
    WebSocket: class { constructor(){ this.readyState = 0; } send(){} close(){} },
    fetch: () => Promise.reject(new Error("no config in harness")),
    SpeechSynthesisUtterance: class { constructor(t){ this.text = t; } },
  };
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(
    fs.readFileSync(path.join(__dirname, "..", "..", "frontend", "app.js"), "utf8"),
    sandbox
  );

  // Capture anything surfaced to the user as a system message.
  const originalAdd = sandbox.addMessage;
  sandbox.addMessage = (role, text) => {
    if (role === "system") systemMessages.push(text);
    return originalAdd(role, text);
  };

  return {
    sandbox, elements, systemMessages,
    rec: () => recInstance,
    diag: () => elements["mic-diag"].textContent,
    run: (code) => vm.runInContext(code, sandbox),
  };
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const fakeTrack = { label: "MacBook Pro Microphone", stop(){} };
const fakeStream = { getAudioTracks: () => [fakeTrack], getTracks: () => [fakeTrack] };

let pass = true;
const check = (label, cond) => {
  console.log(`${cond ? "PASS" : "FAIL"}  ${label}`);
  if (!cond) pass = false;
};

(async () => {
  // --- 1. Permission denied -------------------------------------------
  {
    const denied = Object.assign(new Error("denied"), { name: "NotAllowedError" });
    const h = buildSandbox({ getUserMedia: () => Promise.reject(denied) });
    h.run("startMic()");
    await wait(80);

    check("denied permission -> mic engine never starts", !h.rec() || !h.rec().started);
    const msg = h.systemMessages.join(" ");
    check("denied permission -> names the padlock fix", /padlock/i.test(msg));
    check("denied permission -> names macOS privacy settings", /Privacy & Security/i.test(msg));
    check("denied permission -> points at the text box", /text box/i.test(msg));
    check("denied permission -> logged to diagnostics", /getUserMedia failed/.test(h.diag()));
  }

  // --- 2. Device busy --------------------------------------------------
  {
    const busy = Object.assign(new Error("busy"), { name: "NotReadableError" });
    const h = buildSandbox({ getUserMedia: () => Promise.reject(busy) });
    h.run("startMic()");
    await wait(80);
    check(
      "busy device -> suggests closing other apps",
      /in use by another application/i.test(h.systemMessages.join(" "))
    );
  }

  // --- 3. Mic opens but captures silence -------------------------------
  {
    const h = buildSandbox({ getUserMedia: () => Promise.resolve(fakeStream) });
    h.run("startMic()");
    await wait(80);
    check("granted permission -> engine starts", h.rec() && h.rec().started);
    check("granted permission -> device logged", /mic acquired/.test(h.diag()));

    h.rec().onaudiostart();               // mic opened...
    await wait(50);
    check("audiostart logged", /audiostart/.test(h.diag()));
    check("no silence hint yet", !/no sound is reaching/i.test(h.systemMessages.join(" ")));

    await wait(8300);                     // ...but no soundstart ever follows
    check(
      "silent mic -> explains OS permission / input device",
      /no sound is reaching it/i.test(h.systemMessages.join(" "))
    );
    check("silent mic -> logged to diagnostics", /no sound detected/.test(h.diag()));
  }

  // --- 4. Speech heard but recogniser returns nothing ------------------
  {
    const h = buildSandbox({ getUserMedia: () => Promise.resolve(fakeStream) });
    h.run("startMic()");
    await wait(80);
    h.rec().onaudiostart();
    h.rec().onsoundstart();
    h.rec().onspeechstart();
    await wait(50);
    check("soundstart cancels the silence hint", /soundstart/.test(h.diag()));

    // First failure: retry in single-utterance mode rather than giving up.
    await wait(7300);
    let msg = h.systemMessages.join(" ");
    check(
      "speech but no text -> first retries single-utterance mode",
      /single-utterance mode/i.test(msg)
    );
    check("silence hint did NOT fire (sound was heard)", !/no sound is reaching it/i.test(msg));

    // Second failure, now in the fallback mode: blame the speech service.
    await wait(600);
    h.rec().onaudiostart();
    h.rec().onsoundstart();
    h.rec().onspeechstart();
    await wait(7300);
    msg = h.systemMessages.join(" ");
    check(
      "second failure -> blames the speech service, not the mic",
      /Speech was detected but the recogniser returned no text/i.test(msg)
    );
    check("second failure -> names the locale", /en-IN/.test(msg));
  }

  // --- 5. Happy path: a result clears the pending hint ------------------
  {
    const h = buildSandbox({ getUserMedia: () => Promise.resolve(fakeStream) });
    h.run("startMic()");
    await wait(80);
    h.rec().onaudiostart();
    h.rec().onsoundstart();
    h.rec().onspeechstart();
    h.rec().onresult({
      resultIndex: 0,
      results: [Object.assign([{ transcript: "Telugu" }], { isFinal: true })],
    });
    await wait(7300);
    check(
      "a real result suppresses the no-result hint",
      !/returned no text/i.test(h.systemMessages.join(" "))
    );
    check("result logged to diagnostics", /result: "Telugu"/.test(h.diag()));
  }

  console.log(pass ? "\nALL MIC DIAGNOSTIC CHECKS PASSED" : "\nSOME CHECKS FAILED");
  process.exit(pass ? 0 : 1);
})();
