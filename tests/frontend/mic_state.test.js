/**
 * Headless regression test for the browser mic state machine.
 *
 * Runs frontend/app.js against stubbed Web Speech APIs and replays the
 * sequences that are easy to get wrong and impossible to unit-test in a real
 * browser — most importantly: clicking "Start mic" while the agent is still
 * speaking, which used to leave the UI claiming "listening" while the
 * recognition engine had never been started.
 *
 * Run:  node tests/frontend/mic_state.test.js     (or: make test-ui)
 */
const fs = require("fs");
const vm = require("vm");

function makeEl() {
  const el = {
    textContent: "", innerHTML: "", value: "", disabled: false,
    className: "", style: {},
    classList: { add(){}, remove(){}, toggle(){}, contains(){return false} },
    addEventListener(ev, fn){ (this._h ||= {})[ev] = fn; },
    appendChild(){}, remove(){}, scrollTop: 0, scrollHeight: 0,
    parentElement: null,
  };
  return el;
}
const elements = {};
const getEl = (id) => (elements[id] ||= makeEl());

let recInstance = null;
class FakeRecognition {
  constructor(){ this.started=false; recInstance=this; }
  start(){
    if (this.started) throw new Error("InvalidStateError");
    this.started = true;
    setTimeout(()=> this.onstart && this.onstart(), 0);
  }
  stop(){
    if (!this.started) return;
    this.started = false;
    setTimeout(()=> this.onend && this.onend(), 0);
  }
}

let utterInstance = null;
class FakeUtterance {
  constructor(text){ this.text=text; utterInstance=this; }
}
const speechSynthesis = {
  cancel(){}, getVoices(){ return []; },
  fireOnEnd: true,
  speak(u){ if (speechSynthesis.fireOnEnd) setTimeout(()=> u.onend && u.onend(), 500); },
  addEventListener(){},
};

const sandbox = {
  console,
  setTimeout, clearTimeout, setInterval, clearInterval,
  document: {
    getElementById: getEl,
    createElement: () => makeEl(),
  },
  window: {
    isSecureContext: true,
    SpeechRecognition: FakeRecognition,
    speechSynthesis,
    addEventListener(){},
  },
  location: { protocol: "http:", host: "127.0.0.1:8000" },
  WebSocket: class { constructor(){ this.readyState=0; } send(){} close(){} },
  fetch: () => Promise.reject(new Error("no config in harness")),
  SpeechSynthesisUtterance: FakeUtterance,
};
sandbox.window.SpeechSynthesisUtterance = FakeUtterance;
sandbox.globalThis = sandbox;

vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(require("path").join(__dirname, "..", "..", "frontend", "app.js"), "utf8"), sandbox);

const micStatus = () => elements["mic-status"].textContent;
const engineRunning = () => !!(recInstance && recInstance.started);

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  let pass = true;
  const check = (label, cond) => {
    console.log(`${cond ? "PASS" : "FAIL"}  ${label}`);
    if (!cond) pass = false;
  };

  // --- Replay the reported failure -------------------------------------
  // 1. Agent greeting starts playing (speak() is called by the socket handler).
  vm.runInContext('speak("Welcome to Local Mama! You can call me Mami.", "en-IN")', sandbox);
  await wait(20);
  check("greeting playing -> engine off", !engineRunning());
  check("status shows mic off", micStatus() === "Mic off");

  // 2. User clicks Start mic DURING the greeting (the trigger for the bug).
  vm.runInContext("startMic()", sandbox);
  await wait(50);
  check("clicked mic during speech -> engine still paused", !engineRunning());
  check("status explains the pause", micStatus() === "Paused — Mami is speaking");

  // 3. Greeting ends normally (onend fires at 500ms; resume is +250ms).
  await wait(900);
  check("mic STARTED after speech ended (was the deadlock)", engineRunning());
  check("status shows listening", micStatus() === "Listening…");

  // --- Normal turn: mic already on, agent replies ----------------------
  vm.runInContext('speak("Got it, Mama! May I know your name?", "en-IN")', sandbox);
  await wait(50);
  check("mic muted while agent speaks", !engineRunning());
  await wait(900);
  check("mic auto-resumes after agent reply", engineRunning());

  // --- Watchdog path: speechSynthesis never fires onend ----------------
  speechSynthesis.fireOnEnd = false;
  vm.runInContext('speak("No voice installed for this language.", "en-IN")', sandbox);
  await wait(50);
  check("mic muted when onend will never fire", !engineRunning());
  await wait(8000);   // watchdog upper bound for a short utterance
  check("watchdog recovers the mic (no permanent deadlock)", engineRunning());
  speechSynthesis.fireOnEnd = true;

  // --- Stale-callback guard -------------------------------------------
  vm.runInContext('speak("first utterance", "en-IN")', sandbox);
  await wait(10);
  const stale = utterInstance;
  vm.runInContext('speak("second utterance", "en-IN")', sandbox);
  await wait(10);
  stale.onend && stale.onend();          // late callback from the cancelled one
  await wait(100);
  check("stale onend cannot unmute mid-speech", !engineRunning());

  // --- User stop is respected -----------------------------------------
  await wait(1200);
  vm.runInContext("stopMic()", sandbox);
  await wait(50);
  check("stopMic stops the engine", !engineRunning());
  check("status shows mic off", micStatus() === "Mic off");

  console.log(pass ? "\nALL MIC STATE CHECKS PASSED" : "\nSOME CHECKS FAILED");
  process.exit(pass ? 0 : 1);
})();
