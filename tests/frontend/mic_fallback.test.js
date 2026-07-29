/**
 * Headless test for recognition-layer recovery.
 *
 * Covers the "level bar moves but nothing is transcribed" failure: audio
 * capture is fine, but Chrome's SpeechRecognition returns no results. Two
 * defences are asserted here —
 *
 *   1. the getUserMedia stream used for the level check is released before
 *      recognition starts, so two consumers never compete for the microphone;
 *   2. if speech is heard but yields no text, the app retries once in
 *      single-utterance mode (the most widely supported configuration) before
 *      telling the user to fall back to text.
 *
 * Run:  node tests/frontend/mic_fallback.test.js     (or: make test-ui)
 */
const fs=require("fs"), path=require("path"), vm=require("vm");
const mk=()=>({textContent:"",innerHTML:"",value:"",disabled:false,className:"",style:{},
  classList:{add(){},remove(){},toggle(){},contains(){return false}},
  addEventListener(){},appendChild(){},remove(){},scrollTop:0,scrollHeight:0,parentElement:null});

let liveStreams=0;
const track={label:"MacBook Pro Microphone",stop(){liveStreams--;}};
const stream={getAudioTracks:()=>[track],getTracks:()=>[track]};

const els={}, sys=[];
let rec=null, built=0;
class R{
  constructor(){this.started=false;this.continuous=undefined;rec=this;built++;}
  start(){if(this.started)throw new Error("x");this.started=true;setTimeout(()=>this.onstart&&this.onstart(),0);}
  stop(){if(!this.started)return;this.started=false;setTimeout(()=>this.onend&&this.onend(),0);}
  abort(){this.started=false;}
}
const sb={console,setTimeout,clearTimeout,setInterval,clearInterval,Date,Math,JSON,
  document:{getElementById:id=>(els[id]||=mk()),createElement:mk},
  window:{isSecureContext:true,SpeechRecognition:R,
    speechSynthesis:{cancel(){},getVoices(){return[]},speak(){},addEventListener(){}},
    addEventListener(){}},
  navigator:{mediaDevices:{getUserMedia:()=>{liveStreams++;return Promise.resolve(stream);}}},
  location:{protocol:"http:",host:"h"},
  WebSocket:class{constructor(){this.readyState=0}send(){}close(){}},
  fetch:()=>Promise.reject(new Error("n/a")),
  SpeechSynthesisUtterance:class{constructor(t){this.text=t}}};
sb.globalThis=sb; vm.createContext(sb);
vm.runInContext(fs.readFileSync(path.join(__dirname,"..","..","frontend","app.js"),"utf8"),sb);
const orig=sb.addMessage; sb.addMessage=(r,t)=>{if(r==="system")sys.push(t);return orig(r,t);};

const wait=ms=>new Promise(r=>setTimeout(r,ms));
let pass=true; const check=(l,c)=>{console.log(`${c?"PASS":"FAIL"}  ${l}`);if(!c)pass=false;};

(async()=>{
  vm.runInContext("startMic()",sb);
  await wait(100);
  check("recognition starts in continuous mode", rec.continuous===true);
  check("getUserMedia stream RELEASED before recognition (no mic contention)", liveStreams===0);
  check("device still logged from the permission check", /mic acquired/.test(els["mic-diag"].textContent));

  // Speech is heard, but the service returns nothing.
  rec.onaudiostart(); rec.onsoundstart(); rec.onspeechstart();
  await wait(7400);
  check("auto-switched to single-utterance mode", /single-utterance/.test(els["mic-diag"].textContent));
  check("told the user what changed", sys.some(m=>/single-utterance mode/i.test(m)));
  check("recogniser rebuilt", built>=2);
  await wait(500);
  check("new recogniser is non-continuous", rec.continuous===false);
  check("recognition running again after the switch", rec.started===true);

  // Second failure in the fallback mode -> advise text mode.
  rec.onaudiostart(); rec.onsoundstart(); rec.onspeechstart();
  await wait(7400);
  check("second failure blames the speech service", sys.some(m=>/returned no text/i.test(m)));
  check("second failure points at the text box", sys.some(m=>/text box/i.test(m)));

  console.log(pass?"\nFALLBACK CHECKS PASSED":"\nFAILED"); process.exit(pass?0:1);
})();
