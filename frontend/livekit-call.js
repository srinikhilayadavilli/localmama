/**
 * Local Mama — real voice call over LiveKit.
 *
 * This is the OTHER audio path, and it is deliberately a separate file from
 * app.js. That file is the Web Speech console: your browser recognises and
 * speaks, the server never hears audio, and Mami comes out in whatever system
 * voice you have — which is why she sounds robotic there. Here the browser
 * publishes real microphone audio to LiveKit, the worker in backend/app/agent.py
 * runs it through OpenAI (gpt-4o-transcribe in, gpt-4o-mini-tts out with the
 * Indian-accent instructions), and Mami's actual voice comes back as an audio
 * track.
 *
 * The two paths are mutually exclusive by design: both want the microphone, and
 * running them together means two Mamis answering the same person. Starting a
 * call here stops the Web Speech mic first.
 *
 * The worker must be running somewhere — `make agent` on your machine is enough,
 * since it and this page meet through your LiveKit project rather than through
 * this server. If nobody is running it, you connect to an empty room and hear
 * silence, so that case is detected and reported rather than left a mystery.
 */

(function () {
  "use strict";

  const byId = (id) => document.getElementById(id);

  const ui = {
    call: byId("btn-lk-call"),
    hangup: byId("btn-lk-hangup"),
    status: byId("lk-status"),
    hint: byId("lk-hint"),
    audio: byId("lk-audio"),
  };

  // The page is also opened without these controls in some layouts; bail
  // quietly rather than throwing on every load.
  if (!ui.call) return;

  /** How long to wait for the agent to join before saying something is wrong. */
  const AGENT_JOIN_TIMEOUT_MS = 8000;

  let room = null;
  let agentWatchdog = null;

  function setStatus(state, label) {
    ui.status.className = `badge ${state}`;
    ui.status.textContent = label;
  }

  function setHint(text) {
    ui.hint.textContent = text;
  }

  /** Reuse the console's transcript pane so a call reads like a conversation. */
  function show(role, text) {
    if (typeof addMessage === "function") {
      addMessage(role, text);
    } else {
      console.log(`[${role}] ${text}`);
    }
  }

  /**
   * A fresh room per call.
   *
   * The worker registers unnamed, so it auto-joins whatever room is opened. If
   * two people both used "local-mama" they would land in the same room with one
   * Mami between them, hearing each other's call.
   */
  function newRoomName() {
    const rand = Math.random().toString(36).slice(2, 8);
    return `local-mama-web-${Date.now().toString(36)}-${rand}`;
  }

  /** Silence the Web Speech console so the two paths never share the mic. */
  function stopBrowserSpeech() {
    try {
      if (typeof stopMic === "function") stopMic();
      if (typeof window.speechSynthesis !== "undefined") {
        window.speechSynthesis.cancel();
      }
    } catch (err) {
      console.warn("could not stop the Web Speech console cleanly", err);
    }
  }

  async function mintToken(roomName) {
    const res = await fetch("/api/livekit/token", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ room: roomName, identity: "web-caller" }),
    });
    if (!res.ok) {
      // 503 here means the server has no LiveKit credentials, which is a
      // different problem from a failed connection — say which.
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `token endpoint returned ${res.status}`);
    }
    return res.json();
  }

  function attachAgentAudio(track) {
    const el = track.attach();
    el.autoplay = true;
    ui.audio.appendChild(el);
    // Connecting was a click, so the autoplay policy is satisfied; this only
    // matters when the browser suspended the audio context beforehand.
    if (room && typeof room.startAudio === "function") {
      room.startAudio().catch(() => {});
    }
  }

  function watchForAgent() {
    clearTimeout(agentWatchdog);
    agentWatchdog = setTimeout(() => {
      if (!room || room.remoteParticipants.size > 0) return;
      setStatus("wait", "no agent");
      setHint(
        "Connected, but no agent joined this room. The voice worker is not " +
          "running — start it with `make agent` and call again."
      );
      show("system", "No agent joined the room. Is `make agent` running?");
    }, AGENT_JOIN_TIMEOUT_MS);
  }

  async function startCall() {
    if (typeof LivekitClient === "undefined") {
      setStatus("off", "unavailable");
      setHint("The LiveKit client library did not load; check your connection.");
      return;
    }

    const { Room, RoomEvent, Track } = LivekitClient;

    ui.call.disabled = true;
    setStatus("wait", "connecting…");
    setHint("Requesting a token and joining the room…");
    stopBrowserSpeech();

    try {
      const { token, url, agent_name: agentName, dispatched } = await mintToken(newRoomName());

      // A named worker joins only when dispatched. If that failed, nothing will
      // be in the room — say so now rather than after eight seconds of silence.
      if (agentName && !dispatched) {
        setStatus("wait", "no agent");
        setHint(
          `No worker is registered as "${agentName}", so nothing was dispatched ` +
            "to this room. Deploy or start that agent and try again."
        );
        show("system", `Dispatch to "${agentName}" failed — no such worker registered.`);
        ui.call.disabled = false;
        return;
      }

      room = new Room({ adaptiveStream: true, dynacast: true });

      room.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind === Track.Kind.Audio) attachAgentAudio(track);
      });

      room.on(RoomEvent.ParticipantConnected, () => {
        clearTimeout(agentWatchdog);
        setStatus("live", "on call");
        setHint("Mami is on the line. She greets you first — let her finish, then answer.");
      });

      // The agent publishes both sides of the conversation as transcriptions,
      // so the call is readable, not just audible.
      if (RoomEvent.TranscriptionReceived) {
        room.on(RoomEvent.TranscriptionReceived, (segments, participant) => {
          const fromAgent = !participant || participant.identity !== room.localParticipant.identity;
          segments
            .filter((seg) => seg.final && seg.text.trim())
            .forEach((seg) => show(fromAgent ? "agent" : "user", seg.text.trim()));
        });
      }

      room.on(RoomEvent.Disconnected, () => endCall({ silent: true }));

      await room.connect(url, token);
      await room.localParticipant.setMicrophoneEnabled(true);

      ui.hangup.disabled = false;
      setStatus("live", "on call");
      setHint("Speak normally. Say a language, your name, the service you need, and your area.");
      show("system", "Voice call connected — this is the OpenAI voice, not the browser's.");
      watchForAgent();
    } catch (err) {
      const message = (err && err.message) || String(err);
      setStatus("off", "failed");
      // Permission errors are by far the most common and have a specific fix.
      setHint(
        /permission|denied|NotAllowed/i.test(message)
          ? `Microphone permission was denied (${message}). Allow it and try again.`
          : `Could not start the call: ${message}`
      );
      show("system", `Voice call failed: ${message}`);
      await endCall({ silent: true });
      ui.call.disabled = false;
    }
  }

  async function endCall({ silent = false } = {}) {
    clearTimeout(agentWatchdog);
    if (room) {
      try {
        await room.disconnect();
      } catch (err) {
        console.warn("disconnect failed", err);
      }
      room = null;
    }
    ui.audio.replaceChildren();
    ui.call.disabled = false;
    ui.hangup.disabled = true;
    if (!silent) {
      show("system", "Voice call ended.");
    }
    setStatus("off", "idle");
    setHint("Calls the LiveKit worker, so you hear Mami's real OpenAI voice.");
  }

  ui.call.addEventListener("click", startCall);
  ui.hangup.addEventListener("click", () => endCall());

  // A page unload mid-call otherwise leaves the room occupied until LiveKit
  // times the participant out.
  window.addEventListener("beforeunload", () => {
    if (room) room.disconnect();
  });
})();
