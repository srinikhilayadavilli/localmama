# The agent

Everything in `agent/`. It runs on LiveKit Cloud in ap-south, answers the
phone, and does nothing else.

**1,840 lines. Five pinned dependencies. No database driver.**

---

## 1. What it is responsible for

Exactly four things:

1. **Hold the call.** Answer, run a speech-to-speech model, sound like a person
   from Hyderabad, end the call cleanly.
2. **Follow the flow.** Language, name, service, city — asked in that order by
   default, skipping anything the caller has already given.
3. **Capture four values in the caller's own words.** Raw. In their script.
   Untranslated, unnormalised, uncorrected.
4. **Answer one question during the call:** what is business X's phone number.

Everything else belongs to the backend.

### What it deliberately does not do

| | why not |
|---|---|
| Translate or transliterate | It cost up to 6s of live call in network I/O, and a value normalised at capture can never be re-normalised later. |
| Match the vendor catalogue | Needs the database. Runs after the caller has hung up, where latency is free. |
| Send WhatsApp | Same. Also needs retries, an outbox, and a scheduler. |
| Store anything durably | It has no database driver in its image at all. |
| Decide whether a value is *plausible* | Refusing values in a tool put callers in re-prompt loops at the moment they expected to hang up. |

The test for whether something belongs here: **does the caller hear it if it is
slow?**

---

## 2. The workflow, one call

```
LiveKit dispatches a job
        │
        ├─ ctx.connect()
        ├─ wait_for_participant()          ← the caller is now on the line
        │
        ├─ EventQueue starts               ← background flusher
        ├─ Recorder built                  ← call_id minted, caller number read
        │                                     off the SIP participant
        ├─ POST call.started               ← queued, not awaited
        │
        ├─ build the realtime model        ← gpt-realtime, semantic VAD
        ├─ session.start()                 ← ~0.2s
        ├─ generate_reply(GREETING)        ← the accent travels with this
        │
        │   ┌─────────────────────────────────────────┐
        │   │  the conversation                       │
        │   │                                         │
        │   │  caller speaks → transcript recorded    │
        │   │  model calls a tool → value captured    │
        │   │                     → POST call.captured│
        │   │  model reads back → caller agrees       │
        │   │  model calls save_lead                  │
        │   └─────────────────────────────────────────┘
        │
        ├─ hang-up task notices `saved`
        ├─ waits for the read-back utterance to finish
        ├─ waits for the goodbye to start, then end
        ├─ 1.5s tail so the last frames land
        ├─ ctx.delete_room()               ← the agent hangs up, not the caller
        │
        └─ on_shutdown:
             POST call.ended               ← status, confirmed, transcript, gaps
             drain the queue (≤10s)
```

Two things run for the whole call alongside that:

- **`user_input_transcribed`** — every final caller transcript is kept. It is a
  *second, independent decode* of the same audio the model heard, and it is the
  only evidence the backend's audit has that a captured value came from the
  caller rather than the model's imagination.
- **`conversation_item_added`** — every assistant utterance is kept, and the gap
  between the caller finishing and Mami replying is measured. That gap is the
  only latency number that reflects what the caller actually felt.

---

## 3. The tools

Six. Every one is a dictionary write. Only one awaits the network.

| tool | what it does | network |
|---|---|---|
| `set_language` | Resolves "తెలుగు" → `telugu`. The one value the agent interprets, because it must know what to speak next. | none |
| `set_name` | Sanitises, stores raw, queues an event. | none |
| `set_service` | Same, plus an optional `trade` — the model's reading when the caller *described* a problem rather than naming one. | none |
| `set_city` | Same as name. | none |
| `lookup_vendor_contact` | `GET /v1/vendors`. **The only call a caller waits on.** | 800ms, bounded |
| `save_lead` | Local completeness check. No network at all. | none |

### Three rules the tools enforce locally

**Completeness.** `save_lead` refuses while a mandatory field is missing, so
the model cannot declare a call finished early. A dictionary check — no round
trip. A business the caller asked for by name stands in for the service: naming
a business is a more specific answer than naming a trade, not a missing one.

**A language change must be asked for twice.** A caller said "ఏది?" — Telugu
for "which?" — and the model switched the entire conversation to Hindi off that
one word. A genuine request survives being asked to confirm; a misheard one
does not.

**Every tool result restates everything held.** The model tracks the
conversation in its own context, and a reply naming only the field just written
made it re-ask for details it already had:

```
[HELD: language=telugu, name=రవి, service=my tap is leaking (= plumber).
 STILL NEEDED: city. Do not ask again for anything HELD.]
```

---

## 4. What crosses the boundary

Two HTTP calls, and they are treated completely differently.

**Events — fire and forget.** Queued and flushed by a background task. No tool
ever awaits one. If the backend is slow, restarting or unreachable, the queue
holds and retries with backoff; every event is idempotent by `event_id`, so a
retry that turns out to have landed is a no-op rather than a duplicate lead.
Three kinds: `call.started`, `call.captured` (one per field, as it happens),
`call.ended`.

Capture is a stream rather than one submission at the end because **an
abandoned call is still a lead**. A caller who gives a name and a service and
then hangs up is already recorded.

**The vendor lookup — the one thing a caller waits on.** Hard 800ms ceiling. It
degrades to "I can't look that up right now" rather than to dead air, and the
backend returns *prose to speak* rather than a record: digit grouping and the
decision that a match is too weak to read out are directory policy, and belong
with the directory.

---

## 5. What it asks for but cannot enforce

This is the honest weakness of a speech-to-speech agent, and it is why the
backend audits everything afterwards.

- **The order of questions** is a prompt instruction.
- **The read-back happening at all** is a prompt instruction.
- **The accent** is a prompt instruction — no OpenAI voice is Indian, and there
  is no TTS object to retune when the caller switches language. Tests pin the
  instruction; nothing can pin the audio.
- **Not inventing a value** is a prompt instruction. The backend scores every
  captured value against the transcript and flags what it cannot corroborate.

`confirmed` is the one piece of verification the agent can honestly produce,
and it means *a caller turn followed the read-back* — not merely that
`save_lead` was reached. The model once read the details back and saved two
seconds later, and the lead went out marked confirmed by nobody.

---

## 6. How it fails

| what breaks | what the caller gets |
|---|---|
| Backend unreachable | The whole call works. Events queue and retry; the lead lands late. |
| Backend still unreachable at hang-up | The queue drains for 10s, then the lead is genuinely lost — logged loudly. |
| Vendor lookup slow or down | "I can't look that up right now." |
| Model cannot be built | The call ends immediately rather than sitting in silence. |
| Model loops, or the caller goes silent | `MAX_CALL_SECONDS` (600) ends the call. Both the SIP leg and the realtime session bill by the minute and neither stops on its own. |
| Model never speaks the goodbye | `OUTRO_START_WAIT` (4s), then hang up anyway. |

---

## 7. Configuration

Environment variables, read once at import. There is no runtime config store —
there was one, and it existed to avoid a redeploy back when the image took
minutes to build. It is now five dependencies and no model weights.

Required: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`,
`LIVEKIT_AGENT_NAME`, `OPENAI_API_KEY`, `BACKEND_URL`, `BACKEND_TOKEN`.

`LIVEKIT_AGENT_NAME` has **no default** on purpose: with one, a laptop running
`make agent` against a production `.env` becomes eligible for production
dispatch and real callers land on it.

Worth knowing about, from real calls:

- `OPENAI_REALTIME_EAGERNESS` — `auto`. Was `high` for minimum dead air, and the
  trade came due: background noise kept starting turns and cutting Mami off.
- `OPENAI_REALTIME_INTERRUPT` — set to `0` to make her uninterruptible. Her turns
  are one or two sentences, so little is lost, and noise can no longer cut in.
- `MAX_CALL_SECONDS` — the only thing that ends a call which never completes.

Startup reports every misconfiguration that would otherwise fail quietly at the
point of use.
