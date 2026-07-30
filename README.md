# Local Mama

**Mami**, a warm local-services voice agent for Indian callers. She answers the
phone, asks which of six languages the caller would like, collects their name,
the service they need and their city, reads it back, and hands off a structured
lead over WhatsApp.

Languages: **English, Hindi, Bengali, Telugu, Tamil, Kannada.**

---

## 1. Two deployables

The agent is the UI. Everything else is backend.

```
   PSTN ──▶ SIP trunk ──▶ LiveKit Cloud (ap-south)
                              │
                    ┌─────────▼──────────┐
                    │   AGENT            │  gpt-realtime, speech to speech
                    │   agent/           │  talks. holds no data.
                    └─────────┬──────────┘
                              │ HTTPS · bearer token
                              │ events (async) · vendor lookup (sync, 800ms)
                    ┌─────────▼──────────┐
     Render ────────│  BACKEND           │  FastAPI
                    │  backend/          │  normalise · audit · match · notify
                    ├────────────────────┤
                    │  outbox worker     │  WhatsApp retries, transcript expiry
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Neon (Postgres)   │  vendors · leads · events
                    └────────────────────┘
                              │
                     CampaignBot ─▶ WhatsApp
```

| | Lines | Dependencies | Deploys to |
|---|---|---|---|
| `agent/` | 1,734 | 5 | LiveKit Cloud |
| `backend/` | 3,209 | 4 | Render |
| `contract/` | 309 | — | imported by both |

The test for which side a piece of code belongs on: **does the caller hear the
difference if it is slow?** Speech, turn-taking and the vendor lookup pass.
Translation, vendor matching, WhatsApp and Postgres do not.

### The agent holds nothing

Every tool is a dictionary write plus a queued event. No database driver, no
translation, no catalogue, no WhatsApp, no embedding model. Captured values
leave **raw** — `"రవి"`, not `"Ravi"` — and the backend makes a lead of them
after the caller has hung up.

This is a reversal of how it used to work, and the reason is measured: the
tools transliterated a name, translated a service and matched a catalogue while
the caller waited, which cost up to six seconds of network I/O per call. A
value normalised at capture also can never be re-normalised when the normaliser
improves.

### The contract

`contract/schema.py`, imported by both sides, so the payload cannot drift.

```
POST /v1/events    { schema_version, events: [ … ] }   → 202 EventAck
GET  /v1/vendors   ?name=…&city=…                      → 200 VendorReply
GET  /healthz
```

Three events, all idempotent by `event_id`:

- `call.started` — answered; caller phone, DID
- `call.captured` — **one per field, fire-and-forget**, raw value
- `call.ended` — status, `confirmed`, transcript, turn gaps

Capture is a stream rather than one submission at the end because **an
abandoned call is still a lead** — arguably the most interesting kind. A caller
who gives a name and a service and then hangs up used to be written to an
ephemeral container disk and lost on the next deploy.

`VendorReply` hands the agent a line to speak, not a record. Digit grouping,
and the decision that a match is too weak to read out, are directory policy and
live with the directory. On an approximate match the reply carries **no title
and no phone number at all**, so a model that ignores the instruction still has
nothing to leak.

---

## 2. The flow

`agent/prompts/flow.py`. Seven steps, warm rather than transactional.

1. **Welcome** — "Welcome to Local Mama! You can call me Mami."
2. **Language** — "Which Indian language would you like to speak with me?"
3. **Name** — "Got it, Mama! May I know your name?"
4. **Service** — "Nice to meet you, [Name]! What service are you looking for today?"
5. **City** — "Got it. Which city are you looking for this service in?"
6. **Confirmation** — "Perfect, Mama! So that's an AC repair in Madhapur for Ravi — I'll send the best matching details to your WhatsApp in a few moments."
7. **Closing** — "Thank you for choosing Local Mama. Whenever you need any local service, just call Local Mama. Have a wonderful day, Mama!"

Two things in there are deliberate and measured:

**Step 2 does not recite the six languages.** Reading them aloud takes about
ten seconds and callers hang up during it. Cutting it took the greeting from
12.98s to 5.88s. The list is offered only if the caller hesitates or asks.

**Step 6 carries the read-back as one clause.** It is the only moment a caller
can correct a misheard name, and a wrong name on a phone line is the most
common failure this system has. Folding it into the same sentence costs no
extra turn. Reaching `save_lead` is what sets `confirmed=true`.

---

## 3. Accuracy

The backend scores every captured value against the transcript of the caller's
own audio — a *second, independent decode* from a different model than the one
hearing the call. Two decoders agree on a name the caller said and disagree on
one that was invented.

This used to be a gate inside the agent, and that was wrong twice over. It ran
at save time, so a caller could be sent back to repeat their name at the moment
they expected to hang up. And it ran against a transcript that had not arrived
yet — the realtime model calls a tool the moment it hears an answer, while
transcription is a slower pass — so shipping it inline refused good values on
every live call.

Now it is a score:

- **Names and cities** are proper nouns: scored per word.
- **A service** is a description and may fairly carry a word the caller never
  said — "AC repair" for "मुझे एसी ठीक करवाना है". One anchor is enough, and a
  value the catalogue rules independently derive from the transcript is
  accepted outright. Without that second route, every rephrased Indic call
  would be flagged.
- A call whose transcript never arrived at all is flagged too — not because a
  value looks invented, but because the independent second decode this check
  depends on never happened, which is an operational fault worth seeing.
- Anything below `REVIEW_THRESHOLD`, **or any call that was never read back**,
  sets `needs_review` with a reason.

Running after `call.ended` is what keeps this simple: the whole transcript is
either present or it never came, so there is no mid-call race to reason about.

---

## 4. Setup

```bash
make setup          # venv + both sides + dev deps
cp .env.example .env
```

```bash
make test           # 60 tests, offline, ~1.4s
make backend        # FastAPI on :8000
make agent          # the voice agent, in dev mode
make migrate        # apply database migrations
make outbox         # what is still owed to callers
```

You can exercise the whole lead pipeline **without making a phone call**:

```bash
curl -X POST localhost:8000/v1/events \
  -H "Authorization: Bearer $AGENT_TOKEN" \
  -H 'Content-Type: application/json' \
  -d @tests/fixtures/telugu-call.json
```

That runs normalisation, the audit, vendor matching and the WhatsApp handoff
end to end. None of it was testable without dialling in before the split.

---

## 5. Deployment

**Agent → LiveKit Cloud.** `python -m agent.worker start`. The SIP trunk and
dispatch rule are in [setup/README.md](setup/README.md).

**Backend → Render**, via `render.yaml`: a web service and a background worker
off one image, `preDeployCommand: python -m backend.migrate`.

Migrations run **once per deploy, by one process**. They used to be issued
lazily from inside a live call's save path, once per job process — concurrent
DDL against Neon at the exact moment a caller was hanging up.

### Environment

**Agent** — `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`,
`LIVEKIT_AGENT_NAME`, `OPENAI_API_KEY`, `BACKEND_URL`, `BACKEND_TOKEN`.
Voice knobs: `OPENAI_REALTIME_VOICE`, `OPENAI_REALTIME_EAGERNESS`,
`MAX_CALL_SECONDS`. See `agent/config.py`.

**Backend** — `DATABASE_URL`, `AGENT_TOKEN`, `SARVAM_API_KEY`, `WHATSAPP_*`,
`BRAIN_OWNER_ID`, `BRAIN_AGENT_ID`, `REVIEW_THRESHOLD`,
`TRANSCRIPT_RETENTION_DAYS`. See `backend/config.py`.

`LIVEKIT_AGENT_NAME` has **no default**, deliberately. With one, a laptop
running `make agent` against a production `.env` becomes eligible for
production dispatch and real callers land on it.

---

## 6. What is deliberately not here

**No runtime config store.** There was one — a table, a history table, a CLI
and an endpoint — and it existed to avoid a redeploy back when the agent's
image took minutes to build. That image is now five pinned dependencies and no
model weights. Changing a voice is an environment variable and a restart, which
is faster than the machinery built to avoid it.

**No semantic vendor search.** Matching is literal. Embedding search always
returns its nearest neighbour, so "electrician" matched an EV charging company
at 0.59 — a real business with a real phone number, offered to someone who
wanted their wiring fixed. Finding nothing is a valid outcome, and the WhatsApp
template falls back to "our team is shortlisting", which is true.

**No returning-caller memory.** A prefilled name is asserted to the caller as
fact — the agent never asks, and reads it back among details it collected — so
a name captured wrongly once is repeated on every later call from that number,
and anyone behind a shared handset is greeted as somebody else.

**No deterministic pipeline.** A state machine drove the conversation and
guaranteed question order and the read-back. It was deleted, not lost: it is in
git history, and the guarantees it gave are now split between `save_lead`'s
completeness check (enforced) and the flow prompt (asked for).

---

## 7. Known limits

**Question order and the read-back are asked for, not enforced.** The model can
ask things out of order or skip the read-back and nothing stops it.
`save_lead` refuses while a mandatory field is missing, so a call cannot be
declared finished early — but that is a weaker guarantee than the state machine
gave.

**WhatsApp cannot send.** CampaignBot resets every connection from every
network tested, including their own website. Credentials and payload are
correct and verified; the moment they are reachable it works with no redeploy.
Owed leads accumulate in the outbox and are swept every
`OUTBOX_SWEEP_SECONDS`.

**Indic languages are anglicised.** `gpt-realtime` generates its own audio and
none of its voices is natively Indic — the accent comes from the prompt. Native
Indic speech needs a TTS path with Sarvam's bulbul voices, which costs the
free-form feel of speech-to-speech. That trade has not been resolved, only
deferred.

**The SIP trunk is open.** It accepts a call for the DID from any address,
since Vobiz has not said whether they use fixed IPs or digest auth.

**Tuning is per-tenant, in code.** The service catalogue, the accent block and
the thresholds are Local Mama's. A second customer needs them behind an agent
spec.

---

Measured findings and the reasoning behind specific values are in
**[docs/TUNING.md](docs/TUNING.md)**. Read it before changing a timing value,
a threshold, or the voice.
