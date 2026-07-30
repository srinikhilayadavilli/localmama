# Tuning log

Every setting here was changed because a real call went wrong, and most of the
numbers came from measuring rather than guessing. The reasoning matters more
than the values: if you change one, this tells you what evidence to overturn.

Read `make latency` alongside this — it prints the live distribution per stage
next to the setting that moves it.

---

## What is actually deployed

| | |
|---|---|
| Worker | LiveKit Cloud agent `CA_HvZQi3uySbtD`, region `ap-south` |
| Dispatch name | `local-mama-cloud` — **named**, so it only joins rooms dispatched to it |
| Pipeline | `AGENT_MODULE=agent_realtime` — speech-to-speech (`gpt-realtime`, voice `marin`) |
| Phone | `+918071581496` via Vobiz → trunk `ST_xZhVG8X6KYPR` → rule `SDR_cn5WSYVL2pTD` |
| Leads | `localmama.leads` in Neon, plus ephemeral JSON on the container |
| Knowledge | `utter.knowledge` — one store, 120 vendors with phones (`localmama.businesses` survives for Vaani only) |

**The deterministic pipeline (`agent.py`) is not deployed.** It exists, is
tested, and is a switch away — `AGENT_MODULE=agent`. This matters constantly
when reading the settings below: anything Sarvam-related is **inert** on the
speech-to-speech path, which generates its own audio.

---

## Turn-taking

| Setting | Value | Why |
|---|---|---|
| `OPENAI_REALTIME_EAGERNESS` | `high` | Semantic VAD commits sooner to "they have finished". Measured: model `ttft` is 0.34s p50, but the gap a caller feels was 4.03s p50 / 16.07s p95 — the wait was never compute. **Trade: `high` is likelier to answer over someone who pauses mid-thought.** Drop to `auto` if people get cut off. |
| `INTERRUPT_MIN_WORDS` | `1` | LiveKit defaults this to **0**, so duration alone decided and any 0.4s of sound stopped Mami — a cough, traffic, or her own voice off the caller's speakers. 1 ignores noise (it produces no transcript) while still letting a caller cut in with a single "Telugu". 2+ would block one-word answers, which are most of what callers say here. |
| `BACKCHANNEL_BOUNDARY` | `1.0` | A short "haan" while she is talking is agreement, not a stop request. Widen if she still stops for it. |
| `ENDPOINTING_MODE` | `dynamic` | Lets the detector answer sooner when confident, within the min/max bounds. |
| `HANGUP_GRACE_SECONDS` | `1.5` | **Changed meaning.** It used to be a flat 6s counted from `save_lead` — which fires *before* the goodbye is spoken, so it raced a sentence of unknown length and clipped it. Now the agent waits for the outro to start and finish; this is only the tail so the last frames reach the caller. |
| `HANGUP_MAX_WAIT_SECONDS` | `25` | Backstop for a closing line that never comes. Without it a silent model leaves the caller on a live, metered line. |

---

## Voice

| Setting | Value | Why |
|---|---|---|
| `OPENAI_REALTIME_VOICE` | `marin` | Timbre only. **No OpenAI voice is Indian**, and on the speech-to-speech path the accent can only come from the system prompt. |
| `TTS_PROVIDER` | `sarvam` | Native Indic voices. `gpt-4o-mini-tts` is English-phonetic: `instructions` steer *delivery*, never phonemes, so it reads Telugu in an English speaker's mouth. **Inert while `AGENT_MODULE=agent_realtime`.** |
| `SARVAM_SPEAKER` | `priya` | Chosen by ear against 6 voices; also what Vaani settled on. |
| `SARVAM_PACE` / `SARVAM_TEMPERATURE` | `1.0` / `0.85` in code | The plugin defaults temperature to 0.6, which is flat and monotone over a phone line — heard as both "robotic" and "unclear". Raising pace to compensate only trades clarity for speed. **The deployed secret still says pace 1.15; unresolved, and inert anyway.** |

Measured, Sarvam vs OpenAI TTS: **0.19s** to first audio over a websocket
versus **1.29s**. That, plus the accent, is why the deterministic path uses it.

### The accent rule

`prompts/voice_style.py` is the single source. It names American, British and
Australian in order to exclude them, forbids mirroring a caller who sounds
American, and re-anchors after **interruptions, corrections and language
switches** — the three moments drift was reported. Tests pin the instruction;
nothing can pin the audio.

---

## Latency

Measured on real calls, and the shape of it is the point:

| Stage | p50 | p95 |
|---|---|---|
| Model `ttft` | 0.34s | 0.60s |
| **TurnGap** (caller stops → Mami finishes) | **4.03s** | **16.07s** |

`TurnGap` **includes speaking time**, so a long reply inflates it. That is the
finding: the caller's wait is not the model thinking, it is Mami talking.

Two fixes followed. The greeting went from **12.98s to 5.88s** by not reciting
all six languages — a caller who wants Telugu simply says "Telugu", and
`LANGUAGE_NOT_UNDERSTOOD` still lists them for anyone who did not answer. And
the prompt now states that a long reply *is* dead air.

Nothing durable runs on the caller's clock. Persisting to Postgres and the
WhatsApp handoff both reach the network — a psycopg round trip and an HTTPS
POST that retries three times when the provider is down — and the model cannot
speak its closing line until the turn returns. Inline, that measured **15.26s**
of silence before the goodbye. Both paths now write the lead to disk, return,
and finish the remote work in a background task. Measured after: **2ms** on the
speech-to-speech path.

On the deterministic path, phrasing is prefetched with placeholders
(`{{NAME}}`, `{{SERVICE}}`, `{{LOCATION}}`) so it can be generated before the
caller supplies the values. Hit rate went 1-in-6 → 3-in-5, and total phrasing
wait 6.34s → 3.63s.

---

## Extraction and language

- **Fuzzy service matching** needs comparable length for Latin tokens
  (`"service"` is 7 of the 10 characters in `"ac service"` and scored 0.82, so
  every "car service" became AC repair) and a stricter 0.80 cutoff for Indic,
  where one character is a whole syllable (`"టెన్షన్"` / tension scored 0.71
  against `"ట్యూషన్"` / tuition and filed a plumbing call as a tutor). Neither
  guard is a blanket ban — the first attempt at each was, and both lost real
  matches.
- **Language names are recognised in every Indian script.** A caller saying
  "తెలుగు" transcribed as Devanagari `"तेलुगु"` matched nothing, so script
  detection asserted Hindi, STT was pinned to `hi`, and the call stalled.
- **Language changes need asking twice.** One caller said "ఏది?" ("which?") and
  the model switched the whole call to Hindi off that word.
- **The STT prompt carries no vocabulary.** It listed Indian names to bias
  decoding; Whisper-family models emit prompt content as transcription on
  silence, so callers saw phantom turns reading "Suresh" and "Lakshmi" — and a
  phantom name is captured as the caller's name.

---

## Knowledge and contacts

One store: **`brain.py` → `utter.knowledge`**, scoped `localmama/localmama`,
120 rows, every one with a phone, a category and curated keywords. Phones used
to live in a separate `localmama.businesses` table, which meant two stores and
two matching strategies; `scripts/backfill_brain_phones.py` copied them across.
That table is still there because Vaani's matcher reads it directly — nothing
in Local Mama does, so the two can drift and nothing will say so.

Two lookups, and **both are literal — neither embeds**:

- **`find_business(name)`** — a caller asking "what is X's number?" wants an
  exact record. Ordered exact → prefix → substring. Semantic search matched
  "electrician" to Elecsyn Energy, an EV charging company, at 0.59, because the
  words look alike; that is a stranger's number read out to a caller.
- **`matches_for_service(service)`** — who actually does this work, for the
  WhatsApp options line. Category hits win outright over keyword hits, or
  "cleaning" returns a dental clinic on the strength of "teeth cleaning".
  Requires a phone: a name the caller cannot ring is a teaser, not an answer.

**Finding nothing is a valid outcome.** There is no electrician in the
catalogue, so the template falls back to "our team is shortlisting", which is
true, rather than naming a business that is wrong.

### Every captured detail is stored in English

The catalogue is English, and matching is literal, so a phrase in the caller's
own script matches nothing at all — `'car wash' LIKE '%కార్ వాష్%'` is false,
and the caller is told we found nothing for a trade that is in the directory.

`SERVICE_CATALOG` canonicalises the trades it lists in every language it lists,
but it holds 18 home-services trades and **only 4 of the 50 categories are
reachable from one** — nothing for bakeries, jewellers, hotels or mobile shops.
So `services/translate.py` converts the value through Sarvam
(`TRANSLATE_ENABLED`, `SARVAM_API_KEY`). Unlike everything else Sarvam here,
this is **not inert on the speech-to-speech path.**

The conversion happens **at capture, not at lookup**. It used to translate the
service for the vendor query only, which left the lead, the WhatsApp message
and the Postgres row holding a script that neither the matcher nor the person
actioning the lead could read. What is stored is now what is matched.

- **Names and places are transliterated; services are translated.** A
  translator renders meaning, which is right for a trade and a disaster for a
  proper noun: "आशा" comes back as "Hope". `english_name`/`english_place` use
  Sarvam's `/transliterate`; `english_service` uses `/translate`.
- Only non-Latin text is sent. Romanised input ("naaku plumber kaavali") is
  already Latin and a round trip could only degrade it, and the rule extractor
  has usually already produced an English label for free.
- `source_language_code=auto`, not the session language: the two disagree, and
  transcripts wander across scripts mid-call.
- **A failure never leaves a value in the caller's script.** `translit.py` is
  an offline Indic→Latin table covering all five scripts, and it takes over
  when Sarvam is disabled, slow (`LIVE_TIMEOUT`, 2s) or down. "राहुल" → Rahul,
  "కార్ వాష్" → kar vash. Rougher, always Latin, never a network dependency.
- These now run while the caller waits, which is why the timeout is short. A
  service the rule catalogue knows costs no call at all.

The caller's city is passed to `matches_for_service` as well, on the same terms
`retrieve` uses: listings with no city stay eligible, because most have none.

Then a **fuzzy pass, and only after a literal miss** (`CATEGORY_CUTOFF=0.75`):
translation returns American spelling where the catalogue is British, so
"Jewelry store" has to reach `jewellery stores` (0.86), and "mobile shop" has
to reach `mobile shops` (0.96). The cutoff is what keeps "photographer" from
being answered as `professional services` (0.46) when no photographer is
listed. A literal hit is never overridden by a close one.

A **category** is not a business: "wash" or "tutors" asks which business they
mean, by name, and never reads out a list — that would volunteer vendors.
Categories were tidied 59 → 50 (`scripts/tidy_categories.py`, dry run by
default). Tidying fixed spellings, not misfilings: Equibillbook is still filed
under `Car Wash` and 100Marketeers under `Tutors`, and a category hit is
trusted outright, so those two are answered confidently and wrongly.

`BRAIN_MIN_SCORE=0.35` and the embedding model are **inert on the call path.**
`retrieve()` still exists and still needs that floor if anything calls it —
hybrid retrieval returns its best rows however bad, and "fix my geyser"
surfaced a tutoring service at 0.22 — but nothing does. Taking it off the call
path is also what stopped a ~500MB model cold-loading mid-conversation: one
call spent 12.43s on a single turn waiting for it, and the job process grew
760MB past the memory warning.

---

## Known limits

### When post-call work fails

Persisting the lead and the WhatsApp handoff run after the turn returns, so a
failure there is invisible to the caller — which is exactly why it needs a
paper trail. `localmama.leads.whatsapp_status` is the outbox: `pending` until a
send succeeds (`sent`) or there was no number to send to (`skipped`).

- Task exceptions are logged. `add_done_callback(discard)` never inspected
  `.exception()`, so failures reached nobody.
- Shutdown waits up to `SHUTDOWN_DRAIN_SECONDS` (10). The agent hangs up ~1.5s
  after the outro and the process exits, while a WhatsApp attempt takes several
  seconds — so this work was being killed mid-flight.
- The worker drains the outbox **on startup and every `OUTBOX_SWEEP_SECONDS`**
  (300), so a provider that recovers mid-deployment does not wait for a restart.
  `make outbox` reports what is owed. Retries stop after 25 attempts so a
  permanently bad number is not tried forever.
- Rows are **claimed**, not just read — `UPDATE ... RETURNING` with `FOR UPDATE
  SKIP LOCKED` — so two replicas sweeping at once take disjoint batches. A claim
  stranded by a dead process is reclaimed after 10 minutes: a duplicate message
  is recoverable, a lead silently never sent is not.
- The sweep runs in a daemon thread, safe because LiveKit spawns job processes
  with `spawn`/`forkserver` rather than plain `fork`, so a call in progress never
  inherits it.

Three in-call retries were the wrong shape for the actual failure — a provider
down for hours wants trying again later, not trying harder now.

- **WhatsApp cannot send.** CampaignBot resets every connection from every
  network tested, including their own website. Vaani's telemetry shows the last
  successful send was 2026-07-28 18:11 UTC. Credentials and payload are correct
  and verified; the moment they are reachable it works with no redeploy.
- **Telugu, Tamil, Bengali and Kannada are anglicised** on the deployed path,
  because `gpt-realtime` generates its own audio. Native Indic speech requires
  the deterministic pipeline with Sarvam, which costs the free-form feel.
- **The trunk is open.** It accepts a call for the DID from any address, since
  Vobiz has not said whether they use fixed IPs or digest auth.
- **Tuning is per-tenant, in code.** The service catalogue, message tables,
  accent block and thresholds are Local Mama's. A second customer needs these
  behind an agent spec — see the note in the README.
