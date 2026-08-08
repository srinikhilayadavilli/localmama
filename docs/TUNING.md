# Tuning log

Every value that was chosen rather than defaulted, and the measurement behind
it. Read this before changing a timing value, a threshold, or the voice —
most of these look arbitrary and are not.

Findings that outlived the code they were found in are kept. Where a number
came from a version of the system that no longer exists, that is said.

---

## What is deployed

Two processes. The agent runs on **LiveKit Cloud** (ap-south) on
`gpt-realtime`, speech to speech. The backend runs on **Render** (Singapore)
and does everything that is not talking. Calls arrive over a Vobiz SIP trunk.

---

## Turn-taking

All on the agent, all set by ear on real calls.

| Setting | Value | Why |
|---|---|---|
| `OPENAI_REALTIME_EAGERNESS` | `high` | Semantic VAD commits sooner to "they have finished". Measured: model `ttft` is 0.34s p50, but the gap a caller feels was 4.03s p50 / 16.07s p95 — **the wait was never compute.** Trade: `high` is likelier to answer over someone who pauses mid-thought. Drop to `auto` if people get cut off. |
| `OPENAI_REALTIME_NOISE_REDUCTION` | `near_field` | A phone handset. `far_field` is for a laptop mic. |
| `HANGUP_GRACE_SECONDS` | `1.5` | **Changed meaning.** It used to be a flat 6s counted from `save_lead` — which fires *before* the goodbye is spoken, so it raced a sentence of unknown length and clipped it. The agent now waits for the outro to start and finish; this is only the tail, so the last frames reach the caller. |
| `HANGUP_MAX_WAIT_SECONDS` | `25` | Backstop for a closing line that never comes. Without it a silent model leaves the caller on a live, metered line. |
| `MAX_CALL_SECONDS` | `600` | Hard ceiling on a call that never completes. Nothing else ends one — a looping model or a silent caller holds both the SIP leg and the realtime session, and both bill by the minute. |

**Gemini Live ignores all of the above.** It runs its own server-side turn
detection (`GEMINI_SILENCE_MS`, `GEMINI_PREFIX_PADDING_MS`,
`GEMINI_END_SENSITIVITY`), which is why tuning the OpenAI values changed
nothing on that path. Its native-audio models sound best and measured ~6s from
stop-speaking to reply; the latency-optimised variant measured ~0.5–1s but
refuses `generate_reply`, so the caller has to speak first.

---

## Voice and accent

`OPENAI_REALTIME_VOICE=marin` is timbre only. **No OpenAI voice is Indian**,
and on a speech-to-speech model the accent can only come from the system
prompt — there is no TTS object to retune when the caller switches language.

`agent/prompts/voice_style.py` is the single source. It names American, British
and Australian in order to exclude them, forbids mirroring a caller who sounds
American, and re-anchors after **interruptions, corrections and language
switches** — the three moments drift was reported. Tests pin the instruction;
nothing can pin the audio.

**The unresolved trade.** Sarvam's bulbul voices are recorded natively per
language and measured **0.19s to first audio** over a websocket against
OpenAI TTS's **1.29s**. Using them needs an STT→LLM→TTS pipeline, which costs
the free-form feel of speech-to-speech. That path was deleted rather than
maintained unshipped; the trade has not gone away, only been deferred.

---

## Latency

Measured on real calls, and the shape of it is the point:

| Stage | p50 | p95 |
|---|---|---|
| Model `ttft` | 0.34s | 0.60s |
| **TurnGap** (caller stops → Mami finishes) | **4.03s** | **16.07s** |

`TurnGap` **includes speaking time**, so a long reply inflates it. That is the
finding: the caller's wait is not the model thinking, it is Mami talking. The
flow prompt now states that a long reply *is* dead air.

**Not reciting the six languages took the greeting from 12.98s to 5.88s.** A
caller who wants Telugu simply says "Telugu"; the list is offered only if they
hesitate or ask.

**Nothing durable runs on the caller's clock.** Persisting the lead and the
webhook handoff both reach the network, and the model cannot speak its closing
line until the tool returns. Inline, that measured **15.26s** of silence before
the goodbye.

The split finished the job. Capture tools used to transliterate a name,
translate a service and match a catalogue while the caller waited — up to six
seconds of network I/O per call at a 2s timeout each. Every tool is now a
dictionary write, and the only backend call anyone waits on is a vendor lookup,
bounded at **800ms** and degrading to "I can't look that up right now."

---

## Accuracy

### Grounding is a score, not a gate

Captured values are checked against the transcript of the caller's own audio —
a second, independent decode from a different model than the one hearing the
call. Two decoders agree on a name the caller said and disagree on an invented
one.

The cutoff (`0.75`) is applied to a **sound key**, not raw spelling. On raw
spelling the two populations overlap and no single cutoff separates them:
"lakshmi"/"laxmi" is 0.67 and "phani"/"fani" is 0.67, while the unrelated
"asha"/"usha" is 0.75. Keyed, the same pairs are 1.00 and unrelated names stay
where they were ("suresh"/"rahul" 0.18, "delhi"/"mumbai" 0.18).

**It ran inside the agent twice, and both placements were wrong.** At capture,
it compared a value against a transcript that had not arrived yet — the model
calls a tool the moment it hears an answer, transcription is a slower pass — so
it refused every field on every live call. Moved to save time, it worked, but
it sent callers back to repeat their name at the moment they expected to hang
up, capped at two refusals to stop the loop.

It is now a backend score. The **read-back** is what corrects a bad value while
the caller is still on the line; the score is what flags one that got past it.

Moving it also let the guard go. While it ran mid-call it carried a per-field
count of how much had been transcribed at capture, and skipped any field whose
count had not moved. Running after `call.ended`, that count is meaningless —
the whole transcript has arrived or it never will — and it was actively wrong:
"the transcript for this answer has not landed yet" and "this answer was the
last thing transcribed" produce identical counts, so real checkable values went
unaudited. **Found by pointing the smoke test at a live deployment**, where a
city sitting plainly in the transcript came back unscored. The guard is now
simply "did any transcript arrive", and a call with none is flagged as the
operational fault it is.

### Two routes to a service, not one

A description may fairly carry a word the caller never said. "AC repair" for
"मुझे एसी ठीक करवाना है" shares **no word** with the transcript and scores 0.0
on grounding alone — the catalogue rules independently map "एसी" to `ac repair`
straight off the transcript, and that corroboration is what rescues it.
Dropping that second route flags every rephrased Indic call for review.

### Fuzzy service matching

Needs comparable length for Latin tokens (`"service"` is 7 of the 10 characters
in `"ac service"` and scored 0.82, so every "car service" became AC repair) and
a stricter 0.80 cutoff for Indic, where one character is a whole syllable
(`"టెన్షన్"` / tension scored 0.71 against `"ట్యూషన్"` / tuition and filed a
plumbing call as a tutor). Neither guard is a blanket ban — the first attempt
at each was, and both lost real matches.

It also needs to know that **the words framing a request are never the trade**
(`_NOT_A_TRADE`). The rescue runs on the whole utterance, so the caller's verb
is a candidate too, and measured against the full synonym table five ordinary
request words clear both guards above and reach a wrong trade:

| Word | Nearest synonym | Ratio | Filed as |
|---|---|---|---|
| `looking` | `cooking` | 0.86 | cook |
| `booking` | `cooking` | 0.86 | cook |
| `wanting` | `painting` | 0.80 | painter |
| `searching` | `coaching` | 0.71 | tutor |
| `number` | `plumber` | 0.77 | plumber |

Neither existing guard can catch these and neither is wrong about what it sees:
"looking" and "cooking" really are one character apart and really are the same
length. What makes it coincidence rather than a slip is that nobody asks for a
"looking" — knowledge about the request, not about the spelling.

**This only ever bit a trade the rule catalog does not list**, because for the
eighteen it does `match_service` finds the real one first. That is exactly the
set the 120-row vendor catalogue exists to serve: "I am looking for mental
health help" was filed as `cook` and matched three restaurants, at 0.85 — above
`MIN_CONFIDENCE`, so nothing re-prompted and the caller heard "cook" read back.
Words are skipped rather than the utterance abandoned, so "I am looking for a
plummer" steps over "looking" and still reaches the plumber.

### Language

- **Recognised in every Indian script.** A caller saying "తెలుగు" transcribed
  as Devanagari `"तेलुगु"` matched nothing, so script detection asserted Hindi
  and the whole call continued in Hindi.
- **Changes need asking twice.** One caller said "ఏది?" ("which?") and the
  model switched the entire call to Hindi off that one word.
- **The STT prompt carries no vocabulary.** It listed Indian names to bias
  decoding; Whisper-family models emit prompt content as transcription on
  silence, so callers saw phantom turns reading "Suresh" and "Lakshmi" — and a
  phantom name is captured as the caller's name.

---

## Vendors

One store: **`backend/services/brain.py` → `utter.knowledge`**, scoped
`localmama/localmama`, 120 rows, every one with a phone, a category and curated
keywords. Read-only — ingest and curation stay in the Vaani dashboard.

Two lookups, and **both are literal — neither embeds**:

- **`find_business(name)`** — a caller asking "what is X's number?" wants an
  exact record. Ordered exact → prefix → substring, then edit distance at
  `NAME_CUTOFF=0.82`, which keeps "Pax Jewellers" → "Pax Jwellers" (0.96) and
  "Cleanmates" → "Clean Mates" (0.95) while rejecting "Speed Autos" → "Speed
  Kawasaki Kolkata".
- **`matches_for_service(service)`** — who actually does this work, for the
  vendor list. Category hits win outright over keyword hits, or
  "cleaning" returns a dental clinic on the strength of "teeth cleaning".
  Requires a phone: a name the caller cannot ring is a teaser, not an answer.

**The city means opposite things to the two lookups, deliberately.** To a
service match it is a hard filter — a plumber in Chennai is not a weaker answer
for a Hyderabad caller, it is the wrong one. To a name lookup it only decides
between hits: a caller who names a business has named it, and answering "I
don't have them listed" because our `city` column disagrees is a lie about a
business we hold. So it narrows the list only when narrowing leaves something,
the same shape as a category hit narrowing a service match, and what it buys is
the ambiguous case — two branches of nearly one name, one of them local, is an
answer rather than "could you say the full name?".

`/v1/vendors` accepted the city and dropped it; the agent had been sending it
all along. It is transliterated alongside the name and concurrently with it —
both are free for a Latin value, and when they are not, one round trip on the
only call a caller waits on rather than two.

**120 of the 121 rows have no city at all**, so this narrows almost nothing
today. A listing with no city is treated as unknown rather than local: it never
survives the narrowing and is always in the fallback, so it can never be
withheld and can never displace a real local match.

**Semantic search was removed, not merely unused.** It matched "electrician" to
Elecsyn Energy, an EV charging company, at 0.59 — because the words look alike.
That is a stranger's phone number read out to a caller. The neighbourhood of a
spelling is a handful of near-identical strings; the neighbourhood of a meaning
is the whole catalogue.

Removing it also took a ~500MB embedding model out of the agent's image. It had
cold-loaded mid-conversation once: **12.43s on a single turn**, and the job
process grew 760MB past the memory warning.

**Finding nothing is a valid outcome.** There is no electrician in the
catalogue, so the template falls back to "our team is shortlisting", which is
true, rather than naming a business that is wrong.

A fuzzy category pass runs **only after a literal miss** (`CATEGORY_CUTOFF=0.75`):
translation returns American spelling where the catalogue is British, so
"Jewelry store" has to reach `jewellery stores` (0.86) and "mobile shop" has to
reach `mobile shops` (0.96). The cutoff keeps "photographer" from being
answered as `professional services` (0.46) when no photographer is listed.

Categories were tidied 59 → 50 (`scripts/tidy_categories.py`). Tidying fixed
spellings, not misfilings: **Equibillbook is still filed under `Car Wash` and
100Marketeers under `Tutors`**, and a category hit is trusted outright, so
those two are answered confidently and wrongly.

---

## Storing values in English

The catalogue is English and matching is literal, so a phrase in the caller's
own script matches nothing — `'car wash' LIKE '%కార్ వాష్%'` is false, and the
caller is told we found nothing for a trade that is in the directory.

The rule catalogue canonicalises the trades it lists in every language it
lists, but it holds 18 home-services trades and **only 4 of the 50 categories
are reachable from one** — nothing for bakeries, jewellers, hotels or mobile
shops. So the backend converts through Sarvam.

- **Names and places are transliterated; services are translated.** A
  translator renders meaning, which is right for a trade and a disaster for a
  proper noun: "आशा" comes back as "Hope".
- **Raw and English are both stored.** The raw value is what the caller
  actually said and the only thing that can be re-processed when the normaliser
  improves.
- Only non-Latin text is sent. Romanised input ("naaku plumber kaavali") is
  already Latin and a round trip could only degrade it.
- `source_language_code=auto`, not the session language: the two disagree, and
  transcripts wander across scripts mid-call.
- **A failure never leaves a value in the caller's script.** `translit.py` is
  an offline Indic→Latin table covering all five scripts and takes over when
  Sarvam is disabled, slow or down. "राहुल" → Rahul, "కార్ వాష్" → kar vash.
  Rougher, always Latin, never a network dependency.

`TRANSLATE_TIMEOUT` is **8s**, not the 2s it was in the agent. The trade
reversed with the split: nothing is on a phone line here, so a better
romanisation is worth waiting for and the offline table is a genuine last
resort rather than a routine outcome.

---

## When post-call work fails

Everything after the caller hangs up is invisible to them, which is exactly why
it needs a paper trail. Each channel has its own outbox columns —
`whatsapp_status` and `handoff_status` — claimed and swept independently:
`pending` until a delivery succeeds (`sent`), or `skipped` when there was
nothing to deliver to or nothing to deliver with. `handoff_response` keeps the
receiver's HTTP status — `sent` means it answered 2xx and nothing more, so a
webhook that 200s into a black hole looks exactly like one that works.

- **Rows are claimed, not just read** — `UPDATE ... RETURNING` with `FOR UPDATE
  SKIP LOCKED` — so two workers sweeping at once take disjoint batches. A claim
  stranded by a dead process is reclaimed after 10 minutes: a duplicate message
  is recoverable, a lead silently never sent is not.
- **One attempt per lead per pass.** The sweep *is* the retry. Three in-call
  retries were the wrong shape for the actual failure — a provider down for
  hours wants trying again later, not trying harder now, and 29 owed leads at
  three attempts each is minutes of work to learn one fact.
- Retries stop after `OUTBOX_MAX_ATTEMPTS` (25), so a permanently bad number is
  not tried forever.
- **An unconfigured webhook stays `pending`, not `skipped`.** Whether the
  handoff is configured is a state of the deployment, not of the lead: it
  changes the moment someone sets `WEBHOOK_URL` and `WEBHOOK_SECRET`, and every
  lead that arrived meanwhile is still owed. The guard lives in the sender
  itself because there is no second entry point to hold it — without it an
  unconfigured deployment POSTs to an empty URL three times per call before
  leaving the lead to be retried 25 more times.
- **A 4xx is terminal, a 429 is not.** A rejected body or a dead secret cannot
  be fixed by trying again; rate limiting is the receiver asking for later,
  which is exactly what the sweep provides.

The agent's event queue has the same shape: events are idempotent by
`event_id`, so a retry that turns out to have landed is a no-op rather than a
duplicate lead. `DRAIN_SECONDS` (10) is how long shutdown waits for the queue
to flush — the agent hangs up ~1.5s after the outro and the process exits
shortly after.

---

## Known limits

Two handoff channels run at once as of 2026-08-08: WhatsApp, which the caller
receives, and a signed webhook being proven alongside it. `whatsapp_status` is
what the customer actually got — it is the one costing counts as delivered.
`handoff_status` is the webhook's own record. Both on the lead row are the
signal to trust, not this file.

- **Indic languages are anglicised**, per the voice section above.
- **Question order and the read-back are asked for, not enforced.**
  `save_lead` refuses while a mandatory field is missing, which stops a call
  being declared finished early — but the model can still ask out of order or
  skip the read-back.
- **The SIP trunk is open.** It accepts a call for the DID from any address,
  since Vobiz has not said whether they use fixed IPs or digest auth.
- **Tuning is per-tenant, in code.** The service catalogue, the accent block
  and the thresholds are Local Mama's. A second customer needs them behind an
  agent spec.
