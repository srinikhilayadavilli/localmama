# Monitoring and per-call cost

What a call costs, end to end, and how much of that number to believe.

---

## 1. The problem this solves

Before this, the production path was unmonitored and unpriced.

The hub (`voiceai/bridge`) has had a live dashboard at `/platform/monitor` for a
while, fed by `bridge/agents/localmama/telemetry.py`. That emitter is imported
by exactly one file — `bridge/agents/localmama/convo/realtime.py`, the legacy
cascaded engine. **The agent that actually takes the calls emitted nothing.**
The dashboard was watching a path that no longer receives traffic.

And nothing anywhere recorded cost. Not a token, not a rate card. The obvious
substitute — call duration times a per-minute figure — is precisely the wrong
model for a realtime API, which re-bills the whole conversation on every
response. Two three-minute calls can differ threefold.

Meanwhile every number needed was already being emitted and dropped:
`livekit-agents` reports the provider's own `response.usage`, split by modality
and by cached-versus-fresh, and `agent/worker.py` never subscribed.

---

## 2. The shape

```
agent (LiveKit)                  backend (Render)                 hub (Render)
──────────────────               ─────────────────                ────────────
session_usage_updated ─┐
metrics_collected ─────┤
room clock ────────────┘
        │
        └─ Meter ──── call.ended.usage[] ──→ localmama.usage ──┐
                      call.ended.turns[] ──→ localmama.call_turns
                                                               ├─→ usage_priced
Sarvam / WhatsApp ──── meter.metered() ────→ localmama.usage ──┘   (view)
                                                                        │
                                            localmama.rate_card ────────┤
                                                                        ▼
                                                        /metrics · /v1/ops/*
                                                        /platform/spend (hub)
                                                        make cost (CLI)
```

**Units are stored; money is derived.** The agent and backend record
quantities. `localmama.rate_card` says what a unit costs *and when*, and
`localmama.usage_priced` multiplies. That means a rate correction re-prices
history, and "what would last month have cost on the mini model?" is a query
rather than a guess.

---

## 3. The arithmetic that matters

A realtime model re-bills the entire conversation on every response. Most of it
is served from cache, and **cached audio costs $0.40 per million against $32
for fresh — 80x.**

Every provider, and the framework's own `ModelUsageCollector`, reports the
*total* with the cached count nested inside it as a subset. Add the two and
bill each at its own rate, and a call is overstated by up to eighty times.
Nothing about that failure looks wrong on a dashboard; the number is just big.

`agent/metering.py` subtracts. `tests/test_agent_metering.py` and
`tests/test_backend_costing.py` both hold it down, from opposite ends.

---

## 4. Where each number comes from

| Charge | Source | Measured? |
|---|---|---|
| Realtime model tokens | `session_usage_updated` | yes |
| LiveKit session + SIP leg | room clock at `call.ended` | yes |
| Sarvam translate/transliterate | `meter.metered` at the HTTP choke point | yes |
| WhatsApp handoff | `meter.metered` around the send | yes |
| **Transcription** | **derived — see below** | **no** |

### The one estimate

`input_audio_transcription` on a realtime session is a separate charge, and
OpenAI does report it — on `conversation.item.input_audio_transcription.completed`.
The LiveKit plugin **discards that event's `usage` field**
(`livekit/plugins/openai/realtime/realtime_model.py`, the completed handler),
so it reaches no metric, no usage snapshot and no dashboard.

Under-reporting a real charge is worse than estimating it and saying so, so it
is derived: inbound audio encodes at ten tokens per second, and the *fresh*
audio input tokens across a call are the caller's audio counted exactly once.
Every such row carries `estimated = true` all the way to the dashboard.

Two things retire it: the plugin forwarding that usage upstream, or a real STT
node appearing — which is why the estimate yields nothing when one has.

---

## 5. Reading the coverage line

A costing system's worst failure is not being wrong. It is being wrong
**quietly, in the direction of cheap** — a model with no rate row costs zero on
every chart, indistinguishable from a model that is genuinely free.

So every surface reports its own gaps:

| Signal | Means |
|---|---|
| `unmetered_calls` | No ledger at all. `call.ended` never arrived. **Totals are a floor.** |
| `unpriced_units` | No rate card row. Contributing zero. |
| `placeholder_units` | A rate nobody has confirmed — CampaignBot ships this way. |
| `estimated_units` | Derived, not reported. See above. |
| `inherited_units` | Priced by model-name prefix. Right for a dated snapshot; wrong the day a new model ships under a familiar name. |

`confidence_pct` is the one-number summary: the share of quantities carrying a
rate somebody actually confirmed.

**The CampaignBot rate is a deliberate placeholder of zero.** Replace it from
an invoice:

```sql
UPDATE localmama.rate_card SET effective_to = now()
 WHERE provider = 'campaignbot' AND placeholder;
INSERT INTO localmama.rate_card
    (provider, operation, unit, per_units, unit_price_usd, effective_from, source)
VALUES ('campaignbot', 'handoff', 'messages', 1, <rate>, now(), '<invoice ref>');
```

Never edit a rate in place. Editing silently restates every call ever made at
today's price.

---

## 6. Where to look

| | |
|---|---|
| `make cost` | Last 24h from a terminal. `ARGS="<call_id>"` prices one call line by line. |
| `/platform/spend` | The hub page: cost per delivered lead, spend by provider and model, coverage banner, per-call drill-down. Basic-gated. |
| `/metrics` | Prometheus exposition. Needs `OPS_TOKEN`. |
| `/v1/ops/summary` | The same data as JSON. |

### The two numbers people confuse

**Cost per call** is the easy one and it flatters. It treats a wrong number and
a delivered lead as the same unit of production.

**Cost per delivered lead** is what the business runs on — total spend over
leads that reached a customer. It is larger by the entire conversion loss. Both
are shown everywhere, the real one first.

### Diagnosing an expensive call

Open it in `/platform/spend/<call_id>` or `make cost ARGS="<call_id>"`. The
priced lines say *what*; the turn series says *which turn*. On a realtime model
the answer is nearly always a context that grew without being trimmed, and it
reads as a slope in `input_tokens` rather than as any single bad number.

### Latency: two numbers, deliberately side by side

**turn gap** is the silence the caller actually sat through. **ttft** is the
model's share of it. A healthy ttft under an unhealthy gap is turn detection or
the network — and no amount of switching models will move it.

---

## 7. Configuration

| Variable | Default | |
|---|---|---|
| `OPS_TOKEN` | — | Gates `/metrics` and `/v1/ops/*`. Unset **refuses**, never opens. |
| `INR_PER_USD` | `88` | Display only. Nothing stored is ever converted. |
| `TURN_RETENTION_DAYS` | `90` | Swept by the outbox worker. The usage ledger is never swept — it is the billing record. |
| `COST_ALERT_USD` | `0.75` | Per-call figure worth a human look. |

---

## 8. Deploy order

Backend first, then the agent. Usage rides as **optional fields on
`call.ended`**, not as a new event type, and that is a compatibility decision
rather than a tidy one: `EventBatch` is a discriminated union, so pydantic
rejects a batch containing any unknown `type`. A new event type deployed to the
agent ahead of the backend would 422 the whole batch and take that call's
`call.ended` down with it — losing the lead in order to add a cost number.

An added optional field is safe either way round: an old backend ignores it, a
new backend sees `[]`. `tests/test_contract_usage.py` asserts both directions,
and asserts the hazard itself.

Then: `python -m backend.migrate`.
