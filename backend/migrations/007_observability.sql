-- Observability: what each call consumed, and what that costs.
--
-- Three tables and two views. The organising rule is that **quantities are
-- stored and money is derived**. A price is a fact about the day a call
-- happened: rates change, published rates turn out to have been misread, and
-- the question "what would last month have cost on the mini model?" is only
-- answerable if the quantity outlived the arithmetic. So the agent and the
-- backend record units, `rate_card` says what a unit costs and when, and
-- `usage_priced` multiplies. Correcting a rate re-prices history.
--
-- Nothing here is on a caller's clock — every write happens after the call has
-- ended — and nothing here holds personal data: token counts and durations,
-- keyed by a call_id that resolves to a lead row elsewhere.

-- One metered quantity. The unit ledger; every cost in the system derives
-- from a row in here.
--
-- Written from two places. The agent ships whole-call totals on `call.ended`
-- with ref='session', so a retried event REPLACES rather than doubles. The
-- backend writes one row per provider call it makes after the fact, each with
-- its own ref, so several translations of one lead each count.
CREATE TABLE IF NOT EXISTS localmama.usage (
    call_id    TEXT NOT NULL,
    -- Deduplication scope within a call. 'session' for an agent total; a uuid
    -- for one backend provider call.
    ref        TEXT NOT NULL DEFAULT 'session',
    -- 'agent' | 'backend'. Which side of the split observed this.
    source     TEXT NOT NULL,
    provider   TEXT NOT NULL,
    model      TEXT NOT NULL DEFAULT '',
    operation  TEXT NOT NULL DEFAULT '',
    unit       TEXT NOT NULL,
    quantity   DOUBLE PRECISION NOT NULL CHECK (quantity >= 0),
    -- False when a provider reported this, true when we derived it. Kept all
    -- the way to the dashboard: a total that is part inferred is a different
    -- number from one that is wholly measured, and anyone choosing a model on
    -- the strength of it needs to know which they are reading.
    estimated  BOOLEAN NOT NULL DEFAULT false,
    -- Operational outcome, for backend rows only. A provider call that failed
    -- still consumed something and still gets a row — a retry storm against a
    -- down provider is a cost, and one that is invisible if failures are not
    -- recorded.
    ok         BOOLEAN,
    latency_ms INTEGER,
    error      TEXT,
    at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- `operation` is in the key on purpose: LiveKit bills the agent session
    -- and the SIP leg as separate per-minute charges with the same provider,
    -- the same (empty) model and the same unit. Without it one silently
    -- overwrites the other and every call is billed half its phone time.
    PRIMARY KEY (call_id, ref, provider, model, operation, unit)
);
CREATE INDEX IF NOT EXISTS idx_usage_at       ON localmama.usage(at DESC);
CREATE INDEX IF NOT EXISTS idx_usage_provider ON localmama.usage(provider, at DESC);

-- One model response: what it cost in context, and what it cost in time.
--
-- Separate from the ledger because it answers a different question. The ledger
-- says what a call cost; this says which turn made it expensive. On a realtime
-- model that is the whole diagnostic — input tokens are re-billed against the
-- growing conversation on every response, so a call that costs triple usually
-- looks like an ordinary call whose context never stopped growing.
CREATE TABLE IF NOT EXISTS localmama.call_turns (
    call_id       TEXT NOT NULL,
    ref           TEXT NOT NULL,          -- the provider's response id
    at            DOUBLE PRECISION NOT NULL,   -- seconds since the call was answered
    -- -1 means the response carried no audio. Distinct from 0, and averaging
    -- the two together makes a text-only turn look like the fastest on the call.
    ttft          DOUBLE PRECISION,
    duration      DOUBLE PRECISION,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    -- A barged-in response is still billed for what it generated. A call full
    -- of them is a turn-detection problem presenting as a cost problem.
    cancelled     BOOLEAN NOT NULL DEFAULT false,
    PRIMARY KEY (call_id, ref)
);
CREATE INDEX IF NOT EXISTS idx_call_turns_call ON localmama.call_turns(call_id, at);

-- What a unit costs, and when it cost that.
--
-- Effective-dated rather than overwritten. Editing a price in place silently
-- restates every call ever made at today's rate, which makes last quarter's
-- numbers change every time a vendor does — so a change is a new row plus an
-- `effective_to` on the old one, and history stays what it was.
CREATE TABLE IF NOT EXISTS localmama.rate_card (
    provider       TEXT NOT NULL,
    -- '' matches any model for this provider. A non-empty value matches
    -- exactly, or as a prefix — which is what makes a dated snapshot like
    -- gpt-realtime-2.1-2026-01-15 price correctly against gpt-realtime-2.1
    -- instead of silently costing nothing. Exact beats prefix, and the longest
    -- prefix beats a shorter one; see the `usage_priced` view.
    model          TEXT NOT NULL DEFAULT '',
    -- '' matches any operation. Set it to price two charges that differ only
    -- by what they are for — the LiveKit session and the SIP leg.
    operation      TEXT NOT NULL DEFAULT '',
    unit           TEXT NOT NULL,
    -- The denominator the price is quoted in: 1000000 for a per-million-token
    -- rate, 1 for a per-second one. Stored rather than folded into the price
    -- so a row reads the way the vendor's own pricing page does.
    per_units      DOUBLE PRECISION NOT NULL CHECK (per_units > 0),
    unit_price_usd NUMERIC(20, 12) NOT NULL CHECK (unit_price_usd >= 0),
    effective_from TIMESTAMPTZ NOT NULL DEFAULT '-infinity',
    effective_to   TIMESTAMPTZ,
    -- True for a rate nobody has confirmed yet. Surfaced on the dashboard as
    -- needing attention, so an unset price reads as "unknown" rather than as
    -- "free" — the failure mode where a provider quietly costs nothing on
    -- every chart is the one worth engineering against.
    placeholder    BOOLEAN NOT NULL DEFAULT false,
    source         TEXT NOT NULL DEFAULT '',
    note           TEXT,
    PRIMARY KEY (provider, model, operation, unit, effective_from)
);

-- Published rates as read on 2026-08-03. `effective_from` is -infinity so that
-- calls recorded before this migration price too; when a vendor changes a
-- price, close the row with `effective_to` and insert the new one rather than
-- editing this.
INSERT INTO localmama.rate_card
    (provider, model, operation, unit, per_units, unit_price_usd, placeholder, source, note)
VALUES
    -- OpenAI Realtime. Cached audio is 1/80th of fresh, which is why the
    -- producer of these units subtracts the cached subset out of the total
    -- rather than letting both rates apply to the same tokens.
    ('openai', 'gpt-realtime', '', 'audio_input_tokens',         1000000, 32.00, false, 'developers.openai.com/api/docs/pricing', NULL),
    ('openai', 'gpt-realtime', '', 'audio_input_cached_tokens',  1000000,  0.40, false, 'developers.openai.com/api/docs/pricing', NULL),
    ('openai', 'gpt-realtime', '', 'audio_output_tokens',        1000000, 64.00, false, 'developers.openai.com/api/docs/pricing', NULL),
    ('openai', 'gpt-realtime', '', 'text_input_tokens',          1000000,  4.00, false, 'developers.openai.com/api/docs/pricing', NULL),
    ('openai', 'gpt-realtime', '', 'text_input_cached_tokens',   1000000,  0.40, false, 'developers.openai.com/api/docs/pricing', NULL),
    ('openai', 'gpt-realtime', '', 'text_output_tokens',         1000000, 16.00, false, 'developers.openai.com/api/docs/pricing', NULL),

    ('openai', 'gpt-realtime-2.1', '', 'audio_input_tokens',        1000000, 32.00, false, 'developers.openai.com/api/docs/pricing', NULL),
    ('openai', 'gpt-realtime-2.1', '', 'audio_input_cached_tokens', 1000000,  0.40, false, 'developers.openai.com/api/docs/pricing', NULL),
    ('openai', 'gpt-realtime-2.1', '', 'audio_output_tokens',       1000000, 64.00, false, 'developers.openai.com/api/docs/pricing', NULL),
    ('openai', 'gpt-realtime-2.1', '', 'text_input_tokens',         1000000,  4.00, false, 'developers.openai.com/api/docs/pricing', NULL),
    ('openai', 'gpt-realtime-2.1', '', 'text_input_cached_tokens',  1000000,  0.40, false, 'developers.openai.com/api/docs/pricing', NULL),
    ('openai', 'gpt-realtime-2.1', '', 'text_output_tokens',        1000000, 24.00, false, 'developers.openai.com/api/docs/pricing', NULL),

    ('openai', 'gpt-realtime-2.1-mini', '', 'audio_input_tokens',        1000000, 10.00, false, 'developers.openai.com/api/docs/pricing', NULL),
    ('openai', 'gpt-realtime-2.1-mini', '', 'audio_input_cached_tokens', 1000000,  0.30, false, 'developers.openai.com/api/docs/pricing', NULL),
    ('openai', 'gpt-realtime-2.1-mini', '', 'audio_output_tokens',       1000000, 20.00, false, 'developers.openai.com/api/docs/pricing', NULL),
    ('openai', 'gpt-realtime-2.1-mini', '', 'text_input_tokens',         1000000,  0.60, false, 'developers.openai.com/api/docs/pricing', NULL),
    ('openai', 'gpt-realtime-2.1-mini', '', 'text_input_cached_tokens',  1000000,  0.06, false, 'developers.openai.com/api/docs/pricing', NULL),
    ('openai', 'gpt-realtime-2.1-mini', '', 'text_output_tokens',        1000000,  2.40, false, 'developers.openai.com/api/docs/pricing', NULL),

    -- Transcription, priced per second off OpenAI's own per-minute figure.
    -- The quantity behind this is currently derived rather than reported —
    -- see `agent/metering.py`; the LiveKit plugin discards the usage OpenAI
    -- sends on the transcription-completed event.
    ('openai', 'gpt-4o-transcribe',      'transcription', 'seconds', 1, 0.000100, false, 'developers.openai.com/api/docs/pricing', '$0.006/min'),
    ('openai', 'gpt-4o-mini-transcribe', 'transcription', 'seconds', 1, 0.000050, false, 'developers.openai.com/api/docs/pricing', '$0.003/min'),

    -- LiveKit Cloud: the agent session and the SIP leg, each $0.01/min. Two
    -- rows and not one, so that moving telephony to another carrier re-rates
    -- only the half that moved.
    ('livekit', '', 'session',   'seconds', 1, 0.000166666667, false, 'livekit.com/pricing', '$0.01/min agent session'),
    ('livekit', '', 'telephony', 'seconds', 1, 0.000166666667, false, 'livekit.com/pricing', '$0.01/min telephony'),

    -- Sarvam bills translate and transliterate per character at one rate.
    ('sarvam', '', '', 'characters', 10000, 0.23, false, 'sarvam.ai/api-pricing', 'Rs 20 / 10k characters'),

    -- CampaignBot is a reseller and publishes no rate, so this is knowingly
    -- wrong rather than accidentally missing. `placeholder` puts it on the
    -- dashboard as needing attention; replace it from an invoice with:
    --   UPDATE localmama.rate_card SET effective_to = now()
    --    WHERE provider = 'campaignbot' AND placeholder;
    --   INSERT INTO localmama.rate_card (provider, operation, unit, per_units,
    --                                    unit_price_usd, effective_from, source)
    --   VALUES ('campaignbot', 'handoff', 'messages', 1, <rate>, now(), '<invoice>');
    ('campaignbot', '', 'handoff', 'messages', 1, 0.0, true, '', 'PLACEHOLDER — set from your CampaignBot invoice')
ON CONFLICT DO NOTHING;

-- Every metered quantity with the price that applied when it happened.
--
-- The rate is chosen by specificity, and the ordering is the whole point: an
-- exact model match beats a prefix, a longer prefix beats a shorter one, and a
-- rule naming an operation beats one that does not. Without that, gpt-realtime
-- would win the row for gpt-realtime-2.1-mini and undercharge it threefold.
--
-- A quantity with no matching rate prices as NULL, never as zero. `unpriced`
-- carries that out to the dashboard, because a provider silently costing
-- nothing on every chart is the failure this whole table exists to prevent.
CREATE OR REPLACE VIEW localmama.usage_priced AS
SELECT u.call_id, u.ref, u.source, u.provider, u.model, u.operation, u.unit,
       u.quantity, u.estimated, u.ok, u.latency_ms, u.error, u.at,
       r.unit_price_usd, r.per_units, r.placeholder, r.model AS rate_model,
       CASE WHEN r.unit_price_usd IS NULL THEN NULL
            ELSE (u.quantity / r.per_units) * r.unit_price_usd
       END AS cost_usd,
       (r.unit_price_usd IS NULL AND u.quantity > 0) AS unpriced,
       -- Priced by prefix rather than by name: a model nobody has entered a
       -- rate for, inheriting its ancestor's. Correct for a dated snapshot of
       -- a known model, and a silent mispricing the day a genuinely new model
       -- ships under a familiar prefix. Surfaced so the second case is a line
       -- on the dashboard rather than a number that quietly drifts.
       (r.model <> '' AND r.model <> u.model) AS rate_inherited
FROM localmama.usage u
LEFT JOIN LATERAL (
    SELECT c.unit_price_usd, c.per_units, c.placeholder, c.model
    FROM localmama.rate_card c
    WHERE c.provider = u.provider
      AND c.unit = u.unit
      AND (c.model = '' OR u.model = c.model OR u.model LIKE c.model || '%')
      AND (c.operation = '' OR c.operation = u.operation)
      AND u.at >= c.effective_from
      AND (c.effective_to IS NULL OR u.at < c.effective_to)
    ORDER BY (u.model = c.model) DESC,
             length(c.model) DESC,
             (c.operation <> '') DESC,
             c.effective_from DESC
    LIMIT 1
) r ON TRUE;

-- One row per call: what it cost, split by provider, and how much of that
-- number to trust.
CREATE OR REPLACE VIEW localmama.call_cost AS
WITH per_provider AS (
    SELECT call_id,
           provider,
           sum(cost_usd)                                       AS cost_usd,
           count(*) FILTER (WHERE unpriced)                    AS unpriced,
           count(*) FILTER (WHERE estimated AND quantity > 0)  AS estimated,
           count(*) FILTER (WHERE placeholder)                 AS placeholder,
           count(*) FILTER (WHERE rate_inherited)              AS inherited
    FROM localmama.usage_priced
    GROUP BY call_id, provider
)
SELECT call_id,
       round(coalesce(sum(cost_usd), 0)::numeric, 8)                    AS cost_usd,
       sum(unpriced)                                                    AS unpriced_units,
       sum(estimated)                                                   AS estimated_units,
       sum(placeholder)                                                 AS placeholder_units,
       sum(inherited)                                                   AS inherited_units,
       jsonb_object_agg(provider,
                        round(coalesce(cost_usd, 0)::numeric, 8))       AS by_provider
FROM per_provider
GROUP BY call_id;
