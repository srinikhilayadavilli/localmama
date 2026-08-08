-- Price the realtime model, which has been costing zero.
--
-- `livekit-agents` fills a usage record's `provider` from the client's base URL
-- for some plugins, so the realtime model arrived as `api.openai.com` while the
-- transcription path — which sets it explicitly — arrived as `openai`. The rate
-- card prices `openai`. A hostname prices nothing.
--
-- The result was not a gap anybody would notice from the total: call 5467f3fb
-- reported $0.0222 with `gpt-realtime` at **$0.0000** against 613 audio output
-- tokens and 4,159 fresh text input tokens. Only the coverage line said so —
-- 30%, naming all six unpriced units.
--
-- Renaming the recorded units rather than teaching the rate card a hostname.
-- These are the same provider; one of the two names is simply wrong, and
-- carrying both forward would mean every future rate change had to be written
-- twice or silently apply to half the data. `agent/metering.py` now normalises
-- at the source, so this is a one-off correction of what is already stored.
--
-- Safe to run more than once, and safe on a deployment that never recorded the
-- hostname: both are no-ops. Verified beforehand that no row would collide with
-- the unique key (call_id, ref, provider, model, operation, unit).

UPDATE localmama.usage
   SET provider = 'openai'
 WHERE provider = 'api.openai.com'
   AND NOT EXISTS (
       SELECT 1 FROM localmama.usage other
        WHERE other.call_id   = localmama.usage.call_id
          AND other.ref       = localmama.usage.ref
          AND other.model     = localmama.usage.model
          AND other.operation = localmama.usage.operation
          AND other.unit      = localmama.usage.unit
          AND other.provider  = 'openai'
   );
