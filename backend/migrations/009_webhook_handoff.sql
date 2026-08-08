-- A second handoff channel: a signed webhook, alongside WhatsApp.
--
-- New columns rather than renamed ones, because **both channels run at once**.
-- WhatsApp stays the channel the caller actually hears from until the webhook
-- has been proven against a real receiver; the two are tracked separately and
-- swept separately, so a lead whose message landed and whose webhook 500'd is
-- owed the webhook alone. Share one status column and the next sweep messages
-- that customer a second time to fix a problem they never had.
--
-- When WhatsApp is retired: drop its columns and `idx_leads_outbox` in a
-- migration of their own, once nothing reads them.

ALTER TABLE localmama.leads
    ADD COLUMN IF NOT EXISTS handoff_status   TEXT NOT NULL DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS handoff_attempts INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS handoff_error    TEXT,
    ADD COLUMN IF NOT EXISTS handoff_at       TIMESTAMPTZ,
    -- The receiver's HTTP status. `sent` means it answered 2xx and nothing
    -- more: what it then did with the lead is its own business, and a webhook
    -- that 200s into a black hole is indistinguishable from one that works.
    ADD COLUMN IF NOT EXISTS handoff_response INTEGER;

-- Every lead that already exists predates the webhook, so the default above
-- would make all of them owed at once and the first sweep would POST the entire
-- history to a brand-new endpoint. Close them instead. Anything genuinely still
-- owed can be re-opened deliberately:
--
--   UPDATE localmama.leads SET handoff_status = 'pending', handoff_attempts = 0
--    WHERE call_id IN (...);
UPDATE localmama.leads
   SET handoff_status = 'skipped',
       handoff_error  = 'predates the webhook handoff'
 WHERE handoff_status = 'pending';

-- Mirrors idx_leads_outbox, which indexes the WhatsApp columns and still
-- serves that channel's claim. The webhook's claim scans this predicate.
CREATE INDEX IF NOT EXISTS idx_leads_handoff_outbox
    ON localmama.leads(agent_id, handoff_status)
    WHERE handoff_status IN ('pending', 'sending');

-- Priced at zero, not left unpriced. Delivering a lead now costs an HTTP
-- request to our own receiver, so the true rate is nothing — but the units are
-- still metered, because latency and failure on this hop are worth seeing. An
-- absent rate would report as `unpriced` and drag down the coverage figure that
-- exists to tell us when the ledger is lying.
INSERT INTO localmama.rate_card
    (provider, model, operation, unit, per_units, unit_price_usd, placeholder, source, note)
VALUES
    ('webhook', '', 'handoff', 'deliveries', 1, 0, false, 'self-hosted',
     'our own endpoint; metered for latency and failures, not for spend')
ON CONFLICT DO NOTHING;
