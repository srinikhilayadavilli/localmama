-- Local Mama lead backend — initial schema.
--
-- Run by `python -m backend.migrate`, which Render executes as a pre-deploy
-- command. Deliberately NOT run at first use: the agent used to issue
-- CREATE TABLE / ALTER TABLE lazily from inside a live call's save path, once
-- per job process, which meant concurrent DDL against Neon at the exact moment
-- a caller was hanging up.
--
-- `localmama` is this product's schema. The knowledge base lives in `utter`
-- and is shared with the Vaani bridge, which owns its migrations — nothing
-- here touches it.

CREATE SCHEMA IF NOT EXISTS localmama;

-- The deployed agent already wrote a `localmama.leads`, keyed by `session_id`
-- and shaped for the pipeline that no longer exists. `CREATE TABLE IF NOT
-- EXISTS` below would find it, do nothing, and leave the old columns in place
-- — and then every insert would fail on a missing `call_id`, at the moment a
-- caller hung up.
--
-- So it is renamed rather than replaced. Those rows are real leads from real
-- calls: few enough to move by hand if anyone wants them, and not worth
-- dropping to save a table.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'localmama'
          AND table_name = 'leads'
          AND column_name = 'session_id'
    ) THEN
        ALTER TABLE localmama.leads RENAME TO leads_v1;
        RAISE NOTICE 'renamed the legacy localmama.leads to leads_v1';
    END IF;
END $$;

-- Every event the agent has sent us, exactly as received.
--
-- The raw log is kept rather than only the assembled lead, for two reasons: it
-- is the deduplication key (an agent may retry any event freely), and when a
-- lead comes out wrong it is the only record of what the agent actually
-- reported versus what we made of it.
CREATE TABLE IF NOT EXISTS localmama.call_events (
    event_id    TEXT PRIMARY KEY,
    call_id     TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    type        TEXT NOT NULL,
    payload     JSONB NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_call_events_call ON localmama.call_events(call_id, seq);

-- One row per call, assembled from the events.
--
-- Raw and normalised values are both kept. The raw ones are what the caller
-- actually said and the only thing that can be re-processed when the
-- normaliser improves; the English ones are what the catalogue is matched
-- against and what a human reads.
CREATE TABLE IF NOT EXISTS localmama.leads (
    call_id       TEXT PRIMARY KEY,
    agent_id      TEXT NOT NULL,
    caller_phone  TEXT,
    dialled       TEXT,
    language      TEXT,

    raw           JSONB NOT NULL DEFAULT '{}'::jsonb,
    name          TEXT,
    service       TEXT,
    city          TEXT,

    status        TEXT NOT NULL DEFAULT 'in_progress',
    -- NULL means no read-back happened, which is not the same as the caller
    -- saying the details were wrong. The review queue runs on that distinction.
    confirmed     BOOLEAN,
    confidence    JSONB NOT NULL DEFAULT '{}'::jsonb,
    needs_review  BOOLEAN NOT NULL DEFAULT false,
    review_reason TEXT,

    vendors       JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- The outbox. A lead whose WhatsApp handoff has not succeeded is work
    -- still owed to a customer, so the state lives on the lead rather than in
    -- a process that dies when the call ends.
    whatsapp_status   TEXT NOT NULL DEFAULT 'pending',
    whatsapp_attempts INTEGER NOT NULL DEFAULT 0,
    whatsapp_error    TEXT,
    whatsapp_at       TIMESTAMPTZ,

    transcript    JSONB NOT NULL DEFAULT '[]'::jsonb,
    turn_gaps     JSONB NOT NULL DEFAULT '[]'::jsonb,
    started_at    TIMESTAMPTZ,
    ended_at      TIMESTAMPTZ,
    processed_at  TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_leads_agent  ON localmama.leads(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_leads_phone  ON localmama.leads(caller_phone);
CREATE INDEX IF NOT EXISTS idx_leads_review ON localmama.leads(agent_id, created_at DESC)
    WHERE needs_review;
CREATE INDEX IF NOT EXISTS idx_leads_outbox ON localmama.leads(agent_id, whatsapp_status)
    WHERE whatsapp_status IN ('pending', 'sending');
-- Transcript expiry sweeps on this; a partial index keeps it cheap once most
-- rows have already been cleared.
CREATE INDEX IF NOT EXISTS idx_leads_transcript_gc ON localmama.leads(ended_at)
    WHERE transcript <> '[]'::jsonb;
