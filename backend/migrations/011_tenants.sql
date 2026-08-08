-- Who a lead belongs to, and how the system works it out.
--
-- `agent_id` has been on every lead since 001, and it has always been the same
-- string, because it came from `BRAIN_AGENT_ID` — a deployment fact, not a
-- property of the call. One deployment, one tenant, by construction.
--
-- The routing key was already being captured and thrown away: `leads.dialled`,
-- the number the caller rang, described in `contract/schema.py` as "for routing
-- and attribution". Nothing routed on it. These two tables are what cash that.

CREATE TABLE IF NOT EXISTS localmama.tenants (
    agent_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    city        TEXT,
    -- Inactive keeps the history readable while stopping new work: leads
    -- already filed stay attributed, and the sweep skips the tenant. Deleting
    -- a tenant would orphan every lead they ever received.
    active      BOOLEAN NOT NULL DEFAULT true,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per phone number, not a column on the tenant: a business may run
-- several numbers — a service line and a sales line — and each can point at a
-- different agent later. The DID is the primary key because a number can only
-- ever belong to one tenant at a time, and that is the invariant routing
-- depends on.
CREATE TABLE IF NOT EXISTS localmama.tenant_numbers (
    did         TEXT PRIMARY KEY,
    agent_id    TEXT NOT NULL REFERENCES localmama.tenants(agent_id),
    active      BOOLEAN NOT NULL DEFAULT true,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tenant_numbers_agent
    ON localmama.tenant_numbers(agent_id);

-- The tenant that has existed all along, named rather than implied.
INSERT INTO localmama.tenants (agent_id, name, city)
VALUES ('localmama', 'Local Mama', 'Hyderabad')
ON CONFLICT (agent_id) DO NOTHING;

-- Every agent_id already on a lead becomes a tenant, so the foreign key below
-- cannot fail on history. In practice this is the one row above; it is written
-- as a query because "in practice" is how orphans happen.
INSERT INTO localmama.tenants (agent_id, name)
SELECT DISTINCT agent_id, agent_id FROM localmama.leads
 WHERE agent_id IS NOT NULL
ON CONFLICT (agent_id) DO NOTHING;

-- The live DID, from setup/inbound-trunk.json.
INSERT INTO localmama.tenant_numbers (did, agent_id)
VALUES ('+918071581496', 'localmama')
ON CONFLICT (did) DO NOTHING;

-- A subscription now has to belong to a tenant that exists. Added after the
-- seeds above for that reason.
ALTER TABLE localmama.webhook_subscriptions
    DROP CONSTRAINT IF EXISTS webhook_subscriptions_agent_fk;
ALTER TABLE localmama.webhook_subscriptions
    ADD CONSTRAINT webhook_subscriptions_agent_fk
    FOREIGN KEY (agent_id) REFERENCES localmama.tenants(agent_id);

COMMENT ON TABLE localmama.tenants IS
    'Businesses whose calls this deployment answers. Resolved from the dialled '
    'DID via tenant_numbers; BRAIN_AGENT_ID is the fallback for an unmapped number.';
