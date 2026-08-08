-- Where the lead webhook points, configured by the customer rather than by a
-- redeploy.
--
-- `WEBHOOK_URL` and `WEBHOOK_SECRET` were the right shape for one destination
-- set by us. They are the wrong shape the moment a business sets their own —
-- an environment variable cannot be edited from a dashboard, and a customer
-- who rotates their secret should not be waiting on our deploy queue.
--
-- In `localmama`, not the shared `utter` schema. `utter.webhook_subscriptions`
-- already exists and belongs to the Vaani bridge; pointing lead delivery at it
-- would couple two products that only share a database by accident of hosting.
--
-- The env vars remain as the fallback, so a deployment with no rows behaves
-- exactly as it does today.

CREATE TABLE IF NOT EXISTS localmama.webhook_subscriptions (
    id          TEXT PRIMARY KEY
                DEFAULT 'sub_' || replace(gen_random_uuid()::text, '-', ''),

    -- Which agent's leads go here. One column, and the whole of multi-tenancy
    -- later: today every row is `localmama`, and the day a second business
    -- answers its own number this is what tells their leads apart.
    agent_id    TEXT NOT NULL DEFAULT 'localmama',

    url         TEXT NOT NULL CHECK (url ~ '^https?://'),

    -- **Generated, not optional.** The dashboard offers the secret as optional
    -- and it must not be: this payload carries a caller's name and phone
    -- number, and an unsigned endpoint believes whoever finds the URL. Left
    -- blank, the database mints one — so "optional" to a customer means "we
    -- will make you one", never "you will be sent unsigned traffic".
    secret      TEXT NOT NULL
                DEFAULT 'lm_whsec_' || replace(gen_random_uuid()::text, '-', '')
                        || replace(gen_random_uuid()::text, '-', ''),

    -- Unused today: every delivery is `lead.captured`. Here because the shape
    -- of a subscription is hard to change once customers hold one, and adding
    -- a filter later is easier than explaining why an endpoint started getting
    -- events it never asked for.
    events      TEXT[],

    active      BOOLEAN NOT NULL DEFAULT true,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT webhook_secret_is_real CHECK (length(secret) >= 16)
);

-- **One active subscription per agent**, enforced rather than assumed.
--
-- Delivering to several endpoints sounds like a small step and is not: the
-- lead carries one `handoff_status`, so with two subscriptions a failure at
-- either one leaves the lead `pending`, and the next sweep re-delivers to the
-- endpoint that already succeeded. Fan-out needs delivery state per
-- subscription — its own table — and until that exists this index is what
-- stops the schema promising something the sweep cannot honour.
CREATE UNIQUE INDEX IF NOT EXISTS idx_webhook_one_active_per_agent
    ON localmama.webhook_subscriptions(agent_id)
    WHERE active;

COMMENT ON TABLE localmama.webhook_subscriptions IS
    'Customer-configured lead delivery endpoints. One active row per agent; '
    'falls back to WEBHOOK_URL/WEBHOOK_SECRET when empty.';
