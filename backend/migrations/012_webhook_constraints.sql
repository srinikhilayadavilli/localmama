-- Three corrections to 010 and 011, found by review after they had already
-- been applied. Written as a migration rather than edits to those files,
-- because they have run against production and a checksum that changes under
-- an applied migration is worse than an extra file.

-- 1. **https only.**
--
-- The original CHECK accepted `^https?://`, so a customer typing an http URL
-- into the dashboard got a row that delivers their callers' names and phone
-- numbers in cleartext — and `follow_redirects` is on, so the request may be
-- carried onward before anyone notices. There is no reason for a lead webhook
-- to be plaintext, and no way for a customer to know that from the dialog.
ALTER TABLE localmama.webhook_subscriptions
    DROP CONSTRAINT IF EXISTS webhook_subscriptions_url_check;
ALTER TABLE localmama.webhook_subscriptions
    ADD CONSTRAINT webhook_url_is_https CHECK (url ~ '^https://');

-- 2. **No hard-coded default tenant.**
--
-- `agent_id DEFAULT 'localmama'` was a literal in SQL while every lookup uses
-- `settings.brain_agent_id`. A deployment with a different `BRAIN_AGENT_ID`
-- would write subscriptions under one name and read them under another, and
-- deliver nothing — silently, since an empty result is indistinguishable from
-- "not configured yet". Requiring the column makes the caller say which
-- business it is, which is the only party that actually knows.
ALTER TABLE localmama.webhook_subscriptions
    ALTER COLUMN agent_id DROP DEFAULT;

-- 3. **Tenants exist for every agent that is referenced, not only for those
--    that have already taken a call.**
--
-- 011 backfilled `tenants` from `leads` alone, then added a foreign key from
-- `webhook_subscriptions`. A subscription written for a business onboarded
-- before its first call — the normal order — has no matching tenant, so the
-- foreign key would abort the whole pre-deploy migration. It survived only
-- because the subscriptions table was empty at the time.
INSERT INTO localmama.tenants (agent_id, name)
SELECT DISTINCT s.agent_id, s.agent_id
  FROM localmama.webhook_subscriptions s
 WHERE NOT EXISTS (
       SELECT 1 FROM localmama.tenants t WHERE t.agent_id = s.agent_id)
ON CONFLICT (agent_id) DO NOTHING;

-- 4. **A DID is stored canonically**, so the digits comparison the resolver
-- does has something predictable on the other side. The trunk's formatting is
-- not ours to control; what we store is.
UPDATE localmama.tenant_numbers
   SET did = '+' || regexp_replace(did, '\D', '', 'g')
 WHERE did !~ '^\+[0-9]+$';

ALTER TABLE localmama.tenant_numbers
    DROP CONSTRAINT IF EXISTS tenant_number_is_e164;
ALTER TABLE localmama.tenant_numbers
    ADD CONSTRAINT tenant_number_is_e164 CHECK (did ~ '^\+[0-9]{8,15}$');
