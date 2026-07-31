-- What the caller actually called the trade, in English.
--
-- `service` is canonical, because matching needs one label per trade: someone
-- asking for a "beauty parlor" is stored as `salon` so the query reaches the
-- salons. The WhatsApp message then read "here are some salon options" back to
-- a caller who never said the word — they said one thing and the message said
-- another, which reads as though nobody was listening.
--
-- So both are kept. `service` is what we match on; `service_said` is what the
-- customer reads.

ALTER TABLE localmama.leads
    ADD COLUMN IF NOT EXISTS service_said TEXT;
