-- The trade the agent worked out from a description.
--
-- Callers do not think in catalogue categories. They say "my tap is leaking",
-- "my geyser is not working", "I want to look good for my wedding". Matched
-- literally those reached a plumber by luck, a co-working space by accident,
-- and a devotional singer by nothing at all.
--
-- Kept apart from `service` and `service_said` because it is a different kind
-- of thing: not what they asked for and not what we filed it as, but what we
-- understood them to mean. A lead matched this way is flagged for review, so
-- the guess can be watched before it is trusted.

ALTER TABLE localmama.leads
    ADD COLUMN IF NOT EXISTS service_inferred TEXT;
