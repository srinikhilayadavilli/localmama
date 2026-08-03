-- Close the retention hole in the raw event log.
--
-- `store.expire_transcripts` has always cleared `localmama.leads.transcript`
-- past the retention window, and the reasoning it states is right: a transcript
-- is everything the caller said, which is personal data under the DPDP Act, and
-- once the confidence audit has run it is a liability rather than an asset.
--
-- But the same transcript arrives a second time, in `call_events.payload` — the
-- raw log keeps every event exactly as received, and the `call.ended` event
-- carries the whole transcript inside it. Nothing swept that. So the sweep
-- reported clearing transcripts, the lead row genuinely lost them, and a
-- complete copy of every word every caller had ever said stayed in the event
-- log indefinitely. A deletion that leaves the data in the next table over is
-- worse than no deletion at all, because it stops anyone looking.
--
-- The rows themselves are kept and redacted in place rather than deleted:
-- `event_id` is the deduplication key, and an event whose row has gone would be
-- treated as new if it were ever replayed, re-applying a capture and re-running
-- a pipeline. Only the transcript is removed. Everything else in the payload —
-- the captured values, the phone number — mirrors columns on the lead row that
-- are deliberately retained as the business record, so removing them here would
-- not be a retention policy, just an inconsistency pointing the other way.

-- The sweep runs every five minutes forever and matches almost nothing once it
-- has caught up. A partial index keeps that cheap: it holds only the events
-- that still carry a transcript, and empties itself as they are redacted.
CREATE INDEX IF NOT EXISTS idx_call_events_transcript_gc
    ON localmama.call_events(received_at)
    WHERE jsonb_exists(payload, 'transcript');
