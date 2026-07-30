-- Businesses the caller asked about by name, during the call.
--
-- `/v1/vendors` answered these and immediately forgot them: the lookup carried
-- no call_id, so by the time the pipeline built the WhatsApp message it had no
-- idea the caller had ever asked. The one thing they explicitly wanted was the
-- one thing that never reached them in writing.
--
-- Separate from `vendors`, which holds what we matched *for them* off the
-- service. These were requested by name, so they lead the message.

ALTER TABLE localmama.leads
    ADD COLUMN IF NOT EXISTS asked_vendors JSONB NOT NULL DEFAULT '[]'::jsonb;
