-- The provider's id for the message we sent.
--
-- `whatsapp_status = 'sent'` only means CampaignBot accepted the request. A
-- caller reported a message that never arrived on a lead marked sent, and
-- there was nothing to hand the vendor to ask what became of it.

ALTER TABLE localmama.leads
    ADD COLUMN IF NOT EXISTS whatsapp_message_id TEXT;
