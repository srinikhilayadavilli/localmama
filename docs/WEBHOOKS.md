# Webhooks

How a lead reaches a customer's own system, and what the dashboard must write
for that to happen.

---

## The direction

**The customer supplies the URL. We supply the secret.** Nobody is issued a
webhook URL by us — a webhook is us calling them, at an address they own.

| | Who provides it |
|---|---|
| Endpoint URL | the customer |
| Signing secret | generated here, shown once |
| Which customer a lead belongs to | the DID the caller dialled |

A customer with no system of their own never sees this. They get WhatsApp and
the dashboard. The webhook is opt-in for customers with somewhere to put a lead.

---

## Tables

```
localmama.tenants           agent_id (pk), name, city, active
localmama.tenant_numbers    did (pk), agent_id → tenants, active
localmama.webhook_subscriptions
                            id (pk), agent_id → tenants, url, secret,
                            events, active
```

`did` is the primary key of `tenant_numbers` because a number can belong to one
tenant at a time, and routing depends on that being true. Store it in E.164
(`+918071581496`) — a `CHECK` enforces it. Resolution compares digits only,
because the trunk may present the same number as `91…`, `0…` or
`sip:+91…@host`, and an exact string match silently files those calls under the
default tenant.

A **unique partial index** allows one active subscription per tenant.
Deactivate-then-insert is how rotation works. Fan-out to several endpoints is
deliberately not supported: a lead carries a single `handoff_status`, so with
two endpoints a failure at either leaves it `pending` and the next sweep
re-delivers to the one that already succeeded. Real fan-out needs delivery
state per subscription — its own table.

---

## What the dashboard must write

The "Add Webhook" dialog currently writes to **`utter.webhook_subscriptions`**,
which belongs to the Vaani bridge. Lead delivery reads
**`localmama.webhook_subscriptions`**. Until the dialog is repointed, saving it
has no effect on where leads go.

```sql
-- with a secret the customer typed
INSERT INTO localmama.webhook_subscriptions (agent_id, url, secret)
VALUES ($1, $2, $3);

-- blank: omit the column entirely
INSERT INTO localmama.webhook_subscriptions (agent_id, url)
VALUES ($1, $2);
```

Three rules the dashboard has to honour:

**Send `agent_id`.** It is which customer this is, and it is required — the
column had a hard-coded `'localmama'` default, which meant a row written
without it silently belonged to whoever holds the deployment's default tenant.
The insert now fails instead, which is the correct outcome: the dashboard is
the only party that knows whose subscription this is.

**Omit the `secret` column entirely when the user leaves the field blank** —
two statements, not one with `NULLIF`. Postgres applies a column default only
when the column is absent from the INSERT; an explicit NULL is inserted as
NULL and fails `NOT NULL`. An empty string fails the length `CHECK`. Both
refusals are deliberate: "optional" in the dialog must mean *we will make you
one*, never *you will be sent unsigned traffic*.

**Deactivate before inserting.** One active row per tenant is enforced by an
index, so a second insert fails rather than silently splitting delivery:

```sql
UPDATE localmama.webhook_subscriptions SET active = false
 WHERE agent_id = $1 AND active;
```

Show the secret once, on creation. It cannot be recovered afterwards — only
replaced.

---

## What a receiver gets

```
POST <their url>
Content-Type: application/json
X-Localmama-Signature: v1=<hmac-sha256 hex>
X-Localmama-Timestamp: <unix seconds>
X-Localmama-Version:   1
```

```json
{
  "version": "1",
  "event": "lead.captured",
  "call_id": "…",
  "caller":  { "phone": "+91…", "language": "telugu", "dialled": "+91…" },
  "lead":    { "name": "Ravi", "service": "ac repair",
               "service_said": "…", "service_inferred": null,
               "city": "Madapur", "subject": "ac repair" },
  "vendors": [ { "title": "…", "phone": "+91…", "category": "…", "city": "…" } ],
  "confidence": { "name": {"score": 0.9, "evidence": true}, "…": {} },
  "review":  { "needs_review": true, "reason": "no transcript to check against" },
  "call":    { "status": "completed", "confirmed": true,
               "started_at": "…", "ended_at": "…" }
}
```

**The service appears three ways** because they differ and the difference
matters: `service` is the canonical label that was matched, `service_said` is
the caller's own phrasing, and `service_inferred` is what the agent understood
when their words matched nothing in the catalogue.

**`confidence` is per field, not one number.** A lead can be certain about the
service and unsure of the name, and a receiver routing on an average would
never know which.

**Never included:** the transcript, and the WhatsApp columns. The transcript is
everything the caller said — personal data under the DPDP Act — and it stays
here under a retention sweep rather than being copied to a third party with its
own retention rules and its own breaches. A receiver that needs it can ask for
the `call_id`.

---

## Verifying

Recompute the HMAC over `"<timestamp>.<raw body>"` — the timestamp is inside
the signed string, not merely beside it, so a captured request cannot be
replayed later under a fresh header.

```python
expected = "v1=" + hmac.new(secret.encode(),
                            f"{ts}.".encode() + raw_body,
                            hashlib.sha256).hexdigest()
```

Compare with `hmac.compare_digest`, and reject anything whose timestamp is more
than a few minutes old.

---

## Responses, and what they mean to us

| You return | We do |
|---|---|
| **2xx** | done — recorded `sent` with your status code |
| **4xx** | give up. You are refusing this lead, not failing to receive it |
| **429** | retry in five minutes. You are asking for later |
| **5xx**, timeout, connection error | retry |
| **3xx** | followed automatically |

`sent` means you answered 2xx and nothing more. What you then do with the lead
is your business — a webhook that 200s into a black hole is indistinguishable
from one that works.

Retries come from a sweep every five minutes, one attempt per lead per pass,
stopping after `OUTBOX_MAX_ATTEMPTS` (25) — roughly two hours of trying.
Deliveries are **not** guaranteed to be unique: a delivery that succeeded but
whose result we failed to record is retried. Treat `call_id` as an idempotency
key.
