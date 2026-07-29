# Telephony — attaching a phone number

Turns the agent from "browser callers only" into something a person can dial.
It also switches on two features that are dormant without a phone number: the
WhatsApp handoff (no recipient until now) and returning-caller memory (keyed by
a hash of the caller id).

Everything below happens in the **`localmama` LiveKit project**. Check first:

```bash
lk project list          # the * marks the active one
lk sip inbound list --project localmama
```

## 1. Buy the number (dashboard — needs a card)

Not scriptable: `lk sip` manages trunks and rules, but numbers are bought in
the dashboard.

**LiveKit Cloud → Telephony → Numbers → Buy a number**, in the `localmama`
project. Pick a region your testers can dial cheaply; an Indian number keeps
the media path close to the `ap-south` worker.

Buying through LiveKit Cloud usually **creates the inbound trunk for you** — so
check step 2 before creating a second one.

> Bringing your own number instead (Twilio/Telnyx/Plivo): point that provider's
> SIP trunk at `sip:<project>.sip.livekit.cloud` and fill `auth_username` /
> `auth_password` in `inbound-trunk.json`.

## 2. Inbound trunk

```bash
lk sip inbound list --project localmama          # did buying the number create one?
```

If it did, note the `ST_xxxx` id and skip ahead. Otherwise put your number in
`setup/inbound-trunk.json` and:

```bash
lk sip inbound create setup/inbound-trunk.json --project localmama
lk sip inbound list --project localmama          # note the ST_xxxx id
```

## 3. Dispatch rule

Put the `ST_xxxx` id into `trunk_ids` in `setup/dispatch-rule.json`, then:

```bash
lk sip dispatch create setup/dispatch-rule.json --project localmama
lk sip dispatch list --project localmama
```

Two things in that file are load-bearing:

- **`dispatchRuleIndividual`** gives every call its own room. A shared room
  would drop unrelated callers into one conversation with one agent.
- **`agent_name: local-mama-cloud`** must match `LIVEKIT_AGENT_NAME` on the
  deployed worker. The worker is *named*, so it only joins rooms dispatched to
  it — that is what stops it wandering into another agent's calls, and it is
  also why a missing dispatch rule means the caller hears silence.

## 4. Call it

Dial the number. Expect in `lk agent logs --project localmama`:

```
SIP call from +91****0092 to +91XXXXXXXXXX
session=xxxxxxxx  room=local-mama_xxxx  caller=+91****0092  starting
```

`caller=(anonymous)` means the caller id did not arrive — check the trunk, not
the agent. Numbers are masked in logs on purpose.

## 5. WhatsApp, once calls are landing

The handoff has been wired all along and skipping for want of a recipient. With
a real caller id it needs only credentials:

```ini
WHATSAPP_ENABLED=true
WHATSAPP_API_KEY=...
WHATSAPP_TEMPLATE_NAME=we_found_these_for_you
```

Set them as agent secrets (`lk agent update-secrets`), not in the image.

## Rollback

```bash
lk sip dispatch delete <SDR_xxxx> --project localmama
```

Deleting the dispatch rule stops calls reaching the agent while leaving the
number and trunk intact — the quickest way to stand down without giving up the
DID.
