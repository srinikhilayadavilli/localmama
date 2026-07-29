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

## Current state (Vobiz DID, live)

| | |
|---|---|
| DID | `+918071581496` (Vobiz) |
| Inbound trunk | `ST_xZhVG8X6KYPR` — "local-mama inbound (vobiz)" |
| Dispatch rule | `SDR_cn5WSYVL2pTD` → agent `local-mama-cloud` |
| Rooms | `local-mama_<caller>_<random>`, one per call |

**What Vobiz must be told** — this is the remaining half, and it is on their
side, not ours:

```
Origination URI:  4xe6vzxnl90.sip.livekit.cloud
Transport:        UDP
```

That host is the LiveKit **project id** minus its `p_` prefix
(`p_4xe6vzxnl90`), NOT the project subdomain. The two are unrelated: this
project's subdomain is `localmama-iuyu1598`, and the sibling Vaani project is
`vaaniai-ulbv5i95` against a SIP host of `50bjr5kmu1p.sip.livekit.cloud`.

Do not try to confirm the host with DNS. `*.sip.livekit.cloud` is a wildcard
onto one shared SIP edge, so **every** name under it resolves — including a
wrong one — and routing is decided by the SIP domain, not the address. A bad
host looks perfectly healthy right up until calls silently fail to arrive.

Until that route exists at Vobiz, dialling the number reaches Vobiz and stops
there — LiveKit never sees it, and the agent logs stay silent. Nothing on our
side can detect that.

### Security: the trunk is currently open

It accepts a call for that number from *any* source address, because we do not
yet know how Vobiz authenticates. Anyone who learns the number and the SIP host
could place calls into it, and every one of them costs OpenAI and Sarvam usage.

Close it as soon as Vobiz tells you which they use:

- **IP allowlist** — they send from fixed signalling IPs:
  `lk sip inbound update ST_xZhVG8X6KYPR` with `allowed_addresses`
- **Digest auth** — they authenticate: fill `auth_username` / `auth_password`
  in `setup/inbound-trunk.json` and recreate

## Buying another number (dashboard — needs a card)

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
