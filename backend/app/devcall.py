"""Set up a browser test call against the LiveKit worker.

Mints a room token and, when the worker is registered under a name, dispatches
it explicitly — the LiveKit Agents Playground cannot do that for you, so a
named agent simply never joins and the room sits silent.

    python -m backend.app.devcall

Then paste the printed URL and token into https://agents-playground.livekit.io
(choose "Manual" connection) and talk to Mami with your own microphone.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .config import settings
from .logger import setup_logging

PLAYGROUND = "https://agents-playground.livekit.io"


async def setup_call(room: str, identity: str, dispatch: bool) -> int:
    if not settings.livekit_configured:
        print(
            "LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET are not all set "
            "in .env.",
            file=sys.stderr,
        )
        return 1

    from livekit import api

    lk = api.LiveKitAPI(
        url=settings.livekit_url.replace("wss://", "https://"),
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret,
    )
    try:
        try:
            await lk.room.create_room(api.CreateRoomRequest(name=room))
        except Exception:
            pass  # already exists — fine

        agent_name = settings.livekit_agent_name.strip()
        if dispatch and agent_name:
            try:
                result = await lk.agent_dispatch.create_dispatch(
                    api.CreateAgentDispatchRequest(agent_name=agent_name, room=room)
                )
                print(f"dispatched agent {agent_name!r} to room {room!r} ({result.id})")
            except Exception as exc:  # noqa: BLE001
                print(f"could not dispatch agent: {exc}", file=sys.stderr)
                print(
                    "Is the worker running? Start it in another terminal:\n"
                    "    make agent",
                    file=sys.stderr,
                )
                return 1
        elif not agent_name:
            print("agent is unnamed, so it auto-joins — no dispatch needed")

        token = (
            api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
            .with_identity(identity)
            .with_name(identity)
            .with_grants(api.VideoGrants(room_join=True, room=room))
            .to_jwt()
        )
    finally:
        await lk.aclose()

    print()
    print("=" * 72)
    print("Open the LiveKit Agents Playground and choose 'Manual' connection:")
    print(f"  {PLAYGROUND}")
    print()
    print(f"  LiveKit URL : {settings.livekit_url}")
    print(f"  Room        : {room}")
    print(f"  Token       : {token}")
    print("=" * 72)
    print(
        "\nAllow the microphone when prompted. Mami greets you first — wait for\n"
        "her to finish, then answer. Say a language ('English', 'Telugu'), your\n"
        "name, the service you need, and your area. She reads the details back;\n"
        "say 'yes' to finish or correct her ('no, I need an electrician').\n"
    )
    if settings.livekit_agent_name.strip():
        print(
            "Note: this dispatch is for ONE call. Re-run this command for each\n"
            "new call, or set LIVEKIT_AGENT_NAME= (empty) in .env so the agent\n"
            "auto-joins any room you open.\n"
        )
    return 0


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(
        prog="python -m backend.app.devcall",
        description="Mint a token (and dispatch the agent) for a browser test call.",
    )
    parser.add_argument("--room", default="local-mama-test", help="room name")
    parser.add_argument("--identity", default="you", help="your participant identity")
    parser.add_argument(
        "--no-dispatch",
        action="store_true",
        help="skip dispatch (use when the agent is unnamed and auto-joins)",
    )
    args = parser.parse_args()
    return asyncio.run(setup_call(args.room, args.identity, not args.no_dispatch))


if __name__ == "__main__":
    raise SystemExit(main())
