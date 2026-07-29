"""Terminal test console — the whole agent, no browser.

The browser path depends on Chrome's Web Speech API, which is a cloud service
with no SLA and fails outright on some machines and networks. This module drives
the exact same `ConversationManager` from a terminal instead, so the workflow,
extraction, multilingual rendering, and lead capture can be exercised
independently of any of that.

Speech output uses the operating system's own synthesiser — `say` on macOS,
`espeak-ng` on Linux — which needs no API key and, on macOS, ships genuinely
good Indian-language voices. Input is typed, which is the point: it removes the
single most unreliable component from the loop.

    python -m backend.app.cli                  interactive, speaks replies
    python -m backend.app.cli --no-speak       silent
    python -m backend.app.cli --demo           scripted runs in all 6 languages
    python -m backend.app.cli --script FILE    replay one utterance per line
    python -m backend.app.cli --list-voices    show detected system voices
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import re
import shutil
import subprocess
import sys

from .languages import ENDONYM, Language
from .models import ConversationState
from .services.conversation_manager import ConversationManager, abandon

# --- terminal colours (disabled when piped) ------------------------------
_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


BOLD = lambda t: _c("1", t)          # noqa: E731
DIM = lambda t: _c("2", t)           # noqa: E731
AGENT = lambda t: _c("38;5;215", t)  # noqa: E731
USER = lambda t: _c("38;5;117", t)   # noqa: E731
OK = lambda t: _c("38;5;114", t)     # noqa: E731
WARN = lambda t: _c("38;5;179", t)   # noqa: E731

# Preferred system voice per language, best first. Chosen from the voices
# macOS actually ships for these locales; the first installed one wins.
VOICE_PREFERENCES: dict[Language, tuple[str, ...]] = {
    Language.ENGLISH: ("Rishi", "Tara", "Aman", "Samantha", "Alex"),
    Language.HINDI: ("Lekha",),
    Language.BENGALI: ("Piya",),
    Language.TELUGU: ("Geeta",),
    Language.TAMIL: ("Vani",),
    Language.KANNADA: ("Soumya",),
}

LOCALE_TAG: dict[Language, str] = {
    Language.ENGLISH: "en_IN",
    Language.HINDI: "hi_IN",
    Language.BENGALI: "bn_IN",
    Language.TELUGU: "te_IN",
    Language.TAMIL: "ta_IN",
    Language.KANNADA: "kn_IN",
}


# --------------------------------------------------------------- speech ----


def detect_system_voices() -> dict[str, str]:
    """Map voice name -> locale tag, from the OS synthesiser."""
    if platform.system() != "Darwin" or not shutil.which("say"):
        return {}
    try:
        output = subprocess.run(
            ["say", "-v", "?"], capture_output=True, text=True, timeout=10
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return {}

    voices: dict[str, str] = {}
    for line in output.splitlines():
        # "Lekha                hi_IN    # नमस्ते, मेरा नाम लेखा है।"
        match = re.match(r"^(.+?)\s{2,}([a-z]{2}_[A-Z]{2})\s", line)
        if match:
            voices[match.group(1).strip()] = match.group(2)
    return voices


class Speaker:
    """Speaks text through the OS synthesiser, picking a per-language voice."""

    def __init__(self, enabled: bool = True, rate: int | None = None) -> None:
        self.system = platform.system()
        self.rate = rate
        self.voices = detect_system_voices()
        self.backend = self._detect_backend() if enabled else None
        self.enabled = self.backend is not None
        self._warned: set[Language] = set()

    def _detect_backend(self) -> str | None:
        if self.system == "Darwin" and shutil.which("say"):
            return "say"
        if shutil.which("espeak-ng"):
            return "espeak-ng"
        if shutil.which("espeak"):
            return "espeak"
        return None

    def voice_for(self, language: Language) -> str | None:
        for candidate in VOICE_PREFERENCES.get(language, ()):
            if candidate in self.voices:
                return candidate
        # Fall back to any installed voice for the right locale.
        tag = LOCALE_TAG[language]
        for name, locale in self.voices.items():
            if locale == tag:
                return name
        return None

    def describe(self) -> str:
        if not self.enabled:
            return "text only (no system synthesiser found)"
        parts = []
        for language in Language:
            voice = self.voice_for(language)
            parts.append(f"{language.value}={voice or '—'}")
        return f"{self.backend}: " + ", ".join(parts)

    def say(self, text: str, language: Language) -> None:
        if not self.enabled or not text.strip():
            return
        if self.backend == "say":
            voice = self.voice_for(language)
            if voice is None and language not in self._warned:
                self._warned.add(language)
                print(
                    WARN(
                        f"  (no {language.value} voice installed — speaking with the "
                        f"system default; install one in System Settings → "
                        f"Accessibility → Spoken Content → Manage Voices)"
                    )
                )
            cmd = ["say"]
            if voice:
                cmd += ["-v", voice]
            if self.rate:
                cmd += ["-r", str(self.rate)]
            cmd.append(text)
        else:
            cmd = [self.backend, "-v", LOCALE_TAG[language][:2], text]

        try:
            subprocess.run(cmd, check=False, timeout=90)
        except (subprocess.SubprocessError, OSError) as exc:
            print(WARN(f"  (speech failed: {exc})"))


# ------------------------------------------------------------ rendering ----


def show_agent(text: str, state: ConversationState) -> None:
    print(f"\n{AGENT('Mami')} {DIM(f'[{state.value}]')}\n  {text}")


def show_captured(manager: ConversationManager) -> None:
    session = manager.session
    fields = {
        "language": session.selected_language.value if session.selected_language else None,
        "name": session.user_name,
        "service": session.requested_service,
        "area": session.city_or_area,
    }
    filled = {k: v for k, v in fields.items() if v}
    if filled:
        print(DIM("  captured: " + ", ".join(f"{k}={v}" for k, v in filled.items())))


def show_lead(lead) -> None:
    print("\n" + OK("── Captured lead ──"))
    print(json.dumps(lead.model_dump(mode="json"), indent=2, ensure_ascii=False))
    print(
        OK(
            f"\n[SIMULATED WHATSAPP] best {lead.requested_service} matches in "
            f"{lead.city_or_area} → {lead.user_name}"
        )
    )


# ------------------------------------------------------------- sessions ----


async def run_turns(
    manager: ConversationManager, speaker: Speaker, turns: list[str], echo: bool = True
) -> None:
    """Drive a conversation from a fixed list of utterances."""
    opening = manager.start()
    show_agent(opening.reply, opening.state)
    speaker.say(opening.reply, opening.language)

    for text in turns:
        if echo:
            print(f"\n{USER('You')}\n  {text}")
        result = await manager.handle(text)
        show_agent(result.reply, result.state)
        show_captured(manager)
        speaker.say(result.reply, result.language)
        if result.lead:
            show_lead(result.lead)
        if result.is_final:
            return


async def interactive(speaker: Speaker) -> None:
    manager = ConversationManager()
    print(BOLD("\nLocal Mama — terminal test console"))
    print(DIM(f"voices: {speaker.describe()}"))
    print(DIM("type your answers; Ctrl-C or 'quit' to end\n"))

    opening = manager.start()
    show_agent(opening.reply, opening.state)
    speaker.say(opening.reply, opening.language)

    while True:
        try:
            text = input(f"\n{USER('You')}\n  ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {"quit", "exit", ":q"}:
            break
        if not text:
            continue

        result = await manager.handle(text)
        show_agent(result.reply, result.state)
        show_captured(manager)
        speaker.say(result.reply, result.language)

        if result.lead:
            show_lead(result.lead)
        if result.is_final:
            print(DIM("\ncall complete"))
            return

    if manager.session.state is not ConversationState.COMPLETED:
        abandon(manager.session)
        print(DIM("call abandoned; partial lead saved"))


DEMOS: list[tuple[str, list[str]]] = [
    # Each scenario ends by agreeing to the read-back — nothing is committed
    # until the caller confirms what we heard.
    ("English — one field per turn",
     ["English", "My name is Ravi", "I need an electrician", "Madhapur Hyderabad", "yes"]),
    ("Hindi — romanised input",
     ["Hindi", "mera naam Suresh hai", "plumber chahiye", "Pune mein", "haan ji"]),
    ("Telugu — native script",
     ["తెలుగు", "నా పేరు రవి", "ఎలక్ట్రీషియన్", "మాధాపూర్ హైదరాబాద్", "avunu"]),
    ("Tamil — mixed script",
     ["Tamil", "en peyar Karthik", "AC repair", "Coimbatore", "aamaam"]),
    ("Bengali — native script",
     ["বাংলা", "আমার নাম রাহুল", "carpenter", "Howrah Kolkata", "hyan"]),
    ("Kannada — romanised",
     ["Kannada", "nanna hesaru Kiran", "painter", "Indiranagar Bangalore", "houdu"]),
    ("English — everything in one utterance",
     ["English", "I am Priya and I need a plumber in Chennai", "yes"]),
    ("English — noisy speech, needs a re-prompt",
     ["English", "...", "umm haan ji Anil", "pest control", "Andheri Mumbai", "correct"]),
    ("English — read-back catches a mistranscription",
     ["English", "My name is Ravi", "I need a plumber", "Madhapur Hyderabad",
      "no, I need an electrician", "yes"]),
]


async def run_demo(speaker: Speaker) -> int:
    print(BOLD("\nLocal Mama — scripted demo across all supported languages"))
    print(DIM(f"voices: {speaker.describe()}\n"))
    failures = 0

    for title, turns in DEMOS:
        print("\n" + BOLD("═" * 68))
        print(BOLD(title))
        print(BOLD("═" * 68))
        manager = ConversationManager()
        await run_turns(manager, speaker, turns)

        session = manager.session
        if session.state is ConversationState.COMPLETED:
            print(OK("\n✓ completed"))
        else:
            failures += 1
            print(WARN(f"\n✗ ended in {session.state.value} (expected COMPLETED)"))

    print("\n" + BOLD("═" * 68))
    if failures:
        print(WARN(f"{len(DEMOS) - failures}/{len(DEMOS)} scenarios completed"))
    else:
        print(OK(f"all {len(DEMOS)} scenarios completed and captured a lead"))
    return 1 if failures else 0


# ----------------------------------------------------------------- main ----


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.app.cli",
        description="Terminal test console for the Local Mama voice agent.",
    )
    parser.add_argument("--no-speak", action="store_true", help="disable speech output")
    parser.add_argument("--demo", action="store_true", help="run scripted demos in all languages")
    parser.add_argument("--script", metavar="FILE", help="replay utterances, one per line")
    parser.add_argument("--list-voices", action="store_true", help="show detected system voices")
    parser.add_argument("--rate", type=int, help="speech rate in words per minute (macOS)")
    args = parser.parse_args()

    if args.list_voices:
        voices = detect_system_voices()
        if not voices:
            print("No system voices detected (macOS `say` not available).")
            return 1
        print(BOLD("Voices installed for supported languages:\n"))
        for language in Language:
            tag = LOCALE_TAG[language]
            matches = [n for n, loc in voices.items() if loc == tag]
            label = f"{language.value} ({ENDONYM[language]})"
            print(f"  {label:24} {tag}  " + (", ".join(matches) if matches else WARN("none")))
        print(DIM(f"\n{len(voices)} voices total. Add more in System Settings → "
                  "Accessibility → Spoken Content → Manage Voices."))
        return 0

    speaker = Speaker(enabled=not args.no_speak, rate=args.rate)

    if args.demo:
        return asyncio.run(run_demo(speaker))

    if args.script:
        try:
            lines = [
                ln.strip()
                for ln in open(args.script, encoding="utf-8")
                if ln.strip() and not ln.startswith("#")
            ]
        except OSError as exc:
            print(f"could not read script: {exc}", file=sys.stderr)
            return 1
        manager = ConversationManager()
        asyncio.run(run_turns(manager, speaker, lines))
        return 0

    asyncio.run(interactive(speaker))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
