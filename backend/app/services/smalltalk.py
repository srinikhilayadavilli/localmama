"""Casual conversation handling — warmth without giving up control.

Callers do not stay on script. They greet, ask who they are talking to, vent
about the flooded bathroom, say thank you. Answering those with a flat "Sorry,
I missed your name" is what makes an agent feel robotic.

This module classifies such utterances so the conversation manager can
acknowledge them warmly and then re-ask the pending question. Two properties
make it safe by construction:

  * It is **pure classification**. It returns a `MessageKey` and nothing else —
    it cannot write to the session, cannot change state, and cannot mark a
    field captured. Whatever a caller says, the workflow is unmoved.
  * It runs **only after extraction has already failed** for the field we
    asked for, so a real answer is never mistaken for chit-chat.

Consecutive small talk is capped by the caller (see `MAX_CONSECUTIVE`), so a
caller who only chats cannot keep a session alive indefinitely.
"""

from __future__ import annotations

import re

from ..languages import normalize_text
from ..prompts.messages import MessageKey

#: After this many small-talk turns in a row, go back to a plain re-prompt.
MAX_CONSECUTIVE = 3

# Matching is done on the normalised utterance (casefolded, punctuation
# stripped), so entries here are written in that form. Native-script terms are
# included alongside romanised ones because callers mix freely.
_PATTERNS: tuple[tuple[MessageKey, tuple[str, ...]], ...] = (
    # Distress first: "I'm so stressed" must read as distress, not a greeting.
    (
        MessageKey.SMALLTALK_EMPATHY,
        (
            "stress", "stressed", "tension", "worried", "worry", "upset", "frustrat",
            "terrible", "horrible", "awful", "urgent", "emergency", "help me please",
            "very bad", "so bad", "not good", "bad day", "tired", "exhausted",
            "angry", "annoyed", "fed up", "disaster", "flooded", "broke down",
            "pareshan", "chinta", "bahut bura", "problem hai", "taklif",
            "परेशान", "चिंता", "मुश्किल", "কষ্ট", "চিন্তা",
            "కష్టం", "బాధ", "కంగారు", "கஷ்டம்", "கவலை", "ಕಷ್ಟ", "ಚಿಂತೆ",
        ),
    ),
    (
        MessageKey.SMALLTALK_ABUSE,
        (
            "stupid", "useless", "idiot", "nonsense", "rubbish", "worst",
            "shut up", "hate you", "bakwas", "bekar", "bewakoof",
            "बकवास", "बेकार", "फालतू",
        ),
    ),
    (
        MessageKey.SMALLTALK_HOW_ARE_YOU,
        (
            "how are you", "how r u", "hows it going", "how do you do",
            "kaise ho", "kaisi ho", "kaise hain", "kem cho",
            "कैसे हो", "कैसी हो", "ela unnaru", "ఎలా ఉన్నారు",
            "eppadi irukkinga", "எப்படி இருக்கீங்க", "hegiddira", "ಹೇಗಿದ್ದೀರಾ",
            "kemon achen", "কেমন আছেন",
        ),
    ),
    (
        MessageKey.SMALLTALK_IDENTITY,
        (
            "who are you", "what are you", "your name", "are you a robot",
            "are you human", "are you real", "are you ai", "are you a bot",
            "aap kaun", "tum kaun", "आप कौन", "तुम कौन", "तुम्हारा नाम",
            "meeru evaru", "మీరు ఎవరు", "nee yaar", "நீங்கள் யார்",
            "neevu yaru", "ನೀವು ಯಾರು", "apni ke", "আপনি কে",
        ),
    ),
    (
        MessageKey.SMALLTALK_THANKS,
        (
            "thank you", "thanks", "thank u", "thx", "shukriya", "dhanyavaad",
            "dhanyawad", "धन्यवाद", "शुक्रिया", "ধন্যবাদ",
            "ధన్యవాదాలు", "நன்றி", "ಧನ್ಯವಾದ",
        ),
    ),
    (
        MessageKey.SMALLTALK_GREETING,
        (
            "hello", "hii", "hey there", "good morning", "good afternoon",
            "good evening", "namaste", "namaskar", "namaskara", "vanakkam",
            "nomoshkar", "adaab", "salaam", "नमस्ते", "नमस्कार",
            "নমস্কার", "నమస్తే", "வணக்கம்", "ನಮಸ್ಕಾರ",
        ),
    ),
)

# Whole-word triggers, kept separate so "hi" does not fire inside "this".
_WORD_TRIGGERS: dict[str, MessageKey] = {
    "hi": MessageKey.SMALLTALK_GREETING,
    "hey": MessageKey.SMALLTALK_GREETING,
    "yo": MessageKey.SMALLTALK_GREETING,
    "sad": MessageKey.SMALLTALK_EMPATHY,
    "help": MessageKey.SMALLTALK_EMPATHY,
}


def classify(text: str) -> MessageKey | None:
    """Return the small-talk category for an utterance, or None.

    None means "this is not chit-chat" — the caller should fall through to a
    normal re-prompt.
    """
    if not text:
        return None
    normalized = normalize_text(text)
    if not normalized:
        return None

    for key, phrases in _PATTERNS:
        for phrase in phrases:
            needle = normalize_text(phrase)
            if needle and needle in normalized:
                return key

    tokens = set(normalized.split())
    for word, key in _WORD_TRIGGERS.items():
        if word in tokens:
            return key

    return None


def is_question(text: str) -> bool:
    """Whether the caller asked something, rather than answering."""
    if not text:
        return False
    if "?" in text:
        return True
    normalized = normalize_text(text)
    return bool(
        re.match(
            r"^(what|who|where|when|why|how|can you|do you|are you|will you|is it)\b",
            normalized,
        )
    )
