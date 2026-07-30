"""Does this utterance read as chit-chat rather than an answer?

Callers do not stay on script. They greet, ask who they are talking to, vent
about the flooded bathroom, say thank you. The realtime model handles all of
that conversationally on its own — what it cannot do is stop such a phrase
being *filed as a data point*.

That is this module's only job now, and it has exactly one caller:
`entity_extractor._looks_like_name`. The "I am X" pattern collides directly
with how people express distress, and "I am so stressed" was captured as a
caller named "So Stressed". Recognising the whole class of chit-chat rules that
out in one place rather than one phrase at a time.

Pure classification: it returns a bool and touches nothing.
"""

from __future__ import annotations

from ..text import normalize_text

# Matching is done on the normalised utterance (casefolded, punctuation
# stripped), so entries here are written in that form. Native-script terms are
# included alongside romanised ones because callers mix freely.
_PHRASES: tuple[str, ...] = (
    # Distress. This is the group that actually collides with name capture.
    "stress", "stressed", "tension", "worried", "worry", "upset", "frustrat",
    "terrible", "horrible", "awful", "urgent", "emergency", "help me please",
    "very bad", "so bad", "not good", "bad day", "tired", "exhausted",
    "angry", "annoyed", "fed up", "disaster", "flooded", "broke down",
    "pareshan", "chinta", "bahut bura", "problem hai", "taklif",
    "परेशान", "चिंता", "मुश्किल", "কষ্ট", "চিন্তা",
    "కష్టం", "బాధ", "కంగారు", "கஷ்டம்", "கவலை", "ಕಷ್ಟ", "ಚಿಂತೆ",
    # Abuse.
    "stupid", "useless", "idiot", "nonsense", "rubbish", "worst",
    "shut up", "hate you", "bakwas", "bekar", "bewakoof",
    "बकवास", "बेकार", "फालतू",
    # "How are you".
    "how are you", "how r u", "hows it going", "how do you do",
    "kaise ho", "kaisi ho", "kaise hain", "kem cho",
    "कैसे हो", "कैसी हो", "ela unnaru", "ఎలా ఉన్నారు",
    "eppadi irukkinga", "எப்படி இருக்கீங்க", "hegiddira", "ಹೇಗಿದ್ದೀರಾ",
    "kemon achen", "কেমন আছেন",
    # "Who are you".
    "who are you", "what are you", "your name", "are you a robot",
    "are you human", "are you real", "are you ai", "are you a bot",
    "aap kaun", "tum kaun", "आप कौन", "तुम कौन", "तुम्हारा नाम",
    "meeru evaru", "మీరు ఎవరు", "nee yaar", "நீங்கள் யார்",
    "neevu yaru", "ನೀವು ಯಾರು", "apni ke", "আপনি কে",
    # Thanks.
    "thank you", "thanks", "thank u", "thx", "shukriya", "dhanyavaad",
    "dhanyawad", "धन्यवाद", "शुक्रिया", "ধন্যবাদ",
    "ధన్యవాదాలు", "நன்றி", "ಧನ್ಯವಾದ",
    # Greetings.
    "hello", "hii", "hey there", "good morning", "good afternoon",
    "good evening", "namaste", "namaskar", "namaskara", "vanakkam",
    "nomoshkar", "adaab", "salaam", "नमस्ते", "नमस्कार",
    "নমস্কার", "నమస్తే", "வணக்கம்", "ನಮಸ್ಕಾರ",
)

#: Whole-word triggers, kept separate so "hi" does not fire inside "this".
_WORD_TRIGGERS: frozenset[str] = frozenset({"hi", "hey", "yo", "sad", "help"})


def looks_like_chitchat(text: str) -> bool:
    """Whether an utterance is conversation rather than an answer."""
    if not text:
        return False
    normalized = normalize_text(text)
    if not normalized:
        return False

    for phrase in _PHRASES:
        needle = normalize_text(phrase)
        if needle and needle in normalized:
            return True

    return bool(_WORD_TRIGGERS & set(normalized.split()))
