"""Rule-based entity extraction for name, service, and location.

Strategy: rules first, LLM only as a fallback (see `llm_extractor.py`). Rules
are deterministic, instant, and free — they handle the overwhelming majority of
real answers, which are short and formulaic ("Ravi", "my name is Ravi",
"naa peru Ravi"). The LLM is reserved for utterances the rules can't parse.

Confidence conventions:
  0.9  explicit pattern match ("my name is X", "electrician" in the catalog)
  0.6  positional/bare-phrase inference when we asked for exactly this field
  0.0  nothing found
The conversation manager treats anything at or above `MIN_CONFIDENCE` as
acceptable and re-prompts otherwise.
"""

from __future__ import annotations

import difflib
import re

from ..languages import normalize_text, resolve_language
from ..models import ConversationState, Extraction

MIN_CONFIDENCE = 0.5

#: A service phrase that matches nothing in the catalog, even fuzzily. Reported
#: below MIN_CONFIDENCE so the agent re-prompts once rather than silently
#: committing it: a real call produced "I need a puma" (STT for "plumber"), and
#: "puma" reached the lead as the requested service. Genuinely novel trades are
#: still accepted on the second attempt — see `ConversationManager`.
UNKNOWN_SERVICE_CONFIDENCE = 0.45

# Filler words and politeness markers that commonly precede the real answer.
# Stripped from the front of an utterance before positional inference.
_FILLERS = {
    "um", "uh", "hmm", "haan", "han", "ha", "ji", "ok", "okay", "yes", "yeah",
    "well", "so", "actually", "basically", "please", "kindly", "sir", "madam",
    "namaste", "namaskar", "hello", "hi", "hey", "arre", "acha", "achha",
    "matlab", "woh", "vo", "aa", "ante", "adhu", "andre", "mane",
    "er", "erm", "oh", "ah", "eh", "hm", "mmm", "like", "just",
}


def _is_filler(token: str) -> bool:
    """Filler check tolerant of elongated speech disfluencies ("ummm", "uhhh").

    Speech-to-text renders hesitation with variable letter runs, so collapse
    repeated characters before comparing against the filler set.
    """
    cleaned = token.strip(".,!?;:").casefold()
    if not cleaned:
        return True
    if cleaned in _FILLERS:
        return True
    collapsed = re.sub(r"(.)\1+", r"\1", cleaned)
    return collapsed in _FILLERS

# Words that must never be accepted as a person's name.
_NAME_STOPWORDS = {
    "name", "naam", "peru", "hesaru", "nam", "pera", "my", "is", "am", "i",
    "the", "a", "an", "this", "that", "it", "service", "need", "want", "looking",
    "help", "please", "call", "yes", "no", "ok", "okay", "hello", "hi",
    "sorry", "what", "who", "where", "which", "how", "mama", "mami",
    # Politeness and greetings — these arrive constantly and must never be
    # mistaken for a name by positional inference.
    "thank", "thanks", "thankyou", "welcome", "namaste", "namaskar", "hey",
    "good", "morning", "afternoon", "evening", "much", "very", "fine", "well",
    "shukriya", "dhanyavaad", "vanakkam",
}

# --- Name patterns -------------------------------------------------------
# Each pattern captures the name in group "v". Ordered most-explicit-first.
_NAME_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE | re.UNICODE)
    for p in (
        # English
        r"\bmy\s+name\s+is\s+(?P<v>.+)",
        r"\bname\s+is\s+(?P<v>.+)",
        r"\bi\s+am\s+(?P<v>.+)",
        r"\bi'?m\s+(?P<v>.+)",
        r"\bthis\s+is\s+(?P<v>.+)",
        r"\bmyself\s+(?P<v>.+)",
        r"\bcall\s+me\s+(?P<v>.+)",
        # Hindi  — "mera naam X hai" / "मेरा नाम X है"
        r"\bmera\s+naam\s+(?P<v>.+?)(?:\s+hai)?$",
        r"मेरा\s*नाम\s*(?P<v>.+?)(?:\s*है)?$",
        # Telugu — "naa peru X" / "నా పేరు X"
        r"\bna+\s*peru\s+(?P<v>.+)",
        r"నా\s*పేరు\s*(?P<v>.+)",
        # Tamil  — "en peru X" / "என் பெயர் X"
        r"\ben\s+peyar\s+(?P<v>.+)",
        r"\ben\s+peru\s+(?P<v>.+)",
        r"என்\s*பெயர்\s*(?P<v>.+)",
        # Kannada — "nanna hesaru X" / "ನನ್ನ ಹೆಸರು X"
        r"\bnanna\s+hesaru\s+(?P<v>.+)",
        r"ನನ್ನ\s*ಹೆಸರು\s*(?P<v>.+)",
        # Bengali — "amar nam X" / "আমার নাম X"
        r"\bamar\s+na?am\s+(?P<v>.+)",
        r"আমার\s*নাম\s*(?P<v>.+)",
    )
)

# --- Service catalog -----------------------------------------------------
# canonical English label -> recognised surface forms (any language/script).
# Matching is substring-based on the normalised utterance, so romanised and
# native-script variants both work.
SERVICE_CATALOG: dict[str, tuple[str, ...]] = {
    "electrician": (
        "electrician", "electric", "electrical", "wiring", "bijli", "बिजली",
        "इलेक्ट्रीशियन", "కరెంట్", "ఎలక్ట్రీషియన్", "மின்சாரம்", "எலக்ட்ரீஷியன்",
        "ಎಲೆಕ್ಟ್ರೀಷಿಯನ್", "ইলেকট্রিশিয়ান", "কারেন্ট", "current work",
    ),
    "plumber": (
        "plumber", "plumbing", "tap", "pipe", "leakage", "nal", "नल", "प्लंबर",
        "ప్లంబర్", "పైపు", "பிளம்பர்", "குழாய்", "ಪ್ಲಂಬರ್", "প্লাম্বার",
    ),
    "carpenter": (
        "carpenter", "carpentry", "furniture", "badhai", "बढ़ई", "कारपेंटर",
        "వడ్రంగి", "కార్పెంటర్", "தச்சர்", "ಬಡಗಿ", "ছুতার", "কার্পেন্টার",
    ),
    "ac repair": (
        "ac repair", "a c repair", "air conditioner", "ac service", "ac servicing",
        "ac mechanic", "एसी", "ఏసీ", "ஏசி", "ಎಸಿ", "এসি",
    ),
    "appliance repair": (
        "fridge", "refrigerator", "washing machine", "microwave", "geyser",
        "appliance", "फ्रिज", "వాషింగ్ మెషిన్", "ವಾಷಿಂಗ್ ಮೆಷಿನ್",
    ),
    "painter": ("painter", "painting", "paint", "पेंटर", "పెయింటర్", "ಪೇಂಟರ್", "রঙ"),
    "pest control": ("pest control", "pest", "cockroach", "termite", "पेस्ट कंट्रोल"),
    "cleaning": (
        "cleaning", "cleaner", "housekeeping", "deep clean", "safai", "सफाई",
        "క్లీనింగ్", "ಕ್ಲೀನಿಂಗ್", "পরিষ্কার",
    ),
    "packers and movers": (
        "packers", "movers", "shifting", "house shifting", "relocation",
        "पैकर्स", "షిఫ్టింగ్",
    ),
    "tutor": (
        "tutor", "tuition", "teacher", "coaching", "ट्यूशन", "ట్యూషన్",
        "ಟ್ಯೂಷನ್", "টিউশন",
    ),
    "salon": (
        "salon", "beautician", "parlour", "parlor", "haircut", "सैलून",
        "పార్లర్", "ಸಲೂನ್", "পার্লার",
    ),
    "cook": ("cook", "chef", "maharaj", "khana banane", "వంట", "সমরকারী", "cooking"),
    "maid": ("maid", "housemaid", "domestic help", "kaam wali", "बाई", "పనిమనిషి"),
    "driver": ("driver", "chauffeur", "ड्राइवर", "డ్రైవర్", "ಚಾಲಕ", "ড্রাইভার"),
    "mechanic": ("mechanic", "bike repair", "car repair", "मैकेनिक", "మెకానిక్"),
    "doctor": ("doctor", "physician", "डॉक्टर", "డాక్టర్", "ವೈದ್ಯ", "ডাক্তার"),
    "lawyer": ("lawyer", "advocate", "वकील", "న్యాయవాది", "ವಕೀಲ", "উকিল"),
    "tailor": ("tailor", "stitching", "darzi", "दर्जी", "టైలర్", "ದರ್ಜಿ"),
}

# Verbs/nouns that signal a service request even when the specific trade is
# not in the catalog — used to extract an unknown service phrase.
_SERVICE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE | re.UNICODE)
    for p in (
        # The article group needs its own trailing space. Without it, "some" was
        # matched inside "someone" and "I need someone for my bathroom" captured
        # "one for my bathroom" as the requested service.
        r"\b(?:i\s+)?(?:need|want|looking\s+for|require|searching\s+for)\s+(?:(?:a|an|the|some)\s+)?(?P<v>.+)",
        r"\b(?:mujhe|muje)\s+(?P<v>.+?)\s+chahiye",
        r"\b(?:naaku|naku)\s+(?P<v>.+?)\s+kavali",
        r"\b(?:enakku)\s+(?P<v>.+?)\s+vendum",
        r"\b(?:nanage)\s+(?P<v>.+?)\s+beku",
        r"\b(?:amar)\s+(?P<v>.+?)\s+dorkar",
        # Native-script request forms. Without these the whole sentence became
        # the service on a real Telugu call: "నేను ఎలక్ట్రిషన్ కోసం
        # వెతుకుతున్నాను" was stored verbatim as the requested service.
        r"(?P<v>.+?)\s*కోసం",           # Telugu:  X kosam (for X)
        r"(?P<v>.+?)\s*కావాలి",          # Telugu:  X kavali (want X)
        r"(?P<v>.+?)\s*చేయాలి",
        r"(?P<v>.+?)\s*चाहिए",           # Hindi:   X chahiye
        r"(?P<v>.+?)\s*वाला\s*चाहिए",
        r"(?P<v>.+?)\s*வேண்டும்",        # Tamil:   X vendum
        r"(?P<v>.+?)\s*ಬೇಕು",            # Kannada: X beku
        r"(?P<v>.+?)\s*দরকার",           # Bengali: X dorkar
    )
)

#: Pronouns/verbs that open an Indic request and are not part of the trade name.
_INDIC_REQUEST_PREFIXES = (
    "నేను", "నాకు", "మాకు", "मुझे", "मुझको", "हमें", "எனக்கு", "நான்",
    "ನನಗೆ", "ನಾನು", "আমার", "আমি",
)

# --- Location ------------------------------------------------------------
# A seed list of common Indian cities. This is deliberately small: the
# positional and preposition rules below carry most of the load, and unknown
# localities ("Madhapur", "Indiranagar") must still work.
CITY_SEEDS: tuple[str, ...] = (
    "hyderabad", "bangalore", "bengaluru", "chennai", "mumbai", "bombay",
    "delhi", "new delhi", "kolkata", "calcutta", "pune", "ahmedabad", "surat",
    "jaipur", "lucknow", "kanpur", "nagpur", "indore", "bhopal", "patna",
    "vadodara", "coimbatore", "madurai", "kochi", "cochin", "trivandrum",
    "thiruvananthapuram", "visakhapatnam", "vizag", "vijayawada", "warangal",
    "guntur", "nellore", "tirupati", "mysore", "mysuru", "mangalore", "hubli",
    "belgaum", "salem", "trichy", "tiruchirappalli", "erode", "tirunelveli",
    "howrah", "siliguri", "durgapur", "asansol", "noida", "gurgaon", "gurugram",
    "ghaziabad", "faridabad", "thane", "navi mumbai", "chandigarh", "ludhiana",
    "amritsar", "agra", "varanasi", "meerut", "rajkot", "nashik", "aurangabad",
)

# Locality suffixes/prepositions marking a place in each language.
#: Prepositional forms. Safe to run in any state — "in Hyderabad" is
#: unambiguous.
_LOCATION_PREPOSITIONS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:in|at|near|around|from)\s+(?P<v>.+)", re.IGNORECASE | re.UNICODE),
)

# Postpositional forms ("Pune mein", "Vizag lo"). These are short particles and
# match far too eagerly, so they carry two safeguards:
#   * `\s+` — the particle must be a standalone token. With `\s*` the Telugu
#     "lo" matched inside the English word "hel-lo", turning a greeting into a
#     city named "Hel".
#   * they run only when a location was actually asked for (see
#     `extract_location`), so they cannot fire during the name or language step.
_LOCATION_POSTPOSITIONS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE | re.UNICODE)
    for p in (
        r"(?P<v>.+?)\s+(?:mein|me)\b",       # Hindi:   X mein
        r"(?P<v>.+?)\s+(?:lo|loni)\b",       # Telugu:  X lo
        r"(?P<v>.+?)\s+(?:il|la)\b",         # Tamil:   X il
        r"(?P<v>.+?)\s+(?:alli|nalli)\b",    # Kannada: X alli
        r"(?P<v>.+?)\s+(?:e|te)\b",          # Bengali: X e
        r"(?P<v>.+?)\s*में",
        r"(?P<v>.+?)\s*లో",
        r"(?P<v>.+?)\s*ல்",
        r"(?P<v>.+?)\s*ಅಲ್ಲಿ",
    )
)


def _strip_fillers(text: str) -> str:
    tokens = text.split()
    while tokens and _is_filler(tokens[0]):
        tokens.pop(0)
    return " ".join(tokens)


#: Possessives and determiners that survive the request patterns and end up
#: glued to the front of a captured service. "I need my car serviced" stored
#: "my car serviced" as the trade, which is what reaches the lead and the
#: knowledge-base lookup.
_LEADING_DETERMINERS = {"my", "our", "mera", "meri", "naa", "na", "en", "nanna", "amar"}


def _clean_value(value: str) -> str:
    """Trim politeness, punctuation and leading determiners from a span."""
    value = value.strip(" .,!?;:।")
    tokens = value.split()
    while tokens and _is_filler(tokens[-1]):
        tokens.pop()
    while tokens and tokens[0].casefold() in _LEADING_DETERMINERS:
        tokens.pop(0)
    return " ".join(tokens).strip()


def _looks_like_name(candidate: str) -> bool:
    """Reject obvious non-names before accepting a phrase as a name."""
    if not candidate:
        return False
    tokens = candidate.split()
    if not 1 <= len(tokens) <= 4:
        return False
    if any(t.casefold().strip(".,") in _NAME_STOPWORDS for t in tokens):
        return False
    if any(ch.isdigit() for ch in candidate):
        return False
    # A phrase that is really a service request is not a name.
    if match_service(candidate):
        return False
    # Nor is an emotional state. The "I am X" pattern collides directly with
    # how people express distress — "I am so stressed" was being captured as a
    # caller named "So Stressed". Anything that reads as chit-chat is not a
    # name, which covers this whole class rather than one phrase at a time.
    from . import smalltalk  # local import: smalltalk must not import us back

    if smalltalk.classify(candidate) is not None:
        return False
    return True


def format_name(candidate: str) -> str:
    """"rahul sharma" -> "Rahul Sharma". Non-Latin parts are left alone."""
    return " ".join(part.capitalize() if part.isascii() else part for part in candidate.split())


# Bounds for rescuing a misheard trade name, chosen from a parameter sweep over
# real STT errors versus real false-positive risks rather than by intuition.
#
# Short words are the danger: a live call transcribed "plumber" as "palm oil",
# and at cutoff 0.55 "palm" matched "salon" at 0.85 confidence — the lead said
# salon. Requiring six characters removes that whole class ("palm", "puma",
# "oil", "anil", "area" are all too short to guess from), and 0.70 leaves margin
# on the similarity axis too. Both sit inside a region where all of
# "plummer"/"electrican"/"carpentar"/"plumbr"/"electricion"/"painterr" are still
# rescued and nothing spurious matches.
_FUZZY_CUTOFF = 0.70
_FUZZY_MIN_LENGTH = 6
#: How close in length a token and a synonym must be to count as a typo rather
#: than a different (usually shorter, more generic) word. "plumbr"/"plumber" is
#: A one-character slip on the 6-char minimum gives 6/7 = 0.857, so 0.85 is the
#: natural line. "service"/"ac service" is 0.70 and "serviced"/"ac service" is
#: 0.80 — both are different words, and both must miss.
_FUZZY_LENGTH_RATIO = 0.85
#: Indic scripts pack a syllable into each character, so the same edit distance
#: means a much bigger difference in meaning. Measured on real pairs: the
#: legitimate variant "ఎలక్ట్రిషన్"/"ఎలక్ట్రీషియన్" scores 0.83, while the
#: damaging "టెన్షన్"/"ట్యూషన్" (tension/tuition) scores 0.71.
_FUZZY_CUTOFF_INDIC = 0.80

#: Flat lookup of every synonym, built once for fuzzy matching.
#:
#: Indic spellings are included, not just ASCII. A real Telugu call said
#: "ఎలక్ట్రిషన్" where the catalog has "ఎలక్ట్రీషియన్" — 0.83 similar, but with
#: an ASCII-only table it got no rescue at all and the whole sentence was
#: captured as the service instead. A parameter sweep over real Telugu, Hindi,
#: Tamil and Bengali names and places produced no false positives here.
_ALL_SYNONYMS: dict[str, str] = {
    normalize_text(syn): canonical
    for canonical, syns in SERVICE_CATALOG.items()
    for syn in syns
}


def fuzzy_match_service(text: str) -> str | None:
    """Rescue a trade name that speech recognition garbled.

    Only applied word by word and only above `_FUZZY_CUTOFF`; a badly mangled
    word (STT turning "plumber" into "puma") is deliberately *not* rescued,
    because guessing there would put a service in the lead that the caller
    never asked for.

    Two guards keep character similarity from standing in for meaning. Both
    were added after this function silently produced wrong leads:

    1. **A stricter cutoff for Indic scripts**, where one character is a whole
       syllable and a single substitution is a different word rather than a
       slip. "టెన్షన్" (tension, from a caller describing their stress) scored
       0.71 against "ట్యూషన్" (tuition) and filed a plumbing call as a tutor.
       Not a blanket ban: "ఎలక్ట్రిషన్" is a real spelling variant of the
       catalogue's "ఎలక్ట్రీషియన్" and must still be rescued — it scores 0.83,
       so the line sits between them. That margin is thin; widen the catalogue
       with real variants rather than lowering this.

    2. **Comparable length.** A typo barely changes a word's length. A short
       word that merely *sits inside* a longer synonym is not a typo, it is an
       incomplete phrase — and "service" is 7 of the 10 characters in "ac
       service", scoring 0.82. That made every "car service", "bike service"
       and "RO service" come out as AC repair.
    """
    for token in normalize_text(text).split():
        if len(token) < _FUZZY_MIN_LENGTH:
            continue
        cutoff = _FUZZY_CUTOFF if token.isascii() else _FUZZY_CUTOFF_INDIC
        close = difflib.get_close_matches(
            token, list(_ALL_SYNONYMS), n=1, cutoff=cutoff
        )
        if not close:
            continue
        candidate = close[0]
        # The length guard targets a Latin, multi-word failure: a generic word
        # sitting inside a longer synonym ("service" in "ac service"). Indic
        # synonyms here are single words, and applying it there would reject
        # "ఎలక్ట్రిషన్"/"ఎలక్ట్రీషియన్" at 0.846 — a real variant. On that side
        # the stricter cutoff above is what separates good from bad.
        if token.isascii():
            length_ratio = min(len(token), len(candidate)) / max(len(token), len(candidate))
            if length_ratio < _FUZZY_LENGTH_RATIO:
                continue
        return _ALL_SYNONYMS[candidate]
    return None


def match_service(text: str) -> str | None:
    """Return the canonical service label if the text mentions a known trade."""
    normalized = normalize_text(text)
    if not normalized:
        return None
    best: tuple[int, str] | None = None
    for canonical, synonyms in SERVICE_CATALOG.items():
        for synonym in synonyms:
            key = normalize_text(synonym)
            if key and key in normalized:
                # Prefer the longest matching synonym so "ac repair" beats "ac".
                if best is None or len(key) > best[0]:
                    best = (len(key), canonical)
    return best[1] if best else None


def match_city(text: str) -> str | None:
    """Return a seed city mentioned anywhere in the text."""
    normalized = normalize_text(text)
    best: tuple[int, str] | None = None
    for city in CITY_SEEDS:
        if re.search(rf"\b{re.escape(city)}\b", normalized):
            if best is None or len(city) > best[0]:
                best = (len(city), city)
    return best[1].title() if best else None


def extract_name(text: str, *, expecting: bool) -> tuple[str | None, float]:
    for pattern in _NAME_PATTERNS:
        match = pattern.search(text)
        if match:
            candidate = _clean_value(match.group("v"))
            # Keep only the leading name-ish span: "Ravi and I need a plumber"
            candidate = re.split(
                r"\b(?:and|aur|mariyu|matrum|ebong|,)\b", candidate, maxsplit=1
            )[0].strip()
            if _looks_like_name(candidate):
                return format_name(candidate), 0.9
    if expecting:
        candidate = _clean_value(_strip_fillers(text))
        if _looks_like_name(candidate):
            return format_name(candidate), 0.6
    return None, 0.0


def extract_service(text: str, *, expecting: bool) -> tuple[str | None, float]:
    # A caller who answers the service question with a place name has
    # misheard the question — "Madhapur, Hyderabad" was being stored as the
    # requested service *and* the area from one utterance.
    if match_city(text) and not match_service(text):
        return None, 0.0

    canonical = match_service(text)
    if canonical:
        return canonical, 0.9
    fuzzy = fuzzy_match_service(text)
    if fuzzy:
        return fuzzy, 0.85
    for pattern in _SERVICE_PATTERNS:
        match = pattern.search(text)
        if match:
            candidate = _clean_value(match.group("v"))
            # Drop a trailing location clause: "a plumber in Madhapur".
            candidate = re.split(r"\b(?:in|at|near|around)\b", candidate, maxsplit=1)[0]
            candidate = _clean_value(candidate)
            for prefix in _INDIC_REQUEST_PREFIXES:
                if candidate.startswith(prefix):
                    candidate = candidate[len(prefix):].strip()
                    break
            # The catalog may now match what the verb was hiding.
            inner = match_service(candidate) or fuzzy_match_service(candidate)
            if inner:
                return inner, 0.9
            if candidate and len(candidate.split()) <= 5:
                return candidate.lower(), UNKNOWN_SERVICE_CONFIDENCE
    if expecting:
        candidate = _clean_value(_strip_fillers(text))
        if candidate and len(candidate.split()) <= 5:
            return candidate.lower(), UNKNOWN_SERVICE_CONFIDENCE
    return None, 0.0


def extract_location(text: str, *, expecting: bool) -> tuple[str | None, float]:
    city = match_city(text)
    if city:
        # Include a locality qualifier if one precedes the city name.
        match = re.search(
            rf"([\w\s]{{0,30}}?)\b{re.escape(city.lower())}\b", normalize_text(text)
        )
        if match:
            prefix = _clean_value(_strip_fillers(match.group(1)))
            prefix = re.split(r"\b(?:in|at|near|around|from)\b", prefix)[-1].strip()
            if prefix and 1 <= len(prefix.split()) <= 2 and _looks_like_place(prefix):
                return f"{prefix.title()} {city}", 0.9
        return city, 0.9

    # Postpositions only when a location was asked for — they are too eager to
    # trust in other states.
    patterns = (
        _LOCATION_PREPOSITIONS + _LOCATION_POSTPOSITIONS
        if expecting
        else _LOCATION_PREPOSITIONS
    )
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            candidate = _clean_value(match.group("v"))
            candidate = _clean_value(_strip_fillers(candidate))
            if candidate and 1 <= len(candidate.split()) <= 4 and _looks_like_place(candidate):
                return candidate.title(), 0.8
    if expecting:
        candidate = _clean_value(_strip_fillers(text))
        if candidate and 1 <= len(candidate.split()) <= 4 and _looks_like_place(candidate):
            return candidate.title(), 0.6
    return None, 0.0


#: Words that make a phrase a sentence fragment rather than a place name. A
#: live call produced "I need a please" (STT for "I need a plumber please") at
#: the location step, and "I Need A" was captured as the city. Checking the
#: whole string against the stopword set missed it — the check has to be
#: per-token.
_NOT_A_PLACE_TOKENS = {
    "i", "a", "an", "the", "need", "needed", "want", "wanted", "looking",
    "require", "please", "service", "help", "give", "send", "get", "find",
    "is", "are", "am", "was", "do", "does", "can", "could", "would", "will",
    "chahiye", "kavali", "vendum", "beku", "dorkar",
}


def _looks_like_place(candidate: str) -> bool:
    if any(ch.isdigit() for ch in candidate):
        return False
    if match_service(candidate):
        return False
    tokens = [t.casefold().strip(".,!?") for t in candidate.split()]
    if any(t in _NOT_A_PLACE_TOKENS for t in tokens):
        return False
    lowered = candidate.casefold()
    return lowered not in _NAME_STOPWORDS and lowered not in _FILLERS


def extract(text: str, expecting: ConversationState | None = None) -> Extraction:
    """Extract every field we can find, biased toward the field we asked for.

    `expecting` enables positional inference for exactly one field — if we just
    asked "what is your name?", a bare "Ravi" is a name. Without it, a bare
    phrase is too ambiguous to assign and is left alone.
    """
    text = (text or "").strip()
    if not text:
        return Extraction(raw_text=text, source="none")

    name, name_conf = extract_name(text, expecting=expecting is ConversationState.ASK_NAME)
    service, service_conf = extract_service(
        text, expecting=expecting is ConversationState.ASK_SERVICE
    )
    location, location_conf = extract_location(
        text, expecting=expecting is ConversationState.ASK_LOCATION
    )

    # A bare answer to one question must not bleed into the other fields.
    if expecting is ConversationState.ASK_NAME and name_conf >= 0.6:
        if service_conf < 0.9:
            service, service_conf = None, 0.0
        if location_conf < 0.9:
            location, location_conf = None, 0.0
    if expecting is ConversationState.ASK_LOCATION and location_conf >= 0.6:
        if name_conf < 0.9:
            name, name_conf = None, 0.0
    if expecting is ConversationState.ASK_SERVICE and service_conf >= 0.6:
        if name_conf < 0.9:
            name, name_conf = None, 0.0

    language = (
        resolve_language(text) if expecting is ConversationState.LANGUAGE_SELECTION else None
    )
    # A resolved language is a definite match (explicit name or script), so it
    # carries its own high confidence. Without this the language turn would
    # look unresolved and trigger a pointless LLM fallback.
    language_conf = 0.9 if language else 0.0

    confidences = [
        c for c in (name_conf, service_conf, location_conf, language_conf) if c > 0
    ]
    return Extraction(
        name=name,
        requested_service=service,
        city_or_area=location,
        language=language,
        confidence=max(confidences) if confidences else 0.0,
        raw_text=text,
        source="rules" if confidences or language else "none",
    )
