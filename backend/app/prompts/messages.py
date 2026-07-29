"""Per-language spoken message templates.

Every user-facing string lives here, keyed by `MessageKey` then `Language`.
The state machine references only `MessageKey`, so adding a language is a
matter of adding a column to this table — no control flow changes.

Translations were authored for the MVP and should be reviewed by native
speakers before production use. Keep utterances short: these are spoken aloud,
so no markdown, no emoji, no lists.
"""

from __future__ import annotations

from enum import Enum

from ..languages import Language


class MessageKey(str, Enum):
    WELCOME = "welcome"
    WELCOME_BACK = "welcome_back"
    ASK_LANGUAGE = "ask_language"
    LANGUAGE_NOT_UNDERSTOOD = "language_not_understood"
    ASK_NAME = "ask_name"
    REPROMPT_NAME = "reprompt_name"
    ASK_SERVICE = "ask_service"
    ASK_SERVICE_RETURNING = "ask_service_returning"
    REPROMPT_SERVICE = "reprompt_service"
    ASK_LOCATION = "ask_location"
    REPROMPT_LOCATION = "reprompt_location"
    REVIEW = "review"
    REVIEW_WHAT_TO_CHANGE = "review_what_to_change"
    CONFIRMATION = "confirmation"
    CLOSING = "closing"
    DIDNT_CATCH = "didnt_catch"
    # Small talk. These are acknowledgements only — the current question is
    # appended by the conversation manager, so the flow never advances on them.
    SMALLTALK_GREETING = "smalltalk_greeting"
    SMALLTALK_HOW_ARE_YOU = "smalltalk_how_are_you"
    SMALLTALK_IDENTITY = "smalltalk_identity"
    SMALLTALK_THANKS = "smalltalk_thanks"
    SMALLTALK_EMPATHY = "smalltalk_empathy"
    SMALLTALK_ABUSE = "smalltalk_abuse"
    SMALLTALK_REDIRECT = "smalltalk_redirect"


_MESSAGES: dict[MessageKey, dict[Language, str]] = {
    MessageKey.WELCOME: {
        Language.ENGLISH: "Welcome to Local Mama! You can call me Mami.",
        Language.HINDI: "लोकल मामा में आपका स्वागत है! आप मुझे मामी कह सकते हैं।",
        Language.BENGALI: "লোকাল মামায় আপনাকে স্বাগতম! আপনি আমাকে মামি বলে ডাকতে পারেন।",
        Language.TELUGU: "లోకల్ మామాకు స్వాగతం! మీరు నన్ను మామి అని పిలవచ్చు.",
        Language.TAMIL: "லோக்கல் மாமாவிற்கு வரவேற்கிறோம்! நீங்கள் என்னை மாமி என்று அழைக்கலாம்.",
        Language.KANNADA: "ಲೋಕಲ್ ಮಾಮಾಗೆ ಸ್ವಾಗತ! ನೀವು ನನ್ನನ್ನು ಮಾಮಿ ಎಂದು ಕರೆಯಬಹುದು.",
    },
    MessageKey.WELCOME_BACK: {
        Language.ENGLISH: "Welcome back to Local Mama, {name}! Mami here.",
        Language.HINDI: "लोकल मामा में आपका फिर से स्वागत है, {name}! मामी बोल रही हूँ।",
        Language.BENGALI: "লোকাল মামায় আবার স্বাগতম, {name}! আমি মামি বলছি।",
        Language.TELUGU: "లోకల్ మామాకు మళ్ళీ స్వాగతం, {name}! నేను మామిని.",
        Language.TAMIL: "லோக்கல் மாமாவிற்கு மீண்டும் வரவேற்கிறோம், {name}! நான் மாமி பேசுகிறேன்.",
        Language.KANNADA: "ಲೋಕಲ್ ಮಾಮಾಗೆ ಮತ್ತೆ ಸ್ವಾಗತ, {name}! ನಾನು ಮಾಮಿ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ.",
    },
    # Deliberately does NOT recite the six languages.
    #
    # It used to, and the greeting took 10.8 seconds to speak — measured off a
    # live call. Callers hung up during it: on one recorded call the caller
    # spoke 4s in and disconnected before Mami finished. The list is also
    # unnecessary, because a caller who wants Telugu simply says "Telugu"; the
    # only person who needs the menu is one who did not answer, and
    # LANGUAGE_NOT_UNDERSTOOD gives them the full list.
    MessageKey.ASK_LANGUAGE: {
        Language.ENGLISH: "Which language would you like to speak?",
        Language.HINDI: "आप किस भाषा में बात करना चाहेंगे?",
        Language.BENGALI: "আপনি কোন ভাষায় কথা বলতে চান?",
        Language.TELUGU: "మీరు ఏ భాషలో మాట్లాడాలనుకుంటున్నారు?",
        Language.TAMIL: "நீங்கள் எந்த மொழியில் பேச விரும்புகிறீர்கள்?",
        Language.KANNADA: "ನೀವು ಯಾವ ಭಾಷೆಯಲ್ಲಿ ಮಾತನಾಡಲು ಬಯಸುತ್ತೀರಿ?",
    },
    MessageKey.LANGUAGE_NOT_UNDERSTOOD: {
        Language.ENGLISH: (
            "Sorry Mama, I did not catch that. Please say English, Hindi, "
            "Bengali, Telugu, Tamil, or Kannada."
        ),
        Language.HINDI: (
            "माफ़ कीजिए मामा, मैं समझ नहीं पाई। कृपया कहिए अंग्रेज़ी, हिंदी, "
            "बंगाली, तेलुगु, तमिल या कन्नड़।"
        ),
        Language.BENGALI: (
            "দুঃখিত মামা, আমি বুঝতে পারিনি। অনুগ্রহ করে বলুন ইংরেজি, হিন্দি, "
            "বাংলা, তেলুগু, তামিল বা কন্নড়।"
        ),
        Language.TELUGU: (
            "క్షమించండి మామా, నాకు అర్థం కాలేదు. దయచేసి ఇంగ్లీష్, హిందీ, "
            "బెంగాలీ, తెలుగు, తమిళం లేదా కన్నడ అని చెప్పండి."
        ),
        Language.TAMIL: (
            "மன்னிக்கவும் மாமா, எனக்குப் புரியவில்லை. தயவுசெய்து ஆங்கிலம், இந்தி, "
            "வங்காளம், தெலுங்கு, தமிழ் அல்லது கன்னடம் என்று சொல்லுங்கள்."
        ),
        Language.KANNADA: (
            "ಕ್ಷಮಿಸಿ ಮಾಮಾ, ನನಗೆ ಅರ್ಥವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಇಂಗ್ಲಿಷ್, ಹಿಂದಿ, "
            "ಬಂಗಾಳಿ, ತೆಲುಗು, ತಮಿಳು ಅಥವಾ ಕನ್ನಡ ಎಂದು ಹೇಳಿ."
        ),
    },
    MessageKey.ASK_NAME: {
        Language.ENGLISH: "Got it, Mama! May I know your name?",
        Language.HINDI: "ठीक है मामा! क्या मैं आपका नाम जान सकती हूँ?",
        Language.BENGALI: "ঠিক আছে মামা! আপনার নামটা জানতে পারি?",
        Language.TELUGU: "సరే మామా! మీ పేరు తెలుసుకోవచ్చా?",
        Language.TAMIL: "சரி மாமா! உங்கள் பெயரைத் தெரிந்து கொள்ளலாமா?",
        Language.KANNADA: "ಸರಿ ಮಾಮಾ! ನಿಮ್ಮ ಹೆಸರು ತಿಳಿಯಬಹುದೇ?",
    },
    MessageKey.REPROMPT_NAME: {
        Language.ENGLISH: "Sorry Mama, I missed your name. Could you say just your name?",
        Language.HINDI: "माफ़ कीजिए मामा, आपका नाम समझ नहीं आया। सिर्फ़ अपना नाम बताइए।",
        Language.BENGALI: "দুঃখিত মামা, আপনার নামটা বুঝতে পারিনি। শুধু আপনার নামটা বলুন।",
        Language.TELUGU: "క్షమించండి మామా, మీ పేరు వినిపించలేదు. మీ పేరు మాత్రం చెప్పండి.",
        Language.TAMIL: "மன்னிக்கவும் மாமா, உங்கள் பெயர் புரியவில்லை. உங்கள் பெயரை மட்டும் சொல்லுங்கள்.",
        Language.KANNADA: "ಕ್ಷಮಿಸಿ ಮಾಮಾ, ನಿಮ್ಮ ಹೆಸರು ಕೇಳಿಸಲಿಲ್ಲ. ನಿಮ್ಮ ಹೆಸರನ್ನು ಮಾತ್ರ ಹೇಳಿ.",
    },
    MessageKey.ASK_SERVICE: {
        Language.ENGLISH: "Nice to meet you, {name}! What service are you looking for today?",
        Language.HINDI: "आपसे मिलकर अच्छा लगा, {name}! आज आपको कौन सी सेवा चाहिए?",
        Language.BENGALI: "আপনার সঙ্গে পরিচিত হয়ে ভালো লাগল, {name}! আজ আপনার কোন পরিষেবা দরকার?",
        Language.TELUGU: "మిమ్మల్ని కలవడం సంతోషం, {name}! ఈరోజు మీకు ఏ సేవ కావాలి?",
        Language.TAMIL: "உங்களைச் சந்தித்ததில் மகிழ்ச்சி, {name}! இன்று உங்களுக்கு என்ன சேவை தேவை?",
        Language.KANNADA: "ನಿಮ್ಮನ್ನು ಭೇಟಿಯಾಗಿ ಸಂತೋಷವಾಯಿತು, {name}! ಇಂದು ನಿಮಗೆ ಯಾವ ಸೇವೆ ಬೇಕು?",
    },
    # A returning caller has already been met and already been named in the
    # welcome-back line — repeating "nice to meet you, {name}" reads as amnesia.
    MessageKey.ASK_SERVICE_RETURNING: {
        Language.ENGLISH: "What service do you need today?",
        Language.HINDI: "आज आपको कौन सी सेवा चाहिए?",
        Language.BENGALI: "আজ আপনার কোন পরিষেবা দরকার?",
        Language.TELUGU: "ఈరోజు మీకు ఏ సేవ కావాలి?",
        Language.TAMIL: "இன்று உங்களுக்கு என்ன சேவை தேவை?",
        Language.KANNADA: "ಇಂದು ನಿಮಗೆ ಯಾವ ಸೇವೆ ಬೇಕು?",
    },
    MessageKey.REPROMPT_SERVICE: {
        Language.ENGLISH: (
            "Sorry Mama, which service do you need? For example an electrician, "
            "a plumber, or a carpenter."
        ),
        Language.HINDI: (
            "माफ़ कीजिए मामा, आपको कौन सी सेवा चाहिए? जैसे बिजली मिस्त्री, "
            "प्लंबर या बढ़ई।"
        ),
        Language.BENGALI: (
            "দুঃখিত মামা, আপনার কোন পরিষেবা দরকার? যেমন ইলেকট্রিশিয়ান, "
            "প্লাম্বার বা মিস্ত্রি।"
        ),
        Language.TELUGU: (
            "క్షమించండి మామా, మీకు ఏ సేవ కావాలి? ఉదాహరణకు ఎలక్ట్రీషియన్, "
            "ప్లంబర్ లేదా వడ్రంగి."
        ),
        Language.TAMIL: (
            "மன்னிக்கவும் மாமா, உங்களுக்கு எந்தச் சேவை தேவை? உதாரணமாக மின்சாரம், "
            "பிளம்பர் அல்லது தச்சர்."
        ),
        Language.KANNADA: (
            "ಕ್ಷಮಿಸಿ ಮಾಮಾ, ನಿಮಗೆ ಯಾವ ಸೇವೆ ಬೇಕು? ಉದಾಹರಣೆಗೆ ಎಲೆಕ್ಟ್ರೀಷಿಯನ್, "
            "ಪ್ಲಂಬರ್ ಅಥವಾ ಬಡಗಿ."
        ),
    },
    MessageKey.ASK_LOCATION: {
        Language.ENGLISH: "Got it. Which city or area are you looking for this service in?",
        Language.HINDI: "ठीक है। आपको यह सेवा किस शहर या इलाके में चाहिए?",
        Language.BENGALI: "বুঝেছি। কোন শহরে বা এলাকায় আপনার এই পরিষেবাটি দরকার?",
        Language.TELUGU: "అర్థమైంది. ఈ సేవ మీకు ఏ నగరంలో లేదా ఏ ప్రాంతంలో కావాలి?",
        Language.TAMIL: "புரிந்தது. இந்தச் சேவை உங்களுக்கு எந்த நகரத்தில் அல்லது பகுதியில் தேவை?",
        Language.KANNADA: "ಅರ್ಥವಾಯಿತು. ಈ ಸೇವೆ ನಿಮಗೆ ಯಾವ ನಗರ ಅಥವಾ ಪ್ರದೇಶದಲ್ಲಿ ಬೇಕು?",
    },
    MessageKey.REPROMPT_LOCATION: {
        Language.ENGLISH: "Sorry Mama, which city or area should I look in?",
        Language.HINDI: "माफ़ कीजिए मामा, मैं किस शहर या इलाके में देखूँ?",
        Language.BENGALI: "দুঃখিত মামা, আমি কোন শহরে বা এলাকায় খুঁজব?",
        Language.TELUGU: "క్షమించండి మామా, నేను ఏ నగరంలో లేదా ప్రాంతంలో చూడాలి?",
        Language.TAMIL: "மன்னிக்கவும் மாமா, நான் எந்த நகரத்தில் அல்லது பகுதியில் தேட வேண்டும்?",
        Language.KANNADA: "ಕ್ಷಮಿಸಿ ಮಾಮಾ, ನಾನು ಯಾವ ನಗರ ಅಥವಾ ಪ್ರದೇಶದಲ್ಲಿ ಹುಡುಕಬೇಕು?",
    },
    MessageKey.REVIEW: {
        Language.ENGLISH: "Let me just confirm, Mama. {service} in {location}, for {name}. Is that right?",
        Language.HINDI: "मामा, एक बार पक्का कर लूँ। {location} में {service}, {name} के लिए। सही है?",
        Language.BENGALI: "মামা, একবার নিশ্চিত করে নিই। {location}-এ {service}, {name}-এর জন্য। ঠিক আছে?",
        Language.TELUGU: "మామా, ఒకసారి నిర్ధారించుకుంటాను. {location}లో {service}, {name} గారికి. సరైనదేనా?",
        Language.TAMIL: "மாமா, ஒருமுறை உறுதி செய்கிறேன். {location}ல் {service}, {name} அவர்களுக்கு. சரியா?",
        Language.KANNADA: "ಮಾಮಾ, ಒಮ್ಮೆ ಖಚಿತಪಡಿಸಿಕೊಳ್ಳುತ್ತೇನೆ. {location} ನಲ್ಲಿ {service}, {name} ಅವರಿಗೆ. ಸರಿಯೇ?",
    },
    MessageKey.REVIEW_WHAT_TO_CHANGE: {
        Language.ENGLISH: "Sorry Mama, what should I change? Your name, the service, or the area?",
        Language.HINDI: "माफ़ कीजिए मामा, क्या बदलना है? आपका नाम, सेवा, या इलाका?",
        Language.BENGALI: "দুঃখিত মামা, কী বদলাব? আপনার নাম, পরিষেবা, নাকি এলাকা?",
        Language.TELUGU: "క్షమించండి మామా, ఏది మార్చాలి? మీ పేరు, సేవ, లేదా ప్రాంతమా?",
        Language.TAMIL: "மன்னிக்கவும் மாமா, எதை மாற்ற வேண்டும்? உங்கள் பெயர், சேவை, அல்லது பகுதி?",
        Language.KANNADA: "ಕ್ಷಮಿಸಿ ಮಾಮಾ, ಏನು ಬದಲಾಯಿಸಬೇಕು? ನಿಮ್ಮ ಹೆಸರು, ಸೇವೆ, ಅಥವಾ ಪ್ರದೇಶ?",
    },
    MessageKey.CONFIRMATION: {
        Language.ENGLISH: (
            "Perfect, Mama! I will send the best matching {service} details in "
            "{location} to your WhatsApp in a few moments."
        ),
        Language.HINDI: (
            "बहुत बढ़िया मामा! मैं {location} में {service} की सबसे अच्छी जानकारी "
            "थोड़ी देर में आपके व्हाट्सएप पर भेज दूँगी।"
        ),
        Language.BENGALI: (
            "দারুণ মামা! {location}-এ {service}-এর সেরা তথ্য আমি একটু পরেই "
            "আপনার হোয়াটসঅ্যাপে পাঠিয়ে দেব।"
        ),
        Language.TELUGU: (
            "చాలా బాగుంది మామా! {location}లో {service} కోసం ఉత్తమమైన వివరాలను "
            "కొద్ది సేపట్లో మీ వాట్సాప్‌కు పంపిస్తాను."
        ),
        Language.TAMIL: (
            "அருமை மாமா! {location}ல் {service} க்கான சிறந்த விவரங்களை "
            "சிறிது நேரத்தில் உங்கள் வாட்ஸ்அப்பிற்கு அனுப்புகிறேன்."
        ),
        Language.KANNADA: (
            "ಚೆನ್ನಾಗಿದೆ ಮಾಮಾ! {location} ನಲ್ಲಿ {service} ಗಾಗಿ ಅತ್ಯುತ್ತಮ ವಿವರಗಳನ್ನು "
            "ಸ್ವಲ್ಪ ಸಮಯದಲ್ಲಿ ನಿಮ್ಮ ವಾಟ್ಸಾಪ್‌ಗೆ ಕಳುಹಿಸುತ್ತೇನೆ."
        ),
    },
    MessageKey.CLOSING: {
        Language.ENGLISH: (
            "Thank you for choosing Local Mama. Whenever you need any local "
            "service, just call Local Mama. Have a wonderful day, Mama!"
        ),
        Language.HINDI: (
            "लोकल मामा चुनने के लिए धन्यवाद। जब भी कोई स्थानीय सेवा चाहिए, "
            "बस लोकल मामा को बुलाइए। आपका दिन शुभ हो, मामा!"
        ),
        Language.BENGALI: (
            "লোকাল মামা বেছে নেওয়ার জন্য ধন্যবাদ। যখনই কোনো স্থানীয় পরিষেবা "
            "দরকার হবে, শুধু লোকাল মামাকে ডাকবেন। আপনার দিনটি ভালো কাটুক, মামা!"
        ),
        Language.TELUGU: (
            "లోకల్ మామాను ఎంచుకున్నందుకు ధన్యవాదాలు. ఎప్పుడైనా స్థానిక సేవ "
            "కావాలంటే, లోకల్ మామాకు కాల్ చేయండి. మీ రోజు ఆనందంగా గడవాలి, మామా!"
        ),
        Language.TAMIL: (
            "லோக்கல் மாமாவைத் தேர்ந்தெடுத்ததற்கு நன்றி. எப்போது உள்ளூர் சேவை "
            "தேவைப்பட்டாலும், லோக்கல் மாமாவை அழையுங்கள். இனிய நாள் ஆகட்டும், மாமா!"
        ),
        Language.KANNADA: (
            "ಲೋಕಲ್ ಮಾಮಾವನ್ನು ಆಯ್ಕೆ ಮಾಡಿದ್ದಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು. ಯಾವಾಗ ಬೇಕಾದರೂ "
            "ಸ್ಥಳೀಯ ಸೇವೆ ಬೇಕಿದ್ದರೆ, ಲೋಕಲ್ ಮಾಮಾಗೆ ಕರೆ ಮಾಡಿ. ನಿಮ್ಮ ದಿನ ಶುಭವಾಗಲಿ, ಮಾಮಾ!"
        ),
    },
    MessageKey.DIDNT_CATCH: {
        Language.ENGLISH: "Sorry Mama, I did not catch that. Could you say it again?",
        Language.HINDI: "माफ़ कीजिए मामा, मैं समझ नहीं पाई। कृपया दोबारा कहिए।",
        Language.BENGALI: "দুঃখিত মামা, আমি বুঝতে পারিনি। আবার একবার বলবেন?",
        Language.TELUGU: "క్షమించండి మామా, నాకు అర్థం కాలేదు. మళ్ళీ చెబుతారా?",
        Language.TAMIL: "மன்னிக்கவும் மாமா, எனக்குப் புரியவில்லை. மீண்டும் சொல்லுங்கள்.",
        Language.KANNADA: "ಕ್ಷಮಿಸಿ ಮಾಮಾ, ನನಗೆ ಅರ್ಥವಾಗಲಿಲ್ಲ. ಇನ್ನೊಮ್ಮೆ ಹೇಳುತ್ತೀರಾ?",
    },
    MessageKey.SMALLTALK_GREETING: {
        Language.ENGLISH: "Namaste Mama, lovely to hear from you!",
        Language.HINDI: "नमस्ते मामा, आपसे बात करके अच्छा लगा!",
        Language.BENGALI: "নমস্কার মামা, আপনার সঙ্গে কথা বলে ভালো লাগল!",
        Language.TELUGU: "నమస్తే మామా, మీతో మాట్లాడటం సంతోషం!",
        Language.TAMIL: "வணக்கம் மாமா, உங்களுடன் பேசுவதில் மகிழ்ச்சி!",
        Language.KANNADA: "ನಮಸ್ಕಾರ ಮಾಮಾ, ನಿಮ್ಮೊಂದಿಗೆ ಮಾತನಾಡಿ ಸಂತೋಷವಾಯಿತು!",
    },
    MessageKey.SMALLTALK_HOW_ARE_YOU: {
        Language.ENGLISH: "I am doing well, thank you for asking, Mama!",
        Language.HINDI: "मैं ठीक हूँ मामा, पूछने के लिए धन्यवाद!",
        Language.BENGALI: "আমি ভালো আছি মামা, জিজ্ঞেস করার জন্য ধন্যবাদ!",
        Language.TELUGU: "నేను బాగున్నాను మామా, అడిగినందుకు ధన్యవాదాలు!",
        Language.TAMIL: "நான் நலமாக இருக்கிறேன் மாமா, கேட்டதற்கு நன்றி!",
        Language.KANNADA: "ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ ಮಾಮಾ, ಕೇಳಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದಗಳು!",
    },
    MessageKey.SMALLTALK_IDENTITY: {
        Language.ENGLISH: (
            "I am Mami, your helper from Local Mama. I find trusted local "
            "service people for you."
        ),
        Language.HINDI: (
            "मैं मामी हूँ, लोकल मामा से आपकी मददगार। मैं आपके लिए भरोसेमंद "
            "स्थानीय कारीगर ढूँढती हूँ।"
        ),
        Language.BENGALI: (
            "আমি মামি, লোকাল মামা থেকে আপনার সহায়ক। আমি আপনার জন্য বিশ্বস্ত "
            "স্থানীয় কর্মী খুঁজে দিই।"
        ),
        Language.TELUGU: (
            "నేను మామిని, లోకల్ మామా నుండి మీ సహాయకురాలిని. మీ కోసం నమ్మకమైన "
            "స్థానిక పనివారిని వెతుకుతాను."
        ),
        Language.TAMIL: (
            "நான் மாமி, லோக்கல் மாமாவிலிருந்து உங்கள் உதவியாளர். உங்களுக்கு "
            "நம்பகமான உள்ளூர் பணியாளர்களைத் தேடித் தருகிறேன்."
        ),
        Language.KANNADA: (
            "ನಾನು ಮಾಮಿ, ಲೋಕಲ್ ಮಾಮಾದಿಂದ ನಿಮ್ಮ ಸಹಾಯಕಿ. ನಿಮಗಾಗಿ ವಿಶ್ವಾಸಾರ್ಹ "
            "ಸ್ಥಳೀಯ ಕೆಲಸಗಾರರನ್ನು ಹುಡುಕಿಕೊಡುತ್ತೇನೆ."
        ),
    },
    MessageKey.SMALLTALK_THANKS: {
        Language.ENGLISH: "You are most welcome, Mama!",
        Language.HINDI: "आपका स्वागत है मामा!",
        Language.BENGALI: "আপনাকে স্বাগতম মামা!",
        Language.TELUGU: "మీకు స్వాగతం మామా!",
        Language.TAMIL: "வரவேற்கிறேன் மாமா!",
        Language.KANNADA: "ಸ್ವಾಗತ ಮಾಮಾ!",
    },
    MessageKey.SMALLTALK_EMPATHY: {
        Language.ENGLISH: (
            "Oh no, that sounds stressful, Mama. Do not worry, I will get you "
            "the right help quickly."
        ),
        Language.HINDI: (
            "अरे, यह तो परेशानी की बात है मामा। चिंता मत कीजिए, मैं जल्दी ही "
            "सही मदद दिलवाती हूँ।"
        ),
        Language.BENGALI: (
            "ইশ, এটা তো খুব চিন্তার বিষয় মামা। চিন্তা করবেন না, আমি তাড়াতাড়ি "
            "সঠিক সাহায্য এনে দেব।"
        ),
        Language.TELUGU: (
            "అయ్యో, అది కష్టంగా ఉంది మామా. చింతించకండి, నేను త్వరగా సరైన "
            "సహాయం అందిస్తాను."
        ),
        Language.TAMIL: (
            "ஐயோ, அது கஷ்டமாக இருக்கும் மாமா. கவலைப்படாதீர்கள், நான் விரைவில் "
            "சரியான உதவியை ஏற்பாடு செய்கிறேன்."
        ),
        Language.KANNADA: (
            "ಅಯ್ಯೋ, ಅದು ಕಷ್ಟದ ವಿಷಯ ಮಾಮಾ. ಚಿಂತಿಸಬೇಡಿ, ನಾನು ಬೇಗ ಸರಿಯಾದ "
            "ಸಹಾಯ ಒದಗಿಸುತ್ತೇನೆ."
        ),
    },
    MessageKey.SMALLTALK_ABUSE: {
        Language.ENGLISH: "I understand you are upset, Mama. I am here to help.",
        Language.HINDI: "मैं समझती हूँ आप परेशान हैं मामा। मैं मदद के लिए यहीं हूँ।",
        Language.BENGALI: "আমি বুঝতে পারছি আপনি বিরক্ত মামা। আমি সাহায্য করতে এখানে আছি।",
        Language.TELUGU: "మీరు కలత చెందారని అర్థమైంది మామా. నేను సహాయం చేయడానికి ఉన్నాను.",
        Language.TAMIL: "நீங்கள் வருத்தமாக இருக்கிறீர்கள் என்று புரிகிறது மாமா. உதவ நான் இருக்கிறேன்.",
        Language.KANNADA: "ನೀವು ಬೇಸರಗೊಂಡಿದ್ದೀರಿ ಎಂದು ಅರ್ಥವಾಯಿತು ಮಾಮಾ. ಸಹಾಯ ಮಾಡಲು ನಾನಿದ್ದೇನೆ.",
    },
    MessageKey.SMALLTALK_REDIRECT: {
        Language.ENGLISH: "I can only help with local services, Mama.",
        Language.HINDI: "मैं सिर्फ़ स्थानीय सेवाओं में मदद कर सकती हूँ मामा।",
        Language.BENGALI: "আমি শুধু স্থানীয় পরিষেবায় সাহায্য করতে পারি মামা।",
        Language.TELUGU: "నేను స్థానిక సేవల్లో మాత్రమే సహాయం చేయగలను మామా.",
        Language.TAMIL: "நான் உள்ளூர் சேவைகளில் மட்டுமே உதவ முடியும் மாமா.",
        Language.KANNADA: "ನಾನು ಸ್ಥಳೀಯ ಸೇವೆಗಳಲ್ಲಿ ಮಾತ್ರ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ ಮಾಮಾ.",
    },
}


def get_message(key: MessageKey, language: Language, **kwargs: str) -> str:
    """Render a message template for a language.

    Falls back to English if a translation is missing, so a partially
    translated new language can never crash a live call.
    """
    template = _MESSAGES[key].get(language) or _MESSAGES[key][Language.ENGLISH]
    if kwargs:
        return template.format(**kwargs)
    return template


def raw_template(key: MessageKey, language: Language) -> str:
    """The template before interpolation, so callers can see its placeholders."""
    return _MESSAGES[key].get(language) or _MESSAGES[key][Language.ENGLISH]


def missing_translations() -> dict[str, list[str]]:
    """Report keys lacking a translation. Used by the test suite."""
    gaps: dict[str, list[str]] = {}
    for key, table in _MESSAGES.items():
        absent = [lang.value for lang in Language if lang not in table]
        if absent:
            gaps[key.value] = absent
    return gaps
