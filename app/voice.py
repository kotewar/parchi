"""Speak the counselling out loud, in the patient's own language.

A pharmacist says the same thing twenty times a day, and the patient forgets it
before they reach the door. Written instructions do not help if the patient
reads slowly or not at all, which across this customer base is common enough
that assuming otherwise quietly loses people their treatment. So the counselling
is played at the counter, as audio, from the prescription actually in hand.

This is the ONLY place in Parchi where anything leaves the machine, and that is
worth being blunt about, because the rest of the project is built on the
opposite claim.

What goes to ElevenLabs is the counselling text, which necessarily contains the
medicine names and how often to take them. That IS the counselling; there is no
version of speaking it aloud that leaves it out. What does not go: the page
image, the patient's name, the diagnosis, the prescriber, and any prescription
id. So the disclosure is smaller than the page and considerably larger than
nothing, and it is a deliberate trade that the pharmacist should get to make
rather than one the system should make quietly.

That is why it is opt-in per page and never automatic. The pharmacist presses
play, or nothing leaves the shop.

And if the key is missing, the plan is exhausted, or the network is down, the
text is still returned and the pharmacist still reads it aloud. A dead
text-to-speech key must never cost a patient their instructions.
"""

from __future__ import annotations

import base64
import re

import httpx

from .config import Settings, get_settings
from .formulary import food_timing
from .models import RxLine

_SIG = re.compile(r"^\d+(-\d+){1,3}$")

_SLOTS_HI = ["सुबह", "दोपहर", "रात", "सोने से पहले"]
_SLOTS_EN = ["morning", "afternoon", "night", "at bedtime"]

_COUNT_HI = {1: "एक", 2: "दो", 3: "तीन", 4: "चार"}
_COUNT_EN = {1: "once", 2: "twice", 3: "three times", 4: "four times"}

_SHORTHAND = {
    "sos": ("ज़रूरत पड़ने पर", "if needed"),
    "stat": ("तुरंत", "immediately"),
    "hs": ("सोने से पहले", "at bedtime"),
    "od": ("दिन में एक बार", "once a day"),
    "bd": ("दिन में दो बार", "twice a day"),
    "tds": ("दिन में तीन बार", "three times a day"),
    "qid": ("दिन में चार बार", "four times a day"),
}

_FOOD_HI = {
    "before food": "खाने से पहले",
    "with food": "खाने के साथ",
    "after food": "खाने के बाद",
    "first thing in the morning, on an empty stomach": "सुबह खाली पेट, सबसे पहले",
    "rinse the mouth after each use": "हर बार इस्तेमाल के बाद कुल्ला कीजिए",
    "as the doctor has set out, with the injection": "डॉक्टर के बताए अनुसार, इंजेक्शन के साथ",
    "at the same time each day": "रोज़ एक ही समय पर",
}


class VoiceUnavailable(RuntimeError):
    """Raised when text-to-speech could not be reached."""


def display_name(line: RxLine) -> str:
    """The medicine as the patient will see it printed on the strip.

    The strength is appended only when the name does not already carry it.
    "Glycomet GP2" with a strength of "GP2" is one medicine, not "Glycomet
    GP2 GP2", and a name stuttered aloud is the kind of small wrongness that
    makes a patient stop trusting everything else they were told.
    """
    name = line.drug or ""
    strength = (line.strength or "").strip()
    if not strength:
        return name

    # Compare with punctuation removed, so "Glycomet GP-2" is recognised too.
    compact = re.sub(r"[^a-z0-9]", "", name.lower())
    wanted = re.sub(r"[^a-z0-9]", "", strength.lower())
    if wanted and wanted in compact:
        return name

    return f"{name} {strength}"


def times_of_day(sig: str | None, *, lang: str = "hi") -> str | None:
    """Turn a dosing pattern into the times it names.

    A two-part pattern is deliberately NOT resolved to named slots. Clinics in
    this market use "1-1" to mean different things, and guessing which would be
    inventing an instruction the doctor did not write, so we say "twice a day"
    and leave the pharmacist to say the rest.
    """
    if not sig:
        return None

    hindi = lang == "hi"
    cleaned = sig.strip().lower().replace(" ", "")

    if cleaned in _SHORTHAND:
        return _SHORTHAND[cleaned][0 if hindi else 1]
    if not _SIG.match(cleaned):
        return None

    parts = [int(part) for part in cleaned.split("-")]
    active = [(index, count) for index, count in enumerate(parts) if count > 0]
    if not active:
        return None

    total = sum(count for _, count in active)
    slots = _SLOTS_HI if hindi else _SLOTS_EN

    if len(parts) == 2 or len(active) > len(slots):
        if hindi:
            return f"दिन में {_COUNT_HI.get(total, str(total))} बार"
        word = _COUNT_EN.get(total, f"{total} times")
        return f"{word} a day"

    names = [slots[index] for index, _ in active]
    if len(names) == 1:
        return names[0]
    joiner = " और " if hindi else " and "
    return ", ".join(names[:-1]) + joiner + names[-1]


def medicines_block(lines: list[RxLine]) -> str:
    """Render the prescription for the counselling prompt.

    The times and the food timing are resolved here, in code, and handed to the
    model as settled facts. The model's job is to say them warmly in the
    patient's own language -- not to work out that "1-0-0" means morning, and
    certainly not to guess whether pantoprazole goes before or after a meal.
    """
    out: list[str] = []
    for index, line in enumerate(lines, start=1):
        if not line.drug:
            continue

        bits = [display_name(line)]
        when = times_of_day(line.sig, lang="en")
        if when:
            bits.append(f"taken in the {when}")
        timing = food_timing(line.drug)
        if timing:
            bits.append(timing)
        if line.days:
            bits.append(f"for {line.days} days")

        out.append(f"{index}. " + ", ".join(bits))

    if not out:
        return "(no medicines could be read off this page)"
    return "\n".join(out)


def compose_counselling(
    lines: list[RxLine],
    *,
    patient_name: str | None = None,
    language: str = "Hindi",
) -> str:
    """Deterministic counselling text, used when the model is unavailable.

    Deliberately blunt and repetitive. It states the time of day and the food
    timing for every medicine, because that is the instruction patients most
    often get wrong, and it adds nothing that is not on the prescription.
    """
    if not lines:
        return "इस पर्ची पर कोई दवा नहीं पढ़ी जा सकी। कृपया दुकान पर दिखाइए।"

    opener = f"{patient_name} जी, " if patient_name else ""
    parts = [f"{opener}ये दवाइयाँ इस तरह लेनी हैं।"]

    for index, line in enumerate(lines, start=1):
        if not line.drug:
            continue

        instructions = []
        when = times_of_day(line.sig)
        if when:
            instructions.append(when)
        timing = food_timing(line.drug)
        if timing:
            instructions.append(_FOOD_HI.get(timing, timing))

        sentence = f"{index}. {display_name(line)}"
        if instructions:
            sentence += " — " + ", ".join(instructions)
        if line.days:
            sentence += f"। {line.days} दिन तक"
        parts.append(sentence + "।")

    parts.append("हर दवा रोज़ एक ही समय पर लीजिए।")
    parts.append(
        "कोई खुराक भूल जाएँ तो याद आने पर ले लीजिए, लेकिन अगली खुराक का समय "
        "पास हो तो दो खुराक एक साथ कभी न लें।"
    )
    parts.append(
        "कुछ भी समझ न आए तो स्ट्रिप लेकर दुकान पर आइए। और अगर चकत्ते, सूजन या "
        "साँस लेने में तकलीफ़ हो तो तुरंत डॉक्टर को दिखाइए।"
    )

    return " ".join(parts)


async def speak(text: str, *, settings: Settings | None = None) -> dict:
    """Render counselling text to MP3. Returns metadata, and never raises.

    The caller gets `voice_available: False` and the original text rather than
    an exception, because the text is the part that matters clinically and the
    audio is only how it gets delivered.
    """
    settings = settings or get_settings()

    result = {
        "text": text,
        "audio_base64": None,
        "audio_mime": None,
        "voice_available": False,
        "model": settings.elevenlabs_model,
    }

    if not settings.elevenlabs_api_key or not settings.elevenlabs_voice_id:
        return result
    if not text.strip():
        return result

    payload = {
        "text": text,
        "model_id": settings.elevenlabs_model,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    headers = {
        "xi-api-key": settings.elevenlabs_api_key,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/"
                f"{settings.elevenlabs_voice_id}",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
    except httpx.HTTPError:
        return result

    result["audio_base64"] = base64.b64encode(response.content).decode("ascii")
    result["audio_mime"] = "audio/mpeg"
    result["voice_available"] = True
    return result
