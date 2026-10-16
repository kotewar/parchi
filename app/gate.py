"""The gate: what may be dispensed without a human looking, and what may not.

The design premise, established by measurement rather than taste:

    The OCR engine reported confidence 1.00 on "Ox: HTN, T2OM" -- which is
    "Dx: HTN, T2DM" with two letters misread. It read the handwriting
    perfectly and was confidently, clinically wrong.

So OCR confidence is discarded as a safety signal. The gate runs on the
model's *semantic* confidence plus hard structural checks, and it is
deliberately blunt: a wrong medicine reaching a patient costs far more than a
pharmacist spending four seconds on a line.
"""

from __future__ import annotations

import re

from .config import Settings, get_settings
from .formulary import Formulary, closely_matches, normalise, tokens
from .models import GateDecision, Prescription, RxLine, Verdict

_SIG_PARTS = re.compile(r"^(\d+)(-\d+){1,3}$")

# A strength is digits, optionally a decimal, optionally a unit. Nothing else.
_STRENGTH_RE = re.compile(r"^\d+(\.\d+)?(mg|mcg|mgs|g|gm|ml|iu|units|unit|%)?$")

# Letters that get misread as digits in handwriting, and the digit they are
# pretending to be. "5OO" is 500. "7S" is 75. "2.S" is 2.5.
_DIGIT_LOOKALIKES = {
    "o": "0", "O": "0", "l": "1", "I": "1", "i": "1",
    "s": "5", "S": "5", "b": "6", "B": "8", "z": "2", "Z": "2",
}


def _strength_problems(strength: str | None) -> list[str]:
    """Does the strength actually parse as a strength?

    This closed the largest hole in the whole system, and it was found by
    measurement. On the poisoned eval set, all seven dangerous answers the gate
    failed to catch were the same failure: the model echoed a strength where a
    letter had been misread as a digit, and the gate accepted it because the
    strength field was merely *non-empty*.

        "5OO"   for 500     "65O"  for 650     "7S"   for 75
        "2.S"   for 2.5     "1O"   for 10      "25O"  for 250

    Every one of those is a wrong dose. None of them parses as a strength, and
    so none of them should ever have been accepted.
    """
    if not strength:
        return []

    compact = re.sub(r"\s+", "", strength)
    if _STRENGTH_RE.match(compact.lower()):
        return []

    # If the digits are recoverable, say what we think it was. Suggesting a
    # correction is safe; applying one silently is not.
    guess = "".join(_DIGIT_LOOKALIKES.get(ch, ch) for ch in compact)
    if guess != compact and _STRENGTH_RE.match(guess.lower()):
        return [
            f"strength {strength!r} does not parse as a strength "
            f"(looks like {guess!r} with a letter misread for a digit)"
        ]

    return [f"strength {strength!r} does not parse as a strength"]


def _grounding_problems(line: RxLine) -> list[str]:
    """Does the medicine the model named actually appear on the page?

    Found by measurement, not theory. On the poisoned eval set the base model
    silently swapped Indian brand names for generics and reported 0.95
    confidence while doing it: "Metformin" for Glycomet GP2, "Panadol" for
    Pan 40, "Atorvastatin" for Atorva.

    Those swaps are not cosmetic. Glycomet GP2 contains glimepiride as well as
    metformin, and Panadol is paracetamol where Pan is pantoprazole. A generic
    substitution is a different prescription, and it is invisible in the
    model's JSON output -- except in one place. The name it returned is not
    anywhere on the page.

    The discriminating word is the longest one in the returned name, compared
    against the raw OCR text with an OCR-sized edit tolerance. This check is
    deterministic and caught every substitution the model made.
    """
    if not line.drug or not line.raw:
        return []

    drug_tokens = {t for t in tokens(normalise(line.drug)) if len(t) >= 3}
    if not drug_tokens:
        return []

    source_tokens = tokens(normalise(line.raw))
    anchor = max(drug_tokens, key=len)

    if any(closely_matches(anchor, source) for source in source_tokens):
        return []

    return [
        f"'{line.drug}' does not appear on the page as written "
        f"(OCR line reads: {line.raw.strip()!r})"
    ]


def _sig_shape_problems(sig: str | None) -> list[str]:
    """Check the dose pattern reads like a real dosing schedule."""
    if not sig:
        return ["no dosing pattern (sig)"]

    cleaned = sig.strip().lower().replace(" ", "")
    if cleaned in {"sos", "stat", "hs", "od", "bd", "tds", "qid"}:
        return []
    if not _SIG_PARTS.match(cleaned):
        return [f"dosing pattern '{sig}' is not a morning-afternoon-night pattern"]

    parts = [int(p) for p in cleaned.split("-")]
    if sum(parts) == 0:
        return [f"dosing pattern '{sig}' prescribes nothing"]
    if any(p > 4 for p in parts):
        return [f"dosing pattern '{sig}' has an implausible number of doses"]
    if len(parts) > 2 and parts[-1] > 0 and parts[-2] == 0 and parts[-3] == 0:
        return []
    return []


def _line_problems(
    line: RxLine, threshold: float, formulary: Formulary | None
) -> list[str]:
    """Every reason this line is not safe to fill as-is."""
    problems: list[str] = []

    if line.confidence < threshold:
        problems.append(
            f"model confidence {line.confidence:.2f} is below {threshold:.2f}"
        )

    if not line.drug:
        problems.append("no medicine name recovered from the page")
    else:
        problems.extend(_grounding_problems(line))

    if line.form in ("Tab", "Cap") and not line.strength:
        problems.append("tablet or capsule with no strength")
    else:
        problems.extend(_strength_problems(line.strength))

    problems.extend(_sig_shape_problems(line.sig))

    if line.days is None and line.quantity is None:
        problems.append("neither a duration nor a quantity")

    # An arithmetic check the model cannot bluff its way past: the total
    # quantity has to agree with the schedule and the duration.
    if line.sig and line.days and line.quantity is not None:
        cleaned = line.sig.strip().lower().replace(" ", "")
        if _SIG_PARTS.match(cleaned):
            per_day = sum(int(p) for p in cleaned.split("-"))
            expected = per_day * line.days
            if expected and abs(line.quantity - expected) > max(2, expected * 0.1):
                problems.append(
                    f"quantity {line.quantity} does not match {per_day}/day "
                    f"for {line.days} days (expected about {expected})"
                )

    if formulary is not None and line.drug and not formulary.knows(line.drug):
        suggestions = formulary.nearest(line.drug)
        hint = f" (closest stocked: {', '.join(suggestions)})" if suggestions else ""
        problems.append(f"'{line.drug}' is not in this shop's stock list{hint}")

    return problems


def decide(
    rx: Prescription,
    *,
    settings: Settings | None = None,
    formulary: Formulary | None = None,
) -> GateDecision:
    """Decide what on this page may be filled, and what must be checked."""
    settings = settings or get_settings()
    threshold = settings.confidence_threshold

    reasons: list[str] = []
    review: list[int] = []
    accepted: list[int] = []

    if not rx.lines:
        return GateDecision(
            verdict=Verdict.reject,
            reasons=["no prescription lines were recovered from this page"],
        )

    for index, line in enumerate(rx.lines):
        problems = _line_problems(line, threshold, formulary)
        if problems:
            review.append(index)
            label = line.drug or line.raw or f"line {index + 1}"
            reasons.append(f"line {index + 1} ({label}): " + "; ".join(problems))
        else:
            accepted.append(index)

    if rx.unreadable:
        reasons.append(
            f"{len(rx.unreadable)} part(s) of the page could not be read at all"
        )

    if rx.overall_confidence < threshold:
        reasons.append(
            f"whole-page confidence {rx.overall_confidence:.2f} is below {threshold:.2f}"
        )

    all_unreadable = not accepted and bool(rx.unreadable)
    if all_unreadable:
        verdict = Verdict.reject
    elif review or rx.unreadable or rx.overall_confidence < threshold:
        verdict = Verdict.review
    else:
        verdict = Verdict.accept

    if verdict is Verdict.accept and not reasons:
        reasons.append(
            f"all {len(accepted)} line(s) passed confidence, structure and stock checks"
        )

    return GateDecision(
        verdict=verdict,
        reasons=reasons,
        review_line_indexes=review,
        accepted_line_indexes=accepted,
    )
