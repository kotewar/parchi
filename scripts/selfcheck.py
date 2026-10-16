"""Regression checks for the gate, runnable with nothing but python3.

    python3 scripts/selfcheck.py

Every case here is something that was actually wrong at some point, or something
that would be dangerous if it ever became wrong. It exists because the
strength-parse check shipped a false positive -- it flagged the "GP2" in Glycomet
GP2, a correct answer, as an unparseable strength -- and nothing in the repo
noticed until a human read the demo output.

    A gate that flags correct answers is a gate people learn to click past, and
    then it protects nobody.

No pytest, no fixtures, no setup: a judge should be able to verify these claims
without installing anything. Exits non-zero on the first failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.formulary import food_timing, is_chronic  # noqa: E402
from app.gate import (  # noqa: E402
    _grounding_problems,
    _sig_shape_problems,
    _strength_problems,
    decide,
)
from app.models import Prescription, RxLine, Verdict  # noqa: E402
from app.voice import compose_counselling, display_name  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, label: str, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{label}{': ' + detail if detail else ''}")


def line(**kw) -> RxLine:
    base = dict(drug=None, strength=None, form="Tab", sig=None, days=None,
                quantity=None, confidence=0.95, raw="", flags=[])
    base.update(kw)
    return RxLine(**base)


# ---------------------------------------------------------------------------
# Strengths that must be accepted, because they are correct.
# ---------------------------------------------------------------------------
for strength in ["40mg", "500mg", "2.5mg", "75mcg", "10 units", "500", "0.5mg"]:
    check(not _strength_problems(strength), f"strength {strength!r} is accepted")

# The regression this file exists for. "GP2" is a combination marker, not a
# number: Glycomet GP2 is metformin plus glimepiride. Flagging it is a false
# alarm on a correct answer.
for suffix in ["GP2", "GP1", "gp2", "CV2", "H1"]:
    check(not _strength_problems(suffix), f"strength {suffix!r} is accepted")

# ---------------------------------------------------------------------------
# Strengths that must be flagged, because they are a wrong dose.
# ---------------------------------------------------------------------------
for strength in ["7S", "5Omcg", "2.S", "1O units", "6P2", "SOmg", "l0mg", "xyz"]:
    check(bool(_strength_problems(strength)), f"strength {strength!r} is flagged")

# The four the eval actually produced, and the repair we offer for each. The
# hint must be right, because a wrong suggestion on a dosage is worse than none.
for strength, want in [("7S", "75"), ("5Omcg", "50mcg"), ("2.S", "2.5")]:
    problems = _strength_problems(strength)
    check(bool(problems) and want in problems[0],
          f"strength {strength!r} suggests {want!r}", str(problems))

# "units" ends in "s", which maps to 5. Repairing it naively gives "10unit5".
problems = _strength_problems("1O units")
check(bool(problems) and "10units" in problems[0],
      "strength '1O units' repairs to '10units', not '10unit5'", str(problems))

# ...but a repair is only offered when it lands on an actual dose. "xyz" -> "xy2"
# satisfies the suffix shape and means nothing.
check("looks like" not in " ".join(_strength_problems("xyz")),
      "no nonsense repair hint for a non-dosage string")

# ---------------------------------------------------------------------------
# Grounding: the model naming a medicine that is not on the page.
# ---------------------------------------------------------------------------
sub = _grounding_problems(line(
    drug="Metformin", raw="2. Tab Glycomet 6P2 1-0-1 x 30 days"))
check(bool(sub), "brand-to-generic substitution is caught by grounding")

check(not _grounding_problems(line(
    drug="Glycomet GP2", raw="2. Tab Glycomet GP2 1-0-1 x 30 days")),
    "a name that is on the page is not flagged")

# No raw text to check against is not evidence of a problem.
check(not _grounding_problems(line(drug="Metformin", raw="")),
      "nothing to ground against is not a failure")

# ---------------------------------------------------------------------------
# Structural checks.
# ---------------------------------------------------------------------------
check(not _sig_shape_problems("1-0-1"), "1-0-1 is a valid dosing pattern")
check(bool(_sig_shape_problems("x 30 days")), "prose in the sig is flagged")
check(bool(_sig_shape_problems(None)), "a missing sig is flagged")
check(bool(_sig_shape_problems("0-0-0")), "a sig that prescribes nothing is flagged")
for shorthand in ["SOS", "OD", "BD"]:
    check(not _sig_shape_problems(shorthand), f"shorthand {shorthand!r} is accepted")

# ---------------------------------------------------------------------------
# Formulary. Two real bugs lived here: hyphenated keys were unreachable because
# `normalise` turns hyphens into spaces.
# ---------------------------------------------------------------------------
check(food_timing("Glycomet GP2") == "with food",
      "hyphenated brand reaches its food timing",
      repr(food_timing("Glycomet GP2")))
check(food_timing("Pan 40") == "before food", "Pan 40 is before food")
check(food_timing("Panadol") is None, "Panadol is a different drug and gets no rule")
check(is_chronic("Glycomet GP2"), "Glycomet GP2 is chronic")
check(is_chronic("Telma 40"), "Telma is chronic")
check(not is_chronic("Azithral 500"), "a 5-day antibiotic is not chronic")

# ---------------------------------------------------------------------------
# Counselling names each medicine exactly once. The model sometimes returns the
# full brand as the drug and the suffix again as the strength, and "Glycomet
# GP2 GP2" read aloud at the counter is how a patient stops trusting the rest.
# ---------------------------------------------------------------------------
check(display_name(line(drug="Glycomet GP2", strength="GP2")) == "Glycomet GP2",
      "a strength already in the name is not said twice",
      display_name(line(drug="Glycomet GP2", strength="GP2")))
check(display_name(line(drug="Glycomet", strength="GP2")) == "Glycomet GP2",
      "a strength absent from the name is still appended")
check(display_name(line(drug="Thyronorm 50mcg", strength="50mcg")) == "Thyronorm 50mcg",
      "a name already carrying its unit is not doubled")
check(display_name(line(drug="Telma", strength="40mg")) == "Telma 40mg",
      "an ordinary strength is appended")

spoken = compose_counselling([
    line(drug="Glycomet GP2", strength="GP2", sig="1-0-1", days=30,
         raw="2. Tab Glycomet GP2 1-0-1 x 30 days"),
])
check("GP2 GP2" not in spoken, "the spoken counselling does not stutter a name")

# An unreadable page still returns something the pharmacist can read aloud,
# rather than handing back an empty string at the counter.
check(bool(compose_counselling([]).strip()), "empty prescription still returns text")

# ---------------------------------------------------------------------------
# End to end: the three failure modes the writeup documents, plus one clean
# line that must survive.
# ---------------------------------------------------------------------------
rx = Prescription(
    overall_confidence=0.95,
    lines=[
        line(drug="Metformin", strength="600mg", sig="1-0-1", days=30,
             quantity=60, raw="2. Tab Glycomet 6P2 1-0-1 x 30 days"),
        line(drug="Ecosprin", strength="7S", form="Tab", sig="0-0-1", days=30,
             quantity=30, raw="1. Tab Ecosprin 7S 0-0-1 x 30 days"),
        line(drug="Telma", strength="40mg", sig="1-0-0", days=30,
             quantity=30, raw="1. Tab Telma 40mg 1-0-0 x 30 days"),
    ],
)
decision = decide(rx)
check(decision.verdict is Verdict.review, "a tainted page goes to review")
check(0 not in decision.accepted_line_indexes, "the substituted generic is held back")
check(1 not in decision.accepted_line_indexes, "the corrupted strength is held back")
check(2 in decision.accepted_line_indexes, "the correct line is accepted")

# A page of nothing but correct lines must not be held back, or the gate is noise.
clean = Prescription(
    overall_confidence=0.95,
    lines=[
        line(drug="Telma", strength="40mg", sig="1-0-0", days=30, quantity=30,
             raw="1. Tab Telma 40mg 1-0-0 x 30 days"),
        line(drug="Glycomet GP2", strength="GP2", sig="1-0-1", days=30,
             quantity=60, raw="2. Tab Glycomet GP2 1-0-1 x 30 days"),
    ],
)
clean_decision = decide(clean)
check(clean_decision.verdict is Verdict.accept,
      "a clean page is accepted outright",
      str(clean_decision.reasons))
check(clean_decision.accepted_line_indexes == [0, 1],
      "both clean lines are accepted", str(clean_decision.reasons))

# ---------------------------------------------------------------------------
print(f"{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
if FAILURES:
    print()
    for failure in FAILURES:
        print(f"  FAIL  {failure}")
    sys.exit(1)

print("gate, formulary and end-to-end checks all pass")
