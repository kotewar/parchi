"""The formulary: what this shop actually keeps on the shelf.

This is a quiet but important part of the argument for a local model. Knowing
that "Telma 40" is a real medicine in this shop's stock is not general
knowledge we can look up anywhere — it is the shop's own shelf list. Keeping
it on the machine means the reconciliation can be checked against it on every
page, for free, without shipping the shop's inventory to anyone.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

# A small starter list of medicines common in Indian retail pharmacy, enough to
# run and demo. A real deployment replaces this with the shop's own stock list.
STARTER: set[str] = {
    "telma", "telma-h", "telmisartan", "amlopres", "amlodipine", "telmikind",
    "glycomet", "glycomet-gp1", "glycomet-gp2", "metformin", "glimepiride",
    "januvia", "galvus", "atorva", "atorvastatin", "rosuvas", "rosuvastatin",
    "pan", "pantop", "pantoprazole", "omep", "omeprazole", "rantac",
    "augmentin", "azithral", "azithromycin", "taxim-o", "ceftum",
    "dolo", "dolo-650", "paracetamol", "crocin", "combiflam", "brufen",
    "zerodol", "zerodol-sp", "voveran", "montek", "montair", "levocet",
    "allegra", "cetzine", "avil", "sinarest", "cheston",
    "ecosprin", "clopilet", "clopidogrel", "warf", "warfarin",
    "thyronorm", "eltroxin", "shelcal", "calcirol", "orofer",
    "deriphyllin", "asthalin", "foracort", "budecort", "seroflo",
    "insulin", "lantus", "huminsulin", "mixtard", "novomix",
    "cetzine", "monocef", "mox", "novamox", "clavam", "calpol",
}

# How each medicine is taken relative to food. This is counter knowledge, the
# same kind as the stock list, and it lives here for the same reason: it is not
# something to let a language model guess at.
#
# It matters more than it looks. The dosing pattern says "1-0-0", meaning
# morning -- it does not say "before food", and getting that wrong is the single
# most common instruction error at an Indian pharmacy counter. Pantoprazole on a
# full stomach does not work; metformin on an empty one makes people sick, which
# is how courses get abandoned.
#
# KEYS ARE ALREADY NORMALISED -- lower case, spaces not hyphens, no dose, no
# form. "Glycomet GP2" is the key `glycomet gp2`, not `glycomet-gp2`. This is
# not cosmetic: `match_key` returns a normalised key and indexes this table with
# it, so a hyphenated key here would match and then raise KeyError. The guard
# below `normalise` enforces it at import time.
FOOD_TIMING: dict[str, str] = {
    # Before food: acid suppression needs an empty stomach to work.
    "pan": "before food",
    "pan d": "before food",
    "pantop": "before food",
    "pantoprazole": "before food",
    "omep": "before food",
    "omeprazole": "before food",
    "rantac": "before food",
    # With food: metformin and the sulfonylureas are far better tolerated so.
    "glycomet": "with food",
    "glycomet gp1": "with food",
    "glycomet gp2": "with food",
    "metformin": "with food",
    "glimepiride": "with food",
    "januvia": "with food",
    "galvus": "with food",
    # After food: the NSAIDs, for the stomach lining.
    "zerodol": "after food",
    "zerodol sp": "after food",
    "voveran": "after food",
    "brufen": "after food",
    "combiflam": "after food",
    "dolo": "after food",
    "dolo 650": "after food",
    "calpol": "after food",
    "crocin": "after food",
    "ecosprin": "after food",
    "clopilet": "after food",
    "warf": "after food",
    "warfarin": "after food",
    "shelcal": "after food",
    "calcirol": "after food",
    "orofer": "after food",
    "azithral": "after food",
    "azithromycin": "after food",
    "augmentin": "after food",
    "clavam": "after food",
    "novamox": "after food",
    "taxim o": "after food",
    "monocef": "after food",
    "ceftum": "after food",
    "mox": "after food",
    "levocet": "after food",
    "montek": "after food",
    "montair": "after food",
    "allegra": "after food",
    "cetzine": "after food",
    # Empty stomach, first thing: thyroid replacement, and it must be alone.
    "thyronorm": "first thing in the morning, on an empty stomach",
    "eltroxin": "first thing in the morning, on an empty stomach",
    # Inhaled: the mouth must be rinsed or the patient gets oral thrush.
    "foracort": "rinse the mouth after each use",
    "budecort": "rinse the mouth after each use",
    "seroflo": "rinse the mouth after each use",
    "asthalin": "rinse the mouth after each use",
    # Insulin: the timing is tied to the injection, not to a meal.
    "insulin": "as the doctor has set out, with the injection",
    "lantus": "at the same time each day",
    "huminsulin": "as the doctor has set out, with the injection",
    "mixtard": "as the doctor has set out, with the injection",
    "novomix": "as the doctor has set out, with the injection",
}

# Medicines a chronic patient takes indefinitely. Demand for these is steady,
# and running out is an emergency rather than an inconvenience, because the
# patient is already on them and cannot simply wait a week.
#
# Written in NORMALISED form, because that is what `normalise` produces: no
# hyphens, no dose, no form. "Glycomet GP2" normalises to "glycomet gp2", so an
# entry spelled "glycomet-gp2" would never match anything. `match_key` defends
# against that anyway, but the source of truth should not be lying.
CHRONIC: set[str] = {
    "telma", "telma h", "telmikind", "amlopres", "amlodipine",
    "glycomet", "glycomet gp1", "glycomet gp2", "januvia", "galvus",
    "atorva", "atorvastatin", "rosuvas", "rosuvastatin",
    "thyronorm", "eltroxin",
    "ecosprin", "clopilet", "warf",
    "shelcal", "calcirol", "orofer",
    "insulin", "lantus", "huminsulin", "mixtard", "novomix",
    "deriphyllin", "foracort", "budecort", "seroflo",
}

# Strip the dose and the salutation, keep the molecule or brand stem.
_NOISE = re.compile(
    r"\b(tab|cap|cap\.|syp|syr|inj|inj\.|drops|oint|cream|gel|spray|sr|er|xr|"
    r"forte|plus|ds|dt|mr|pr|od|bd|tds|qid|hs|sos|stat)\b",
    re.IGNORECASE,
)
_DOSE = re.compile(r"\b\d+\s*(mg|mcg|g|ml|iu|%)\b", re.IGNORECASE)
_NONALNUM = re.compile(r"[^a-z0-9]+")


def normalise(name: str | None) -> str:
    """Reduce a printed drug name to a comparable key: 'Tab. Telma 40mg' -> 'telma'."""
    if not name:
        return ""
    text = name.lower()
    text = _DOSE.sub(" ", text)
    text = _NOISE.sub(" ", text)
    text = _NONALNUM.sub(" ", text)
    return " ".join(text.split()).strip()


# Fail loudly at import rather than at 9pm at the counter. Both knowledge tables
# are keyed by normalised names because `match_key` returns one, and a stray
# hyphen in a key would match fine and then raise KeyError on the lookup.
_UNNORMALISED = [
    name
    for name in sorted(list(FOOD_TIMING) + sorted(CHRONIC))
    if normalise(name) != name
]
if _UNNORMALISED:
    raise RuntimeError(
        "formulary keys must be normalised, but these are not: "
        + ", ".join(repr(name) for name in _UNNORMALISED)
    )


def match_key(keys: Iterable[str], drug: str | None) -> str | None:
    """Find which of `keys` this drug name refers to, whatever form it arrived in.

    A drug name reaches us in whatever shape the page used. "Pan" on one
    prescription is "Pan 40" on the next and "Pan-D" on a third, and the OCR may
    have dropped the tail altogether. Both sides are normalised before
    comparing, and candidates are tried longest-first so a specific brand wins
    over its own stem.

    The two prefix rules need different guards. "pan 40" starting with "pan " is
    unambiguous, so it needs no length limit. Starting with "pan" and no space
    does need one, because that would let "pan" claim "panadol" -- and Panadol
    is paracetamol where Pan is pantoprazole.
    """
    key = normalise(drug or "")
    if not key:
        return None

    candidates = {normalise(candidate) for candidate in keys} - {""}
    if key in candidates:
        return key

    for candidate in sorted(candidates, key=len, reverse=True):
        if key.startswith(candidate + " "):
            return candidate
        if len(candidate) >= 4 and key.startswith(candidate):
            return candidate
    return None


def food_timing(drug: str | None) -> str | None:
    """How this medicine is taken relative to food, or None if the shop does not know.

    Deterministic on purpose. Rather than let a language model guess at food
    timing, the shop either knows it for this medicine or says that it does not
    and the pharmacist fills the gap. A wrong guess here is a real medication
    error, and it is not one the model's confidence would reveal.

    None is a legitimate answer rather than a failure. Telmisartan genuinely may
    be taken with or without food, so for Telma the honest reply is that the
    shop has no rule for this one.
    """
    matched = match_key(FOOD_TIMING, drug)
    return FOOD_TIMING[matched] if matched else None


def is_chronic(drug: str | None) -> bool:
    """Is this a medicine a patient takes indefinitely?

    This matters for stock, not for safety: a chronic medicine running out is
    worse than a seasonal one running out, because the patient is already on it.
    """
    return match_key(CHRONIC, drug) is not None


class Formulary:
    def __init__(self, names: set[str] | None = None) -> None:
        self._keys = {normalise(n) for n in (names or STARTER) if normalise(n)}

    @classmethod
    def from_file(cls, path: str | Path) -> "Formulary":
        """Load the shop's own stock list, one brand per line."""
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        names = {l.strip() for l in lines if l.strip() and not l.startswith("#")}
        return cls(names or STARTER)

    def knows(self, name: str | None) -> bool:
        key = normalise(name)
        if not key:
            return False
        if key in self._keys:
            return True
        # A brand read with a dropped or garbled tail ("glycomet gp" vs
        # "glycomet gp2") is close enough to at least not be a stranger.
        return any(
            key.startswith(k) or k.startswith(key)
            for k in self._keys
            if min(len(key), len(k)) >= 4
        )

    def nearest(self, name: str | None) -> list[str]:
        """Stocked names within one edit of this one.

        Used to *suggest* a correction to the pharmacist. Never applied
        silently: the whole point is that the human decides.
        """
        key = normalise(name)
        if not key:
            return []
        out = []
        for candidate in self._keys:
            if abs(len(candidate) - len(key)) > 2:
                continue
            if _edit_distance(key, candidate) <= 2:
                out.append(candidate)
        return sorted(out)[:3]


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,          # deletion
                    current[j - 1] + 1,       # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
        previous = current
    return previous[-1]


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance, exposed for the grounding check."""
    return _edit_distance(a, b)


def tokens(text: str) -> set[str]:
    return set(_NONALNUM.split(text.lower())) - {""}


def closely_matches(token: str, candidate: str) -> bool:
    """Is this token the same word the OCR wrote, allowing for OCR error?

    The tolerance scales with length on purpose. Two edits on a six-letter
    brand name is a plausible misreading; two edits on a three-letter one is a
    different word entirely, and treating those as the same is how "Pan"
    silently becomes "Panadol".
    """
    if token == candidate:
        return True
    longest = max(len(token), len(candidate))
    allowed = 1 if longest <= 4 else 2
    return _edit_distance(token, candidate) <= allowed
