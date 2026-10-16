"""Fixtures for the hosted demo, and the recorded OCR transcript.

The hosted demo runs on a free instance with no GPU and no macOS, so two parts
of the real pipeline genuinely cannot run there: the Vision OCR, which is
macOS-only, and the model, which is 3.4 GB and wants an accelerator.

Rather than quietly fake those or leave them out, the demo does three separate
things and says which is which:

  - the OCR transcript is REAL, recorded from `scripts/ocr_vision.swift` on the
    synthetic page, and stored verbatim below including the misreads;
  - the model's step falls back to the deterministic fixtures in
    `gemma._demo_prescription`, the same path `DEMO_MODE=true` takes locally;
  - everything downstream of those two -- the gate, the stock plan, the
    counselling text -- is the real code running for real.

The transcript is worth keeping faithful because its most interesting line is
`Ox: HTN, T2OM` at confidence 1.00, which is `Dx: HTN, T2DM` misread and is the
observation the whole safety design comes from.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from .models import OcrLine, OcrPage

# Recorded verbatim from the Vision OCR run on the synthetic page. Do not tidy
# these up: the misreads are the point.
SAMPLE_OCR_LINES: list[tuple[str, float, float]] = [
    ("Dr. R.Deshmukh MBBSReg2019/4482", 1.00, 0.886),
    ("Pt: Mr. Suresh Wankhede 42/M", 1.00, 0.812),
    ("Ox: HTN, T2OM", 1.00, 0.729),
    ("Rx", 1.00, 0.580),
    ("1. Tab Telma 40mg 1-0-0 x 30 days", 1.00, 0.499),
    ("2. Tab Glycomet GP2 1-0-1 x 30 days", 1.00, 0.423),
    ("3. Cap Pan 40 1-0-0 x15 days", 1.00, 0.345),
    ("4. Inj. Insulin 0-0-1 x 10 days", 1.00, 0.265),
    ("Adv: review after 15 days", 1.00, 0.190),
]


def sample_page() -> OcrPage:
    """The recorded page, shaped exactly like a live OCR result."""
    return OcrPage(
        image_path="sample/prescription.png",
        image_width=1240,
        image_height=1754,
        observations=11,  # 11 observations collapsed into 9 lines by y-position
        lines=[
            OcrLine(text=text, ocr_confidence=confidence, y=y)
            for text, confidence, y in SAMPLE_OCR_LINES
        ],
        mean_ocr_confidence=1.00,
        ocr_seconds=1.622,
    )


# (drug, strength, typical quantity dispensed per week, is it spiky)
DEMO_SHELF: list[tuple[str, str | None, int, bool]] = [
    ("Glycomet GP2", None, 56, False),
    ("Thyronorm", "50mcg", 30, False),
    ("Telma", "40mg", 28, False),
    ("Ecosprin", "75mg", 28, False),
    ("Shelcal", "500mg", 28, False),
    ("Pan", "40mg", 14, False),
    ("Zerodol-SP", None, 10, True),
    ("Azithral", "500mg", 8, True),
]

DEMO_WEEKS = 14


def seed_dispense_history(store, *, weeks: int = DEMO_WEEKS, seed: int = 7) -> int:
    """Write plausible dispensing history into the shop's own store.

    Deterministic on purpose, so the demo shows the same stock plan every time
    and a judge can check it against the numbers. Real history would come from
    the pharmacist's own dispensing, not from this function.
    """
    rng = random.Random(seed)
    start = date.today() - timedelta(days=weeks * 7)
    written = 0

    for week in range(weeks):
        for drug, strength, weekly, spiky in DEMO_SHELF:
            if spiky and rng.random() < 0.82:
                # An antibiotic or a painkiller does not sell every week.
                continue

            jitter = rng.uniform(0.85, 1.15)
            quantity = max(1, round(weekly * jitter))
            if spiky:
                quantity = max(1, quantity // 2)

            day = start + timedelta(days=week * 7 + rng.randint(0, 6))
            store.record_dispense(drug, quantity, strength=strength, on=day)
            written += 1

    return written
