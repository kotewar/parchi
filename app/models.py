from __future__ import annotations

import enum
import re
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator

_SIG_SHAPE = re.compile(r"^\d+(-\d+){1,3}$")


class Verdict(str, enum.Enum):
    accept = "accept"   # safe to fill without a human looking
    review = "review"   # a human must confirm before it reaches a patient
    reject = "reject"   # cannot be read; hand the paper back


class OcrLine(BaseModel):
    """One physical line as the OCR engine saw it."""

    text: str
    # What the OCR engine thought of its own reading. Measured to be ~1.00 even
    # when the text is garbage ("Ox: HTN, T2OM" scored 1.00). It means "the
    # pixels were legible", not "this is medically real". Never gate on it alone.
    ocr_confidence: float = 0.0
    y: float = 0.0
    x: float = 0.0


class OcrPage(BaseModel):
    image_path: str = ""
    image_width: int = 0
    image_height: int = 0
    observations: int = 0
    lines: list[OcrLine] = Field(default_factory=list)
    mean_ocr_confidence: float = 0.0
    ocr_seconds: float = 0.0


class RxLine(BaseModel):
    """One prescribed item, after the model has made sense of the OCR text."""

    # Which numbered OCR line this came from. The model cites an index rather
    # than echoing the text, because models copy long strings unreliably and
    # refer to enumerated items well. We attach the real text ourselves.
    ocr_line: int | None = None
    raw: str = ""
    drug: str | None = None
    strength: str | None = None
    form: str | None = None          # Tab / Cap / Inj / Syr / Drops
    sig: str | None = None           # 1-0-0 = morning-afternoon-night
    days: int | None = None
    quantity: int | None = None
    # The model's own probability that this line is correct AND real. This is
    # the number the gate uses, because it is the only one that knows whether
    # "Telrna" is a medicine.
    confidence: float = 0.0
    flags: list[str] = Field(default_factory=list)
    in_stock: bool | None = None
    substituted: str | None = None

    @field_validator("drug", "strength", "form", "sig", mode="before")
    @classmethod
    def _stringify(cls, value: object) -> object:
        """The model writes null, a string, or occasionally something else.

        A JSON number where we expect text is not an error worth failing a
        whole prescription over, so we coerce rather than reject.
        """
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip() or None
        return str(value)

    @model_validator(mode="after")
    def _rescue_misplaced_sig(self) -> "RxLine":
        """Move a dosing pattern that was filed under strength.

        Seen on combination products with no numeric strength -- Zerodol-SP,
        Telmikind-H. The model reaches for the next token it can see and puts
        "1-0-1" in the strength column. It is the right information in the
        wrong field, so move it rather than fail the line, and never invent a
        strength to fill the gap.
        """
        if self.strength and _SIG_SHAPE.match(self.strength.strip()):
            if not self.sig:
                self.sig = self.strength.strip()
            self.strength = None
        return self

    @field_validator("days", "quantity", mode="before")
    @classmethod
    def _to_int(cls, value: object) -> object:
        """Accept 30, "30", "30 days", or "null" as typed by the model.

        The prompt asks for null when unknown, but a model that writes 0 or ""
        must not be read as "zero days", which would be a real prescription
        error rather than a parsing one.
        """
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value or None
        if isinstance(value, float):
            return int(value) or None
        if isinstance(value, str):
            digits = "".join(ch for ch in value if ch.isdigit())
            return int(digits) if digits else None
        return None


class Prescription(BaseModel):
    image_path: str = ""
    patient_name: str | None = None
    age: str | None = None
    sex: str | None = None
    prescriber: str | None = None
    registration_no: str | None = None
    diagnosis: str | None = None
    lines: list[RxLine] = Field(default_factory=list)
    unreadable: list[str] = Field(default_factory=list)
    overall_confidence: float = 0.0
    ocr_seconds: float = 0.0
    reconcile_seconds: float = 0.0

    @field_validator(
        "patient_name", "age", "sex", "prescriber", "registration_no", "diagnosis",
        mode="before",
    )
    @classmethod
    def _stringify(cls, value: object) -> object:
        """An age of 42 arriving as a number is fine; it is still an age."""
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip() or None
        return str(value)


class GateDecision(BaseModel):
    verdict: Verdict
    # Human-readable whys, shown to the pharmacist. A gate that says "review"
    # without saying why trains people to click through it.
    reasons: list[str] = Field(default_factory=list)
    review_line_indexes: list[int] = Field(default_factory=list)
    accepted_line_indexes: list[int] = Field(default_factory=list)


class Correction(BaseModel):
    """One pharmacist fix, kept as a training row for the fine-tune.

    This is the asset. A generic hosted model cannot learn "in this shop, the
    OCR string 'Telrna' is Telma" because it never sees these corrections and
    they are not ours to send anywhere. They stay on this machine until we
    choose to fine-tune on them.
    """

    created_at: datetime = Field(default_factory=datetime.utcnow)
    raw_ocr: str
    field: str            # drug | strength | sig | days | quantity
    model_value: str | None = None
    corrected_value: str
    drug_context: str | None = None
    note: str | None = None
