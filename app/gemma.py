from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

from .config import Settings, get_settings
from .models import OcrPage, Prescription, RxLine
from .voice import compose_counselling, medicines_block

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


class GemmaUnavailable(RuntimeError):
    """Raised when the local Ollama endpoint cannot serve the model."""


def _read_prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def strip_fences(raw: str) -> str:
    """Models wrap JSON in ```json fences even when told not to. Strip them."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    return text.strip()


async def chat(
    prompt: str,
    *,
    images: list[str] | None = None,
    settings: Settings | None = None,
) -> str:
    """One turn against the local model.

    Verified on Ollama 0.40.5 + gemma3:4b on an Intel CPU-only Mac. Two things
    here are load-bearing, found by measurement rather than assumption:

    1. /api/chat, NOT /api/generate. gemma3's instruction-following comes from
       its chat template. Through /api/generate the prompt is completed as raw
       text and the model returns an empty object instead of JSON.

    2. NO format="json". On this model+version the JSON grammar constraint
       collapses the output to a bare "{}" for every instruction we tried,
       however simple. We ask for JSON in the prompt and strip fences instead.
    """
    settings = settings or get_settings()

    message: dict = {"role": "user", "content": prompt}
    if images:
        message["images"] = images

    payload = {
        "model": settings.gemma_model,
        "stream": False,
        # temperature 0: this is transcription and arithmetic, not writing.
        "options": {
            "temperature": 0.0,
            "num_ctx": settings.num_ctx,
            "num_predict": settings.num_predict,
        },
        "messages": [message],
    }

    try:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout_s) as client:
            resp = await client.post(f"{settings.ollama_url}/api/chat", json=payload)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise GemmaUnavailable(
            f"Ollama unreachable at {settings.ollama_url} ({exc}). "
            f"Start it, then `ollama pull {settings.gemma_model}`."
        ) from exc

    return resp.json().get("message", {}).get("content", "")


async def reconcile(
    page: OcrPage,
    *,
    settings: Settings | None = None,
) -> Prescription:
    """Turn OCR lines into a prescription the pharmacist can act on.

    The model's only job is the semantic check: is this a real medicine at a
    real strength with a real sig. It is deliberately NOT asked to read pixels;
    a dedicated OCR engine already did that, faster and better.
    """
    settings = settings or get_settings()

    if settings.demo_mode:
        return _demo_prescription(page)

    prompt = _read_prompt("extract.txt").replace(
        "{ocr_lines}", _lines_for_prompt(page)
    )

    raw = await chat(prompt, settings=settings)

    try:
        payload = json.loads(strip_fences(raw))
    except json.JSONDecodeError:
        # A malformed read is itself an unreadable read. Never crash the
        # pharmacist's evening over it, and never silently trust it either.
        return Prescription(
            image_path=page.image_path,
            unreadable=[raw[:2000]],
            overall_confidence=0.0,
            ocr_seconds=page.ocr_seconds,
        )

    lines: list[RxLine] = []
    for item in payload.get("lines", []):
        lines.append(RxLine.model_validate(item))

    _align_to_ocr(lines, page)

    return Prescription(
        image_path=page.image_path,
        patient_name=payload.get("patient_name"),
        age=payload.get("age"),
        sex=payload.get("sex"),
        prescriber=payload.get("prescriber"),
        registration_no=payload.get("registration_no"),
        diagnosis=payload.get("diagnosis"),
        lines=lines,
        unreadable=payload.get("unreadable", []),
        overall_confidence=float(payload.get("overall_confidence", 0.0)),
        ocr_seconds=page.ocr_seconds,
    )


def _lines_for_prompt(page: OcrPage) -> str:
    if not page.lines:
        return "(the OCR found no text on this page)"
    return "\n".join(f"{i}. {line.text}" for i, line in enumerate(page.lines, 1))


_ALNUM = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_ALNUM.findall(text.lower()))


def _match_score(line: RxLine, ocr_text: str) -> float:
    """How well this medicine explains this OCR line.

    The drug name carries almost all the weight. Strength is a small
    tie-breaker only: "40" appears in both a Telma 40 line and a Pan 40 line,
    so on its own it cannot tell two medicines apart.
    """
    drug_tokens = _tokens(line.drug or "")
    if not drug_tokens:
        return 0.0
    haystack = _tokens(ocr_text)
    if not haystack:
        return 0.0

    drug_hits = sum(1 for t in drug_tokens if t in haystack) / len(drug_tokens)
    score = 0.8 * drug_hits

    strength_tokens = _tokens(line.strength or "")
    if strength_tokens:
        strength_hits = sum(1 for t in strength_tokens if t in haystack)
        score += 0.2 * (strength_hits / len(strength_tokens))

    return score


def _align_to_ocr(lines: list[RxLine], page: OcrPage) -> None:
    """Attach each medicine to the OCR line it actually came from.

    We do this here rather than asking the model to cite a line number.
    Asked to count the numbered OCR lines, the model was consistently off by
    two. The raw text is the one field the correction loop and the fine-tune
    dataset absolutely depend on, so the model does not get a vote on it.

    Assignment is best-first across all pairs, which matters: without it, the
    shared "40" lets a Pan line steal a Telma line's match.
    """
    pairs: list[tuple[float, int, int]] = []
    for line_index, line in enumerate(lines):
        for ocr_index, ocr_line in enumerate(page.lines):
            score = _match_score(line, ocr_line.text)
            if score > 0.0:
                pairs.append((score, line_index, ocr_index))
    pairs.sort(key=lambda pair: -pair[0])

    used_lines: set[int] = set()
    used_ocr: set[int] = set()
    for score, line_index, ocr_index in pairs:
        if score < 0.5:
            break
        if line_index in used_lines or ocr_index in used_ocr:
            continue
        used_lines.add(line_index)
        used_ocr.add(ocr_index)
        lines[line_index].ocr_line = ocr_index + 1
        lines[line_index].raw = page.lines[ocr_index].text


async def draft_counselling(
    *,
    prescription: Prescription,
    language: str = "Hindi",
    settings: Settings | None = None,
) -> str:
    """Draft the spoken counselling text the pharmacist plays for the patient.

    The dosing times and the food timing are resolved deterministically by
    `voice.medicines_block` and handed to the model as settled facts. The model
    is doing language here, not pharmacy, because a wrong food timing is a real
    medication error and not one the model's confidence would ever reveal.
    """
    settings = settings or get_settings()

    # The hosted demo has no model, so it gets the deterministic version of the
    # same counselling rather than an apology.
    if settings.demo_mode:
        return compose_counselling(prescription.lines, language=language)

    substitutions = {
        "language": language,
        "patient_name": prescription.patient_name or "(not written on the page)",
        "diagnosis": prescription.diagnosis or "(not written on the page)",
        "medicines": medicines_block(prescription.lines),
    }

    prompt = _read_prompt("counsel.txt")
    for key, value in substitutions.items():
        prompt = prompt.replace(f"{{{{{key}}}}}", value)

    return strip_fences(await chat(prompt, settings=settings))


def _demo_prescription(page: OcrPage) -> Prescription:
    """Deterministic stand-in for the model's answer, for DEMO_MODE.

    Deliberately shaped like a good-but-imperfect model output rather than a
    perfect one: `Pan` comes back at 0.71, which is below the gate's threshold,
    so the hosted demo shows a line actually being sent to the pharmacist
    instead of a page where everything trivially passes.

    Read this next to `demo_data.SAMPLE_OCR_LINES`, which is the real recorded
    OCR transcript for the same page. Note that the diagnosis reads `HTN, T2DM`
    here while the OCR line reads `Ox: HTN, T2OM` -- the reconciliation of that
    line is the point of the whole pipeline.
    """
    return Prescription(
        image_path=page.image_path,
        patient_name="Suresh Wankhede",
        age="42",
        sex="M",
        prescriber="Dr. R. Deshmukh",
        registration_no="2019/4482",
        diagnosis="HTN, T2DM",
        lines=[
            RxLine(raw="1. Tab Telma 40mg 1-0-0 x 30 days", drug="Telma",
                   strength="40mg", form="Tab", sig="1-0-0", days=30, quantity=30,
                   confidence=0.93),
            RxLine(raw="2. Tab Glycomet GP2 1-0-1 x 30 days", drug="Glycomet GP2",
                   strength=None, form="Tab", sig="1-0-1", days=30, quantity=60,
                   confidence=0.88),
            RxLine(raw="3. Cap Pan 40 1-0-0 x15 days", drug="Pan", strength="40mg",
                   form="Cap", sig="1-0-0", days=15, quantity=15, confidence=0.71,
                   flags=["low_confidence"]),
            RxLine(raw="4. Inj. Insulin 0-0-1 x 10 days", drug="Insulin",
                   strength=None, form="Inj", sig="0-0-1", days=10, quantity=10,
                   confidence=0.91),
        ],
        unreadable=[],
        overall_confidence=0.86,
        ocr_seconds=page.ocr_seconds,
    )
