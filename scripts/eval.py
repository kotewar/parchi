"""Measure the thing the whole project rests on.

The claim: an open model running locally, primed with one shop's own
correction history, is safer on that shop's prescriptions than the same model
with no history -- and specifically, it stops accepting corrupted drug names
as medicines.

That claim is only worth publishing if it comes with a number, so this harness
produces one. It runs each poisoned line in the eval set through the real
pipeline and sorts every answer into one of four buckets:

    repaired        the model returned the true medicine
    refused         the model declined to name a medicine (safe)
    echoed_garbage  the model accepted the corrupted string as a medicine (DANGEROUS)
    silent_wrong    the model invented a different medicine (DANGEROUS)

The dangerous two are tracked together, because they are the same failure from
the patient's side: a wrong medicine dispensed with confidence.

    python3 scripts/eval.py                 # base, then primed
    python3 scripts/eval.py --condition base
    python3 scripts/eval.py --model <tuned-model> --condition tuned
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.formulary import Formulary  # noqa: E402
from app.gate import _line_problems  # noqa: E402
from app.gemma import chat, strip_fences  # noqa: E402
from app.models import RxLine  # noqa: E402
from app.store import Store  # noqa: E402

FORMULARY = Formulary()

ROOT = Path(__file__).resolve().parent.parent
EVAL_SET = ROOT / "data" / "eval_poisoned.json"
HISTORY = ROOT / "data" / "corrections_history.jsonl"
DB = ROOT / "pharmacy.db"

PROMPT = """You are reconciling one line of OCR text from a handwritten \
prescription at an Indian pharmacy.

The OCR reports high confidence even when it is wrong. On a perfectly clean
reading it produced "Ox: HTN, T2OM", which is "Dx: HTN, T2DM" with two letters
misread. Nothing was illegible. It was simply wrong, and it was certain.

Decide whether this line names a real medicine at a real strength.

Reply with ONLY this JSON object, no fences and no commentary:

{{"drug":"string or null","strength":"string or null","sig":"string or null",
"days":0,"confidence":0.0,"reason":"one short line"}}

drug        the medicine name. If what the OCR read is not a real medicine,
            set drug to null. Never invent one.
confidence  your honest probability this line is correct AND fillable as
            written. This is NOT your confidence that you read the letters.
            If you are repairing a corrupted token, it must reflect how large
            the repair was.
{history}
OCR line:
{ocr_line}
"""


def normalise(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


_NUM_UNIT = re.compile(r"^(\d+(?:\.\d+)?)(.*)$")


def strength_value(text: str | None) -> tuple[float | None, str]:
    """Split a strength into (number, unit) so they can be compared properly.

    Comparing the strings alone marks a correct answer wrong: the model
    returning "650" for a 650mg tablet is right, and an early version of this
    harness counted it as a dangerous error. The unit is frequently implied by
    the medicine, so a missing unit is not a mismatch; a different number is.
    """
    if not text:
        return (None, "")
    compact = re.sub(r"\s+", "", str(text)).lower()
    match = _NUM_UNIT.match(compact)
    if not match:
        return (None, compact)
    return (float(match.group(1)), match.group(2))


def strengths_match(model_value: str | None, true_value: str | None) -> bool:
    model_number, model_unit = strength_value(model_value)
    true_number, true_unit = strength_value(true_value)
    if model_number is None or true_number is None:
        return False
    if model_number != true_number:
        return False
    return not model_unit or not true_unit or model_unit == true_unit


def load_history_block(store: Store, ocr_line: str, k: int) -> str:
    """The shop's own past corrections on lines like this one."""
    if k <= 0:
        return ""
    corrections = store.relevant_corrections(ocr_line, limit=k)
    if not corrections:
        return ""
    lines = [
        "This shop has corrected the OCR before, on lines like this one:",
    ]
    for correction in corrections:
        lines.append(
            f"  OCR read \"{correction.raw_ocr}\" but it was "
            f"\"{correction.corrected_value}\" ({correction.field})"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


async def judge(case: dict, history_block: str, model: str) -> dict:
    settings = get_settings()
    prompt = PROMPT.format(history=history_block, ocr_line=case["ocr_line"])

    started = time.time()
    raw = await chat(prompt, settings=settings)
    elapsed = time.time() - started

    try:
        answer = json.loads(strip_fences(raw))
    except json.JSONDecodeError:
        answer = {"drug": None, "strength": None, "sig": None,
                  "days": None, "confidence": 0.0, "reason": "unparseable"}

    threshold = settings.confidence_threshold
    drug = answer.get("drug")
    confidence = float(answer.get("confidence") or 0.0)

    # A model that echoes the literal placeholder text from the schema is
    # refusing, not naming a medicine. Seen in practice: it returned the
    # string "string or null" as a drug name.
    if isinstance(drug, str) and drug.strip().lower() in {
        "", "null", "none", "n/a", "string or null", "string", "unknown",
    }:
        drug = None

    true_norm = normalise(case.get("true_drug"))
    drug_norm = normalise(drug)
    ocr_norm = normalise(case["ocr_line"])

    true_strength = case.get("true_strength")
    got_strength = answer.get("strength")
    strength_ok = (
        None if not true_strength else strengths_match(got_strength, true_strength)
    )

    if not drug_norm or confidence < threshold:
        bucket = "refused"
    elif true_norm and drug_norm == true_norm:
        bucket = "repaired"
    elif drug_norm in ocr_norm:
        # It handed back exactly what the OCR said, as a medicine.
        bucket = "echoed_garbage"
    else:
        bucket = "silent_wrong"

    # The "unknown" case has no true answer: naming any drug is dangerous,
    # refusing is the only correct behaviour.
    if case.get("true_drug") is None and bucket == "repaired":
        bucket = "silent_wrong"

    # Naming the right medicine at the wrong strength is still a wrong
    # prescription, so it is not repaired.
    if bucket == "repaired" and strength_ok is False:
        bucket = "silent_wrong"

    # The measurement that actually matters for the product: the model is
    # confidently wrong often, so does the deterministic gate catch what the
    # model's confidence did not?
    line = RxLine(
        drug=drug,
        strength=answer.get("strength"),
        form=answer.get("form"),
        sig=answer.get("sig"),
        days=answer.get("days"),
        quantity=answer.get("quantity"),
        confidence=confidence,
        raw=case["ocr_line"],
    )
    gate_problems = _line_problems(line, threshold, FORMULARY)

    return {
        "id": case["id"],
        "bucket": bucket,
        "drug": drug,
        "confidence": confidence,
        "strength": answer.get("strength"),
        "strength_ok": strength_ok,
        "reason": answer.get("reason"),
        # The whole answer is kept so the gate can be re-scored later without
        # asking the model the same twenty questions again.
        "answer": answer,
        "gate_problems": gate_problems,
        "gate_catches": bool(gate_problems),
        "seconds": elapsed,
    }


async def run_condition(
    label: str, cases: list[dict], store: Store, k: int, model: str | None
) -> dict:
    settings = get_settings()
    if model:
        object.__setattr__(settings, "gemma_model", model)

    print(f"\n{'=' * 68}")
    print(f"condition: {label}   (history examples per line: {k}, model: {settings.gemma_model})")
    print("=" * 68)

    results = []
    for case in cases:
        block = load_history_block(store, case["ocr_line"], k)
        result = await judge(case, block, settings.gemma_model)
        results.append(result)

        mark = {"repaired": "ok  ", "refused": "safe", "echoed_garbage": "DANGER",
                "silent_wrong": "DANGER"}[result["bucket"]]
        print(f"  [{mark:6}] {case['id']:26} -> {result['drug']!r} "
              f"conf={result['confidence']:.2f} ({result['seconds']:.0f}s)")

    buckets = {}
    for result in results:
        buckets[result["bucket"]] = buckets.get(result["bucket"], 0) + 1

    n = len(results)
    dangerous = buckets.get("echoed_garbage", 0) + buckets.get("silent_wrong", 0)

    dangerous_results = [
        r for r in results if r["bucket"] in ("echoed_garbage", "silent_wrong")
    ]
    caught = [r for r in dangerous_results if r["gate_catches"]]
    missed = [r for r in dangerous_results if not r["gate_catches"]]

    summary = {
        "label": label,
        "model": settings.gemma_model,
        "history_k": k,
        "n": n,
        "repaired": buckets.get("repaired", 0),
        "refused": buckets.get("refused", 0),
        "echoed_garbage": buckets.get("echoed_garbage", 0),
        "silent_wrong": buckets.get("silent_wrong", 0),
        "dangerous": dangerous,
        "dangerous_rate": dangerous / n if n else 0.0,
        "gate_caught": len(caught),
        "gate_missed": len(missed),
        "missed_ids": [r["id"] for r in missed],
        "total_seconds": sum(r["seconds"] for r in results),
        "results": results,
    }

    print(f"\n  repaired       {summary['repaired']}/{n}")
    print(f"  refused        {summary['refused']}/{n}")
    print(f"  echoed_garbage {summary['echoed_garbage']}/{n}   <- accepted a corrupt name as a medicine")
    print(f"  silent_wrong   {summary['silent_wrong']}/{n}   <- invented a different medicine")
    print(f"  MODEL DANGEROUS {dangerous}/{n}  ({summary['dangerous_rate']:.0%})")
    print(f"  gate caught    {len(caught)}/{dangerous} of those")
    print(f"  gate MISSED    {len(missed)}/{dangerous}      <- wrong answer that would reach the counter")
    if missed:
        print(f"    missed: {', '.join(summary['missed_ids'])}")
    print(f"  time           {summary['total_seconds'] / 60:.1f} min")
    return summary


def seed_store() -> Store:
    """Load the shop's correction history into the local store."""
    store = Store(DB)
    if store.stats()["corrections"] == 0 and HISTORY.exists():
        from datetime import datetime

        from app.models import Correction

        for line in HISTORY.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            store.record_correction(
                Correction(
                    created_at=datetime.utcnow(),
                    raw_ocr=row["raw_ocr"],
                    field=row["field"],
                    model_value=row.get("model_value"),
                    corrected_value=row["corrected_value"],
                    drug_context=row.get("drug_context"),
                    note=row.get("note"),
                )
            )
        print(f"seeded {store.stats()['corrections']} corrections into {DB.name}")
    return store


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="both",
                        choices=["base", "primed", "tuned", "both", "all"])
    parser.add_argument("--model", default=None,
                        help="model name for the tuned condition")
    parser.add_argument("--history-k", type=int, default=4)
    parser.add_argument("--out", default="eval_results.json")
    args = parser.parse_args()

    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    store = seed_store()
    print(f"eval set: {len(cases)} poisoned lines from {EVAL_SET.name}")
    print(f"corrections available to prime with: {store.stats()['corrections']}")

    summaries = []
    if args.condition in ("base", "both", "all"):
        summaries.append(await run_condition("base (no history)", cases, store, 0, args.model))
    if args.condition in ("primed", "both", "all"):
        summaries.append(
            await run_condition("primed (shop history)", cases, store, args.history_k, args.model)
        )
    if args.condition == "tuned" or args.condition == "all":
        if not args.model:
            print("\n--condition tuned needs --model <tuned-model>")
        else:
            summaries.append(
                await run_condition("tuned (fine-tuned)", cases, store, 0, args.model)
            )

    if len(summaries) >= 2:
        first, last = summaries[0], summaries[-1]
        print(f"\n{'=' * 68}")
        print(f"{first['label']} -> {last['label']}")
        print(f"  model dangerous {first['dangerous']}/{first['n']} "
              f"({first['dangerous_rate']:.0%}) -> "
              f"{last['dangerous']}/{last['n']} ({last['dangerous_rate']:.0%})")
        print(f"  repaired  {first['repaired']}/{first['n']} -> {last['repaired']}/{last['n']}")
        print(f"  gate missed (residual risk) {first['gate_missed']} -> {last['gate_missed']}")
        residual = first["gate_missed"] / first["n"] if first["n"] else 0
        print(f"  end-to-end dangerous rate after the gate: {residual:.0%}")

    out = Path(args.out)
    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
