"""Run one prescription page through the whole pipeline.

    python3 scripts/run_page.py [image] [--demo]

With no image it renders a synthetic page and reads that. Prints the OCR, the
reconciled prescription, the gate decision, and where the time went.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import ocr  # noqa: E402
from app.gate import decide  # noqa: E402
from app.gemma import reconcile  # noqa: E402


async def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    demo = "--demo" in sys.argv

    if demo:
        from app.config import get_settings
        settings = get_settings()
        object.__setattr__(settings, "demo_mode", True)

    image = Path(args[0]) if args else ocr.make_sample()
    print(f"image: {image}")

    t0 = time.time()
    page = ocr.read_page(image)
    ocr_wall = time.time() - t0

    print(f"\n--- OCR ({ocr_wall:.2f}s wall, {page.ocr_seconds:.2f}s engine, "
          f"{page.observations} observations -> {len(page.lines)} lines) ---")
    for line in page.lines:
        print(f"  [conf {line.ocr_confidence:.2f}] {line.text}")

    t0 = time.time()
    rx = await reconcile(page)
    reconcile_wall = time.time() - t0

    print(f"\n--- reconciled prescription ({reconcile_wall:.1f}s wall) ---")
    print(f"  patient:   {rx.patient_name}  {rx.age}/{rx.sex}")
    print(f"  prescriber: {rx.prescriber}  reg {rx.registration_no}")
    print(f"  diagnosis: {rx.diagnosis}")
    for i, line in enumerate(rx.lines):
        print(f"  [{i}] {line.drug} {line.strength or ''} {line.form or ''} "
              f"sig={line.sig} days={line.days} qty={line.quantity} "
              f"conf={line.confidence:.2f} flags={line.flags}")
        print(f"      raw: {line.raw}")
    if rx.unreadable:
        print(f"  unreadable: {rx.unreadable}")
    print(f"  overall confidence: {rx.overall_confidence:.2f}")

    decision = decide(rx)
    print(f"\n--- gate: {decision.verdict.value} ---")
    for reason in decision.reasons:
        print(f"  - {reason}")
    print(f"  accepted lines: {decision.accepted_line_indexes}")
    print(f"  review lines:   {decision.review_line_indexes}")

    print(f"\ntotal: {ocr_wall + reconcile_wall:.1f}s "
          f"(OCR {ocr_wall:.1f}s + model {reconcile_wall:.1f}s)")


if __name__ == "__main__":
    asyncio.run(main())
