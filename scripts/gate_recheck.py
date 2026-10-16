"""Re-score saved eval answers against the current gate. No model calls.

The gate is deterministic, so when it changes there is no reason to spend
another five minutes of CPU re-asking the model the same twenty questions.
This reads the answers out of an eval results file and re-applies whatever
rules live in app/gate.py right now.

    python3 scripts/gate_recheck.py eval_base_gated.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.formulary import Formulary  # noqa: E402
from app.gate import _line_problems  # noqa: E402
from app.models import RxLine  # noqa: E402

EVAL_SET = Path(__file__).resolve().parent.parent / "data" / "eval_poisoned.json"
FORMULARY = Formulary()


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "eval_base_gated.json")
    settings = get_settings()
    threshold = settings.confidence_threshold

    cases = {c["id"]: c for c in json.loads(EVAL_SET.read_text(encoding="utf-8"))}
    summaries = json.loads(path.read_text(encoding="utf-8"))

    for summary in summaries:
        print(f"\n{'=' * 70}")
        print(f"re-scored: {summary['label']}  ({summary['model']})")
        print("=" * 70)

        still_missed: list[str] = []
        newly_caught: list[str] = []
        newly_flagged_safe: list[str] = []

        for result in summary["results"]:
            case = cases[result["id"]]
            dangerous = result["bucket"] in ("echoed_garbage", "silent_wrong")

            # Rebuild the line from the model's FULL answer, not just the two
            # fields that happen to be in the summary. Feeding a half-built
            # line to the gate fires structural checks that have nothing to do
            # with what is being tested, and credits the gate with catches it
            # did not make.
            answer = result.get("answer", {})
            line = RxLine(
                drug=result["drug"],
                strength=answer.get("strength"),
                form=answer.get("form"),
                sig=answer.get("sig"),
                days=answer.get("days"),
                quantity=answer.get("quantity"),
                confidence=result["confidence"],
                raw=case["ocr_line"],
            )
            problems = _line_problems(line, threshold, FORMULARY)
            caught = bool(problems)

            was_caught = result.get("gate_catches", False)

            if dangerous and not caught:
                still_missed.append(result["id"])
                print(f"  [STILL MISSED] {result['id']:26} -> {result['drug']!r} "
                      f"strength={result.get('strength')!r}")
            elif dangerous and caught and not was_caught:
                newly_caught.append(result["id"])
                print(f"  [now caught ] {result['id']:26} -> {result['drug']!r} "
                      f"strength={result.get('strength')!r}")
                for problem in problems:
                    print(f"                 {problem}")
            elif not dangerous and caught and not was_caught:
                newly_flagged_safe.append(result["id"])
                print(f"  [over-flag  ] {result['id']:26} -> {result['drug']!r} "
                      f"(correct answer, now sent to review)")

            result["gate_problems"] = problems
            result["gate_catches"] = caught

        dangerous_total = summary["dangerous"]
        summary["gate_caught"] = dangerous_total - len(still_missed)
        summary["gate_missed"] = len(still_missed)
        summary["missed_ids"] = still_missed

        n = summary["n"]
        print(f"\n  model dangerous          {dangerous_total}/{n}")
        print(f"  gate caught              {summary['gate_caught']}/{dangerous_total}")
        print(f"  gate MISSED              {len(still_missed)}/{dangerous_total}")
        print(f"  newly caught by this fix {len(newly_caught)}")
        print(f"  correct answers now over-flagged {len(newly_flagged_safe)}")
        print(f"  END-TO-END dangerous rate "
              f"{len(still_missed)}/{n} ({len(still_missed) / n:.0%})")
        if still_missed:
            print(f"  still missed: {', '.join(still_missed)}")

    out = Path(path).with_suffix(".rescored.json")
    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
