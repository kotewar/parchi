"""Read a page with the macOS Vision framework.

We call a small compiled Swift binary rather than a Python binding, because the
binary has no dependencies, starts in milliseconds, and cannot be broken by a
pip resolution on the pharmacy's machine. Build it once with:

    swiftc -O scripts/ocr_vision.swift -o bin/ocr_vision
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .models import OcrPage

REPO_ROOT = Path(__file__).resolve().parent.parent
BINARY = REPO_ROOT / "bin" / "ocr_vision"
SOURCE = REPO_ROOT / "scripts" / "ocr_vision.swift"


class OcrUnavailable(RuntimeError):
    pass


def build() -> Path:
    """Compile the OCR binary if it is missing or older than its source."""
    if BINARY.exists() and (
        not SOURCE.exists() or BINARY.stat().st_mtime >= SOURCE.stat().st_mtime
    ):
        return BINARY

    if shutil.which("swiftc") is None:
        raise OcrUnavailable(
            "swiftc not found. Install the Xcode command line tools: "
            "xcode-select --install"
        )

    BINARY.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["swiftc", "-O", str(SOURCE), "-o", str(BINARY)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise OcrUnavailable(f"failed to build ocr_vision:\n{result.stderr}")
    return BINARY


def make_sample(path: str | Path = "/tmp/rx_sample.png") -> Path:
    """Render a synthetic prescription to test the pipeline without a scanner."""
    binary = build()
    result = subprocess.run(
        [str(binary), "--make-sample", str(path)], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise OcrUnavailable(result.stderr)
    return Path(result.stdout.strip())


def read_page(path: str | Path) -> OcrPage:
    """OCR one image into line-grouped text with positions."""
    binary = build()
    result = subprocess.run([str(binary), str(path)], capture_output=True, text=True)

    if result.returncode != 0:
        raise OcrUnavailable(
            f"ocr_vision failed on {path} (exit {result.returncode}): {result.stderr}"
        )

    return OcrPage.model_validate(json.loads(result.stdout))


def lines_as_prompt_text(page: OcrPage) -> str:
    """Render OCR lines for the model prompt.

    The OCR confidence is deliberately withheld from the model. Handing it a
    confident-looking 1.00 next to garbage text is how you get a confidently
    wrong answer back, and 1.00 next to garbage is precisely what this engine
    was measured to produce.
    """
    out = []
    for i, line in enumerate(page.lines, start=1):
        out.append(f"{i}. {line.text}")
    if not out:
        return "(the OCR found no text on this page)"
    return "\n".join(out)
