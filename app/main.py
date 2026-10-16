"""The counter screen.

This is NOT the deployment. The real one runs on the shop's own laptop, with no
network, which is the entire argument of the project. This is the decision layer
made reachable from a browser, so it can be looked at without an Intel Mac and a
3.4 GB model download.

Two parts of the pipeline cannot run on a hosted instance, and the page says so
rather than hiding it:

  - the OCR is macOS Vision, so the transcript shown is a real recorded one
    (`demo_data.SAMPLE_OCR_LINES`), not a live read of a live page;
  - the model is 3.4 GB and a free instance has no accelerator, so its step is
    served by the deterministic fixtures `DEMO_MODE` uses locally.

Everything after those two is live code running on request: the gate, the stock
plan and the counselling text are all computed here, not pasted in.

The single most useful endpoint for a reader is `POST /api/gate`, which takes
model output and returns the gate's decision. It needs no model and no OCR, so
it can be exercised directly against the failure modes described in the write-up.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from html import escape

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import voice
from .config import get_settings
from .demo_data import sample_page, seed_dispense_history
from .formulary import Formulary, food_timing
from .gate import decide
from .gemma import GemmaUnavailable, draft_counselling, reconcile
from .models import RxLine
from .store import Store
from .tabpfn_engine import TABPFN_AVAILABLE, plan_reorders


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.demo_mode:
        store = Store(settings.database_path)
        if store.stats()["dispenses"] == 0:
            seed_dispense_history(store)
        store.close()
    yield


app = FastAPI(
    title="Parchi",
    description="A local-first prescription reader for a neighbourhood pharmacy.",
    lifespan=lifespan,
)

# Observability is opt-in and off unless a DSN is set, which matters more here
# than in most projects: on the shop's own machine, error reporting would ship
# traces off the premises, and keeping things on the premises is the whole
# argument. The pharmacist has to turn this on deliberately.
#
# It is on in the hosted demo, where there is no patient data to leak, because
# an unhandled exception in front of a judge is worth knowing about.
_startup_settings = get_settings()
if _startup_settings.sentry_dsn:
    import sentry_sdk

    sentry_sdk.init(
        dsn=_startup_settings.sentry_dsn,
        traces_sample_rate=1.0,
        environment="demo" if _startup_settings.demo_mode else "pharmacy",
    )


class GateLine(BaseModel):
    """One line of model output, as the gate would receive it."""

    drug: str | None = None
    strength: str | None = None
    form: str | None = None
    sig: str | None = None
    days: int | None = None
    quantity: int | None = None
    confidence: float = 0.0
    raw: str = ""


class GateRequest(BaseModel):
    lines: list[GateLine] = Field(default_factory=list)
    # Optional. When absent it is derived from the lines, so a caller sending
    # only lines does not get a spurious "whole-page confidence 0.00" reason.
    overall_confidence: float | None = None


def _formulary() -> Formulary:
    settings = get_settings()
    if settings.formulary_path:
        return Formulary.from_file(settings.formulary_path)
    return Formulary()


def _render_line(line: RxLine) -> dict:
    return {
        "drug": line.drug,
        "strength": line.strength,
        "form": line.form,
        "sig": line.sig,
        "days": line.days,
        "quantity": line.quantity,
        "confidence": line.confidence,
        "raw": line.raw,
        "food_timing": food_timing(line.drug),
        "in_stock": None if line.drug is None else _formulary().knows(line.drug),
    }


@app.get("/api/health")
async def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "demo_mode": settings.demo_mode,
        "model": settings.gemma_model,
        "ocr": "macOS Vision (recorded transcript in the hosted demo)",
        "tabpfn": TABPFN_AVAILABLE,
        "voice": bool(settings.elevenlabs_api_key and settings.elevenlabs_voice_id),
    }


@app.post("/api/gate")
async def gate(request: GateRequest) -> dict:
    """Run the safety gate over supplied lines. Needs no model, no OCR, no network."""
    settings = get_settings()

    if not request.lines:
        return {
            "verdict": "reject",
            "reasons": ["no prescription lines were supplied"],
            "accepted": [],
            "review": [],
        }

    from .models import Prescription

    lines = [RxLine.model_validate(line.model_dump()) for line in request.lines]
    overall = request.overall_confidence
    if overall is None and lines:
        overall = sum(line.confidence for line in lines) / len(lines)

    rx = Prescription(lines=lines, overall_confidence=overall or 0.0)
    decision = decide(rx, settings=settings, formulary=_formulary())

    return {
        "verdict": decision.verdict.value,
        "reasons": decision.reasons,
        "accepted": decision.accepted_line_indexes,
        "review": decision.review_line_indexes,
        "lines": [_render_line(line) for line in rx.lines],
    }


@app.get("/api/plan")
async def plan() -> dict:
    """What the shop should stock up on, forecast from its own dispensing."""
    settings = get_settings()
    store = Store(settings.database_path)
    try:
        rows = plan_reorders(store)
        method = rows[0]["method"] if rows else "not enough history yet"
        return {"method": method, "reorders": rows}
    finally:
        store.close()


@app.get("/api/stats")
async def stats() -> dict:
    settings = get_settings()
    store = Store(settings.database_path)
    try:
        return store.stats()
    finally:
        store.close()


@app.post("/api/counselling")
async def counselling(include_audio: bool = False) -> dict:
    """Draft the spoken counselling, and optionally render it to audio.

    Audio is opt-in because it is the only step that sends anything off the
    machine, and the text it sends necessarily contains the medicines.
    """
    settings = get_settings()

    rx = await reconcile(sample_page(), settings=settings)
    try:
        text = await draft_counselling(prescription=rx, settings=settings)
    except GemmaUnavailable as exc:
        return {"text": None, "error": str(exc)}

    if not include_audio:
        return {"text": text, "voice_available": False}

    spoken = await voice.speak(text, settings=settings)
    return spoken


def _page() -> str:
    settings = get_settings()
    page = sample_page()
    formulary = _formulary()

    rx = None
    reconcile_error = None
    if settings.demo_mode:
        from .gemma import _demo_prescription

        rx = _demo_prescription(page)

    rows: list[tuple[str, str, str]] = []
    for index, line in enumerate(page.lines, start=1):
        is_misread = line.text.startswith("Ox:")
        style = ' class="flag"' if is_misread else ""
        rows.append(
            f"<tr{style}><td>{index}</td><td>{escape(line.text)}</td>"
            f"<td>{line.ocr_confidence:.2f}</td></tr>"
        )

    ocr_table = (
        "<table><thead><tr><th>#</th><th>OCR read</th><th>OCR conf</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )

    if rx is None:
        gate_block = (
            "<p class='muted'>Live reconciliation needs the local model, which this "
            "instance does not have. Set <code>DEMO_MODE=true</code> to see the "
            "fixtures, or run it on the shop's own machine.</p>"
        )
        counsel_block = ""
    else:
        decision = decide(rx, settings=settings, formulary=formulary)
        reasons = "".join(f"<li>{escape(r)}</li>" for r in decision.reasons)
        line_rows = []
        for index, line in enumerate(rx.lines):
            sent = index in decision.review_line_indexes
            badge = (
                "<span class='review'>to review</span>"
                if sent
                else "<span class='ok'>fill</span>"
            )
            timing = food_timing(line.drug) or "<span class='muted'>no rule</span>"
            line_rows.append(
                f"<tr><td>{index + 1}</td><td>{escape(str(line.drug))}</td>"
                f"<td>{escape(str(line.strength or '—'))}</td>"
                f"<td>{escape(str(line.sig or '—'))}</td>"
                f"<td>{line.confidence:.2f}</td><td>{escape(str(timing))}</td>"
                f"<td>{badge}</td></tr>"
            )
        gate_block = (
            f"<p class='verdict {decision.verdict.value}'>"
            f"verdict: <strong>{decision.verdict.value}</strong></p>"
            f"<table><thead><tr><th>#</th><th>medicine</th><th>strength</th>"
            f"<th>sig</th><th>model conf</th><th>food timing</th><th>gate</th>"
            f"</tr></thead><tbody>{''.join(line_rows)}</tbody></table>"
            f"<h3>Why</h3><ul class='reasons'>{reasons}</ul>"
        )

        counsel_text = voice.compose_counselling(rx.lines)
        counsel_block = (
            "<h3>Spoken counselling, as the pharmacist plays it</h3>"
            f"<blockquote>{escape(counsel_text)}</blockquote>"
            "<p class='muted'>That is Devanagari, and it is written by deterministic "
            "code rather than by a model. The dosing times and the food timing come "
            "from tables, because a wrong food timing is a real medication error and "
            "not one a model's confidence would ever reveal.</p>"
        )

    store = Store(settings.database_path)
    try:
        reorders = plan_reorders(store)
    finally:
        store.close()

    if reorders:
        plan_rows = "".join(
            f"<tr><td>{escape(row['drug'])}</td>"
            f"<td>{escape(str(row['strength'] or '—'))}</td>"
            f"<td>{row['weekly_mean']:g}</td><td>{row['forecast_7d']:g}</td>"
            f"<td>{'yes' if row['chronic'] else 'no'}</td>"
            f"<td>{row['urgency']:g}</td></tr>"
            for row in reorders
        )
        plan_block = (
            f"<p class='muted'>method: {escape(reorders[0]['method'])}</p>"
            "<table><thead><tr><th>medicine</th><th>strength</th><th>weekly mean</th>"
            "<th>forecast next 7d</th><th>chronic</th><th>urgency</th></tr></thead>"
            f"<tbody>{plan_rows}</tbody></table>"
        )
    else:
        plan_block = (
            "<p class='muted'>No dispensing history yet. Run "
            "<code>scripts/seed_demo_data.py</code>.</p>"
        )

    banner = (
        "This instance has <strong>DEMO_MODE on</strong>. The OCR transcript below is "
        "a <em>real recorded read</em> of a synthetic page. The model's step is served "
        "by deterministic fixtures, because this machine has no GPU and the model is "
        "3.4 GB. Everything after that — the gate, the stock plan, the counselling — is "
        "live code running on this request."
        if settings.demo_mode
        else "Running against the local model."
    )

    return _PAGE.replace("{{BANNER}}", banner).replace(
        "{{OCR}}", ocr_table
    ).replace("{{GATE}}", gate_block).replace(
        "{{PLAN}}", plan_block
    ).replace("{{COUNSEL}}", counsel_block).replace(
        "{{MODEL}}", escape(settings.gemma_model)
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _page()


_STYLE = """
:root { color-scheme: light dark; }
body { font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 60rem; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
h1 { font-size: 1.5rem; margin-bottom: .25rem; }
h2 { font-size: 1.05rem; margin-top: 2.5rem; border-bottom: 1px solid rgba(128,128,128,.35);
     padding-bottom: .3rem; }
h3 { font-size: .95rem; margin-top: 1.5rem; }
table { border-collapse: collapse; width: 100%; margin: .75rem 0; font-size: .9rem; }
th, td { text-align: left; padding: .35rem .55rem; border-bottom: 1px solid rgba(128,128,128,.25); }
th { font-weight: 600; opacity: .75; }
code { font-size: .85em; background: rgba(128,128,128,.15); padding: .1em .3em; border-radius: 3px; }
.banner { background: rgba(200,150,0,.13); border-left: 3px solid rgba(200,150,0,.8);
          padding: .7rem .9rem; margin: 1.5rem 0; font-size: .92rem; }
.muted { opacity: .65; font-size: .88rem; }
tr.flag td { background: rgba(200,60,60,.12); font-weight: 600; }
.verdict { font-size: 1rem; }
.verdict.accept strong { color: #1a7f37; }
.verdict.review strong { color: #b7791f; }
.verdict.reject strong { color: #c0392b; }
.ok { color: #1a7f37; font-weight: 600; }
.review { color: #b7791f; font-weight: 600; }
.reasons li { font-size: .9rem; margin: .2rem 0; }
blockquote { margin: .75rem 0; padding: .7rem .9rem; border-left: 3px solid rgba(128,128,128,.4);
             background: rgba(128,128,128,.07); font-size: .95rem; }
footer { margin-top: 3rem; font-size: .85rem; opacity: .6; }
"""

_PAGE = (
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Parchi — a prescription reader that knows when not to be trusted</title>
<style>"""
    + _STYLE
    + """</style>
</head>
<body>
<h1>Parchi</h1>
<p class="muted">A prescription reader for a neighbourhood medical shop. It runs on the
shop's own machine, and its job is deciding which lines a human must check.</p>

<div class="banner">{{BANNER}}</div>

<h2>1. What the OCR read</h2>
<p class="muted">Every line came back at confidence 1.00, including the second one in
the highlighted row. That is the whole reason the OCR's confidence is not used as a
safety signal anywhere in this system.</p>
{{OCR}}

<h2>2. What the gate decided</h2>
{{GATE}}

<h2>3. What to stock up on</h2>
<p class="muted">Forecast from the shop's own dispensing history. No patient, age,
diagnosis or prescriber is involved &mdash; the <code>dispenses</code> table has none of
those columns, on purpose.</p>
{{PLAN}}

<h2>4. What the patient hears</h2>
{{COUNSEL}}

<footer>
Model: <code>{{MODEL}}</code> via Ollama.
Try <code>POST /api/gate</code> with your own lines &mdash; it needs no model and no OCR.
See <code>/docs</code>.
</footer>
</body>
</html>
"""
)
