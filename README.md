# Parchi

A prescription reader for a neighbourhood medical shop in India. It runs on the
shop's own machine, with no network connection at all.

The pharmacist photographs a handwritten prescription. The system reads it,
works out what was actually prescribed, and decides for itself which lines are
safe to fill and which ones a human needs to check against the paper.

The interesting part is not the reading. It is **deciding when not to trust the
reading**.

---

## The measurement this is built on

While validating the OCR path, a test prescription rendered in a handwriting
font was read back flawlessly — **confidence 1.00 on every line**. One of the
lines it read perfectly was this:

```
Ox: HTN, T2OM
```

That is `Dx: HTN, T2DM` with two letters misread. Nothing about it was
illegible. It was wrong, and the OCR was certain.

So OCR confidence measures *"were the pixels legible"*, not *"is this thing
real"*, and it is discarded as a safety signal everywhere in this codebase.

That confidence scores are useless for this is not a new observation — see
[Confidence Is Theater](https://dev.to/sohakanu/confidence-is-theater-benchmarking-nine-local-vlm-pipelines-on-handwritten-clinical-forms-38h6)
by [sohakanu](https://dev.to/sohakanu), who found the same thing on handwritten
clinical forms and solved it with cross-model verification. Parchi takes a
different route, because a pharmacy has something a form does not: a fixed
vocabulary. The shop knows which forty medicines it stocks, and that is
checkable in code, for free.

## Architecture

```
page image
   -> macOS Vision OCR        (reading: ~1.8s/page)
   -> gemma3:4b via Ollama    (meaning: what is this medicine?)
   -> deterministic gate      (safety: may this be filled?)
   -> human review            (only what the gate refuses)
```

The OCR reads. The model understands. The gate decides. Only one of those three
is allowed to be wrong quietly, and it is not the model.

The gate is deterministic and needs no model and no network. It checks, per line:

| check | what it catches |
|---|---|
| grounding | a medicine name that does not appear anywhere on the page |
| strength parse | a dose where a letter was misread as a digit |
| sig shape | a dosing pattern that is not a morning-afternoon-night pattern |
| arithmetic | quantity that disagrees with the sig and the duration |
| stock list | a medicine this shop does not keep |
| confidence | the model's own semantic confidence, as one input among several |

## The eval

Twenty realistic OCR corruptions of real Indian brands (`0` read as `O`, `1` as
`l`, `5` as `S`), in `data/eval_poisoned.json`.

| | base `gemma3:4b` | + the shop's own correction history |
|---|---|---|
| wrong answers | 9/20 (45%) | **6/20 (30%)** |
| correct | 9/20 | **13/20** |
| filled with no human looking | 7/20 | **11/20** |
| wrong answers reaching the counter | **0/20** | **0/20** |

18 of 20 base answers scored 0.90 or above, and **9 of those 18 were wrong**. The
gate caught 9/9 with 0 missed.

Reproduce it, without needing a model for the gate half:

```bash
python3 scripts/selfcheck.py                  # 47 checks, no setup, no model
python3 scripts/eval.py --condition base      # ~6 min on an 8-core CPU
python3 scripts/eval.py --condition primed    # with the shop's corrections
python3 scripts/gate_recheck.py eval_base_v2.json   # re-scores, no model calls
```

`selfcheck.py` needs nothing installed and no model running. It encodes the
cases that were wrong at some point in this project's life, including a false
positive that shipped: the strength check used to flag the `GP2` in
`Glycomet GP2`, which is a correct answer. A gate that flags correct answers is
a gate people learn to click past, and then it protects nobody.

### Two failure modes, both found by measurement

**The model substitutes generics for brands.** Asked about `Glycomet 6P2` it
answered `Metformin`; for `Pan 4O`, `Panadol`; for `Atorva 1O`, `Atorvastatin`.
All at 0.95 confidence. These are not cosmetic: Glycomet GP2 contains
glimepiride as well as metformin, and Panadol is paracetamol where Pan is
pantoprazole. A generic substitution is a different prescription. It is
invisible in the JSON except in one place — **the returned name is not on the
page** — which is what `_grounding_problems` checks.

**All four remaining misses were the same thing**: a correct, grounded drug name
with a corrupted **strength**. `5Omcg` for 50mcg, `7S` for 75, `2.S` for 2.5,
`1O units` for 10 units. The gate had accepted them because it only checked that
the strength field was *non-empty*, not that it was a strength. One regex closed
the gap.

### A result worth keeping

Feeding the shop's own 18 corrections back into the prompt improves the totals
(45% → 30% wrong) but introduces **three regressions**: the model starts snapping
to the nearest familiar name and silently drops suffixes. `Telmikind-H` becomes
`Telmikind`, and Telmikind-H contains a diuretic where Telmikind does not.

So the correction history is a trade, not a free win. The gate catches all three
anyway, which is the point of having a gate that does not care how confident the
model sounds.

## Running it

Requires macOS for the OCR (the Vision framework) and
[Ollama](https://ollama.com) with `gemma3:4b` for the model.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

ollama pull gemma3:4b                     # 3.4 GB
nohup env OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 \
    ollama serve > /tmp/ollama.log 2>&1 < /dev/null &
trap 'pkill -f "ollama serve"' EXIT

swiftc -O scripts/ocr_vision.swift -o bin/ocr_vision   # build the OCR helper

python3 scripts/run_page.py --make-sample              # render a synthetic page
python3 scripts/run_page.py                            # read it end to end
python3 scripts/seed_demo_data.py                      # 14 weeks of dispense history
```

The web app, on the shop's own machine:

```bash
uvicorn app.main:app --reload
```

### Without a model, or without macOS

The pipeline degrades instead of failing:

```bash
DEMO_MODE=true uvicorn app.main:app      # fixtures instead of calling Ollama
```

`DEMO_MODE` is what the hosted demo runs. On Linux the OCR helper cannot be
built at all — `app/ocr.py` raises `OcrUnavailable` — so anything that needs a
live page read will not run there. The gate, the stock plan and the counselling
text do not need it.

## The app

| endpoint | what it does | needs a model? |
|---|---|---|
| `GET /` | the whole pipeline on one page, labelled live-vs-recorded | only for the reconcile step |
| `POST /api/gate` | gate supplied lines, return the decision and reasons | **no** |
| `GET /api/plan` | what to restock, forecast from dispense history | no |
| `POST /api/counselling` | draft the spoken counselling, optionally as audio | yes (falls back) |
| `GET /api/health` | what is actually available on this instance | no |
| `GET /api/stats` | counts from the local store | no |

`POST /api/gate` is the one worth trying, because it needs nothing else:

```bash
curl -s localhost:8000/api/gate -H 'content-type: application/json' -d '{
  "lines": [{"drug":"Metformin","strength":"600 mg","form":"Tab","sig":"1-0-1",
             "days":30,"quantity":60,"confidence":0.95,
             "raw":"2. Tab Glycomet 6P2 1-0-1 x 30 days"}]}'
```

```json
{"verdict":"review","accepted":[],
 "reasons":["line 1 (Metformin): 'Metformin' does not appear on the page as written (OCR line reads: '2. Tab Glycomet 6P2 1-0-1 x 30 days')"]}
```

## Privacy

The store is SQLite on the shop's own disk. A prescription carries a patient's
name, age, diagnosis and medicines, which is the most sensitive record the shop
holds, and the claim being made is that the model runs where the data already
is.

Two things leave the machine, both deliberately:

- **`POST /api/counselling?include_audio=true`** sends the counselling text to
  ElevenLabs. That text necessarily contains the medicine names and how often to
  take them — that *is* the counselling. It does not contain the image, the
  patient's name, the diagnosis or the prescriber. It is opt-in per page.
- **Sentry**, only if `SENTRY_DSN` is set. Off by default.

The `dispenses` table — the only one the TabPFN forecaster reads — holds a
medicine, a count and a day. No patient, no prescriber, no diagnosis. It is the
only table that could leave the machine without carrying anyone's medical
history with it, which is why this forecasts demand *per medicine* rather than
refills *per patient*, even though the second would be the richer signal.

## Two gotchas, found by measurement

Both are commented in `app/gemma.py` because they cost real time to find. On
Ollama 0.40.5 with `gemma3:4b`:

1. **Use `/api/chat`, not `/api/generate`.** gemma3's instruction-following
   lives in its chat template. Through `/api/generate` the prompt is completed
   as raw text and the model returns `{}`.
2. **Do not send `format: "json"`.** The JSON grammar constraint collapses the
   output to a bare `{}` for every instruction tried, however simple. Ask for
   JSON in the prompt and strip the fences.

## Layout

```
app/
  ocr.py             Swift Vision wrapper; builds the helper on first use
  gemma.py           the model call, plus the two gotchas above
  gate.py            the deterministic safety checks
  formulary.py       the shop's stock list, food timing, chronic list
  models.py          pydantic schemas, including the tolerant field validators
  store.py           SQLite: prescriptions, corrections, dispenses
  tabpfn_engine.py   demand forecasting from dispense history
  voice.py           counselling text and text-to-speech
  demo_data.py       the recorded OCR transcript, and demo fixtures
  main.py            FastAPI
scripts/
  ocr_vision.swift   the OCR helper (macOS Vision)
  run_page.py        image -> OCR -> reconcile -> gate
  selfcheck.py       regression checks, runnable with plain python3
  eval.py            the poisoned-name eval harness
  gate_recheck.py    re-apply the gate to saved answers, no model calls
  bench_ollama.py    CPU throughput benchmark
  seed_demo_data.py  fill the store so the stock plan has something to read
data/
  eval_poisoned.json twenty corruptions with ground truth
prompts/
  extract.txt        reconciliation
  counsel.txt        spoken counselling
```

## Status

Working: OCR, reconciliation, the gate, the correction loop, the eval harness,
the store, the app, demand forecasting.

Not yet: the fine-tune. `store.export_training_jsonl()` produces the dataset from
real pharmacist corrections and `scripts/eval.py --condition tuned` will score a
fine-tuned model, but the training run needs an API key. The correction history
is already being collected, which is the part that cannot be synthesised later.

*All names, prescriptions and corrections in this repository are synthetic. No
real patient data was used.*
