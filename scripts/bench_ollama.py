"""Benchmark a local Ollama model on the real task: messy OCR text -> structured JSON.

Run:  python3 scripts/bench_ollama.py [model] [--runs N]

Two hard-won findings baked in here, both verified against Ollama 0.40.5 +
gemma3:4b on an Intel CPU-only Mac. Do not "fix" them back:

  1. Use /api/chat, NOT /api/generate. gemma3's instruction-following comes
     from its chat template. Through /api/generate the prompt is completed as
     plain text and the model returns an empty object.

  2. Do NOT send format="json". On this model+version combination the JSON
     grammar constraint collapses the output to a bare "{}" every time, no
     matter how simple the instruction. Instead, ask for JSON in the prompt
     and strip the markdown fences the model adds.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.request

OLLAMA = "http://localhost:11434/api"

# Deliberately corrupted the way a real handwriting OCR would corrupt it:
# 0/O and 1/l confusion, dropped characters, split words.
MESSY_OCR = """
Dr. R. Deshmukh   MBBS   Reg 2O19/4482
Pt: Mr. Suresh Wankhede   42/M
Dx: HTN, T2DM

Rx
1. Tab Telrna 40mg      1-0-0   x 3O days
2. Tab Glycomet GP2     l-0-l   x 30 days
3. Cap Pan 4O           1-0-0   x l5 days (bef brkfst)
4. Inj. ????             -       -
Adv: repeat after 15 days
"""

PROMPT = f"""You are reading OCR output from a handwritten prescription at a small pharmacy.
The OCR is noisy: 0 and O, 1 and l are confused, characters are dropped.

Extract every line into JSON with EXACTLY this shape:

{{"lines":[{{"drug":"string or null","strength":"string or null","sig":"string or null",
"days":"string or null","confidence":0.0,"raw":"the exact OCR text this came from"}}],
"unreadable":["text you could not decode"],"overall_confidence":0.0}}

Rules:
1. Never invent a drug name. If the name is corrupt, set drug to null,
   keep the corrupted string in "raw", and set confidence below 0.8.
2. Confidence is your honest probability this line is fully correct.
3. Anything you cannot decode goes in "unreadable".

OCR output:
{MESSY_OCR}

Return ONLY the JSON object, with no markdown fences and no commentary."""


def strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def chat(model: str, prompt: str, timeout: int = 900) -> dict:
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 4096},
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        OLLAMA + "/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    data["response"] = data.get("message", {}).get("content", "")
    return data


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "gemma3:4b"
    runs = 3
    if "--runs" in sys.argv:
        runs = int(sys.argv[sys.argv.index("--runs") + 1])

    print(f"model: {model}")
    print("warm-up (first call loads weights into RAM)...")
    t0 = time.time()
    chat(model, "Reply with the single word: ready")
    print(f"  warm-up took {time.time() - t0:.1f}s")

    walls, rates, parsed_ok = [], [], 0
    last = None
    for i in range(runs):
        t0 = time.time()
        out = chat(model, PROMPT)
        wall = time.time() - t0
        walls.append(wall)

        evals = out.get("eval_count", 0)
        nanos = out.get("eval_duration", 1) or 1
        rate = evals / (nanos / 1e9)
        rates.append(rate)
        last = out

        try:
            json.loads(strip_fences(out["response"]))
            parsed_ok += 1
            ok = "valid JSON"
        except json.JSONDecodeError:
            ok = "NOT valid JSON"
        print(f"  run {i + 1}: wall {wall:.1f}s | {evals} tok | {rate:.2f} tok/s | {ok}")

    print()
    print(f"median wall:  {statistics.median(walls):.1f}s  (per prescription page)")
    print(f"median rate:  {statistics.median(rates):.2f} tok/s")
    print(f"valid JSON:   {parsed_ok}/{runs}")
    print()
    print("--- extracted JSON ---")
    print(json.dumps(json.loads(strip_fences(last["response"])), indent=2, ensure_ascii=False)[:2500])


if __name__ == "__main__":
    main()
