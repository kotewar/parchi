"""Local store for prescriptions, corrections and dispense history.

Deliberately SQLite on the pharmacy's own disk. Two reasons, and the second is
the one that matters:

1. A neighbourhood pharmacy has no database administrator. A file is a
   deployment.

2. A prescription carries a patient's name, age, diagnosis and medicine. That
   is the most sensitive record in the shop. The central claim of this project
   is that the model runs where the data is; shipping the data to a hosted
   database to prove the model is local would give away the thing being
   protected.

Nothing in this file talks to a network. Corrections collected here are what
the fine-tune is later built from, and they never leave this machine unless the
pharmacist exports them deliberately.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from .models import Correction, GateDecision, Prescription

SCHEMA = """
CREATE TABLE IF NOT EXISTS prescriptions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,
    image_path          TEXT,
    patient_name        TEXT,
    age                 TEXT,
    sex                 TEXT,
    prescriber          TEXT,
    registration_no     TEXT,
    diagnosis           TEXT,
    overall_confidence  REAL NOT NULL DEFAULT 0,
    verdict             TEXT NOT NULL,
    ocr_seconds         REAL NOT NULL DEFAULT 0,
    model_seconds       REAL NOT NULL DEFAULT 0,
    payload             TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    raw_ocr         TEXT NOT NULL,
    field           TEXT NOT NULL,
    model_value     TEXT,
    corrected_value TEXT NOT NULL,
    drug_context    TEXT,
    note            TEXT
);

CREATE INDEX IF NOT EXISTS corrections_field_idx ON corrections (field);
CREATE INDEX IF NOT EXISTS corrections_raw_idx ON corrections (raw_ocr);

-- Only what is needed to forecast demand: a medicine, a count, a day. No
-- patient, no prescriber, no diagnosis. This is the only table that could ever
-- leave the machine without carrying anyone's medical history with it.
CREATE TABLE IF NOT EXISTS dispenses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dispensed_on TEXT NOT NULL,
    drug        TEXT NOT NULL,
    strength    TEXT,
    quantity    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS dispenses_drug_idx ON dispenses (drug, dispensed_on);
"""


class Store:
    def __init__(self, path: str | Path = "pharmacy.db") -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---------------------------------------------------------------- writes

    def save_prescription(
        self,
        rx: Prescription,
        decision: GateDecision,
        *,
        model_seconds: float = 0.0,
    ) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO prescriptions (
                created_at, image_path, patient_name, age, sex, prescriber,
                registration_no, diagnosis, overall_confidence, verdict,
                ocr_seconds, model_seconds, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(timespec="seconds"),
                rx.image_path,
                rx.patient_name,
                rx.age,
                rx.sex,
                rx.prescriber,
                rx.registration_no,
                rx.diagnosis,
                rx.overall_confidence,
                decision.verdict.value,
                rx.ocr_seconds,
                model_seconds,
                rx.model_dump_json(),
            ),
        )

        # Only the lines a human actually approved become dispense history.
        # A rejected line must never train the forecaster.
        for index in decision.accepted_line_indexes:
            line = rx.lines[index]
            if line.drug and line.quantity:
                self.conn.execute(
                    "INSERT INTO dispenses (dispensed_on, drug, strength, quantity)"
                    " VALUES (?, ?, ?, ?)",
                    (date.today().isoformat(), line.drug, line.strength, line.quantity),
                )

        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def record_correction(self, correction: Correction) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO corrections (
                created_at, raw_ocr, field, model_value, corrected_value,
                drug_context, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                correction.created_at.isoformat(timespec="seconds"),
                correction.raw_ocr,
                correction.field,
                correction.model_value,
                correction.corrected_value,
                correction.drug_context,
                correction.note,
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def record_dispense(
        self, drug: str, quantity: int, *, strength: str | None = None,
        on: date | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO dispenses (dispensed_on, drug, strength, quantity)"
            " VALUES (?, ?, ?, ?)",
            ((on or date.today()).isoformat(), drug, strength, quantity),
        )
        self.conn.commit()

    # ---------------------------------------------------------------- reads

    def corrections(self, limit: int = 200) -> list[Correction]:
        rows = self.conn.execute(
            "SELECT * FROM corrections ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            Correction(
                created_at=datetime.fromisoformat(row["created_at"]),
                raw_ocr=row["raw_ocr"],
                field=row["field"],
                model_value=row["model_value"],
                corrected_value=row["corrected_value"],
                drug_context=row["drug_context"],
                note=row["note"],
            )
            for row in rows
        ]

    def stats(self) -> dict[str, int]:
        def count(table: str) -> int:
            return int(
                self.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            )

        verdicts = self.conn.execute(
            "SELECT verdict, COUNT(*) AS n FROM prescriptions GROUP BY verdict"
        ).fetchall()
        out = {
            "prescriptions": count("prescriptions"),
            "corrections": count("corrections"),
            "dispenses": count("dispenses"),
        }
        for row in verdicts:
            out[f"verdict_{row['verdict']}"] = int(row["n"])
        return out

    def dispense_history(self, days: int = 120) -> list[sqlite3.Row]:
        """Daily totals per medicine, for the demand forecaster."""
        since = (date.today() - timedelta(days=days)).isoformat()
        return self.conn.execute(
            """
            SELECT dispensed_on, drug, strength, SUM(quantity) AS quantity
            FROM dispenses
            WHERE dispensed_on >= ?
            GROUP BY dispensed_on, drug, strength
            ORDER BY dispensed_on
            """,
            (since,),
        ).fetchall()

    # ------------------------------------------------- what the model gets told

    def relevant_corrections(
        self, ocr_text: str, *, limit: int = 5
    ) -> list[Correction]:
        """Corrections that look like they bear on this page.

        This is the cheap, immediately useful half of the fine-tune idea: the
        shop's own history, consulted at read time, before any training run has
        happened. Matching is on token overlap with the OCR text.
        """
        wanted = {
            token
            for token in _tokens(ocr_text)
            if len(token) >= 4 and not token.isdigit()
        }
        if not wanted:
            return []

        scored: list[tuple[int, Correction]] = []
        for correction in self.corrections(limit=500):
            haystack = _tokens(f"{correction.raw_ocr} {correction.drug_context or ''}")
            overlap = len(wanted & haystack)
            if overlap:
                scored.append((overlap, correction))

        scored.sort(key=lambda pair: -pair[0])
        return [correction for _, correction in scored[:limit]]

    def training_rows(self) -> list[dict]:
        """Corrections shaped as chat rows for the fine-tune.

        One row per correction: the OCR text plus the shop's decision. This is
        the dataset. It is small, it is specific to one shop, and it is the
        reason a general model cannot substitute for this one.
        """
        rows = []
        for correction in self.corrections(limit=10_000):
            answer = json.dumps(
                {
                    "field": correction.field,
                    "model_value": correction.model_value,
                    "corrected_value": correction.corrected_value,
                    "drug_context": correction.drug_context,
                },
                ensure_ascii=False,
            )
            rows.append(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You correct OCR text from a small Indian pharmacy's "
                                "prescriptions. Reply with the corrected value only."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"OCR read: {correction.raw_ocr}\n"
                                f"Field: {correction.field}\n"
                                f"The model said: {correction.model_value}\n"
                                "What should it be?"
                            ),
                        },
                        {"role": "assistant", "content": answer},
                    ],
                    "metadata": {"source": "pharmacist_correction"},
                }
            )
        return rows

    def export_training_jsonl(self, path: str | Path) -> int:
        rows = self.training_rows()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(rows)


_ALNUM_RE = None


def _tokens(text: str) -> set[str]:
    global _ALNUM_RE
    if _ALNUM_RE is None:
        import re

        _ALNUM_RE = re.compile(r"[a-z0-9]+")
    return set(_ALNUM_RE.findall(text.lower()))
