"""What the shop needs to stock up on, forecast from its own dispensing history.

Why TabPFN rather than the obvious alternatives: a neighbourhood pharmacy's
dispense history is tiny. A dozen to sixty weekly points per medicine, not the
hundreds of thousands of rows a gradient-boosted model wants. That regime --
few rows, few features, no time to tune anything -- is where TabPFN's in-context
inference is strongest, and it gets there without a hyperparameter search. A
model that has to be tuned is a model the pharmacist cannot retrain himself
when his stock changes, and a model he cannot retrain is one he will stop
trusting.

What this deliberately does NOT use: the patient. The `dispenses` table holds a
medicine, a count and a day -- no name, no age, no diagnosis, no prescriber.
It is the only table in this project that could leave the machine without
carrying anyone's medical history with it, and forecasting must not be the
thing that breaks that. So this forecasts demand *per medicine*, not refills
*per patient*, even though the second would be the richer signal. The privacy
boundary is worth more than the accuracy.

Without tabpfn installed the app still runs end to end: forecasts fall back to
a damped moving average, which is roughly what a pharmacist does in his head.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable

from .formulary import is_chronic

try:  # TabPFN pulls in torch. A dev machine without it still gets a working app.
    from tabpfn import TabPFNRegressor

    TABPFN_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    TabPFNRegressor = None  # type: ignore[assignment]
    TABPFN_AVAILABLE = False


PERIOD_DAYS = 7

# Below this there is not enough history for in-context inference to beat
# arithmetic, and pretending otherwise would just be slower.
MIN_ROWS_FOR_TABPFN = 16

# Whether a medicine is taken indefinitely lives in the formulary, next to the
# rest of the shop's knowledge about its own medicines. See formulary.CHRONIC.

COLUMNS = [
    "lag_1", "lag_2", "lag_3",
    "mean_4", "std_4", "trend",
    "chronic", "weeks",
]


@dataclass
class WeeklyPoint:
    week_start: date
    quantity: int


@dataclass
class RefillFeatures:
    drug: str
    strength: str | None = None
    lag_1: float = 0.0
    lag_2: float = 0.0
    lag_3: float = 0.0
    mean_4: float = 0.0
    std_4: float = 0.0
    trend: float = 0.0
    chronic: int = 0
    weeks: int = 0

    def as_row(self) -> list[float]:
        return [
            self.lag_1, self.lag_2, self.lag_3,
            self.mean_4, self.std_4, self.trend,
            float(self.chronic), float(self.weeks),
        ]


def _get(row: Any, key: str) -> Any:
    """Read a field from a sqlite3.Row or a plain dict."""
    try:
        return row[key]
    except (IndexError, KeyError, TypeError):
        return row.get(key) if hasattr(row, "get") else None


def weekly_series(
    history: Iterable[Any],
) -> dict[tuple[str, str | None], list[WeeklyPoint]]:
    """Pivot dispense rows into a dense weekly series per medicine.

    `history` is what Store.dispense_history() returns: one row per medicine per
    day, already summed. Weeks are anchored to the first day in the history and
    zero-filled, because a gap in the data means "none dispensed", and a lag
    feature that silently skips a week would be reading the wrong history.
    """
    parsed: list[tuple[date, str, str | None, int]] = []
    anchor: date | None = None

    for row in history:
        raw_day = _get(row, "dispensed_on")
        drug = _get(row, "drug")
        if not raw_day or not drug:
            continue
        try:
            day = date.fromisoformat(str(raw_day)[:10])
        except ValueError:
            continue

        anchor = day if anchor is None or day < anchor else anchor
        parsed.append(
            (
                day,
                str(drug),
                _get(row, "strength"),
                int(_get(row, "quantity") or 0),
            )
        )

    if not parsed or anchor is None:
        return {}

    buckets: dict[tuple[str, str | None], dict[int, int]] = {}
    for day, drug, strength, quantity in parsed:
        key = (drug, strength)
        block = (day - anchor).days // PERIOD_DAYS
        per_week = buckets.setdefault(key, {})
        per_week[block] = per_week.get(block, 0) + quantity

    series: dict[tuple[str, str | None], list[WeeklyPoint]] = {}
    for key, per_week in buckets.items():
        last = max(per_week)
        series[key] = [
            WeeklyPoint(
                week_start=anchor + timedelta(days=PERIOD_DAYS * index),
                quantity=per_week.get(index, 0),
            )
            for index in range(last + 1)
        ]
    return series


def features_for(
    key: tuple[str, str | None],
    points: list[WeeklyPoint],
    *,
    lookback: int = 4,
) -> RefillFeatures | None:
    """Features as of the end of `points`. None if there is too little history."""
    if len(points) < lookback:
        return None

    drug, strength = key
    values = [point.quantity for point in points]
    recent = values[-lookback:]
    mean = sum(recent) / len(recent)
    variance = sum((value - mean) ** 2 for value in recent) / len(recent)

    return RefillFeatures(
        drug=drug,
        strength=strength,
        lag_1=float(values[-1]),
        lag_2=float(values[-2]),
        lag_3=float(values[-3]),
        mean_4=mean,
        std_4=variance**0.5,
        trend=float(values[-1] - values[-3]),
        chronic=1 if is_chronic(drug) else 0,
        weeks=len(points),
    )


def training_rows(
    series: dict[tuple[str, str | None], list[WeeklyPoint]],
    *,
    lookback: int = 4,
) -> tuple[list[list[float]], list[float]]:
    """Every honest (past -> next week) pair the shop's own history contains.

    Built by walking each series forward and only ever using weeks that came
    before the target, so the forecaster is never trained on the future it is
    about to be asked to predict.
    """
    features: list[list[float]] = []
    targets: list[float] = []

    for key, points in series.items():
        for end in range(lookback, len(points)):
            row = features_for(key, points[:end], lookback=lookback)
            if row is None:
                continue
            features.append(row.as_row())
            targets.append(float(points[end].quantity))

    return features, targets


class RefillForecaster:
    """Predicts the quantity of a medicine the shop will dispense next week."""

    def __init__(self) -> None:
        self._model = None
        self._rows = 0
        self.method = "damped-mean"

    def fit(self, features: list[list[float]], targets: list[float]) -> "RefillForecaster":
        if TABPFN_AVAILABLE and len(features) >= MIN_ROWS_FOR_TABPFN:
            import numpy as np

            self._model = TabPFNRegressor(device="cpu")
            self._model.fit(
                np.asarray(features, dtype=float),
                np.asarray(targets, dtype=float),
            )
            self._rows = len(features)
            self.method = f"tabpfn ({self._rows} shop rows)"
        return self

    def forecast_7d(self, row: RefillFeatures) -> float:
        if self._model is not None:
            import numpy as np

            prediction = float(
                self._model.predict(np.asarray([row.as_row()], dtype=float))[0]
            )
        else:
            # Damped on purpose: follow half the recent trend, and never
            # extrapolate past double the recent level. A pharmacist who
            # ordered on a spike would be left holding dead stock, which is a
            # real loss in a shop this size.
            prediction = row.mean_4 + 0.5 * row.trend
            prediction = min(prediction, max(row.mean_4 * 2.0, row.mean_4))

        return max(0.0, prediction)


def plan_reorders(store: Any, *, periods: int = 12, lookback: int = 4) -> list[dict]:
    """What to stock up on, ranked, from the shop's own dispensing history.

    Returns an empty list rather than a guess when there is not enough history.
    Telling a pharmacist to order something on no evidence is worse than
    telling him nothing.
    """
    series = weekly_series(store.dispense_history(days=periods * PERIOD_DAYS))
    if not series:
        return []

    features, targets = training_rows(series, lookback=lookback)
    forecaster = RefillForecaster().fit(features, targets)

    plan: list[dict] = []
    for key, points in series.items():
        row = features_for(key, points, lookback=lookback)
        if row is None:
            continue

        forecast = forecaster.forecast_7d(row)
        plan.append(
            {
                "drug": row.drug,
                "strength": row.strength,
                "weekly_mean": round(row.mean_4, 1),
                "forecast_7d": round(forecast, 1),
                "trend": round(row.trend, 1),
                "chronic": bool(row.chronic),
                "weeks_of_history": row.weeks,
                # Running out of a chronic medicine is worse than running out
                # of a seasonal one: the patient is already on it.
                "urgency": round(forecast * (1.6 if row.chronic else 1.0), 1),
                "method": forecaster.method,
            }
        )

    plan.sort(key=lambda entry: -entry["urgency"])
    return plan
