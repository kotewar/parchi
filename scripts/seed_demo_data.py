"""Put demo dispensing history into the shop's store, so the stock plan has
something to forecast from.

    python3 scripts/seed_demo_data.py
    python3 scripts/seed_demo_data.py --force   # add another 14 weeks anyway

Real history would come from the pharmacist's own dispensing. This exists so a
fresh clone shows a working stock plan rather than an empty table.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.demo_data import seed_dispense_history  # noqa: E402
from app.store import Store  # noqa: E402


def main() -> None:
    settings = get_settings()
    store = Store(settings.database_path)

    existing = store.stats()["dispenses"]
    if existing and "--force" not in sys.argv:
        print(
            f"{existing} dispense rows already present in {settings.database_path}; "
            "nothing to do (pass --force to seed another 14 weeks)"
        )
        store.close()
        return

    written = seed_dispense_history(store)
    print(f"seeded {written} dispense rows into {settings.database_path}")
    print(f"store now: {store.stats()}")
    store.close()


if __name__ == "__main__":
    main()
