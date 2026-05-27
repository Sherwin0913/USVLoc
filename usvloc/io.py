from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def save_tsv(path: str | Path, rows: Sequence[Mapping[str, object]]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    rows = list(rows)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
