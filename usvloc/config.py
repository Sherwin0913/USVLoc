from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable

import yaml


def _parse_value(raw: str):
    return yaml.safe_load(raw)


def _set_by_dotted_key(payload: Dict, dotted_key: str, value) -> None:
    target = payload
    segments = dotted_key.split(".")
    for segment in segments[:-1]:
        if segment not in target or not isinstance(target[segment], dict):
            target[segment] = {}
        target = target[segment]
    target[segments[-1]] = value


def apply_overrides(cfg: Dict, overrides: Iterable[str]) -> Dict:
    cfg = deepcopy(cfg)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be KEY=VALUE, got: {override}")
        key, raw = override.split("=", 1)
        _set_by_dotted_key(cfg, key, _parse_value(raw))
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["__config_path__"] = str(path.resolve())
    cfg["__repo_root__"] = str(path.resolve().parents[1])
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg
