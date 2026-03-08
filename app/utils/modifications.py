from __future__ import annotations

from typing import Any


NO_MODIFICATION_TYPE = "no_modification"


def normalize_modification_type(value: Any) -> str:
    return str(value or "").strip().lower()


def is_perturbed_modification(value: Any) -> bool:
    normalized = normalize_modification_type(value)
    return bool(normalized) and normalized != NO_MODIFICATION_TYPE
