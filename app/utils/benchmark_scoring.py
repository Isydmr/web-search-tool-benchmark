from __future__ import annotations

from typing import Any, Mapping


TARGETED_REPAIR_WEIGHT = 0.4
SURFACE_ALIGNMENT_WEIGHT = 0.6


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def clamp_01(value: float) -> float:
    return max(0.0, min(1.0, value))


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def normalize_percentage(value: Any) -> float:
    numeric = to_float(value)
    if numeric > 1.0:
        numeric /= 100.0
    return clamp_01(numeric)


def normalize_geval_overall(value: Any) -> float | None:
    if value in (None, ""):
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if 1.0 <= numeric <= 5.0:
        return round(clamp_01((numeric - 1.0) / 4.0), 4)
    return round(clamp_01(numeric), 4)


def extract_geval_overall_score(result: Mapping[str, Any]) -> float | None:
    geval = result.get("geval_scores") or {}
    if not isinstance(geval, Mapping):
        return None

    for key in ("normalized_overall", "overall_normalized"):
        if geval.get(key) not in (None, ""):
            return round(clamp_01(to_float(geval.get(key))), 4)

    normalized = normalize_geval_overall(geval.get("overall"))
    if normalized is not None:
        return normalized
    return None


def compute_surface_alignment_score(result: Mapping[str, Any]) -> float:
    lexical = result.get("lexical_distance") or {}
    nli = result.get("nli_scores") or {}

    weighted_parts: list[tuple[float, float]] = []

    judge_score = extract_geval_overall_score(result)
    if judge_score is not None:
        weighted_parts.append((judge_score, 0.55))

    weighted_parts.append((clamp_01(to_float(result.get("semantic_similarity"))), 0.15))
    weighted_parts.append((clamp_01(to_float(lexical.get("norm_edit_similarity"))), 0.10))

    if isinstance(nli, Mapping) and nli:
        weighted_parts.append((normalize_percentage(nli.get("entailment")), 0.10))
        weighted_parts.append((1.0 - normalize_percentage(nli.get("contradiction")), 0.10))

    if not weighted_parts:
        return 0.0

    total_weight = sum(weight for _, weight in weighted_parts)
    if total_weight <= 0:
        return 0.0

    score = sum(value * weight for value, weight in weighted_parts) / total_weight
    return round(clamp_01(score), 4)


def compute_overall_score(result: Mapping[str, Any]) -> float:
    surface_alignment = clamp_01(to_float(result.get("surface_alignment_score")))
    targeted_score = (result.get("targeted_correction") or {}).get("score")
    if targeted_score is None:
        return surface_alignment

    total_weight = SURFACE_ALIGNMENT_WEIGHT + TARGETED_REPAIR_WEIGHT
    blended_score = (
        surface_alignment * SURFACE_ALIGNMENT_WEIGHT
        + clamp_01(to_float(targeted_score)) * TARGETED_REPAIR_WEIGHT
    ) / max(total_weight, 1e-9)
    return round(clamp_01(blended_score), 4)
