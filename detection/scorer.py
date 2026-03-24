from __future__ import annotations

from typing import Dict

WEIGHTS = {
    "S1_hedging": 8,
    "S2_failure_absence": 9,
    "S3_temporal_vagueness": 8,
    "S4_tribal_vocab": 9,
    "S5_tool_polarity": 7,
    "S6_length_uniformity": 7,
    "S7_structural_symmetry": 6,
    "S8_depth_breadth": 6,
}

TRIPWIRE_MULTIPLIER = {
    "BOTH": 1.5,
    "ONE": 1.2,
    "NONE": 1.0,
}


def _band(fps: float) -> str:
    if fps < 0.35:
        return "CLEAN"
    if fps < 0.60:
        return "BORDERLINE"
    if fps < 0.80:
        return "HIGH_SUSPICION"
    return "FRAUD"


def _recommended_action(band: str) -> str:
    return {
        "CLEAN": "PASS",
        "BORDERLINE": "REQUEST_FOLLOW_UP",
        "HIGH_SUSPICION": "REQUEST_FOLLOW_UP_OR_REVIEW",
        "FRAUD": "FLAG",
    }[band]


def compute_fps(signal_scores: Dict[str, float], tripwire_result: str) -> dict:
    """
    FPS = min(weighted_average(signals) * tripwire_multiplier, 1.0)
    """
    weighted_sum = 0.0
    weight_total = float(sum(WEIGHTS.values()))

    contributions: dict[str, float] = {}
    for key, weight in WEIGHTS.items():
        score = float(signal_scores.get(key, 0.0))
        weighted_component = score * weight
        weighted_sum += weighted_component
        contributions[key] = weighted_component / weight_total

    base_score = weighted_sum / weight_total
    multiplier = TRIPWIRE_MULTIPLIER.get(tripwire_result, 1.0)
    fps = min(base_score * multiplier, 1.0)
    band = _band(fps)

    sorted_contrib = dict(
        sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    )

    return {
        "base_score": round(base_score, 4),
        "multiplier": multiplier,
        "fps": round(fps, 4),
        "band": band,
        "recommended_action": _recommended_action(band),
        "contributions": sorted_contrib,
    }
