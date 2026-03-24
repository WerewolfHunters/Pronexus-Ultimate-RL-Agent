from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TripwireResult:
    fired: bool
    value: float
    reason: str


def _word_count(text: str) -> int:
    return len([w for w in text.strip().split() if w])


def check_word_count_trap(answer: str, stated_limit: int, tolerance: int = 2) -> dict:
    """
    T1: Fires when an answer lands too close to the instructed word count.
    """
    word_count = _word_count(answer)
    delta = abs(word_count - stated_limit)
    fired = delta <= tolerance
    return {
        "fired": fired,
        "word_count": word_count,
        "stated_limit": stated_limit,
        "delta": delta,
        "reason": (
            f"Answer length ({word_count}) is within +/-{tolerance} of limit ({stated_limit})."
            if fired
            else f"Answer length ({word_count}) is not near the stated limit ({stated_limit})."
        ),
    }


def check_submission_velocity(
    answer: str,
    submission_time_seconds: float,
    threshold_seconds_per_word: float = 1.5,
) -> dict:
    """
    T2: Fires when submission is implausibly fast for the number of words.
    """
    word_count = max(_word_count(answer), 1)
    seconds_per_word = float(submission_time_seconds) / word_count
    fired = seconds_per_word < threshold_seconds_per_word
    return {
        "fired": fired,
        "word_count": word_count,
        "submission_time_seconds": float(submission_time_seconds),
        "seconds_per_word": seconds_per_word,
        "threshold_seconds_per_word": threshold_seconds_per_word,
        "reason": (
            f"Submission velocity {seconds_per_word:.2f}s/word is faster than threshold {threshold_seconds_per_word:.2f}s/word."
            if fired
            else f"Submission velocity {seconds_per_word:.2f}s/word is within normal range."
        ),
    }


def evaluate_tripwires(t1_results: Iterable[dict], t2_results: Iterable[dict]) -> dict:
    """
    Aggregates per-question tripwire outputs into NONE / ONE / BOTH and multiplier.
    """
    t1_any = any(bool(r.get("fired")) for r in t1_results)
    t2_any = any(bool(r.get("fired")) for r in t2_results)

    if t1_any and t2_any:
        result = "BOTH"
        multiplier = 1.5
    elif t1_any or t2_any:
        result = "ONE"
        multiplier = 1.2
    else:
        result = "NONE"
        multiplier = 1.0

    return {
        "t1_fired": t1_any,
        "t2_fired": t2_any,
        "tripwire_result": result,
        "multiplier": multiplier,
    }
