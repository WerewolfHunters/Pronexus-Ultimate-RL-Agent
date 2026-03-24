from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from detection.scorer import compute_fps


def _bounded_normal(mu: float, sigma: float) -> float:
    return float(np.clip(np.random.normal(mu, sigma), 0.0, 1.0))


def generate_candidate(label: str = "FRAUD", n_questions: int = 5) -> dict:
    questions = []

    for _ in range(n_questions):
        if label == "FRAUD":
            t1 = random.random() < 0.80
            t2 = random.random() < 0.75
            signals = {
                "S1_hedging": _bounded_normal(0.75, 0.12),
                "S2_failure_absence": _bounded_normal(0.85, 0.10),
                "S3_temporal_vagueness": _bounded_normal(0.80, 0.12),
                "S4_tribal_vocab": _bounded_normal(0.78, 0.10),
                "S5_tool_polarity": _bounded_normal(0.70, 0.15),
                "S6_length_uniformity": _bounded_normal(0.72, 0.12),
                "S7_structural_symmetry": _bounded_normal(0.68, 0.15),
                "S8_depth_breadth": _bounded_normal(0.65, 0.15),
            }
        else:
            t1 = random.random() < 0.12
            t2 = random.random() < 0.10
            signals = {
                "S1_hedging": _bounded_normal(0.25, 0.15),
                "S2_failure_absence": _bounded_normal(0.20, 0.12),
                "S3_temporal_vagueness": _bounded_normal(0.22, 0.12),
                "S4_tribal_vocab": _bounded_normal(0.18, 0.12),
                "S5_tool_polarity": _bounded_normal(0.30, 0.15),
                "S6_length_uniformity": _bounded_normal(0.28, 0.15),
                "S7_structural_symmetry": _bounded_normal(0.25, 0.15),
                "S8_depth_breadth": _bounded_normal(0.30, 0.15),
            }

        tripwire = "BOTH" if (t1 and t2) else "ONE" if (t1 or t2) else "NONE"
        fps_result = compute_fps(signals, tripwire)

        questions.append(
            {
                "t1_fired": t1,
                "t2_fired": t2,
                "signals": signals,
                "tripwire": tripwire,
                "fps": fps_result["fps"],
            }
        )

    return {
        "label": label,
        "candidate_id": f"{label[:1]}-{random.randint(10000, 99999)}",
        "questions": questions,
    }


def generate_dataset(n_fraud: int = 500, n_genuine: int = 500, n_questions: int = 5) -> list[dict]:
    data = [generate_candidate("FRAUD", n_questions=n_questions) for _ in range(n_fraud)]
    data.extend(generate_candidate("GENUINE", n_questions=n_questions) for _ in range(n_genuine))
    random.shuffle(data)
    return data


def save_dataset(path: str = "data/sample_candidates.json", n_fraud: int = 500, n_genuine: int = 500) -> Path:
    dataset = generate_dataset(n_fraud=n_fraud, n_genuine=n_genuine)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    return out_path
