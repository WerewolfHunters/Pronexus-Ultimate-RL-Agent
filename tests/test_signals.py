from detection.signals import score_all_signals


def test_score_all_signals_returns_expected_shape():
    answers = [
        "I recently worked on Kafka pipelines, generally with Spark and Airflow.",
        "Usually I optimize reliability, though it depends on the workload.",
        "In 2024 we had an outage and I wrote an RCA and rollback plan.",
    ]
    result = score_all_signals(answers)
    assert len(result) == 8
    assert all(0.0 <= v <= 1.0 for v in result.values())
