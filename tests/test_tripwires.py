from detection.tripwires import check_submission_velocity, check_word_count_trap, evaluate_tripwires


def test_word_count_tripwire_fires_near_limit():
    result = check_word_count_trap("word " * 100, 100)
    assert result["fired"] is True


def test_submission_velocity_fires_for_fast_input():
    result = check_submission_velocity("word " * 100, submission_time_seconds=60)
    assert result["fired"] is True


def test_tripwire_aggregation():
    result = evaluate_tripwires([{"fired": True}], [{"fired": False}])
    assert result["tripwire_result"] == "ONE"
    assert result["multiplier"] == 1.2
