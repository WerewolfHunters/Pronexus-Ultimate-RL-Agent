from detection.scorer import compute_fps


def test_compute_fps_caps_at_one():
    signal_scores = {
        "S1_hedging": 1.0,
        "S2_failure_absence": 1.0,
        "S3_temporal_vagueness": 1.0,
        "S4_tribal_vocab": 1.0,
        "S5_tool_polarity": 1.0,
        "S6_length_uniformity": 1.0,
        "S7_structural_symmetry": 1.0,
        "S8_depth_breadth": 1.0,
    }
    result = compute_fps(signal_scores, "BOTH")
    assert result["fps"] == 1.0
    assert result["band"] == "FRAUD"
