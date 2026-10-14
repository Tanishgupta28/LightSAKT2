import numpy as np
import pytest

from scripts.verify_selected_inference import compare_predictions


def test_independent_inference_checks_every_probability_and_target():
    target = np.array([0,1]); probability = np.array([.2,.8], dtype=np.float32)
    result = compare_predictions(target, probability, target, probability+1e-7)
    assert result['max_absolute_probability_difference'] < 2e-7
    assert result['threshold_classification_disagreements'] == 0
    with pytest.raises(AssertionError):
        compare_predictions(target, probability, target, np.array([.21,.79]))
    with pytest.raises(AssertionError):
        compare_predictions(target, probability, target[::-1], probability)
