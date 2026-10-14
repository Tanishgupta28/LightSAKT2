import numpy as np
import pytest

from scripts.verify_selected_inference import compare_predictions,store_record


def test_independent_inference_checks_every_probability_and_target():
    target = np.array([0,1]); probability = np.array([.2,.8], dtype=np.float32)
    result = compare_predictions(target, probability, target, probability+1e-7)
    assert result['max_absolute_probability_difference'] < 2e-7
    assert result['threshold_classification_disagreements'] == 0
    with pytest.raises(AssertionError):
        compare_predictions(target, probability, target, np.array([.21,.79]))
    with pytest.raises(AssertionError):
        compare_predictions(target, probability, target[::-1], probability)


def test_inference_proof_updates_preserve_other_completed_records():
    records=[{'run_id':'earlier','verified':True},{'run_id':'later','verified':True}]
    store_record(records,{'run_id':'earlier','verified':True,'new_device':'cuda'})
    assert len(records)==2 and records[1]=={'run_id':'later','verified':True}
    store_record(records,{'run_id':'new','verified':True})
    assert len(records)==3 and len({r['run_id'] for r in records})==3
