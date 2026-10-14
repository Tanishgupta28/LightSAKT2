"""Protect genuine benchmark observations and the manuscript's measured claims."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.efficiency_evidence import manuscript_findings, paired_measurements, verify_saved_measurements


PROJECT = Path(__file__).resolve().parents[1]


def measurements():
    return pd.read_csv(PROJECT / 'results/tables/efficiency.csv')


def test_comparisons_reject_missing_duplicate_and_nonfinite_measurements():
    frame = measurements()
    with pytest.raises(ValueError, match='Unmatched'):
        paired_measurements(frame.iloc[1:])
    with pytest.raises(ValueError, match='Duplicate'):
        paired_measurements(pd.concat([frame, frame.iloc[:1]]))
    invalid = frame.copy()
    invalid.loc[0, 'latency_ms_mean'] = np.inf
    with pytest.raises(ValueError, match='finite and positive'):
        paired_measurements(invalid)


def test_pairing_uses_all_identities_and_preserves_negative_latency_findings():
    frame = measurements()
    pairs = paired_measurements(frame)
    pd.testing.assert_frame_equal(pairs, paired_measurements(frame.sample(frac=1, random_state=2026)))
    assert len(frame) == 120 and len(pairs) == 60
    assert (pairs.latency_ratio_light_over_sakt > 1).all()
    assert (pairs.peak_allocated_ratio_light_over_sakt < 1).sum() == 7
    assert (pairs.loc['documented_complete'].peak_allocated_ratio_light_over_sakt > 1).all()


def test_manuscript_uses_measurements_and_keeps_efficiency_concepts_separate():
    frame = measurements()
    parameters = pd.read_csv(PROJECT / 'results/tables/parameters.csv')
    text = manuscript_findings(frame, parameters)
    assert 'higher mean latency in 60 of 60 matched comparisons' in text
    assert 'lower peak allocated memory in 7 of 60' in text
    for term in ['**Measured runtime:**', '**Measured memory:**', '**Parameter efficiency:**',
                 '**Theoretical scaling and interpretation:**', 'not independent student or training-seed',
                 'baseline allocations', '|ASSIST2009|148673|112201|', '|ASSIST2015|134849|101401|',
                 '|Algebra2005|108449|197501|']:
        assert term in text
    # Reporting must follow observations rather than hardcoding a desired conclusion.
    changed = frame.copy()
    key = changed.iloc[0][['scope', 'length', 'batch', 'operation']]
    matched = changed[(changed['scope'] == key.scope) & (changed.length == key.length)
                      & (changed.batch == key.batch) & (changed.operation == key.operation)]
    changed.loc[matched[matched.model == 'LightSAKT'].index, 'latency_ms_mean'] = matched[matched.model == 'SAKT'].latency_ms_mean.iloc[0] / 2
    assert 'higher mean latency in 59 of 60' in manuscript_findings(changed, parameters)


def test_verification_rederives_every_raw_repeat_and_rejects_edited_table(tmp_path):
    for relative in ['results/tables/efficiency.csv', 'results/tables/parameters.csv',
                     'experiments/efficiency/raw_timings.json', 'experiments/efficiency/environment.json']:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / relative, target)
    result = verify_saved_measurements(tmp_path)
    assert result['verified'] and result['timed_repetitions_verified'] == 1200
    assert result['higher_light_latency'] == 60 and result['lower_light_latency'] == 0
    assert len(result['artifacts']) == 4
    path = tmp_path / 'results/tables/efficiency.csv'
    frame = pd.read_csv(path)
    frame.loc[0, 'latency_ms_mean'] /= 2
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        verify_saved_measurements(tmp_path)
