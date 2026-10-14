import copy

import numpy as np
import pytest

from scripts.light_final_report import METRICS, SEEDS, eligible_result, seed_statistics
from src.training.engine import write_json


def test_report_scope_includes_final_models_and_baseline_primaries_only():
    assert eligible_result({'group': 'final', 'model': 'LightSAKT'})
    assert eligible_result({'group': 'final', 'model': 'DKT'})
    assert eligible_result({'group': 'primary', 'model': 'DKT'})
    assert not eligible_result({'group': 'primary', 'model': 'LightSAKT'})
    assert not eligible_result({'group': 'baseline_tuning', 'model': 'DKT'})
    assert not eligible_result({'group': 'grid', 'model': 'LightSAKT'})
    assert not eligible_result({'group': 'ablation', 'model': 'LightSAKT'})


def test_final_summary_requires_all_seeds_and_sample_standard_deviation():
    records = [{'seed': seed, 'test': {metric: value for metric in METRICS}}
               for seed, value in zip(SEEDS, [.1, .2, .3, .4, .5])]
    summary = seed_statistics(records)
    for metric in METRICS:
        np.testing.assert_allclose(summary[metric + '_mean'], .3)
        np.testing.assert_allclose(summary[metric + '_std'], np.std([.1, .2, .3, .4, .5], ddof=1))
    with pytest.raises(ValueError, match='prespecified seed'):
        seed_statistics(records[:-1])
    duplicate = copy.deepcopy(records); duplicate[-1]['seed'] = SEEDS[0]
    with pytest.raises(ValueError, match='prespecified seed'):
        seed_statistics(duplicate)
    invalid = copy.deepcopy(records); invalid[0]['test']['auc'] = float('nan')
    with pytest.raises(ValueError, match='Undefined'):
        seed_statistics(invalid)


def test_isolated_eta_recalculation_preserves_live_report(tmp_path, monkeypatch):
    from scripts import eta
    monkeypatch.chdir(tmp_path)
    config = {'run_id': 'done', 'dataset': 'ASSIST2009', 'model': 'LightSAKT', 'group': 'primary'}
    write_json('results/raw/done/config.json', config)
    write_json('results/raw/done/history.json', [{'epoch': 1, 'epoch_seconds': 10}])
    write_json('results/raw/done/metrics.json', {'epochs_trained': 1, 'training_seconds': 10})
    write_json('results/raw/done/status.json', {'status': 'completed'})
    write_json('configs/done.json', config)
    write_json('experiments/manifest.json', {'jobs': [{'run_id': 'done', 'phase': 5, 'status': 'completed', 'config': 'configs/done.json'}]})
    write_json('reports/ETA.json', {'live_worker_owned': True})
    before = (tmp_path / 'reports/ETA.json').read_bytes()
    monkeypatch.setattr(eta, 'snapshot', lambda: {'counts': {'completed': 1}, 'remaining': 0})
    monkeypatch.setattr(eta, 'work_counts', lambda: {})
    report = eta.estimate('reports/summary_eta/test')
    assert report['remaining_training_gpu_reserved_hours']['central'] == 0
    assert (tmp_path / 'reports/ETA.json').read_bytes() == before
    assert (tmp_path / 'reports/summary_eta/test/ETA_REPORT.md').exists()
