import pandas as pd

from scripts import eta_history


def test_history_reports_actual_nearest_milestone_and_scoped_concurrency(tmp_path, monkeypatch):
    archived = [{'git_commit': 'archived', 'updated_utc': '2026-01-01T00:00:00Z',
                 'completed_entries': 255, 'remaining_entries': 702,
                 'serial_equivalent_hours': 100., 'occupied_mig_hours': 80.}]
    monkeypatch.setattr(eta_history, 'archived_estimates', lambda: archived)
    remaining = pd.DataFrame([
        {'run_id': 'pam', 'phase': 'baseline_tuning', 'model': 'GKT',
         'dataset': 'toy', 'action': 'training', 'central_hours': 10.},
        {'run_id': 'mha', 'phase': 'baseline_tuning', 'model': 'GKT_MHA',
         'dataset': 'toy', 'action': 'training', 'central_hours': 12.},
        {'run_id': 'other', 'phase': 'repeated_seeds', 'model': 'DKVMN',
         'dataset': 'toy', 'action': 'training', 'central_hours': 2.}])
    eta = {'two_worker_measured_throughput_speedup': 1.5,
           'updated_utc': '2026-01-02T00:00:00Z', 'counts': {'completed': 300},
           'remaining_slots': 657, 'remaining_training_gpu_reserved_hours': {'central': 24.},
           'two_worker_training_wall_hours': {'central': 20., 'low': 10., 'high': 40.},
           'conditional_two_worker_training_finish_utc': {'central': '2026-01-02T20:00:00Z'}}
    document = eta_history.generate(eta, remaining, tmp_path)
    milestones = pd.read_csv(tmp_path / 'ETA_HISTORY_MILESTONES.csv')
    assert milestones.iloc[1].requested_completed_milestone == '254'
    assert milestones.iloc[1].completed_entries == 255
    grouped = pd.read_csv(tmp_path / 'ETA_CATEGORY_OCCUPIED_HOURS.csv').set_index('model')
    assert grouped.loc['GKT', 'occupied_mig_hours'] == 10.
    assert grouped.loc['GKT_MHA', 'occupied_mig_hours'] == 8.
    assert grouped.loc['DKVMN', 'occupied_mig_hours'] == 2.
    assert '18.000/20.000 occupied MIG hours (90.00%' in document
    assert 'not a confidence interval' in document
