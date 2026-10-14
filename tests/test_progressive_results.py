"""Completed research tables must follow saved run evidence, not stale ledgers."""
import json
import shutil
from pathlib import Path

import pandas as pd

from scripts.summarize import summarize


def test_grid_table_reads_live_status_without_mutating_scheduler(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    manifest = json.loads((project / 'experiments/manifest.json').read_text())
    job = next(j for j in manifest['jobs'] if j['phase'] == 6 and j['run_id'].startswith('ASSIST2009-LightSAKT-grid-'))
    job = {**job, 'status': 'pending'}
    copied = [job['config'], f"results/raw/{job['run_id']}/metrics.json", f"results/raw/{job['run_id']}/config.json"]
    copied += [f'reports/{d}_statistics.json' for d in ['ASSIST2009', 'ASSIST2015', 'Algebra2005']]
    for name in copied:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / name, target)
    state = tmp_path / f"results/raw/{job['run_id']}/status.json"
    state.write_text(json.dumps({'status': 'completed'}))
    ledger = tmp_path / 'experiments/manifest.json'
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({'jobs': [job]}))
    before = ledger.read_bytes()
    monkeypatch.chdir(tmp_path)
    summarize()
    table = pd.read_csv('results/tables/Table_5.csv')
    assert table.status.tolist() == ['completed']
    assert table.validation_auc.notna().all()
    assert ledger.read_bytes() == before
