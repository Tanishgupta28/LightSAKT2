"""Verify the completed 108-case grid and derive validation-only selections."""
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.training.engine import write_json
from scripts.verify_artifacts import verify_stored_result


def generate():
    manifest = json.load(open('experiments/manifest.json'))
    verified, selected, checked = [], [], set()
    inference_path=Path('reports/light_grid_inference_verification.json')
    inferred={r['run_id']:r for r in json.load(open(inference_path))['runs']} if inference_path.exists() else {}
    expected = set(itertools.product([.0005, .001, .005], [32, 64, 128], [2, 4], [.1, .2]))
    for dataset, table in [('Algebra2005', 4), ('ASSIST2009', 5), ('ASSIST2015', 6)]:
        jobs = [j for j in manifest['jobs'] if j['phase'] == 6 and j['run_id'].startswith(dataset + '-LightSAKT-grid-')]
        assert len(jobs) == 36, f'{dataset}: incomplete schedule'
        identities, candidates = [], []
        for job in jobs:
            path = Path('results/raw') / job['run_id']
            assert json.load(open(path / 'status.json'))['status'] == 'completed', f'Grid still pending: {job["run_id"]}'
            result, config, stats = verify_stored_result(path / 'metrics.json', checked)
            mc = config['model_config']
            identities.append((config['learning_rate'], mc['embed_dim'], mc['num_heads'], mc['dropout']))
            assert config['seed'] == 42 and config['evaluate_test'] is False
            candidates.append((-result['validation']['auc'], job['run_id'], config))
            proof=inferred.get(job['run_id'],{})
            inference_checked=bool(proof.get('verified') and proof.get('metrics_sha256')==__import__('src.datasets.preprocess',fromlist=['sha256']).sha256(path/'metrics.json')
                                   and proof.get('checkpoint_sha256')==result['checkpoint_sha256'] and proof.get('split_sha256')==result['split_sha256'])
            verified.append({'run_id': job['run_id'], 'dataset': dataset,
                             'checkpoint_sha256': result['checkpoint_sha256'], 'split_sha256': result['split_sha256'],
                             'validation_auc': result['validation']['auc'],
                             'reused_training_run': result.get('reused_training_run'), 'storage_verified': True,
                             'validation_checkpoint_inference_recomputed':inference_checked})
        assert len(set(identities)) == 36 and set(identities) == expected, f'{dataset}: duplicate or changed grid'
        frame = pd.read_csv(f'results/tables/Table_{table}.csv')
        assert len(frame) == 36 and (frame.status == 'completed').all() and frame.validation_auc.notna().all()
        score, best, config = sorted(candidates, key=lambda x: (x[0], x[1]))[0]
        frozen = Path('experiments/selections') / (dataset + '-LightSAKT.json')
        if frozen.exists():
            record = json.load(open(frozen))
            assert record['selected_run_id'] == best and record['test_access_for_selection'] is False
            assert set(record['all_candidates']) == {j['run_id'] for j in jobs}
        selected.append({'dataset': dataset, 'validation_best_run_id': best, 'validation_auc': -score,
                         'learning_rate': config['learning_rate'], **config['model_config'],
                         'split_sha256': stats['split_sha256'], 'selection_metric': 'validation_auc',
                         'test_used_for_selection': False})
    pd.DataFrame(selected).to_csv('results/tables/LightSAKT_selected_configurations.csv', index=False)
    record = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'verified': True,
              'completed_configurations': len(verified),
              'fresh_training_runs': sum(not r['reused_training_run'] for r in verified),
              'all_36_unique_per_dataset': True, 'seed': 42,
              'selection': 'All 36 prespecified candidates; validation AUC descending, run ID lexical tie-break; no test selection',
              'validation_checkpoint_inference_recomputed': all(r['validation_checkpoint_inference_recomputed'] for r in verified),
              'scope': 'Saved checkpoint, data/split hashes and complete table coverage verified; independent original-device validation proofs are linked when available',
              'reproduce_command': 'CUDA_VISIBLE_DEVICES= python -m scripts.light_grid_results', 'runs': verified}
    write_json('reports/light_grid_verification.json', record)
    return record


if __name__ == '__main__':
    report = generate()
    print('Verified LightSAKT grid configurations:', report['completed_configurations'])
