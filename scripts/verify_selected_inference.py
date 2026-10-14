"""Independently reproduce completed selected-final inference on CPU only."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.locked_test_evidence import LockedTestEvidence
from scripts.sequence_cache import CachedSequences
from scripts.verify_artifacts import verify_stored_result
from src.datasets.preprocess import sha256
from src.models.factory import build_model
from src.training.engine import evaluate, write_json
from src.training.reproducibility import environment, seed_everything


def compare_predictions(target, probability, saved_target, saved_probability):
    np.testing.assert_array_equal(target, saved_target)
    np.testing.assert_allclose(probability, saved_probability, atol=1e-6, rtol=1e-6)
    return {'max_absolute_probability_difference': float(np.abs(probability-saved_probability).max()),
            'threshold_classification_disagreements': int(((probability >= .5) != (saved_probability >= .5)).sum())}


def verify(models):
    assert not torch.cuda.is_available(), 'Run with CUDA_VISIBLE_DEVICES empty; active GPU training must remain untouched'
    output = Path('reports/selected_final_inference_verification.json')
    previous = json.load(open(output)) if output.exists() else {'runs': []}
    old = {r['run_id']: r for r in previous['runs']}
    datasets = {}; data_checked = set(); locked = LockedTestEvidence(); records = []
    fingerprint = environment()['training_code_sha256']
    for path in sorted(Path('results/raw').glob('*/metrics.json')):
        raw = json.load(open(path))
        if raw['group'] != 'final' or raw['model'] not in models:
            continue
        assert json.load(open(path.parent/'status.json'))['status'] == 'completed'
        result, config, stats = verify_stored_result(path, data_checked)
        assert result['training_code_sha256'] == fingerprint, 'Training source changed since the completed run'
        proof = locked.check(result, config)
        identity = {'metrics_sha256': sha256(path), 'checkpoint_sha256': result['checkpoint_sha256'],
                    'prediction_sha256': result['prediction_sha256'], 'split_sha256': result['split_sha256'],
                    'processed_sha256': result['processed_sha256'], 'config_sha256': sha256(path.parent/'config.json'),
                    'training_code_sha256': fingerprint}
        prior = old.get(result['run_id'])
        if prior and all(prior.get(k) == v for k,v in identity.items()) and prior.get('verified'):
            records.append(prior)
            continue
        started = time.perf_counter(); seed_everything(config['seed'])
        model = build_model(config['model'], stats['num_q'], config['model_config']).eval()
        state = torch.load(result['checkpoint'], map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        assert state['epoch'] == result['best_epoch']
        regenerated = {}
        for split in ['validation', 'test']:
            key = (config['dataset'], split, config['seq_len'])
            if key not in datasets:
                datasets[key] = CachedSequences(f'data/processed/{config["dataset"]}/interactions.csv', split, config['seq_len'])
            metrics, target, probability = evaluate(model, config['model'],
                DataLoader(datasets[key], batch_size=config['batch_size'], shuffle=False), 'cpu')
            for metric in ['auc','accuracy','f1','mse','bce']:
                np.testing.assert_allclose(metrics[metric], result[split][metric], atol=1e-7, rtol=1e-6,
                                           err_msg=f'{result["run_id"]} {split} {metric}')
            regenerated[split] = metrics
            if split == 'test':
                with np.load(result['prediction_path']) as saved:
                    differences = compare_predictions(target, probability, saved['target'], saved['probability'])
        record = {'run_id': result['run_id'], 'dataset': result['dataset'], 'model': result['model'],
                  'seed': result['seed'], 'verified': True, 'device': 'cpu', 'selected_validation_epoch': state['epoch'],
                  'validation_recomputed': True, 'test_probabilities_recomputed': True,
                  'verified_utc': datetime.now(timezone.utc).isoformat(), 'seconds': time.perf_counter()-started,
                  'training_git_commit': result['git_commit'], 'checkpoint': result['checkpoint'],
                  'prediction_path': result['prediction_path'], 'regenerated_metrics': regenerated,
                  'probability_atol': 1e-6, 'probability_rtol': 1e-6,
                  'metric_atol': 1e-7, 'metric_rtol': 1e-6, **identity, **proof, **differences}
        records.append(record)
        # Preserve independently verified observations after every model. Source
        # checkpoints, probabilities and historical metrics are never rewritten.
        write_json(output, {'verified': True, 'verified_utc': record['verified_utc'],
                           'scope': 'Completed selected final runs for requested models; complete research verification remains pending',
                           'requested_models': models, 'runs': records})
        print(f'{result["run_id"]}: CPU validation/test verified in {record["seconds"]:.3f}s', flush=True)
        del model, state
    write_json(output, {'verified': True, 'verified_utc': datetime.now(timezone.utc).isoformat(),
                       'scope': 'Completed selected final runs for requested models; complete research verification remains pending',
                       'requested_models': models, 'runs': records})
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['LightSAKT','DKT','DKT+','SAKT'])
    print('Verified selected final runs:', len(verify(parser.parse_args().models)))
