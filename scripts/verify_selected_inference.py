"""Independently reproduce completed selected-final inference on CPU only."""
import argparse
import fcntl
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


def store_record(records, record):
    """Commit one proof without dropping other completed proofs on a failure."""
    for index, old in enumerate(records):
        if old['run_id']==record['run_id']:
            records[index]=record;return
    records.append(record)


def verify(models, groups=('final',), device='cpu'):
    Path('experiments').mkdir(exist_ok=True)
    with open('experiments/inference_verification.lock','a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX)
        return _verify(models,groups,device)


def _verify(models, groups, device):
    if device=='cpu':
        assert not torch.cuda.is_available(), 'Run with CUDA_VISIBLE_DEVICES empty; active GPU training must remain untouched'
    else:assert device=='cuda' and torch.cuda.is_available()
    output = Path('reports/selected_final_inference_verification.json')
    previous = json.load(open(output)) if output.exists() else {'runs': []}
    old = {r['run_id']: r for r in previous['runs']}
    datasets = {}; data_checked = set(); locked = LockedTestEvidence()
    records = list(previous['runs'])
    inference_cache = {}
    fingerprint = environment()['training_code_sha256']
    for path in sorted(Path('results/raw').glob('*/metrics.json')):
        raw = json.load(open(path))
        if raw['group'] not in groups or raw['model'] not in models:
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
        cache_key = (result['checkpoint_sha256'], result['processed_sha256'], result['split_sha256'],
                     json.dumps(config['model_config'],sort_keys=True),config['seq_len'],config['batch_size'],result['prediction_sha256'])
        if prior and prior.get('device')==device and all(prior.get(k) == v for k,v in identity.items()) and prior.get('verified'):
            store_record(records,prior)
            inference_cache[cache_key]=prior
            continue
        if cache_key in inference_cache:
            peer=inference_cache[cache_key]
            for split in ['validation','test']:
                for metric in ['auc','accuracy','f1','mse','bce']:
                    np.testing.assert_allclose(peer['regenerated_metrics'][split][metric],result[split][metric],atol=1e-7,rtol=1e-6)
            record={**peer,**identity,**proof,'run_id':result['run_id'],'group':result['group'],
                    'dataset':result['dataset'],'model':result['model'],'seed':result['seed'],
                    'training_git_commit':result['git_commit'],'checkpoint':result['checkpoint'],
                    'prediction_path':result['prediction_path'],'seconds':0.,'inference_reused_from':peer['run_id']}
            store_record(records,record);continue
        started = time.perf_counter(); seed_everything(config['seed'])
        model = build_model(config['model'], stats['num_q'], config['model_config']).to(device).eval()
        state = torch.load(result['checkpoint'], map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        assert state['epoch'] == result['best_epoch']
        regenerated = {}
        for split in ['validation', 'test']:
            key = (config['dataset'], split, config['seq_len'])
            if key not in datasets:
                datasets[key] = CachedSequences(f'data/processed/{config["dataset"]}/interactions.csv', split, config['seq_len'])
            metrics, target, probability = evaluate(model, config['model'],
                DataLoader(datasets[key], batch_size=config['batch_size'], shuffle=False), device)
            for metric in ['auc','accuracy','f1','mse','bce']:
                np.testing.assert_allclose(metrics[metric], result[split][metric], atol=1e-7, rtol=1e-6,
                                           err_msg=f'{result["run_id"]} {split} {metric}')
            regenerated[split] = metrics
            if split == 'test':
                with np.load(result['prediction_path']) as saved:
                    differences = compare_predictions(target, probability, saved['target'], saved['probability'])
        record = {'run_id': result['run_id'], 'dataset': result['dataset'], 'model': result['model'], 'group':result['group'],
                  'seed': result['seed'], 'verified': True, 'device': device, 'selected_validation_epoch': state['epoch'],
                  'validation_recomputed': True, 'test_probabilities_recomputed': True,
                  'verified_utc': datetime.now(timezone.utc).isoformat(), 'seconds': time.perf_counter()-started,
                  'training_git_commit': result['git_commit'], 'checkpoint': result['checkpoint'],
                  'prediction_path': result['prediction_path'], 'regenerated_metrics': regenerated,
                  'probability_atol': 1e-6, 'probability_rtol': 1e-6,
                  'metric_atol': 1e-7, 'metric_rtol': 1e-6, **identity, **proof, **differences}
        store_record(records,record)
        inference_cache[cache_key]=record
        # Preserve independently verified observations after every model. Source
        # checkpoints, probabilities and historical metrics are never rewritten.
        write_json(output, {'verified': True, 'verified_utc': record['verified_utc'],
                           'scope': 'Completed final/ablation runs; identical aliases reuse verified checkpoint inference; complete research verification remains pending',
                           'requested_models': models,'requested_groups':list(groups), 'runs': records})
        print(f'{result["run_id"]}: {device} validation/test verified in {record["seconds"]:.3f}s', flush=True)
        del model, state
    write_json(output, {'verified': True, 'verified_utc': datetime.now(timezone.utc).isoformat(),
                       'scope': 'Completed final/ablation runs; identical aliases reuse verified checkpoint inference; complete research verification remains pending',
                       'requested_models': models,'requested_groups':list(groups), 'runs': records})
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['LightSAKT','DKT','DKT+','SAKT'])
    parser.add_argument('--groups',nargs='+',default=['final'],choices=['final','ablation'])
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    print('Verified completed inference records:', len(verify(args.models,args.groups,args.device)))
