"""Original-device validation inference for completed LightSAKT grid rows."""
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.sequence_cache import CachedSequences
from scripts.verify_artifacts import verify_stored_result
from scripts.reuse_training import training_code_compatible
from src.models.factory import build_model
from src.training.engine import evaluate,write_json
from src.training.reproducibility import seed_everything,environment
from src.datasets.preprocess import sha256


def main():
    assert torch.cuda.is_available(), 'Original-device checkpoint verification requires CUDA'
    output=Path('reports/light_grid_inference_verification.json')
    previous={r['run_id']:r for r in json.load(open(output))['runs']} if output.exists() else {}
    records=[];datasets={};checked=set();cache={};fingerprint=environment()['training_code_sha256']
    jobs=[j for j in json.load(open('experiments/manifest.json'))['jobs'] if j['phase']==6 and '-LightSAKT-grid-' in j['run_id']]
    assert len(jobs)==108
    for job in jobs:
        path=Path('results/raw')/job['run_id']/'metrics.json'
        result,config,stats=verify_stored_result(path,checked)
        assert training_code_compatible(result,fingerprint)
        assert config['evaluate_test'] is False and result['test'] is None
        identity={'checkpoint_sha256':result['checkpoint_sha256'],'metrics_sha256':sha256(path),
                  'split_sha256':result['split_sha256'],'processed_sha256':result['processed_sha256'],
                  'config_sha256':sha256(path.parent/'config.json'),'training_code_sha256':fingerprint}
        old=previous.get(job['run_id'])
        if old and all(old.get(k)==v for k,v in identity.items()):records.append(old);continue
        key=(result['checkpoint_sha256'],result['split_sha256'],result['processed_sha256'],
             json.dumps(config['model_config'],sort_keys=True),config['seq_len'],config['batch_size'])
        if key not in cache:
            seed_everything(config['seed']);model=build_model('LightSAKT',stats['num_q'],config['model_config']).cuda()
            state=torch.load(result['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True)
            assert state['epoch']==result['best_epoch']
            dataset_key=(config['dataset'],config['seq_len'])
            if dataset_key not in datasets:
                datasets[dataset_key]=CachedSequences(f'data/processed/{config["dataset"]}/interactions.csv','validation',config['seq_len'])
            regenerated,_,_=evaluate(model,'LightSAKT',DataLoader(datasets[dataset_key],batch_size=config['batch_size']), 'cuda')
            cache[key]=regenerated
            del model,state
        regenerated=cache[key]
        for metric in ['auc','accuracy','f1','mse','bce']:
            np.testing.assert_allclose(regenerated[metric],result['validation'][metric],atol=1e-7,rtol=1e-6)
        records.append({'run_id':job['run_id'],'dataset':config['dataset'],'verified':True,'device':'cuda',
                        'validation_recomputed':True,'test_evaluated':False,'regenerated_validation':regenerated,
                        'training_git_commit':result['git_commit'],**identity})
        write_json(output,{'verified':len(records)==108,'verified_utc':datetime.now(timezone.utc).isoformat(),'runs':records,
                           'test_used_for_selection':False,'retrained_runs':0})
    write_json(output,{'verified':True,'verified_utc':datetime.now(timezone.utc).isoformat(),'runs':records,
                       'test_used_for_selection':False,'retrained_runs':0})
    print('Independently verified LightSAKT validation grid checkpoints:',len(records))


if __name__=='__main__':main()
