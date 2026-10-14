import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.datasets.preprocess import sha256
from src.datasets.sequences import Sequences
from src.models.factory import build_model
from src.training.engine import evaluate,write_json
from src.training.reproducibility import seed_everything


def verify_stored_result(path, data_verified=None):
    """Verify immutable saved evidence without allocating GPU memory."""
    path=Path(path);result=json.load(open(path));config=json.load(open(path.parent/'config.json'))
    assert sha256(result['checkpoint'])==result['checkpoint_sha256'], f'Checkpoint checksum: {path}'
    stats=json.load(open(f"reports/{result['dataset']}_statistics.json"))
    assert stats['processed_sha256']==result['processed_sha256'] and stats['split_sha256']==result['split_sha256']
    identity=(result['dataset'],result['processed_sha256'],result['split_sha256'])
    if data_verified is None or identity not in data_verified:
        assert sha256(f"data/processed/{result['dataset']}/interactions.csv")==result['processed_sha256']
        assert sha256(f"data/splits/{result['dataset']}.json")==result['split_sha256']
        if data_verified is not None:data_verified.add(identity)
    if result.get('prediction_path'):
        assert sha256(result['prediction_path'])==result['prediction_sha256']
        with np.load(result['prediction_path']) as arrays:
            from src.evaluation.metrics import metrics
            assert metrics(arrays['target'],arrays['probability'])==result['test']
    return result,config,stats


def main(stored_only=False):
    verified=[];data_verified=set();inference_cache={}
    for path in sorted(Path('results/raw').glob('*/metrics.json')):
        result,config,stats=verify_stored_result(path,data_verified)
        if stored_only:
            verified.append({'run_id':result['run_id'],'checkpoint':result['checkpoint'],'checksum_verified':True,
                             'validation_recomputed':False,'test_prediction_metrics_recomputed':bool(result.get('prediction_path'))})
            continue
        device='cuda' if torch.cuda.is_available() else 'cpu'
        identity=(result['checkpoint_sha256'],result['processed_sha256'],result['split_sha256'],
                  config['model'],stats['num_q'],json.dumps(config['model_config'],sort_keys=True),
                  config['seq_len'],config['batch_size'],device)
        reused_inference=identity in inference_cache
        if not reused_inference:
            seed_everything(config['seed'])
            model=build_model(config['model'],stats['num_q'],config['model_config']).to(device)
            checkpoint=torch.load(result['checkpoint'],map_location=device,weights_only=False)
            model.load_state_dict(checkpoint['model'])
            data_path=f"data/processed/{result['dataset']}/interactions.csv"
            from scripts.sequence_cache import CachedSequences,location
            cache=location(data_path,'validation',config['seq_len'])
            dataset=(CachedSequences(data_path,'validation',config['seq_len']) if (cache/'manifest.json').exists()
                     else Sequences(data_path,'validation',config['seq_len']))
            regenerated,_,_=evaluate(model,config['model'],DataLoader(dataset,batch_size=config['batch_size']),device)
            inference_cache[identity]=regenerated
            del model,checkpoint,dataset
        regenerated=inference_cache[identity]
        for key in ['auc','accuracy','f1','mse','bce']:
            assert np.isclose(regenerated[key],result['validation'][key],atol=1e-7,rtol=1e-6)
        verified.append({'run_id':result['run_id'],'checkpoint':result['checkpoint'],'checksum_verified':True,'validation_recomputed':True,'reused_identical_checkpoint_inference':reused_inference,'test_prediction_metrics_recomputed':bool(result.get('prediction_path'))})
    write_json('reports/stored_artifact_verification.json' if stored_only else 'reports/artifact_verification.json',verified)
    print('Verified',len(verified),'completed runs', '(stored evidence; no GPU inference)' if stored_only else '(checkpoint inference and stored evidence)')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stored-only',action='store_true')
    main(parser.parse_args().stored_only)
