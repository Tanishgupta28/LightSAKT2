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


def main():
    verified=[]
    for path in sorted(Path('results/raw').glob('*/metrics.json')):
        result=json.load(open(path));config=json.load(open(path.parent/'config.json'))
        assert sha256(result['checkpoint'])==result['checkpoint_sha256']
        stats=json.load(open(f"reports/{result['dataset']}_statistics.json"))
        assert stats['processed_sha256']==result['processed_sha256'] and stats['split_sha256']==result['split_sha256']
        if result.get('prediction_path'):
            assert sha256(result['prediction_path'])==result['prediction_sha256']
            arrays=np.load(result['prediction_path'])
            from src.evaluation.metrics import metrics
            assert metrics(arrays['target'],arrays['probability'])==result['test']
        seed_everything(config['seed']);device='cuda' if torch.cuda.is_available() else 'cpu'
        model=build_model(config['model'],stats['num_q'],config['model_config']).to(device)
        checkpoint=torch.load(result['checkpoint'],map_location=device,weights_only=False)
        model.load_state_dict(checkpoint['model'])
        dataset=Sequences(f"data/processed/{result['dataset']}/interactions.csv",'validation',config['seq_len'])
        regenerated,_,_=evaluate(model,config['model'],DataLoader(dataset,batch_size=config['batch_size']),device)
        for key in ['auc','accuracy','f1','mse','bce']:
            assert np.isclose(regenerated[key],result['validation'][key],atol=1e-7,rtol=1e-6)
        verified.append({'run_id':result['run_id'],'checkpoint':result['checkpoint'],'checksum_verified':True,'validation_recomputed':True,'test_prediction_metrics_recomputed':bool(result.get('prediction_path'))})
    write_json('reports/artifact_verification.json',verified)
    print('Verified',len(verified),'completed runs')

if __name__=='__main__':main()
