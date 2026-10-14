"""Explain cross-device AUC sensitivity without relaxing inference checks."""
import json
from datetime import datetime,timezone
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.sequence_cache import CachedSequences
from src.models.factory import build_model
from src.training.engine import evaluate,write_json
from src.training.reproducibility import seed_everything
from src.datasets.preprocess import sha256


def rank_uncertainty_bound(target,probability,maximum_probability_difference):
    negative=np.sort(probability[target==0]);positive=probability[target==1]
    assert len(negative) and len(positive)
    radius=2*maximum_probability_difference
    near=np.searchsorted(negative,positive+radius,side='right')-np.searchsorted(negative,positive-radius,side='left')
    return int(near.sum()),float(near.sum())/(len(positive)*len(negative))


def main():
    run_id='ASSIST2009-LightSAKT-no_position-123';root='results/raw/'+run_id
    result=json.load(open(root+'/metrics.json'));config=json.load(open(root+'/config.json'))
    stats=json.load(open('reports/ASSIST2009_statistics.json'));seed_everything(config['seed'])
    state=torch.load(result['checkpoint'],map_location='cpu',weights_only=False)
    model=build_model('LightSAKT',stats['num_q'],config['model_config']);model.load_state_dict(state['model'],strict=True)
    dataset=CachedSequences('data/processed/ASSIST2009/interactions.csv','validation',config['seq_len'])
    loader=DataLoader(dataset,batch_size=config['batch_size'],shuffle=False)
    cpu,y,cpu_p=evaluate(model,'LightSAKT',loader,'cpu')
    gpu,y_gpu,gpu_p=evaluate(model.cuda(),'LightSAKT',loader,'cuda')
    np.testing.assert_array_equal(y,y_gpu)
    np.testing.assert_allclose(cpu_p,gpu_p,atol=1e-6,rtol=1e-6)
    assert gpu==result['validation'], 'Original-device validation inference failed exact metric reproduction'
    delta=float(np.max(np.abs(cpu_p-gpu_p)));pairs,bound=rank_uncertainty_bound(y,gpu_p,delta)
    difference=cpu['auc']-gpu['auc'];assert abs(difference)<=bound+1e-15
    record={'verified':True,'verified_utc':datetime.now(timezone.utc).isoformat(),'run_id':run_id,
            'checkpoint':result['checkpoint'],'checkpoint_sha256':sha256(result['checkpoint']),
            'split_sha256':result['split_sha256'],'training_git_commit':result['git_commit'],
            'cpu_validation_auc':cpu['auc'],'gpu_validation_auc':gpu['auc'],
            'cpu_minus_gpu_auc':difference,'maximum_probability_difference':delta,
            'near_tie_positive_negative_pairs':pairs,'auc_rank_uncertainty_bound':bound,
            'gpu_metrics_exactly_match_original':True,'probability_atol':1e-6,'probability_rtol':1e-6,
            'cpu_metric_tolerance_relaxed':False,'experimental_metrics_changed':False,
            'explanation':'AUC is discontinuous at score ties. Only pairs within twice the measured probability difference can change ordering; the observed AUC difference lies within that measured pair bound.'}
    write_json('reports/ablation_cpu_gpu_ranking_diagnostic.json',record);print(json.dumps(record,indent=2))


if __name__=='__main__':main()
