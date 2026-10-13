"""Evaluate a frozen validation-selected tuning checkpoint once, reusing seed 42."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.training.engine import write_json,evaluate
from src.training.reproducibility import seed_everything
from src.datasets.preprocess import sha256
from src.datasets.sequences import Sequences
from src.models.factory import build_model


def main(job_id):
    job=next(j for j in json.load(open('experiments/manifest.json'))['jobs'] if j['run_id']==job_id)
    source=job['reuse_selected_run'];directory=Path('results/raw')/job_id
    if (directory/'metrics.json').exists():return
    selection=json.load(open(job['selection_path']))
    assert selection['selected_run_id']==source and selection['selection_metric']=='validation_auc'
    c=json.load(open(Path('results/raw')/source/'config.json'));stats=json.load(open(f"reports/{c['dataset']}_statistics.json"))
    path=Path('data/processed')/c['dataset']/'interactions.csv'
    assert sha256(path)==stats['processed_sha256'] and sha256(f"data/splits/{c['dataset']}.json")==stats['split_sha256']
    seed_everything(42);device='cuda' if torch.cuda.is_available() else 'cpu'
    model=build_model(c['model'],stats['num_q'],c['model_config']).to(device)
    result=json.load(open(Path('results/raw')/source/'metrics.json'))
    assert sha256(result['checkpoint'])==result['checkpoint_sha256']
    state=torch.load(result['checkpoint'],map_location=device,weights_only=False);model.load_state_dict(state['model'])
    loader=DataLoader(Sequences(path,'test',c['seq_len']),batch_size=c['batch_size'])
    result['test'],y,p=evaluate(model,c['model'],loader,device)
    dest=Path('results/predictions')/(job_id+'.npz');np.savez_compressed(dest,target=y,probability=p)
    result.update(run_id=job_id,group='final',reused_training_run=source,selection_path=job['selection_path'],prediction_path=str(dest),prediction_sha256=sha256(dest))
    write_json(directory/'metrics.json',result);write_json(directory/'config.json',c)
    write_json(directory/'prediction_identity.json',loader.dataset.identities)
    write_json(directory/'status.json',{'run_id':job_id,'status':'completed','last_completed_epoch':result['epochs_trained'],'reused_training_run':source})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);main(parser.parse_args().job)
