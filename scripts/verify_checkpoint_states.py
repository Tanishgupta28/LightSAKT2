"""Load completed and live checkpoint bytes consistently on CPU without inference."""
import hashlib
import io
import json
from datetime import datetime,timezone
from pathlib import Path
import random
import numpy as np
import torch
from src.models.factory import build_model
from src.training.engine import write_json
from scripts.progress import snapshot


def verify(path,config,stats):
    payload=Path(path).read_bytes();state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
    model=build_model(config['model'],stats['num_q'],config['model_config']);model.load_state_dict(state['model'],strict=True)
    assert all(torch.isfinite(value).all() for value in state['model'].values() if value.is_floating_point())
    assert state['processed_sha256']==stats['processed_sha256'] and state['split_sha256']==stats['split_sha256']
    latest='optimizer' in state
    if latest:
        optimizer=torch.optim.Adam(model.parameters());optimizer.load_state_dict(state['optimizer'])
        assert all(torch.isfinite(v).all() for item in state['optimizer']['state'].values() for v in item.values() if isinstance(v,torch.Tensor) and v.is_floating_point())
        random.Random().setstate(state['rng']['python']);np.random.RandomState().set_state(state['rng']['numpy'])
        torch.Generator().set_state(state['rng']['torch']);torch.Generator().set_state(state['loader_rng'])
        assert all(v.dtype==torch.uint8 and v.numel()>0 for v in state['rng']['cuda'])
    return {'path':str(path),'sha256':hashlib.sha256(payload).hexdigest(),'epoch':state['epoch'],
            'strict_state_load':True,'finite_parameters':True,'split_data_match':True,'optimizer_rng_verified':latest}


def main():
    assert not torch.cuda.is_available(),'Run with CUDA_VISIBLE_DEVICES empty to protect live GPU work'
    completed=[];seen=set()
    for path in sorted(Path('results/raw').glob('*/metrics.json')):
        result=json.load(open(path));config=json.load(open(path.parent/'config.json'))
        directory=Path(result['checkpoint']).parent
        for name in ['best.pt','latest.pt']:
            p=directory/name
            if str(p) in seen:continue
            seen.add(str(p));stats=json.load(open(f"reports/{config['dataset']}_statistics.json"))
            completed.append(verify(p,config,stats))
    write_json('reports/completed_checkpoint_loads.json',{'verified_utc':datetime.now(timezone.utc).isoformat(),'verified':True,'checkpoints':completed})
    active=[]
    for job in snapshot()['running_jobs']:
        root=Path('results/raw')/job['run_id']
        if not (root/'config.json').exists():continue
        config=json.load(open(root/'config.json'));stats=json.load(open(f"reports/{config['dataset']}_statistics.json"))
        for name in ['best.pt','latest.pt']:
            p=Path('checkpoints')/job['run_id']/name
            if p.exists():active.append(verify(p,config,stats))
    write_json('reports/active_checkpoint_verification.json',active)
    print('Strict CPU checkpoint loads:',len(completed),'completed states and',len(active),'active states')


if __name__=='__main__':main()
