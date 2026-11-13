"""CPU evaluation only of two already-selected GKT seed42 checkpoints.

Never trains, accesses CUDA, changes selection, or writes historical artifacts.
Checkpoint/dataset bindings are frozen in the published November10 inventory.
Raw prediction arrays stay in the ignored local .tools directory, not Git/ZIP.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import ASSETS,ROOT,sha,write
from collect_verified_sources import metrics
from src.datasets.sequences import Sequences
from src.models.factory import build_model,predict


def main():
    torch.set_num_threads(1)
    source=ROOT.parent/'LightSAKT2'
    selected=json.loads((ROOT/'reports/master_comparison_20261110/selected_reusable_gkt_checkpoints.json').read_text())
    work=ROOT/'.tools/paper_update_assets_20261112/replays';work.mkdir(parents=True,exist_ok=True)
    destination=ASSETS/'verification/checkpoint_verification/new_GKT_CPU_replays.json'
    existing=json.loads(destination.read_text()) if destination.exists() else []
    for r in selected:
        if any(x['checkpoint_sha256']==r['checkpoint_sha256'] for x in existing):
            for x in existing:
                for a in x['prediction_artifacts'].values():assert sha(a['path'])==a['sha256']
            continue
        cfg=r['config'];checkpoint=Path(r['checkpoint']);assert sha(checkpoint)==r['checkpoint_sha256']
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['config']==cfg
        data=source/'data/processed/ASSIST2009/interactions.csv'
        assert sha(data)==state['processed_sha256'] and sha(source/'data/splits/ASSIST2009.json')==state['split_sha256']
        stats=json.loads((source/'reports/ASSIST2009_statistics.json').read_text())
        model=build_model(cfg['model'],stats['num_q'],cfg['model_config']).cpu();model.load_state_dict(state['model'],strict=True);model.eval()
        raw=Path(r['source']).parent;recorded=json.loads((raw/'metrics.json').read_text())
        entry={**recorded,'checkpoint':str(checkpoint),'config':cfg,'config_file':str(raw/'config.json'),
          'config_sha256':sha(raw/'config.json'),'recorded_metrics_path':str(raw/'metrics.json'),
          'recorded_metrics_sha256':sha(raw/'metrics.json'),'model':cfg['model'],
          'cpu_replay':{},'prediction_artifacts':{},'inference_device':'CPU','new_training':False,
          'selection_frozen_before_test':True,'selection_inventory_commit':'db8e2a5bc57ba9aa228fbf895ce9d972aba2b6ac',
          'selected_git_commit':recorded['git_commit']}
        start=time.monotonic()
        for split in ['validation','test']:
            loader=DataLoader(Sequences(data,split,cfg['seq_len']),batch_size=8,shuffle=False,num_workers=0)
            ys,ps=[],[]
            with torch.inference_mode():
                for i,batch in enumerate(loader):
                    if time.monotonic()-start>900:raise RuntimeError('CPU replay900second window exceeded; no completed result published')
                    probabilities,_=predict(model,cfg['model'],batch);valid=batch[-1]
                    ys.append(batch[3][valid].numpy());ps.append(probabilities[valid].numpy())
                    if i%50==0:print(cfg['model'],split,i,'batches',round(time.monotonic()-start,1),'s',flush=True)
            y,p=np.concatenate(ys),np.concatenate(ps);m=metrics(y,p)
            if split=='validation':assert abs(m['auc']-r['best_validation_auc'])<2e-6
            dest=work/f'{cfg["model"]}-42-{split}.npz';np.savez_compressed(dest,target=y,probability=p)
            entry['cpu_replay'][split]=m;entry['prediction_artifacts'][split]={'path':str(dest),'sha256':sha(dest)}
        entry['test']=entry['cpu_replay']['test'];entry['new_CPU_evaluation_seconds']=time.monotonic()-start
        existing.append(entry);write(destination,existing)
        print('VERIFIED',cfg['model'],'test_AUC',entry['test']['auc'],flush=True)


if __name__=='__main__':main()
