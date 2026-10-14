"""CPU-only intermediate paper artifacts; final analyses retain their dependencies."""
import argparse
import io
import json
import os
import subprocess
import time
from pathlib import Path
import pandas as pd
import torch
import matplotlib.pyplot as plt
from src.models.factory import build_model
from src.training.reproducibility import seed_everything
from src.training.engine import write_json
from scripts.analyze import save_figure


def update():
    assert not torch.cuda.is_available(),'Progressive artifacts must run CPU-only'
    record_path=Path('reports/progressive_artifacts.json')
    prior=json.load(open(record_path)) if record_path.exists() else {'visualized_runs':[]}
    seed_everything(42)
    if not prior.get('primary_parameters'):
        rows=[]
        for job in json.load(open('experiments/manifest.json'))['jobs']:
            if job['phase']!=5:continue
            c=json.load(open(job['config']));s=json.load(open(f"reports/{c['dataset']}_statistics.json"))
            model=build_model(c['model'],s['num_q'],c['model_config']);buffer=io.BytesIO();torch.save(model.state_dict(),buffer)
            rows.append({'dataset':c['dataset'],'model':c['model'],'parameters':sum(p.numel() for p in model.parameters()),
                         'serialized_bytes':len(buffer.getvalue()),'config_path':job['config']})
        frame=pd.DataFrame(rows);frame.to_csv('results/tables/parameters.csv',index=False)
        for dataset,data in frame.groupby('dataset'):
            data.to_csv(f'results/figures/Figure_8_{dataset}.csv',index=False)
            plt.figure(figsize=(8,4));plt.barh(data.model,data.parameters);plt.xlabel('Trainable parameters')
            plt.title(dataset+' — documented primary configurations');save_figure('Figure_8_'+dataset)
        prior['primary_parameters']=True
    from scripts.visualize import visualize
    for path in sorted(Path('results/raw').glob('*/metrics.json')):
        result=json.load(open(path))
        if result['model']!='LightSAKT' or result['group'] not in ['primary','final'] or result['seed']!=42:continue
        run_id=result['run_id']
        if run_id in prior['visualized_runs']:continue
        visualize(run_id);prior['visualized_runs'].append(run_id);write_json(record_path,prior)
    subprocess.run(['python','-m','scripts.analyze'],check=True)
    subprocess.run(['python','-m','scripts.manuscript'],check=True)
    prior.update(pid=os.getpid(),scope='CPU-only genuine intermediate artifacts; final research gate is pending')
    write_json(record_path,prior)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    if args.watch:
        import fcntl
        lock=open('experiments/progressive.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        Path('experiments/progressive.pid').write_text(str(os.getpid())+'\n')
    previous=None
    while True:
        observed=sorted((p.parent.name,p.stat().st_mtime) for p in Path('results/raw').glob('*/metrics.json'))
        if observed!=previous:
            try:update();previous=observed
            except Exception as error:
                print('Intermediate artifact update failed; research training unchanged:',type(error).__name__,str(error),flush=True)
                if not args.watch:raise
        if not args.watch:break
        time.sleep(300)
