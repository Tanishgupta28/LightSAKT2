"""Controlled untrained compute benchmark; predictive performance is measured separately."""
import argparse
import io
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.attention.linear import LinearAttention
from src.models.light_sakt import LightSAKT
from src.models.sakt import SAKT
from src.models.factory import build_model
from src.training.reproducibility import seed_everything, environment
from scripts.analyze import save_figure
import matplotlib.pyplot as plt


def measure(model,forward,backward,device,repeats):
    def sync():
        if device=='cuda':torch.cuda.synchronize()
    def iteration():
        if backward:
            model.zero_grad(set_to_none=True)
            out=forward();out.square().mean().backward()
        else:
            with torch.no_grad():forward()
    for _ in range(3):iteration()
    sync()
    baseline_allocated=torch.cuda.memory_allocated() if device=='cuda' else None
    baseline_reserved=torch.cuda.memory_reserved() if device=='cuda' else None
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    timings=[]
    for _ in range(repeats):
        sync();start=time.perf_counter();iteration();sync();timings.append(time.perf_counter()-start)
    return {'latency_ms_mean':float(np.mean(timings)*1000),'latency_ms_std':float(np.std(timings,ddof=1)*1000),
            'timings_ms':[float(t*1000) for t in timings],
            'baseline_allocated_bytes':baseline_allocated,'baseline_reserved_bytes':baseline_reserved,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else None}


def main(repeats=10):
    if repeats<2:raise ValueError('At least two timed repeats are required to report variability')
    seed_everything(42);device='cuda' if torch.cuda.is_available() else 'cpu';rows=[]
    Path('experiments/efficiency').mkdir(parents=True,exist_ok=True)
    Path('experiments/efficiency/environment.json').write_text(json.dumps(environment(),indent=2))
    for length in [50,100,200,500,1000]:
        for batch in [1,8]:
            for scope in ['equal_attention','equal_complete','documented_complete']:
                for model_name in ['LightSAKT','SAKT']:
                    if device=='cuda':torch.cuda.empty_cache()
                    config_path=None;dropout=0 if scope=='equal_attention' else .1
                    width=64;heads=2
                    if scope=='documented_complete':
                        jobs=json.load(open('experiments/manifest.json'))['jobs']
                        primary=next(j for j in jobs if j['phase']==5 and j['run_id'].startswith('ASSIST2009-'+model_name+'-primary'))
                        config_path=primary['config'];mc=json.load(open(config_path))['model_config']
                        width=mc['embed_dim'] if model_name=='LightSAKT' else mc['d']
                        heads=mc['num_heads'] if model_name=='LightSAKT' else mc['num_attn_heads'];dropout=mc['dropout']
                    if scope=='equal_attention':
                        model=(LinearAttention(width,heads,0) if model_name=='LightSAKT' else nn.MultiheadAttention(width,heads,dropout=0,batch_first=True)).to(device)
                        q,k,v=[torch.randn(batch,length,width,device=device) for _ in range(3)]
                        mask=torch.ones(length,length,device=device,dtype=torch.bool).triu(1) if model_name=='SAKT' else None
                        forward=(lambda:model(q,k,v)[0]) if model_name=='LightSAKT' else (lambda:model(q,k,v,attn_mask=mask,need_weights=True)[0])
                    else:
                        model=(LightSAKT(137,width,heads,dropout,length) if model_name=='LightSAKT' else SAKT(137,length,width,heads,dropout)).to(device)
                        q=torch.randint(0,137,(batch,length),device=device);r=torch.randint(0,2,(batch,length),device=device);qry=torch.randint(0,137,(batch,length),device=device)
                        forward=lambda:model(q,r,qry)[0]
                    stream=io.BytesIO();torch.save(model.state_dict(),stream)
                    for backward in [False,True]:
                        model.train(backward)
                        result=measure(model,forward,backward,device,repeats)
                        result.update(scope=scope,model=model_name,length=length,batch=batch,width=width,heads=heads,dropout=dropout,config_path=config_path,
                                      capacity='positional capacity equals benchmark length; freshly initialized weights',
                                      implementation='dense MHA need_weights=True' if scope=='equal_attention' and model_name=='SAKT' else 'causal ELU+1 tree prefix scan' if model_name=='LightSAKT' else 'training SAKT MHA need_weights=False',
                                      operation='forward_backward' if backward else 'inference_forward',parameters=sum(p.numel() for p in model.parameters()),
                                      serialized_bytes=len(stream.getvalue()),throughput_tokens_per_second=batch*length/(result['latency_ms_mean']/1000))
                        rows.append(result)
                    del model
                    if scope=='equal_attention': del q,k,v
                    else: del q,r,qry
    frame=pd.DataFrame(rows);frame.to_json('experiments/efficiency/raw_timings.json',orient='records',indent=2)
    frame.drop(columns='timings_ms').to_csv('results/tables/efficiency.csv',index=False)
    for (scope,batch,operation),data in frame.groupby(['scope','batch','operation']):
        for metric in ['latency_ms_mean','peak_allocated_bytes','peak_reserved_bytes','throughput_tokens_per_second']:
            name=f'efficiency_{scope}_batch{batch}_{operation}_{metric}'
            data.drop(columns='timings_ms').to_csv(f'results/figures/{name}.csv',index=False)
            plt.figure(figsize=(6,4))
            for model,group in data.groupby('model'):plt.plot(group.length,group[metric],marker='o',label=model)
            plt.xlabel('Sequence length');plt.ylabel(metric.replace('_',' '));plt.title(f'{scope}, batch {batch}, {operation}');plt.legend();save_figure(name)
    parameter_rows=[]
    for job in json.load(open('experiments/manifest.json'))['jobs']:
        if job['phase']!=5:continue
        c=json.load(open(job['config']));s=json.load(open(f"reports/{c['dataset']}_statistics.json"));m=build_model(c['model'],s['num_q'],c['model_config']);buffer=io.BytesIO();torch.save(m.state_dict(),buffer)
        parameter_rows.append({'dataset':c['dataset'],'model':c['model'],'parameters':sum(p.numel() for p in m.parameters()),'serialized_bytes':len(buffer.getvalue()),'config_path':job['config']})
    pd.DataFrame(parameter_rows).to_csv('results/tables/parameters.csv',index=False)
    for dataset,data in pd.DataFrame(parameter_rows).groupby('dataset'):
        data.to_csv(f'results/figures/Figure_8_{dataset}.csv',index=False)
        plt.figure(figsize=(8,4));plt.barh(data.model,data.parameters);plt.xlabel('Trainable parameters');plt.title(dataset+' — documented primary configurations');save_figure('Figure_8_'+dataset)
    epochs=[]
    for path in sorted(Path('results/raw').glob('*/history.json')):
        config=json.load(open(path.parent/'config.json'))
        for epoch in json.load(open(path)):
            epochs.append({'run_id':config['run_id'],'dataset':config['dataset'],'model':config['model'],
                           'group':config['group'],'epoch':epoch['epoch'],'train_and_validation_epoch_seconds':epoch['epoch_seconds']})
    pd.DataFrame(epochs).to_csv('results/tables/epoch_runtime.csv',index=False)
    complexity=[{'model':'LightSAKT','attention_time':'O(B H L d_k d_v)','attention_prefix_storage':'O(B H L d_k d_v) in the vectorized training scan',
                 'additional_operations':'O(B L d^2) projections and reduced-width FFN; O(L d) learned positional parameters'},
                {'model':'SAKT dense reference','attention_time':'O(B H L^2 d_k)','attention_prefix_storage':'O(B H L^2) dense attention weights',
                 'additional_operations':'O(B L d^2) projections and full-width FFN; backend-specific complete-model memory measured separately'}]
    pd.DataFrame(complexity).to_csv('results/tables/model_complexity.csv',index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repeats',type=int,default=10);main(parser.parse_args().repeats)
