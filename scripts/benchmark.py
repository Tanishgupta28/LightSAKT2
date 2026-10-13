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
        model.zero_grad(set_to_none=True)
        if backward:
            out=forward();out.square().mean().backward()
        else:
            with torch.no_grad():forward()
    for _ in range(3):iteration()
    sync()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    timings=[]
    for _ in range(repeats):
        sync();start=time.perf_counter();iteration();sync();timings.append(time.perf_counter()-start)
    return {'latency_ms_mean':float(np.mean(timings)*1000),'latency_ms_std':float(np.std(timings,ddof=1)*1000),
            'timings_ms':[float(t*1000) for t in timings],
            'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else None}


def main(repeats=10):
    seed_everything(42);device='cuda' if torch.cuda.is_available() else 'cpu';rows=[]
    Path('experiments/efficiency').mkdir(parents=True,exist_ok=True)
    Path('experiments/efficiency/environment.json').write_text(json.dumps(environment(),indent=2))
    for length in [50,100,200,500,1000]:
        for batch in [1,8]:
            for scope in ['equal_attention','equal_complete','documented_complete']:
                for model_name in ['LightSAKT','SAKT']:
                    if device=='cuda':torch.cuda.empty_cache()
                    width=64 if scope!='documented_complete' or model_name=='LightSAKT' else 100
                    heads=2 if width==64 else 5
                    if scope=='equal_attention':
                        model=(LinearAttention(width,heads,0) if model_name=='LightSAKT' else nn.MultiheadAttention(width,heads,dropout=0,batch_first=True)).to(device)
                        q,k,v=[torch.randn(batch,length,width,device=device) for _ in range(3)]
                        mask=torch.ones(length,length,device=device,dtype=torch.bool).triu(1) if model_name=='SAKT' else None
                        forward=(lambda:model(q,k,v)[0]) if model_name=='LightSAKT' else (lambda:model(q,k,v,attn_mask=mask,need_weights=True)[0])
                    else:
                        model=(LightSAKT(137,width,heads,0,length) if model_name=='LightSAKT' else SAKT(137,length,width,heads,0)).to(device)
                        q=torch.randint(0,137,(batch,length),device=device);r=torch.randint(0,2,(batch,length),device=device);qry=torch.randint(0,137,(batch,length),device=device)
                        forward=lambda:model(q,r,qry)[0]
                    stream=io.BytesIO();torch.save(model.state_dict(),stream)
                    for backward in [False,True]:
                        model.train(backward)
                        result=measure(model,forward,backward,device,repeats)
                        result.update(scope=scope,model=model_name,length=length,batch=batch,width=width,heads=heads,
                                      operation='forward_backward' if backward else 'inference_forward',parameters=sum(p.numel() for p in model.parameters()),
                                      serialized_bytes=len(stream.getvalue()),throughput_tokens_per_second=batch*length/(result['latency_ms_mean']/1000))
                        rows.append(result)
                    del model
                    if scope=='equal_attention': del q,k,v
                    else: del q,r,qry
    frame=pd.DataFrame(rows);frame.to_json('experiments/efficiency/raw_timings.json',orient='records',indent=2)
    frame.drop(columns='timings_ms').to_csv('results/tables/efficiency.csv',index=False)
    for metric in ['latency_ms_mean','peak_allocated_bytes','throughput_tokens_per_second']:
        data=frame[(frame.scope=='equal_attention')&(frame.batch==8)&(frame.operation=='inference_forward')]
        plt.figure(figsize=(6,4))
        for model,group in data.groupby('model'):plt.plot(group.length,group[metric],marker='o',label=model)
        plt.xlabel('Sequence length');plt.ylabel(metric.replace('_',' '));plt.title('Equal width 64, 2 heads, batch 8');plt.legend();save_figure('efficiency_'+metric)
    parameter_rows=[]
    for job in json.load(open('experiments/manifest.json'))['jobs']:
        if job['phase']!=5:continue
        c=json.load(open(job['config']));s=json.load(open(f"reports/{c['dataset']}_statistics.json"));m=build_model(c['model'],s['num_q'],c['model_config']);buffer=io.BytesIO();torch.save(m.state_dict(),buffer)
        parameter_rows.append({'dataset':c['dataset'],'model':c['model'],'parameters':sum(p.numel() for p in m.parameters()),'serialized_bytes':len(buffer.getvalue()),'config_path':job['config']})
    pd.DataFrame(parameter_rows).to_csv('results/tables/parameters.csv',index=False)
    for dataset,data in pd.DataFrame(parameter_rows).groupby('dataset'):
        plt.figure(figsize=(8,4));plt.barh(data.model,data.parameters);plt.xlabel('Trainable parameters');plt.title(dataset+' — documented primary configurations');save_figure('Figure_8_'+dataset)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repeats',type=int,default=10);main(parser.parse_args().repeats)
