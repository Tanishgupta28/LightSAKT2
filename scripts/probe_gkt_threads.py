"""Bounded train-only CPU dispatch probe; never enable an unverified policy."""
import hashlib
import json
from pathlib import Path
import time
from datetime import datetime,timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.sequence_cache import CachedSequences
from src.models.factory import build_model,predict
from src.training.engine import training_loss,write_json
from src.training.reproducibility import seed_everything,environment


def digest(values):
    h=hashlib.sha256()
    for name,value in sorted(values.items()):
        h.update(name.encode())
        if value is None:h.update(b'None')
        else:h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def cases():
    jobs=json.load(open('experiments/manifest.json'))['jobs'];result=[]
    for job in jobs:
        if job['phase']!=5:continue
        config=json.load(open(job['config']))
        if config['model'] in ['GKT','GKT_MHA']:result.append((config['dataset']+'-'+config['model']+'-primary',config))
    for model,width,heads in [('GKT',104,2),('GKT_MHA',64,8)]:
        source=next(c for _,c in result if c['dataset']=='Algebra2005' and c['model']==model)
        result.append(('Algebra2005-'+model+'-wide',{**source,'model_config':{**source['model_config'],
                      'hidden_size':width,'num_attn_heads':heads}}))
    return result


def probe():
    assert torch.cuda.is_available(), 'GPU dispatch timings cannot be inferred from CPU'
    rows=[];root=Path('reports/profiling/gkt_threads');root.mkdir(parents=True,exist_ok=True)
    for name,config in cases():
        path=f'data/processed/{config["dataset"]}/interactions.csv'
        dataset=CachedSequences(path,'train',config['seq_len'])
        generator=torch.Generator().manual_seed(config['seed'])
        cpu=next(iter(DataLoader(dataset,batch_size=config['batch_size'],shuffle=True,generator=generator)))
        batch=[tensor.cuda() for tensor in cpu];stats=json.load(open(f'reports/{config["dataset"]}_statistics.json'))
        reference=None
        for threads in [4,1,2]:
            seed_everything(config['seed']);torch.set_num_threads(threads)
            model=build_model(config['model'],stats['num_q'],config['model_config'])
            initial=digest(model.state_dict());initial_rng=torch.get_rng_state().clone();model=model.cuda().train()
            optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],betas=tuple(config['betas']),
                                       eps=config['epsilon'],weight_decay=config['weight_decay'])
            times=[];trajectory=[]
            for step in range(5):
                torch.cuda.synchronize();started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                p,all_p=predict(model,config['model'],batch);loss=training_loss(model,config['model'],batch,p,all_p)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite engineering loss')
                loss.backward()
                if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):
                    raise FloatingPointError('Nonfinite engineering gradient')
                optimizer.step();torch.cuda.synchronize();elapsed=time.perf_counter()-started
                if step>=2:times.append(elapsed)
                trajectory.append({'probabilities':digest({'p':p}),
                                   'gradients':digest({k:v.grad for k,v in model.named_parameters()}),
                                   'parameters':digest(model.state_dict()),
                                   'cuda_rng':digest({'rng':torch.cuda.get_rng_state()})})
            item={'case':name,'dataset':config['dataset'],'model':config['model'],'cpu_threads':threads,
                  'config':config,'train_only':True,'initial_parameters_sha256':initial,
                  'initial_cpu_rng_sha256':digest({'rng':initial_rng}),'trajectory':trajectory,
                  'timed_steps':times,'mean_step_seconds':float(np.mean(times)),
                  'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            if reference is None:reference=item
            item['exact_initialization_and_rng']=initial==reference['initial_parameters_sha256'] and item['initial_cpu_rng_sha256']==reference['initial_cpu_rng_sha256']
            item['exact_step_probabilities_gradients_updates_rng']=trajectory==reference['trajectory']
            item['speedup_against_four_threads']=reference['mean_step_seconds']/item['mean_step_seconds']
            rows.append(item);write_json(root/'measurements.json',rows)
            print(name,'CPU threads',threads,'exact',item['exact_step_probabilities_gradients_updates_rng'],
                  'speedup',item['speedup_against_four_threads'],flush=True)
            del model,optimizer;torch.cuda.empty_cache()
    candidates=[]
    for threads in [1,2]:
        subset=[r for r in rows if r['cpu_threads']==threads]
        candidates.append({'threads':threads,'all_cases_exact':all(r['exact_initialization_and_rng'] and r['exact_step_probabilities_gradients_updates_rng'] for r in subset),
                           'minimum_measured_speedup':min(r['speedup_against_four_threads'] for r in subset),
                           'maximum_measured_speedup':max(r['speedup_against_four_threads'] for r in subset)})
    result={'status':'completed','updated_utc':datetime.now(timezone.utc).isoformat(),'cases':len(rows),
            'warmup_steps':2,'timed_steps_per_case':3,'candidate_summaries':candidates,
            'enabled_for_research':False,'precision':'FP32; original gradients/Adam/inputs/RNG; no AMP/TF32/compile',
            'scope':'Train-only untrained engineering probes; no research performance metrics or reused training evidence',
            'activation_rule':'No policy enabled from short timings alone; full initialization/resume/validation and sustained throughput proof required',
            'environment':environment()}
    write_json(root/'verification.json',result)
    return result


if __name__=='__main__':probe()
