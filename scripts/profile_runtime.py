"""Train-only engineering probes; never produce research-model metrics."""
import argparse
import cProfile
import json
import os
import pstats
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.datasets.sequences import Sequences
from src.models.factory import build_model,predict
from src.training.engine import training_loss,write_json
from src.training.reproducibility import seed_everything,environment

ROOT=Path('reports/profiling')


def step(model,c,batch,optimizer):
    model.train();optimizer.zero_grad(set_to_none=True)
    p,all_p=predict(model,c['model'],batch)
    loss=training_loss(model,c['model'],batch,p,all_p)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite diagnostic loss')
    loss.backward()
    if any(x.grad is not None and not torch.isfinite(x.grad).all() for x in model.parameters()):
        raise FloatingPointError('Nonfinite diagnostic gradient')
    optimizer.step();n=int(batch[-1].sum());value=float(loss.detach())*n
    return value


def prepare(c):
    seed_everything(c['seed']);stats=json.load(open(f"reports/{c['dataset']}_statistics.json"))
    started=time.perf_counter();ds=Sequences(f"data/processed/{c['dataset']}/interactions.csv",'train',c['seq_len'])
    construction=time.perf_counter()-started
    loader=DataLoader(ds,batch_size=c['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(c['seed']))
    started=time.perf_counter();cpu=next(iter(loader));collation=time.perf_counter()-started
    model=build_model(c['model'],stats['num_q'],c['model_config']).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],betas=tuple(c['betas']),eps=c['epsilon'],weight_decay=c['weight_decay'])
    started=time.perf_counter();batch=[x.cuda() for x in cpu];torch.cuda.synchronize();transfer=time.perf_counter()-started
    return model,optimizer,batch,{'dataset_construction_seconds':construction,'batch_collation_seconds':collation,
                                'host_to_device_seconds':transfer,'training_chunks':len(ds),'num_q':stats['num_q']}


def worker(case_path,destination,start_flag=None,steps=10):
    c=json.load(open(case_path));model,optimizer,batch,meta=prepare(c)
    for _ in range(2):step(model,c,batch,optimizer)
    if start_flag:
        Path(str(destination)+'.ready').write_text('ready')
        while not Path(start_flag).exists():time.sleep(.01)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    times=[]
    for _ in range(steps):
        begin=time.perf_counter();step(model,c,batch,optimizer);torch.cuda.synchronize();times.append(time.perf_counter()-begin)
    wall=time.perf_counter()-started
    result={'config':c,'steps':steps,'wall_seconds':wall,'step_seconds':times,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),**meta}
    state={'parameters':{k:v.detach().cpu() for k,v in model.state_dict().items()},
           'gradients':{k:v.grad.detach().cpu() if v.grad is not None else None for k,v in model.named_parameters()}}
    torch.save(state,str(destination)+'.pt');write_json(destination,result)
    return result


def cases():
    jobs=json.load(open('experiments/manifest.json'))['jobs'];result=[]
    for job in jobs:
        if job['phase']!=5:continue
        c=json.load(open(job['config']));result.append((c['dataset']+'-'+c['model'].replace('+','plus')+'-primary',c))
    for dataset in ['ASSIST2009','ASSIST2015','Algebra2005']:
        for model in ['GKT','GKT_MHA']:
            base=next(c for _,c in result if c['dataset']==dataset and c['model']==model)
            c={**base,'model_config':{**base['model_config'],'hidden_size':104 if model=='GKT' else 64,
                                    'num_attn_heads':2 if model=='GKT' else 8}}
            result.append((dataset+'-'+model+'-wide',c))
    return result


def profile():
    if not torch.cuda.is_available():raise RuntimeError('GPU pilot requires CUDA; no invented GPU timings')
    ROOT.mkdir(parents=True,exist_ok=True);records=[]
    for name,c in cases():
        case=ROOT/(name+'-config.json');write_json(case,c)
        model,optimizer,batch,meta=prepare(c)
        for _ in range(2):step(model,c,batch,optimizer)
        torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(3):
            torch.cuda.synchronize();started=time.perf_counter();step(model,c,batch,optimizer);torch.cuda.synchronize();times.append(time.perf_counter()-started)
        model.eval();validation_times=[]
        for _ in range(3):
            torch.cuda.synchronize();started=time.perf_counter()
            with torch.no_grad():
                p,_=predict(model,c['model'],batch)
                _=batch[3][batch[-1]].cpu().numpy();_=p[batch[-1]].cpu().numpy()
            torch.cuda.synchronize();validation_times.append(time.perf_counter()-started)
        row={'case':name,'dataset':c['dataset'],'model':c['model'],'config':c,'train_step_seconds':times,
             'eval_step_seconds':validation_times,'train_step_mean':float(np.mean(times)),
             'eval_step_mean':float(np.mean(validation_times)),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
             'peak_reserved_bytes':torch.cuda.max_memory_reserved(),**meta}
        if c['dataset']=='Algebra2005' and name.endswith('primary'):
            profiler=cProfile.Profile();profiler.enable();step(model,c,batch,optimizer);profiler.disable()
            with open(ROOT/(name+'-python.txt'),'w') as out:pstats.Stats(profiler,stream=out).sort_stats('cumulative').print_stats(25)
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    step(model,c,batch,optimizer);torch.cuda.synchronize()
                events=prof.key_averages()
                row['profile_cpu_self_microseconds']=float(sum(x.self_cpu_time_total for x in events))
                row['profile_cuda_self_microseconds']=float(sum(getattr(x,'self_cuda_time_total',getattr(x,'self_device_time_total',0)) for x in events))
                with open(ROOT/(name+'-operators.txt'),'w') as out:out.write(events.table(sort_by='self_cpu_time_total',row_limit=25))
            except Exception as error:row['profiler_limitation']=f'{type(error).__name__}: {error}'
        records.append(row);write_json(ROOT/'representative_cases.json',records)
        print('Measured engineering probe',name,'train step',row['train_step_mean'],flush=True)
        del model,optimizer,batch;torch.cuda.empty_cache()
    # Timing results are engineering data, never AUC/accuracy or trained research checkpoints.
    write_json(ROOT/'environment.json',{**environment(),'scope':'Untrained train-split probes at a serial inter-run boundary; excludes heldout labels.'})
    parallel=[]
    for dataset in ['ASSIST2009','ASSIST2015','Algebra2005']:
        name=dataset+'-GKT_MHA-primary';case=ROOT/(name+'-config.json')
        reference=ROOT/(name+'-serial.json')
        subprocess.run(['python','-m','scripts.profile_runtime','--worker',str(case),'--output',str(reference)],check=True)
        record=json.load(open(reference));worst=max(r['peak_reserved_bytes'] for r in records if r['model'].startswith('GKT'))
        capacity=torch.cuda.get_device_properties(0).total_memory
        workers=4 if 4*(worst+1024**3)<.8*capacity else 2 if 2*(worst+1024**3)<.8*capacity else 1
        flag=ROOT/(name+'-start.flag');flag.unlink(missing_ok=True);processes=[];outputs=[]
        for i in range(workers):
            output=ROOT/(name+f'-parallel-{i}.json');Path(str(output)+'.ready').unlink(missing_ok=True);outputs.append(output)
            processes.append(subprocess.Popen(['python','-m','scripts.profile_runtime','--worker',str(case),'--output',str(output),'--start-flag',str(flag)]))
        deadline=time.monotonic()+120
        while not all(Path(str(out)+'.ready').exists() for out in outputs):
            if any(p.poll() is not None for p in processes):raise RuntimeError('Concurrent probe exited before ready')
            if time.monotonic()>deadline:raise TimeoutError('Concurrent probe initialization')
            time.sleep(.05)
        began=time.perf_counter();flag.write_text('start')
        for p in processes:
            if p.wait()!=0:raise RuntimeError('Concurrent probe failed')
        wall=time.perf_counter()-began;states=torch.load(str(reference)+'.pt',map_location='cpu',weights_only=False);exact=True
        peers=[]
        for out in outputs:
            state=torch.load(str(out)+'.pt',map_location='cpu',weights_only=False)
            for group in ['parameters','gradients']:
                for key,value in states[group].items():
                    other=state[group][key]
                    if value is None:exact &= other is None
                    else:exact &= torch.equal(value,other)
            peers.append(json.load(open(out)))
        item={'dataset':dataset,'model':'GKT_MHA','workers':workers,'exact_parameters_and_gradients':bool(exact),
              'serial_step_mean':float(np.mean(record['step_seconds'])),'parallel_step_means':[float(np.mean(p['step_seconds'])) for p in peers],
              'loop_throughput_speedup':workers*record['wall_seconds']/max(p['wall_seconds'] for p in peers),
              'parent_observed_parallel_wall_seconds_including_result_serialization':wall,
              'memory_admission_per_worker_bytes':worst+1024**3}
        parallel.append(item);write_json(ROOT/'concurrency_probe.json',parallel)
        print('Concurrent equivalence probe',item,flush=True)
    write_json(ROOT/'pilot_status.json',{'status':'completed','updated_utc':datetime.now(timezone.utc).isoformat(),
        'case_count':len(records),'concurrent_probes':parallel,'precision':'FP32; TF32 off; AMP off',
        'published_research_results':False,'original_architecture_unchanged':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker');parser.add_argument('--output');parser.add_argument('--start-flag');args=parser.parse_args()
    if args.worker:worker(args.worker,args.output,args.start_flag)
    else:profile()
