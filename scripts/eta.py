"""Measured runtime strata with explicit extrapolation and planning scenarios."""
import argparse
import json
import math
import fcntl
import os
import statistics
import time
from collections import defaultdict
from datetime import datetime,timedelta,timezone
from pathlib import Path
import pandas as pd
from src.training.engine import write_json
from scripts.progress import snapshot
from scripts.make_schedule import model_defaults


DATASETS=['ASSIST2009','ASSIST2015','Algebra2005']
MODELS=['DKT','DKT+','DKVMN','GKT','GKT_MHA','SAKT','LightSAKT']


def work_counts():
    cache_path=Path('reports/eta_data_work.json')
    stats={d:json.load(open(f'reports/{d}_statistics.json')) for d in DATASETS}
    cache=json.load(open(cache_path)) if cache_path.exists() else {}
    for d,s in stats.items():
        if d in cache and cache[d]['processed_sha256']==s['processed_sha256'] and cache[d]['split_sha256']==s['split_sha256']:continue
        frame=pd.read_csv(f'data/processed/{d}/interactions.csv',usecols=['student','split'],dtype={'student':str})
        length=200 if d=='Algebra2005' else 100
        values={'num_q':s['num_q'],'length':length,'processed_sha256':s['processed_sha256'],'split_sha256':s['split_sha256']}
        for split in ['train','validation','test']:
            sizes=frame[frame.split==split].groupby('student').size()
            values[split+'_chunks']=int(((sizes-1+length-1)//length).sum())
        cache[d]=values
    write_json(cache_path,cache);return cache


def features(c,work):
    w=work[c['dataset']];mc=c['model_config'];length=c['seq_len'];batch=c['batch_size'];n=w['num_q']
    calls=length*sum(math.ceil(w[s+'_chunks']/batch) for s in ['train','validation'])
    name=c['model']
    if name in ['DKT','DKT+']:
        e,h=mc['emb_size'],mc['hidden_size'];arithmetic=4*h*(e+h)+h*n
    elif name=='DKVMN':
        d,m=mc['dim_s'],mc['size_m'];arithmetic=4*d*d+3*m*d
    elif name.startswith('GKT'):
        h=mc['hidden_size'];arithmetic=n*h*h+(n*n if name=='GKT' else n*mc['num_attn_heads'])
    elif name=='SAKT':
        d=mc['d'];arithmetic=6*d*d+2*length*d
    else:
        d=mc['embed_dim'];heads=1 if mc.get('variant')=='single_head' else mc['num_heads']
        arithmetic=d*d if mc.get('variant')=='no_attention' else 5*d*d+3*d*d/heads
    return calls,calls*batch*arithmetic


def estimate():
    progress=snapshot();work=work_counts();manifest=json.load(open('experiments/manifest.json'))
    known=[];epochs=defaultdict(list);observed_rows=[];observed_groups=defaultdict(list)
    for path in sorted(Path('results/raw').glob('*/history.json')):
        directory=path.parent;c=json.load(open(directory/'config.json'));h=json.load(open(path))
        if not h or c['group']=='smoke':continue
        result=json.load(open(directory/'metrics.json')) if (directory/'metrics.json').exists() else None
        if result and result.get('reused_training_run'):continue
        times=[r['epoch_seconds'] for r in h]
        observation={'config':c,'epoch_seconds':statistics.mean(times),'history_epochs':len(h),'run_id':c['run_id']}
        if result:
            epochs[c['model']].append(result['epochs_trained'])
            observation['run_seconds']=result['training_seconds']
            observation['overhead_seconds']=max(0.,result['training_seconds']-sum(times))
        else:observation['overhead_seconds']=0.
        known.append(observation);observed_groups[(c['dataset'],c['model'])].append(observation)
    for dataset in DATASETS:
        for model in MODELS:
            group=observed_groups[(dataset,model)];complete=[r for r in group if 'run_seconds' in r]
            observed_rows.append({'dataset':dataset,'model':model,'measured_runs':len(group),'completed_runs':len(complete),
                'measured_epochs':sum(r['history_epochs'] for r in group),
                'average_epoch_seconds':sum(r['epoch_seconds']*r['history_epochs'] for r in group)/sum(r['history_epochs'] for r in group) if group else None,
                'average_completed_run_seconds':statistics.mean(r['run_seconds'] for r in complete) if complete else None})
    pd.DataFrame(observed_rows).to_csv('reports/ETA_MEASURED_TIMINGS.csv',index=False)
    configs={j['run_id']:json.load(open(j['config'])) for j in manifest['jobs'] if 'config' in j}
    overlay=json.load(open('experiments/reuse_plan.json'))['jobs'] if Path('experiments/reuse_plan.json').exists() else {}
    def epoch_profile(c):
        same=[r for r in known if r['config']['model']==c['model']]
        if not same:raise ValueError('No measured reference for '+c['model'])
        direct=[r for r in same if r['config']['dataset']==c['dataset']]
        target=features(c,work)
        ref=min(direct or same,key=lambda r:abs(math.log(target[0]/features(r['config'],work)[0])))
        reference=features(ref['config'],work);ratios=[target[i]/reference[i] for i in [0,1]]
        lo,hi=min(ratios),max(ratios)
        return {'low':ref['epoch_seconds']*lo,'central':ref['epoch_seconds']*math.sqrt(lo*hi),
                'high':ref['epoch_seconds']*hi,'ref':ref['run_id'],'direct_stratum':bool(direct),
                'overhead':ref['overhead_seconds']*math.sqrt(lo*hi)}
    estimates=[];now=datetime.now(timezone.utc)
    for job in manifest['jobs']:
        status_path=Path('results/raw')/job['run_id']/'status.json'
        saved=json.load(open(status_path)) if status_path.exists() else {};state=saved.get('status',job['status'])
        if state=='completed':continue
        c=configs.get(job['run_id']);source=overlay.get(job['run_id'])
        label='primary' if job['phase']==5 else 'LightSAKT_grid' if c and c['model']=='LightSAKT' and job['phase']==6 else 'baseline_tuning' if job['phase']==6 else 'repeated_seeds' if job['phase']==7 else 'ablations'
        if source or job.get('variant')=='no_regularization' or job['phase']==7 and job['seed']==42:
            estimates.append({'run_id':job['run_id'],'phase':label,'dataset':c['dataset'] if c else job['dataset'],
                'model':c['model'] if c else job['model'],'action':'reuse/evaluation; zero additional training',
                'low_hours':0.,'central_hours':0.,'high_hours':0.});continue
        if c is None:
            candidates=[configs[j['run_id']] for j in manifest['jobs'] if j['phase']==6 and configs[j['run_id']]['dataset']==job['dataset'] and configs[j['run_id']]['model']==job['model']]
            if job.get('variant'):
                candidates=[{**candidate,'model_config':{**candidate['model_config'],'variant':job['variant']}} for candidate in candidates]
            profiles=[epoch_profile(candidate) for candidate in candidates];model=job['model'];dataset=job['dataset']
            profile={'low':min(p['low'] for p in profiles),'central':statistics.mean(p['central'] for p in profiles),
                     'high':max(p['high'] for p in profiles),'ref':';'.join(sorted(set(p['ref'] for p in profiles))),
                     'overhead':statistics.mean(p['overhead'] for p in profiles),'direct_stratum':all(p['direct_stratum'] for p in profiles)}
            max_epochs=50;patience=5
        else:profile=epoch_profile(c);model=c['model'];dataset=c['dataset'];max_epochs=c['max_epochs'];patience=c['patience']
        done=saved.get('last_completed_epoch',0);bad=0
        history_path=status_path.parent/'history.json'
        if history_path.exists():
            h=json.load(open(history_path));best=-math.inf
            for entry in h:
                if entry['validation']['auc']>best:best=entry['validation']['auc'];bad=0
                else:bad+=1
        floor_remaining=patience-bad if done else patience+1
        mean_epochs=statistics.mean(epochs[model]) if epochs[model] else max_epochs
        central_remaining=max(floor_remaining,min(max_epochs,mean_epochs)-done)
        fraction=0.
        if state=='running' and history_path.exists():
            # Epoch writes are atomic. Elapsed work in the current unsaved epoch is approximate.
            last_write=history_path.stat().st_mtime;fraction=min(1.,max(0.,(time.time()-last_write)/profile['central']))
        values={'low':max(0.,min(max_epochs-done,floor_remaining)-fraction)*profile['low'],
                'central':max(0.,central_remaining-fraction)*profile['central']+profile['overhead'],
                'high':max(0.,max_epochs-done-fraction)*profile['high']+profile['overhead']}
        estimates.append({'run_id':job['run_id'],'phase':label,'dataset':dataset,'model':model,'action':'training',
            'last_completed_epoch':done,'central_expected_total_epochs':mean_epochs,'reference_runs':profile['ref'],
            'dataset_model_timing_measured':profile['direct_stratum'],
            **{k+'_hours':v/3600 for k,v in values.items()}})
    frame=pd.DataFrame(estimates) if estimates else pd.DataFrame(columns=[
        'run_id','phase','dataset','model','action','low_hours','central_hours','high_hours'])
    if not estimates:frame[['low_hours','central_hours','high_hours']]=frame[['low_hours','central_hours','high_hours']].astype(float)
    frame.to_csv('reports/ETA_REMAINING_RUNS.csv',index=False)
    grouped=frame.groupby(['phase','dataset','model'])[['low_hours','central_hours','high_hours']].sum().reset_index()
    grouped.to_csv('reports/ETA_PHASE_MODEL_DATASET.csv',index=False)
    totals={k:float(frame[k+'_hours'].sum()) for k in ['low','central','high']}
    record={'updated_utc':now.isoformat(),'counts':progress['counts'],'remaining_slots':progress['remaining'],
        'mean_completed_run_seconds':statistics.mean(r['run_seconds'] for r in known if 'run_seconds' in r),
        'completed_duration_scope':'Saved cumulative training_seconds include training and validation through checkpoint restoration, excluding subprocess initialization and post-selection test inference. Epoch_seconds directly measure complete train/validation epochs; resumed histories are included once.',
        'remaining_training_gpu_reserved_hours':totals,
        'conditional_training_finish_utc':{k:(now+timedelta(hours=v)).isoformat() for k,v in totals.items()},
        'wall_clock_assumption':'One current MIG worker remains continuously available; no outages or new paid resources.',
        'interpretation':'GPU reserved wall hours from end-to-end measured epochs, not hardware GPU-active time. Scenario envelope, not a statistical confidence interval or guaranteed bound.',
        'method':'Per-model measured epochs; matching dataset references where available. Extrapolate by actual padded batch-timesteps and arithmetic proxies for nodes, widths and sequence lengths. Central is geometric mean of launch and arithmetic ratios. Six-epoch earliest-stop versus 50-epoch maximum scenarios; central epoch counts are completed-run model means. Unknown selected configs use the unweighted mean of candidate runtime estimates, not a claim about selection probabilities.',
        'unknowns':'No Algebra timings yet; most model/dataset/config strata have zero or one completed observation. GKT hidden-width performance and future stopping epochs are unmeasured. Post-selection test inference, subprocess initialization, final checkpoint verification and exclusive GPU benchmarks remain additional work.',
        'postprocessing':{'benchmark':'120 cases, 3 warmups plus 10 repeats each. GPU duration not measured during active training; excluded from training ETA.',
                          'statistics':'18 paired comparisons, five seeds; engineering microbenchmark timing is reported below; real paired research results still require final training.',
                          'visualizations':'Three final attention maps, two ASSIST2009 case figures; measured plot-render pilot below; final-case CPU inference remains unmeasured.',
                          'verification':'Fresh checkpoint inference for each unique source; redundant aliases will use cached inference. Duration not separately measured.'},
        'longest_expected_remaining':frame[frame.action=='training'].nlargest(12,'central_hours').to_dict('records')}
    write_json('reports/ETA.json',record)
    text='# Measured runtime estimate\n\nUpdated UTC: '+record['updated_utc']+'\n\n'
    text+=f"Remaining training: **{totals['central']:,.1f} GPU reserved hours**; scenario range **{totals['low']:,.1f}–{totals['high']:,.1f} hours**. Serial wall time: {totals['central']/24:,.1f} days; continuous-availability training finish {record['conditional_training_finish_utc']['central']}.\n\n"
    text+='This is a provisional measured-reference extrapolation, not a validated confidence interval. Hardware-active GPU utilization is unavailable. Epochs include training and validation, and reserved hours include their CPU work. Unmeasured final benchmarking and artifact verification add time beyond this training-only date.\n\n'
    text+=f"Overall mean completed-run duration: {record['mean_completed_run_seconds']/60:.2f} minutes, over {sum('run_seconds' in r for r in known)} actual independent completed runs. This global average is reported descriptively and is not multiplied by the remaining count.\n\n"
    text+=record['completed_duration_scope']+'\n\n'
    text+='## Observed model/dataset averages\n\n|Dataset|Model|Completed runs|Measured epochs|Mean epoch seconds|Mean completed run minutes|\n|---|---|---|---|---|---|\n'
    for r in observed_rows:
        ep=f"{r['average_epoch_seconds']:.3f}" if r['average_epoch_seconds'] is not None else 'not measured'
        minutes=f"{r['average_completed_run_seconds']/60:.3f}" if r['average_completed_run_seconds'] is not None else 'not measured'
        text+=f"|{r['dataset']}|{r['model']}|{r['completed_runs']}|{r['measured_epochs']}|{ep}|{minutes}|\n"
    text+='\n## Remaining phases\n\n|Phase|Low hours|Central hours|High hours|\n|---|---|---|---|\n'
    for phase,g in frame.groupby('phase'):
        text+='|'+phase+'|'+ '|'.join(f"{g[k+'_hours'].sum():,.2f}" for k in ['low','central','high'])+'|\n'
    text+='\n## Longest forecast runs\n\n|Run|Reference|Central hours|Scenario hours|\n|---|---|---|---|\n'
    for r in record['longest_expected_remaining'][:6]:
        text+=f"|{r['run_id']}|{r['reference_runs']}|{r['central_hours']:.2f}|{r['low_hours']:.2f}–{r['high_hours']:.2f}|\n"
    text+='\n## Dependent analysis duration\n\n'
    for name,description in record['postprocessing'].items():text+=f'* {name}: {description}\n'
    timing=Path('reports/postprocessing_timings.json')
    if timing.exists():
        measured=json.load(open(timing))
        text+='\nMeasured CPU pilot timing (actual completed primary plots; not final research completion): '+json.dumps(measured)+'\n'
        record['postprocessing_cpu_pilot_timings']=measured
    text+='\n'+record['method']+'\n\n'+record['unknowns']+'\n\nThe preserved equal 36-configuration baseline tuning budget includes 216 GKT variant configurations across all datasets; GKT is the main measured bottleneck. No dimensions, batches, split assignments, seeds, learning rates or early-stopping rules were changed to shorten this forecast.\n'
    write_json('reports/ETA.json',record)
    Path('reports/ETA_REPORT.md').write_text(text);print(json.dumps({'remaining_training_hours':totals,'finish_utc':record['conditional_training_finish_utc']},indent=2))
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    lock=open('experiments/eta.lock','w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        if args.watch:raise SystemExit('An ETA worker is already running')
        raise SystemExit(0)
    if args.watch:Path('experiments/eta.pid').write_text(str(os.getpid())+'\n')
    while True:
        estimate()
        if args.watch:
            from scripts.health import inspect
            from scripts.progress import write_progress
            inspect();write_progress()
        if not args.watch:break
        time.sleep(60)
