"""Single-worker resumable scheduler. No GPU contention; completed runs are immutable."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
from src.training.engine import write_json
from scripts.make_schedule import config, write_config


def status(manifest):
    counts={}
    for j in manifest['jobs']:
        p=Path('results/raw')/j['run_id']/'status.json'
        if p.exists():j.update({k:v for k,v in json.loads(p.read_text()).items() if k in ['status','last_completed_epoch','failure_reason','resume_command']})
        counts[j['status']]=counts.get(j['status'],0)+1
    write_json('experiments/manifest.json',manifest)
    Path('reports/EXPERIMENT_STATUS.md').write_text('# Experiment status\n\n'+json.dumps(counts,indent=2)+'\n\nTotal experiment slots: '+str(len(manifest['jobs']))+'\nIncludes 108 LightSAKT grid configurations, equal 36-configuration baseline tuning budgets, and five matched training seeds. Seed-42 final runs reuse selected tuning checkpoints (21 slots); no_regularization aliases no_dropout by user instruction (15 slots). Default ablation reuses the 15 LightSAKT final runs. These reused slots are not independent new trainings. Split seed remains 42 across every model seed. Run details: experiments/manifest.json.\n')
    return counts


def materialize(job,manifest):
    if 'config' in job:return True
    if job.get('variant')=='no_regularization':
        source=next(j for j in manifest['jobs'] if j['phase']==8 and j['dataset']==job['dataset'] and j['seed']==job['seed'] and j.get('variant')=='no_dropout')
        if source['status']!='completed':return False
        job['reuse_alias_run']=source['run_id'];return True
    model=job['model'];dataset=job['dataset']
    deps=[j for j in manifest['jobs'] if j['phase']==6 and j['run_id'].startswith(dataset+'-'+model.replace('+','plus')+'-')]
    if len(deps)!=36 or any(j['status']!='completed' for j in deps):return False
    candidates=[]
    for dep in deps:
        result=json.load(open(Path('results/raw')/dep['run_id']/'metrics.json'))
        candidates.append((result['validation']['auc'],dep['run_id'],dep))
    selected=sorted(candidates,key=lambda x:(-x[0],x[1]))[0][2]
    source=json.load(open(selected['config']))
    selected_path=Path('experiments/selections')/(dataset+'-'+model.replace('+','plus')+'.json')
    record={'selection_metric':'validation_auc','selected_run_id':selected['run_id'],
            'all_candidates':[j['run_id'] for j in deps],'test_access_for_selection':False}
    if selected_path.exists() and json.load(open(selected_path))!=record:raise ValueError('Frozen selection changed')
    write_json(selected_path,record)
    mc=dict(source['model_config'])
    if job.get('variant'):mc['variant']=job['variant']
    # A final training run at the tuning seed is reused rather than needlessly retrained.
    # Evaluation is performed separately after selection and bound to the tuning checkpoint.
    if job['phase']==7 and job['seed']==42:
        job['reuse_selected_run']=selected['run_id'];job['selection_path']=str(selected_path)
        return True
    c=config(dataset,model,'ablation' if job['phase']==8 else 'final',mc,source['learning_rate'],job['seed'],True)
    c['run_id']=job['run_id'];c['selection_path']=str(selected_path)
    job['config']=write_config(c);return True


def main():
    from scripts.user_training_hold import require_training_enabled
    require_training_enabled()
    parser=argparse.ArgumentParser();parser.add_argument('--retry-failed',action='store_true');parser.add_argument('--max-jobs',type=int)
    args=parser.parse_args()
    lock=open('experiments/scheduler.lock','w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('A scheduler is already running')
    Path('experiments/scheduler.pid').write_text(str(os.getpid())+'\n')
    child=None;shutdown_requested=0
    def interrupted(signum,frame):
        nonlocal shutdown_requested
        shutdown_requested=signum
        if child is not None:
            # Never reenter Popen.wait from a signal handler: its wait lock is non-reentrant.
            try:os.kill(child.pid,signal.SIGINT)
            except ProcessLookupError:pass
        else:raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    test_log=Path('reports/test_runs')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+f'-{os.getpid()}.log')
    test_log.parent.mkdir(parents=True,exist_ok=True)
    with open(test_log,'w') as stream:
        rc=subprocess.run(['python','-m','pytest','-q'],stdout=stream,stderr=subprocess.STDOUT).returncode
    write_json('reports/latest_test_run.json',{'log':str(test_log),'tests_passed':rc==0,'exit_code':rc})
    print(test_log.read_text(),flush=True)
    if rc:raise SystemExit(f'Correctness gate failed: {test_log}; training was not started.')
    manifest=json.load(open('experiments/manifest.json'));completed=0
    status(manifest)
    for job in manifest['jobs']:
        if job['status']=='running':
            job['status']='interrupted'
            p=Path('results/raw')/job['run_id']/'status.json'
            if p.exists():
                saved=json.load(open(p));saved['status']='interrupted';saved['failure_reason']='Previous scheduler stopped; resuming last completed epoch';write_json(p,saved)
    attempted=set()
    for phase in [5,6,7,8]:
        while True:
            require_training_enabled()
            status(manifest)
            jobs=[j for j in manifest['jobs'] if j['phase']==phase and j['run_id'] not in attempted and j['status'] in (['pending','interrupted','failed'] if args.retry_failed else ['pending','interrupted'])]
            runnable=next((j for j in jobs if materialize(j,manifest)),None)
            if runnable is None:break
            runnable['status']='running';write_json('experiments/manifest.json',manifest)
            attempted.add(runnable['run_id'])
            log=Path('reports/run_logs')/(runnable['run_id']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            cmd=(['python','-m','scripts.alias_ablation','--job',runnable['run_id']] if runnable.get('reuse_alias_run') else
                 ['python','-m','scripts.evaluate_selected','--job',runnable['run_id']] if runnable.get('reuse_selected_run') else
                 ['python','-m','scripts.run','--config',runnable['config']])
            with open(log,'a') as stream:
                child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
                rc=child.wait();child=None
            runnable['status']='interrupted' if shutdown_requested else 'completed' if rc==0 else 'failed'
            runnable['failure_reason']=None if rc==0 else f'Process exit {rc}; see {log}'
            write_json('experiments/manifest.json',manifest);status(manifest)
            if shutdown_requested:raise SystemExit(128+shutdown_requested)
            subprocess.run(['python','-m','scripts.summarize'],check=False)
            completed+=1
            print('Scheduler finished',runnable['run_id'],'status',runnable['status'],flush=True)
            if args.max_jobs and completed>=args.max_jobs:return
    status(manifest)
    command=['python','-m','scripts.finalize']
    if args.retry_failed:command.append('--retry-failed')
    subprocess.run(command,check=True)

if __name__=='__main__':main()
