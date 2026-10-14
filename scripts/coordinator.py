"""Prioritize the complete immutable research schedule; bounded independent workers."""
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from src.training.engine import write_json
from scripts.schedule import materialize


def refreshed_jobs():
    jobs=json.load(open('experiments/manifest.json'))['jobs']
    for job in jobs:
        extra=Path('experiments/runtime_jobs')/(job['run_id']+'.json')
        if extra.exists():job.update(json.load(open(extra)))
        path=Path('results/raw')/job['run_id']/'status.json'
        if path.exists():job.update(json.load(open(path)))
    return jobs


def rank(job):
    if 'config' in job:config=json.load(open(job['config']));model=config['model']
    else:model=job['model']
    if job['phase']==6 and model=='LightSAKT':return 0
    if job['phase']==7 and model=='LightSAKT':return 1
    if job['phase']==8:return 2
    if job['phase']==6 and model in ['DKT','DKT+','SAKT']:return 3
    if job['phase']==7 and not model.startswith('GKT'):return 4
    if job['phase']==6 and model=='DKVMN':return 5
    if job['phase']==6:return 6
    return 7


def model_name(job):
    return json.load(open(job['config']))['model'] if 'config' in job else job['model']


def eligible(job,jobs):
    if job['phase']==8:
        defaults=[j for j in jobs if j['phase']==7 and j['dataset']==job['dataset'] and j['model']=='LightSAKT']
        if len(defaults)!=5 or any(j['status']!='completed' for j in defaults):return False
    return materialize(job,{'jobs':jobs})


def main():
    if not json.load(open('experiments/optimization_request.json')).get('coordinate_complete_schedule'):return
    lock=open('experiments/coordinator.lock','w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('An execution coordinator already exists')
    Path('experiments/coordinator.pid').write_text(str(os.getpid())+'\n')
    policy=json.load(open('experiments/optimization_request.json'));limit=int(policy.get('validated_gkt_workers',1))
    assert limit in [1,2]
    processes={};stop=False
    def interrupted(signum,frame):
        nonlocal stop
        stop=True
        for process,_ in processes.values():
            if process.poll() is None:process.send_signal(signal.SIGINT)
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    while True:
        for run_id,(process,job) in list(processes.items()):
            rc=process.poll()
            if rc is None:continue
            path=Path('results/raw')/run_id/'status.json';saved=json.load(open(path)) if path.exists() else {'run_id':run_id}
            if rc and saved.get('status')!='completed':
                saved.update(status='interrupted' if stop else 'failed',failure_reason=f'Execution exit {rc}; preserved log reports/run_logs/{run_id}.log')
                write_json(path,saved)
            del processes[run_id]
        if stop and not processes:return
        jobs=refreshed_jobs();pending=[j for j in jobs if j['status'] in ['pending','interrupted']]
        if not stop:
            for job in sorted(pending,key=lambda j:(rank(j),j['run_id'])):
                if job['run_id'] in processes:continue
                name=model_name(job)
                can_share=limit==2 and name.startswith('GKT') and all(model_name(j).startswith('GKT') for _,j in processes.values())
                if len(processes)>=(2 if can_share else 1):continue
                if not eligible(job,jobs):continue
                job_file=Path('experiments/runtime_jobs')/(job['run_id']+'.json');write_json(job_file,job)
                path=Path('results/raw')/job['run_id']/'status.json'
                prior=json.load(open(path)) if path.exists() else {}
                write_json(path,{**prior,'run_id':job['run_id'],'status':'running','execution_coordinator_pid':os.getpid()})
                command=(['python','-m','scripts.alias_ablation','--job',job['run_id'],'--job-file',str(job_file)] if job.get('reuse_alias_run') else
                         ['python','-m','scripts.evaluate_selected','--job',job['run_id'],'--job-file',str(job_file)] if job.get('reuse_selected_run') else
                         ['python','-m','scripts.run','--config',job['config']])
                env=dict(os.environ);env['LIGHTSAKT_COORDINATED_CHILD']='1';env['LIGHTSAKT_MAX_CONCURRENT_GKT']=str(limit)
                log=Path('reports/run_logs')/(job['run_id']+'.log');log.parent.mkdir(exist_ok=True)
                with open(log,'a') as stream:process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
                processes[job['run_id']]=(process,job)
                print('Started required run',job['run_id'],'PID',process.pid,flush=True)
        write_json('reports/optimization/coordinator_status.json',{'coordinator_pid':os.getpid(),'stopping':stop,
            'running':{run_id:process.pid for run_id,(process,_) in processes.items()},'scope':'complete 957-entry schedule; no scope reduction',
            'worker_limit_gkt':limit,'worker_limit_other_models':1,'batch_sizes_unchanged':True})
        if not processes:
            unresolved=[j for j in jobs if j['status']!='completed']
            if unresolved:raise RuntimeError(f'{len(unresolved)} unresolved entries; inspect failures/dependencies; original scheduler will retain evidence')
            return
        time.sleep(1)


if __name__=='__main__':main()
