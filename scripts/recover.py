"""Duplicate-safe process recovery; no signals or automatic platform reboot."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time

from src.training.engine import write_json

PROJECT = Path(__file__).resolve().parents[1]
POLICY = Path('configs/recovery.json')
HELPERS = {
    'preservation': ('scripts.preserve', ['--watch'], 'preservation_watch.lock'),
    'eta': ('scripts.eta', ['--watch'], 'eta.lock'),
    'offsite': ('scripts.offsite', ['--watch'], 'offsite.lock'),
    'progressive': ('scripts.progressive', ['--watch'], 'progressive.lock'),
    'active_recovery': ('scripts.active_recovery', [], 'active_recovery.lock'),
}
GPU_MODULES = {'scripts.run', 'scripts.evaluate_selected', 'scripts.alias_ablation',
               'scripts.finalize', 'scripts.benchmark', 'scripts.profile_runtime',
               'scripts.validate_execution', 'scripts.verify_artifacts'}


def code_fingerprint():
    return hashlib.sha256(b''.join(str(p).encode()+p.read_bytes()
        for p in sorted(Path('src').rglob('*.py')) if 'visualization' not in p.parts)).hexdigest()


@contextmanager
def available_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            yield False;return
        try:yield True
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def processes(root=PROJECT):
    """Inspect only project modules; never report arbitrary argv/credentials."""
    result=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            if (path/'cwd').resolve()!=Path(root).resolve():continue
            state=(path/'stat').read_text().split(') ',1)[1].split()[0]
            if state=='Z':continue
            argv=(path/'cmdline').read_bytes().split(b'\0')
            if b'-m' not in argv:continue
            index=argv.index(b'-m');module=argv[index+1].decode()
            if not module.startswith('scripts.'):continue
            result.append({'pid':int(path.name),'module':module,
                           'watchdog':module=='scripts.recover' and b'--watch' in argv})
        except (FileNotFoundError,PermissionError,ProcessLookupError,IndexError):continue
    return result


def configure():
    if POLICY.exists():return json.load(open(POLICY))
    policy={'version':1,'project_root':str(PROJECT),'training_code_sha256':code_fingerprint(),
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'interval_seconds':30,'retry_cooldown_seconds':120,'max_attempts_without_progress':3,
            'automatic_platform_restart':False,'scope':'Complete existing manifest; completed runs skipped',
            'python_version':platform.python_version(),
            'package_versions':{name:importlib.metadata.version(name) for name in
                                ['torch','numpy','pandas','scikit-learn','scipy']},
            'training_env':{k:os.environ[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG'] if k in os.environ}}
    write_json(POLICY,policy);return policy


def validate_completed_results():
    from scripts.coordinator import refreshed_jobs
    from scripts.verify_artifacts import verify_stored_result
    checked=set();count=0
    for job in refreshed_jobs():
        if job['status']!='completed':continue
        verify_stored_result(Path('results/raw')/job['run_id']/'metrics.json',checked);count+=1
    return count


def gpu_environment(policy):
    env=dict(os.environ);mask=policy.get('cuda_visible_devices')
    if mask is None:env.pop('CUDA_VISIBLE_DEVICES',None)
    else:env['CUDA_VISIBLE_DEVICES']=mask
    env.update(policy.get('training_env',{}));return env


def cpu_environment():
    env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='';env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    return env


def launch(module,args,env):
    log=Path('reports/recovery')/(module.rsplit('.',1)[-1]+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as output:
        child=subprocess.Popen([sys.executable,'-u','-m',module,*args],cwd=PROJECT,env=env,
                               stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    return {'module':module,'pid':child.pid,'log':str(log)}


def validate_resume_states(policy):
    """Check resumable bytes before any new scheduler is allowed to start."""
    from scripts.coordinator import refreshed_jobs
    from scripts.verify_checkpoint_states import verify
    from src.datasets.preprocess import sha256
    assert code_fingerprint()==policy['training_code_sha256'], 'Training code changed: automatic resume blocked'
    if 'python_version' in policy:
        assert platform.python_version()==policy['python_version'], 'Python version changed: automatic resume blocked'
    for name,version in policy.get('package_versions',{}).items():
        assert importlib.metadata.version(name)==version, f'{name} version changed: automatic resume blocked'
    states=[];data_checked=set()
    for job in refreshed_jobs():
        if job['status']=='completed':continue
        directory=Path('results/raw')/job['run_id'];latest=Path('checkpoints')/job['run_id']/'latest.pt'
        if not latest.exists():
            assert not (directory/'history.json').exists() or not json.load(open(directory/'history.json')), \
                f'Completed epoch history lacks latest state: {job["run_id"]}'
            continue
        import torch
        config=json.load(open(directory/'config.json'));stats=json.load(open(f'reports/{config["dataset"]}_statistics.json'))
        if config['dataset'] not in data_checked:
            assert sha256(f'data/processed/{config["dataset"]}/interactions.csv')==stats['processed_sha256']
            assert sha256(f'data/splits/{config["dataset"]}.json')==stats['split_sha256']
            data_checked.add(config['dataset'])
        state=torch.load(latest,map_location='cpu',weights_only=False)
        assert state['config']==config and state['environment']['training_code_sha256']==policy['training_code_sha256']
        assert 'optimizer' in state and 'rng' in state and 'loader_rng' in state
        best=latest.parent/'best.pt';assert best.exists(), f'Best checkpoint missing: {job["run_id"]}'
        best_state=torch.load(best,map_location='cpu',weights_only=False)
        assert best_state['epoch']==state['best_epoch'], f'Checkpoint boundary mismatch: {job["run_id"]}'
        states += [verify(latest,config,stats),verify(best,config,stats)]
    return states


def plan(observed,completed,total):
    modules={p['module'] for p in observed}
    controller=bool(modules & {'scripts.schedule','scripts.coordinator'})
    gpu_worker=bool(modules & GPU_MODULES)
    return {'controller_present':controller,'gpu_worker_present':gpu_worker,
            'start_scheduler':not controller and not gpu_worker and completed<total,
            'start_finalizer':not controller and not gpu_worker and completed==total,
            'leave_healthy_training_untouched':controller or gpu_worker}


def ensure(policy,attempts):
    from scripts.user_training_hold import training_is_held
    if training_is_held():
        return {'observed_utc':datetime.now(timezone.utc).isoformat(),
                'blocked':'Explicit user training hold for HiTSKT priority',
                'launched':[],'signals_sent':0,'completed_training_restarted':False,
                'automatic_platform_restart':False}
    from scripts.progress import snapshot
    progress=snapshot();observed=processes();decision=plan(observed,progress['completed'],progress['total_slots'])
    result={'observed_utc':datetime.now(timezone.utc).isoformat(),'supervisor_pid':os.getpid(),
            'completed':progress['completed'],'remaining':progress['remaining'],'processes':observed,
            **decision,'launched':[],'automatic_platform_restart':False,
            'limitation':'A platform startup hook must invoke scripts/resume_research.sh after replacement; RBAC/startup access is unavailable',
            'signals_sent':0,'completed_training_restarted':False}
    if progress['research_verified_complete']:return result
    modules={p['module'] for p in observed};now=time.monotonic()
    # Independent CPU services can recover even while existing GPU jobs run.
    for name,(module,args,lock) in HELPERS.items():
        if module in modules or progress['all_training_slots_completed'] and name in ['progressive','active_recovery']:continue
        old=attempts.get(module,{'count':0,'time':0})
        if old['count']>=policy['max_attempts_without_progress'] or now-old['time']<policy['retry_cooldown_seconds']:continue
        with available_lock(Path('experiments')/lock) as free:
            if not free:continue
        result['launched'].append(launch(module,args,cpu_environment()))
        attempts[module]={'count':old['count']+1,'time':now}
    if not decision['start_scheduler'] and not decision['start_finalizer']:return result
    module='scripts.schedule' if decision['start_scheduler'] else 'scripts.finalize'
    old=attempts.get(module,{'count':0,'time':0})
    if old['count']>=policy['max_attempts_without_progress']:
        result['blocked']='Restart limit reached without completed-run progress; inspect recovery logs';return result
    if now-old['time']<policy['retry_cooldown_seconds']:return result
    # Lock owners outlive PID files. Orphan children and inter-run probes are
    # conservatively protected; busy resources defer recovery without signals.
    for lock in ['scheduler.lock','coordinator.lock','final_analysis.lock','gpu_admission.lock']:
        with available_lock(Path('experiments')/lock) as free:
            if not free:result['deferred']='Live lock owner: '+lock;return result
    if any(p['module'] in GPU_MODULES | {'scripts.schedule','scripts.coordinator'} for p in processes()):
        result['deferred']='A project GPU/controller process appeared during inspection';return result
    env=gpu_environment(policy)
    check=subprocess.run([sys.executable,'-c','import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'],
                         env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if check.returncode:result['blocked']='CUDA unavailable; full research training is not silently moved to CPU';return result
    try:
        result['resume_states']=validate_resume_states(policy)
        result['completed_artifacts_verified']=validate_completed_results()
    except Exception as error:
        result['blocked']=str(error);return result
    result['launched'].append(launch(module,['--retry-failed'],env))
    attempts[module]={'count':old['count']+1,'time':now}
    return result


def start():
    from scripts.user_training_hold import require_training_enabled
    require_training_enabled()
    policy=configure()
    with available_lock('experiments/recovery_start.lock') as free:
        if not free:return {'status':'Startup already being checked'}
        existing=[p for p in processes() if p['watchdog']]
        if existing:return {'status':'Existing recovery supervisor retained','pid':existing[0]['pid']}
        with available_lock('experiments/recovery_supervisor.lock') as unlocked:
            if not unlocked:return {'status':'Existing recovery lock owner retained'}
        return {'status':'Started CPU-only recovery supervisor',**launch('scripts.recover',['--watch'],cpu_environment())}


def watch():
    from scripts.user_training_hold import require_training_enabled
    require_training_enabled()
    policy=configure();attempts={};previous_completed=None
    with available_lock('experiments/recovery_supervisor.lock') as free:
        if not free:raise SystemExit('A recovery supervisor already exists')
        Path('experiments/recovery_supervisor.pid').write_text(str(os.getpid())+'\n')
        while True:
            require_training_enabled()
            try:
                with available_lock('experiments/recovery_action.lock') as action:
                    if not action:time.sleep(1);continue
                    result=ensure(policy,attempts)
                if result.get('blocked')=='Explicit user training hold for HiTSKT priority':
                    return
                if previous_completed is not None and result['completed']>previous_completed:attempts={}
                previous_completed=result['completed'];write_json('reports/recovery/supervisor.json',result)
            except Exception as error:
                write_json('reports/recovery/failure.json',{'failure_type':type(error).__name__,
                    'observed_utc':datetime.now(timezone.utc).isoformat(),'healthy_processes_not_signalled':True})
            time.sleep(policy['interval_seconds'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true')
    parser.add_argument('--watch',action='store_true');parser.add_argument('--check',action='store_true')
    args=parser.parse_args();assert Path.cwd().resolve()==PROJECT, 'Run inside the existing LightSAKT2 root'
    if args.watch:watch();return
    if args.start:print(json.dumps(start(),indent=2));return
    from scripts.progress import snapshot
    policy=configure();progress=snapshot();result={**plan(processes(),progress['completed'],progress['total_slots']),
        'training_code_matches':code_fingerprint()==policy['training_code_sha256'],
        'resume_states':validate_resume_states(policy),'completed_artifacts_verified':validate_completed_results(),
        'automatic_platform_restart':False,'read_only_training_check':True}
    write_json('reports/recovery/preflight.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
