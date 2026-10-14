"""Run bounded diagnostic work only after the owning research run is saved."""
import fcntl
import gc
import json
import os
import subprocess
from pathlib import Path
from src.training.engine import write_json


def after_run():
    if os.environ.get('LIGHTSAKT_COORDINATED_CHILD'):
        required_efficiency();additional_gkt_probe();verify_ablations_between_runs();return
    request=Path('experiments/optimization_request.json')
    if not request.exists() or not json.load(open(request)).get('profile_between_runs'):return
    status=Path('reports/profiling/pilot_status.json')
    if status.exists() and json.load(open(status)).get('status') in ['completed','failed']:
        validate_execution();coordinate();return
    with open('experiments/profiling.lock','w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        from scripts.progress import snapshot
        if snapshot()['counts'].get('running',0):return
        import torch
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        write_json(status,{'status':'running','scope':'Bounded engineering probes between completed research runs; no test labels'})
        with open('reports/profiling/pilot.log','a') as stream:
            process=subprocess.Popen(['python','-u','-m','scripts.profile_runtime'],stdout=stream,stderr=subprocess.STDOUT)
            Path('experiments/profiling.pid').write_text(str(process.pid)+'\n')
            code=process.wait()
        if code:write_json(status,{'status':'failed','exit_code':code,'log':'reports/profiling/pilot.log','full_research_schedule_preserved':True})
    validate_execution();coordinate()


def validate_execution():
    if not json.load(open('experiments/optimization_request.json')).get('validate_concurrency'):return
    status=Path('reports/profiling/execution_validation.json')
    if status.exists() and json.load(open(status)).get('status') in ['completed','failed']:return
    with open('reports/profiling/execution_validation.log','a') as stream:
        process=subprocess.Popen(['python','-u','-m','scripts.validate_execution'],stdout=stream,stderr=subprocess.STDOUT)
        Path('experiments/profiling.pid').write_text(str(process.pid)+'\n')
        code=process.wait()
    if code:write_json(status,{'status':'failed','exit_code':code,'full_research_schedule_preserved':True})


def coordinate():
    request=json.load(open('experiments/optimization_request.json'))
    if not request.get('coordinate_complete_schedule'):return
    validation=Path('reports/profiling/execution_validation.json')
    if request.get('validated_gkt_workers',1)>1 and (not validation.exists() or json.load(open(validation)).get('status')!='completed'):
        return
    from scripts.progress import snapshot
    if snapshot()['counts'].get('running',0):return
    with open('reports/optimization/coordinator.log','a') as stream:
        process=subprocess.Popen(['python','-u','-m','scripts.coordinator'],stdout=stream,stderr=subprocess.STDOUT)
        try:code=process.wait()
        except BaseException:
            process.send_signal(2);process.wait();raise
    if code:print('Coordinator stopped with unresolved entries; original complete schedule and all evidence remain available',flush=True)


def required_efficiency():
    request=Path('experiments/optimization_request.json')
    if not request.exists() or not json.load(open(request)).get('exclusive_efficiency_between_runs'):return
    status=Path('reports/profiling/efficiency_execution.json')
    if status.exists() and json.load(open(status)).get('status') in ['completed','failed']:return
    with open('experiments/efficiency_execution.lock','a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        from scripts.progress import snapshot
        if snapshot()['counts'].get('running',0):return
        from scripts.gpu_admission import admission
        from datetime import datetime,timezone
        import time
        with admission('exclusive_efficiency_benchmark'):
            write_json(status,{'status':'running','purpose':'Required 120-case controlled efficiency benchmark after completed training checkpoint; no prediction-performance claim'})
            started=time.perf_counter();log=Path('reports/analysis_logs/efficiency.log');log.parent.mkdir(parents=True,exist_ok=True)
            with open(log,'a') as stream:
                process=subprocess.Popen(['python','-u','-m','scripts.benchmark'],stdout=stream,stderr=subprocess.STDOUT)
                Path('experiments/profiling.pid').write_text(str(process.pid)+'\n');code=process.wait()
            completed=datetime.now(timezone.utc).isoformat();elapsed=time.perf_counter()-started
            write_json(status,{'status':'completed' if code==0 else 'failed','exit_code':code,'log':str(log),
                'updated_utc':completed,'actual_execution_seconds':elapsed,'research_training_interrupted':False})
            if code==0:
                import pandas as pd
                assert len(pd.read_csv('results/tables/efficiency.csv'))==120
                ledger=Path('experiments/analyses.json');record=json.load(open(ledger)) if ledger.exists() else {'tasks':{}}
                record['tasks']['efficiency']={'status':'completed','command':['python','-m','scripts.benchmark'],
                    'exit_code':0,'log':str(log),'executed_exclusively_between_runs':True,'completed_utc':completed}
                write_json(ledger,record)


def additional_gkt_probe():
    request=Path('experiments/gkt_thread_probe_request.json')
    if not request.exists() or not json.load(open(request)).get('enabled'):return
    status=Path('reports/profiling/gkt_thread_probe_status.json')
    if status.exists() and json.load(open(status)).get('status') in ['completed','failed']:return
    from scripts.recover import available_lock
    # Nonblocking exclusive admission: never wait on or signal a healthy job.
    # A newly queued worker simply waits for this bounded inter-run probe.
    with available_lock('experiments/extra_gkt_probe.lock') as owner:
        if not owner:return
        with available_lock('experiments/gpu_admission.lock') as idle:
            if not idle:return
            from datetime import datetime,timezone
            write_json(status,{'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
                              'healthy_training_interrupted':False,'enabled_for_research':False})


def verify_ablations_between_runs():
    request=Path('experiments/ablation_gpu_verification_request.json')
    if not request.exists() or not json.load(open(request)).get('enabled'):return
    status=Path('reports/ablation_gpu_verification_status.json')
    prior=json.load(open(status)).get('status') if status.exists() else None
    if prior=='failed':return
    if prior=='completed' and Path('reports/ablation_cpu_gpu_ranking_diagnostic.json').exists() and Path('reports/light_grid_inference_verification.json').exists():return
    from scripts.recover import available_lock
    with available_lock('experiments/ablation_verification.lock') as owner:
        if not owner:return
        with available_lock('experiments/gpu_admission.lock') as idle:
            if not idle:return
            write_json(status,{'status':'running','healthy_training_interrupted':False})
            log=Path('reports/ablation_gpu_verification.log')
            with log.open('a') as output:
                code=0
                if prior!='completed':
                    child=subprocess.Popen(['python','-u','-m','scripts.verify_selected_inference','--models','LightSAKT',
                        '--groups','ablation','--device','cuda'],stdout=output,stderr=subprocess.STDOUT)
                    code=child.wait()
                if code==0 and not Path('reports/ablation_cpu_gpu_ranking_diagnostic.json').exists():
                    code=subprocess.run(['python','-u','-m','scripts.diagnose_ablation_ranking'],stdout=output,stderr=subprocess.STDOUT).returncode
                grid=Path('reports/light_grid_inference_verification.json')
                if code==0 and (not grid.exists() or not json.load(open(grid)).get('verified')):
                    code=subprocess.run(['python','-u','-m','scripts.verify_light_grid_inference'],stdout=output,stderr=subprocess.STDOUT).returncode
            write_json(status,{'status':'completed' if code==0 else 'failed','exit_code':code,'log':str(log),
                              'healthy_training_interrupted':False})
            log=Path('reports/profiling/gkt_threads/probe.log');log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as output:
                child=subprocess.Popen(['python','-u','-m','scripts.probe_gkt_threads'],stdout=output,stderr=subprocess.STDOUT)
                code=child.wait()
            write_json(status,{'status':'completed' if code==0 else 'failed','exit_code':code,'log':str(log),
                              'completed_utc':datetime.now(timezone.utc).isoformat(),
                              'healthy_training_interrupted':False,'enabled_for_research':False})

