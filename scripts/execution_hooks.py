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
        required_efficiency();return
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

