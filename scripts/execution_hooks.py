"""Run bounded diagnostic work only after the owning research run is saved."""
import fcntl
import gc
import json
import os
import subprocess
from pathlib import Path
from src.training.engine import write_json


def after_run():
    if os.environ.get('LIGHTSAKT_COORDINATED_CHILD'):return
    request=Path('experiments/optimization_request.json')
    if not request.exists() or not json.load(open(request)).get('profile_between_runs'):return
    status=Path('reports/profiling/pilot_status.json')
    if status.exists() and json.load(open(status)).get('status') in ['completed','failed']:
        coordinate();return
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
    coordinate()


def coordinate():
    request=json.load(open('experiments/optimization_request.json'))
    if not request.get('coordinate_complete_schedule'):return
    from scripts.progress import snapshot
    if snapshot()['counts'].get('running',0):return
    with open('reports/optimization/coordinator.log','a') as stream:
        process=subprocess.Popen(['python','-u','-m','scripts.coordinator'],stdout=stream,stderr=subprocess.STDOUT)
        try:code=process.wait()
        except BaseException:
            process.send_signal(2);process.wait();raise
    if code:print('Coordinator stopped with unresolved entries; original complete schedule and all evidence remain available',flush=True)

