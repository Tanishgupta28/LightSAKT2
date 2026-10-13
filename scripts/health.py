"""Observe progress without signals, CUDA allocations, or scheduler ledger writes."""
import json
import os
import re
import statistics
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
from scripts.progress import snapshot
from src.training.engine import write_json


def cpu_sample(pid):
    fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
    return {'pid':pid,'cpu_seconds':(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'),
            'process_state':fields[0],'ppid':int(fields[1]),'session_id':int(fields[3])}


def inspect():
    now=time.time();progress=snapshot();pid=progress['processes']['scheduler']['pid']
    samples=[]
    if progress['processes']['scheduler']['alive']:
        children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        for child in children:
            try:
                cmd=Path(f'/proc/{child}/cmdline').read_bytes().replace(b'\0',b' ').decode()
                if 'scripts.run' in cmd:samples.append(cpu_sample(int(child)))
            except (FileNotFoundError,ProcessLookupError):pass
    previous_path=Path('reports/execution_health.json')
    previous=json.load(open(previous_path)) if previous_path.exists() else {}
    if not previous and Path('reports/health_sample_start.json').exists():
        initial=json.load(open('reports/health_sample_start.json'))
        previous={'sample_time':initial['utc'],'training_processes':[{'pid':initial['child'],
            'cpu_seconds':initial['cpu_ticks']/initial['clock_ticks']}]}
    for sample in samples:
        old=next((s for s in previous.get('training_processes',[]) if s['pid']==sample['pid']),None)
        sample['observed_seconds']=now-previous['sample_time'] if old else None
        sample['cpu_seconds_advanced']=sample['cpu_seconds']-old['cpu_seconds'] if old else None
    active=[]
    estimate_path=Path('reports/ETA_REMAINING_RUNS.csv')
    estimates={}
    if estimate_path.exists():
        import csv
        estimates={r['run_id']:r for r in csv.DictReader(estimate_path.open())}
    for job in progress['running_jobs']:
        root=Path('results/raw')/job['run_id'];history_path=root/'history.json'
        history=json.load(open(history_path)) if history_path.exists() else []
        epoch_mean=statistics.mean(h['epoch_seconds'] for h in history[-5:]) if history else None
        age=now-history_path.stat().st_mtime if history else now-(root/'status.json').stat().st_mtime if (root/'status.json').exists() else None
        planned=estimates.get(job['run_id'],{})
        estimated_mean=float(planned['central_hours'])*3600/float(planned['central_expected_total_epochs']) if not history and planned.get('central_expected_total_epochs') else None
        reference_mean=epoch_mean or estimated_mean
        active.append({**job,'recent_mean_epoch_seconds':epoch_mean,'seconds_since_last_epoch':age,
            'startup_epoch_seconds_extrapolated':estimated_mean,
            'age_reference':'history checkpoint' if history else 'initial running status; first checkpoint not yet saved',
            'stall_threshold_seconds':3*reference_mean if reference_mean else None,
            'checkpoint_stalled':age>3*reference_mean if age is not None and reference_mean else None})
    gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True)
    Path('reports/gpu_health_sample.txt').write_text(gpu.stdout+gpu.stderr)
    memory=re.search(r'(\d+)MiB /\s*(\d+)MiB',gpu.stdout)
    offsite_path=Path('reports/offsite_backup.json')
    backup=json.load(open(offsite_path)) if offsite_path.exists() else None
    record={'updated_utc':datetime.now(timezone.utc).isoformat(),'sample_time':now,
        'counts':progress['counts'],'scheduler':progress['processes']['scheduler'],
        'training_processes':samples,'active_runs':active,
        'gpu_memory_mib':int(memory[1]) if memory else None,'gpu_capacity_mib':int(memory[2]) if memory else None,
        'gpu_utilization_percent':None,'gpu_utilization_reason':'Unavailable for this MIG device; process/epoch evidence is used instead.',
        'healthy_progress_observed':progress['processes']['scheduler']['alive'] and
            bool(samples) and any((s.get('cpu_seconds_advanced') or 0)>0 for s in samples) and
            bool(active) and all(r['checkpoint_stalled'] is False for r in active),
        'failed_or_interrupted_runs':sum(progress['counts'].get(k,0) for k in ['failed','interrupted','blocked']),
        'restart_policy':'Detached session survives terminal/client disconnect; no process restart after environment shutdown is configured.',
        'storage':'/workspace is an NFS mount; retention after deprovisioning is not verified. '+
            (f"Private GitHub draft assets: {len(backup['assets'])} independently downloaded/checksum verified; completed-archive synchronization {backup.get('all_available_completed_archives_synced',False)}." if backup else 'Independent offsite artifact copies are not yet verified.')}
    write_json(previous_path,record)
    text='# Execution health\n\nObserved UTC: '+record['updated_utc']+'\n\n'
    text+=f"Scheduler PID {pid}; healthy progress sampled: {record['healthy_progress_observed']}; failed/interrupted/blocked slots: {record['failed_or_interrupted_runs']}.\n\n"
    for s in samples:text+=f"Child PID {s['pid']}: CPU seconds advanced {s['cpu_seconds_advanced']} over {s['observed_seconds']} observed seconds. Session {s['session_id']}, parent {s['ppid']}.\n\n"
    for r in active:text+=f"Run `{r['run_id']}`: epoch {r['last_completed_epoch']}; measured recent epoch average {r['recent_mean_epoch_seconds']}; extrapolated first-epoch duration {r['startup_epoch_seconds_extrapolated']}; progress age {r['seconds_since_last_epoch']} seconds ({r['age_reference']}); stall flag {r['checkpoint_stalled']}. Threshold uses three measured epoch durations, or the provisional ETA epoch estimate before the first checkpoint.\n\n"
    text+=f"MIG memory allocation: {record['gpu_memory_mib']} / {record['gpu_capacity_mib']} MiB. Utilization percentage is unavailable; no percentage is inferred from board power or PID existence.\n\n"
    text+='Best/latest state loads, finite parameters and optimizer/RNG integrity are independently recorded in reports/active_checkpoint_verification.json; saved result verification is in reports/stored_artifact_verification.json. Archive and recovery evidence is in reports/backup_integrity.json and reports/recovery_validation.json.\n\n'
    text+=record['restart_policy']+' '+record['storage']+'\n'
    Path('reports/EXECUTION_HEALTH.md').write_text(text)
    return record


if __name__=='__main__':print(json.dumps(inspect(),indent=2))
