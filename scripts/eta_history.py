"""Forensic ETA history and separate work/occupied-time forecasts from artifacts."""
import json
import subprocess
from pathlib import Path

import pandas as pd

from src.training.engine import write_json


def archived_estimates():
    top=subprocess.run(['git','rev-parse','--show-toplevel'],capture_output=True,text=True)
    if top.returncode or Path(top.stdout.strip()).resolve()!=Path.cwd().resolve():return []
    found=subprocess.run(['git','rev-list','HEAD','--','reports/ETA.json'],capture_output=True,text=True)
    if found.returncode:return []
    commits=found.stdout.splitlines()
    payload=''.join(c+':reports/ETA.json\n' for c in commits)
    raw=subprocess.run(['git','cat-file','--batch'],input=payload.encode(),stdout=subprocess.PIPE,check=True).stdout
    offset=0;rows=[]
    for commit in commits:
        end=raw.index(b'\n',offset);header=raw[offset:end].decode();offset=end+1
        if header.endswith(' missing'):continue
        size=int(header.split()[-1]);body=raw[offset:offset+size];offset+=size+1
        record=json.loads(body);occupied=record.get('two_worker_training_wall_hours',{}).get('central')
        if occupied is None:continue
        rows.append({'git_commit':commit,'updated_utc':record['updated_utc'],
                     'completed_entries':record.get('counts',{}).get('completed',0),
                     'remaining_entries':record.get('remaining_slots'),
                     'serial_equivalent_hours':record['remaining_training_gpu_reserved_hours']['central'],
                     'occupied_mig_hours':occupied})
    return rows


def generate(eta, remaining, output_dir):
    output_dir=Path(output_dir);factor=eta['two_worker_measured_throughput_speedup']
    frame=remaining.copy()
    frame['occupied_mig_hours']=frame.central_hours.where(frame.model!='GKT_MHA',frame.central_hours/factor)
    rows=archived_estimates()
    selected=[]
    for target in [183,254,551]:
        if rows:
            nearest=min(rows,key=lambda r:abs(r['completed_entries']-target))
            selected.append({'requested_completed_milestone':target,**nearest})
    current={'requested_completed_milestone':'current','git_commit':'live saved artifacts',
             'updated_utc':eta['updated_utc'],'completed_entries':eta['counts'].get('completed',0),
             'remaining_entries':eta['remaining_slots'],
             'serial_equivalent_hours':eta['remaining_training_gpu_reserved_hours']['central'],
             'occupied_mig_hours':eta['two_worker_training_wall_hours']['central']}
    selected.append(current)
    pd.DataFrame(selected).to_csv(output_dir/'ETA_HISTORY_MILESTONES.csv',index=False)
    if rows:pd.DataFrame(rows).to_csv(output_dir/'ETA_ARCHIVED_HISTORY.csv',index=False)
    grouped=frame.groupby(['phase','model','dataset']).agg(
        remaining_entries=('run_id','count'),fresh_training_runs=('action',lambda x:int((x=='training').sum())),
        serial_equivalent_hours=('central_hours','sum'),occupied_mig_hours=('occupied_mig_hours','sum')).reset_index()
    grouped.to_csv(output_dir/'ETA_CATEGORY_OCCUPIED_HOURS.csv',index=False)
    phase=frame.groupby('phase').agg(remaining_entries=('run_id','count'),
                                  serial_equivalent_hours=('central_hours','sum'),occupied_mig_hours=('occupied_mig_hours','sum'))
    wall=float(frame.occupied_mig_hours.sum());gkt=float(frame.loc[frame.model.str.startswith('GKT'),'occupied_mig_hours'].sum())
    text=['## Historical ETA audit and runtime categories','',
          'Archived corrected-rebuild ETA snapshots were read from Git objects; these are execution estimates, not historical manuscript results. The nearest saved snapshot to each requested completion count is labelled explicitly.','',
          '|Requested milestone|Archived completed entries|Updated UTC|Serial equivalent hours|Occupied MIG hours|Git commit|',
          '|---|---|---|---|---|---|']
    for r in selected:
        text.append(f'|{r["requested_completed_milestone"]}|{r["completed_entries"]}|{r["updated_utc"]}|{r["serial_equivalent_hours"]:.3f}|{r["occupied_mig_hours"]:.3f}|{r["git_commit"]}|')
    text+=['',
           'The earlier approximately 391-hour number was a continuously refreshed, model/dataset-weighted forecast of remaining work. It was not a global average multiplied by total pending entries. Its small changes were obscured by rounding and by the fact that short prioritized jobs and evidence-reuse entries contributed little remaining time. The expensive GKT configurations remained untrained; new fast-model timings updated their own strata without removing those costs.','',
           f'Current GKT contribution: {gkt:.3f}/{wall:.3f} occupied MIG hours ({100*gkt/wall:.2f}% of the central forecast). Six completed primary GKT dataset/variant runs and real primary/wide engineering profiles still anchor this dominant term. Most future GKT stopping epochs are unknown. The forecast is refreshed rather than stale, but the slow configurations have limited full-run timing evidence.','',
           '|Remaining category|Entries|Serial equivalent hours|Occupied MIG hours|','|---|---|---|---|']
    for category,r in phase.iterrows():text.append(f'|{category}|{int(r.remaining_entries)}|{r.serial_equivalent_hours:.3f}|{r.occupied_mig_hours:.3f}|')
    text+=['',
           f'Isolated serial training work is {eta["remaining_training_gpu_reserved_hours"]["central"]:.3f} hours. With the admitted independent GKT-MHA concurrency, occupied-device time is {wall:.3f} MIG hours. On the one available MIG partition, continuous-availability wall-clock training time is therefore {wall/24:.2f} days, with conditional finish {eta["conditional_two_worker_training_finish_utc"]["central"]}. These are separate quantities; actual hardware-active utilization is unavailable.', '',
           f'Planning envelope: {eta["two_worker_training_wall_hours"]["low"]:.1f}–{eta["two_worker_training_wall_hours"]["high"]:.1f} occupied hours. This is an early-stop/maximum-epoch scenario envelope, not a confidence interval or guaranteed bound. The upper scenario does not assume sustained concurrency benefit. Unselected baseline configurations, GKT stopping counts, contention and environment availability dominate uncertainty.', '',
           'Startup, selected test evaluation and independent final artifact/checkpoint verification add wall time beyond this training-only estimate. Completed LightSAKT grids, selected seeds, ablations, controlled efficiency measurements and trained case figures are retained; they are not scheduled for fresh training again. Remaining numerical graphs and paired statistics are generated progressively on CPU.','',
           'Per-category/model/dataset source: ETA_CATEGORY_OCCUPIED_HOURS.csv. Complete archived trajectory: ETA_ARCHIVED_HISTORY.csv. Current forecasts: ETA.json, ETA_REMAINING_RUNS.csv and ETA_MEASURED_TIMINGS.csv. Live scheduler logs: reports/scheduler.log and reports/optimization/coordinator.log; sampled processes/checkpoints: reports/execution_health.json.']
    document='\n'.join(text)+'\n'
    (output_dir/'ETA_HISTORY_AUDIT.md').write_text(document)
    write_json(output_dir/'ETA_HISTORY_VERIFICATION.json',{'verified':True,'source':'Git-archived ETA.json objects and current measured run timings',
               'snapshots_read':len(rows),'milestones':selected,'gkt_occupied_hours':gkt,'occupied_hours':wall,
               'scope_changed':False,'forecast_algorithm_changed':False})
    return document
