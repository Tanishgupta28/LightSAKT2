"""Resumable, sequential analyses after every required training slot completes."""
import argparse
import json
import subprocess
from pathlib import Path
from scripts.progress import snapshot, write_progress
from src.training.engine import write_json


def commands(manifest):
    tasks=[('tables',['python','-m','scripts.summarize']),
           ('statistics_and_performance_figures',['python','-m','scripts.analyze']),
           ('efficiency',['python','-m','scripts.benchmark'])]
    for dataset in ['ASSIST2009','ASSIST2015','Algebra2005']:
        selected=next(j for j in manifest['jobs'] if j['phase']==7 and j['dataset']==dataset and j['model']=='LightSAKT' and j['seed']==42)
        tasks.append(('visualization_'+dataset,['python','-m','scripts.visualize','--run-id',selected['run_id']]))
    tasks += [('recovery_archives',['python','-m','scripts.preserve']),
              ('manuscript_outputs',['python','-m','scripts.manuscript']),
              ('full_tests',['python','-m','pytest','-q']),
              ('checkpoint_and_predictions',['python','-m','scripts.verify_artifacts']),
              ('research_outputs',['python','-m','scripts.verify_research'])]
    return tasks


def main(retry_failed=False):
    progress=snapshot()
    if not progress['all_training_slots_completed']:
        raise SystemExit(f"Final analyses pending: {progress['remaining']} experiment slots remain; no completion claim.")
    manifest=json.load(open('experiments/manifest.json'));path=Path('experiments/analyses.json')
    ledger=json.load(open(path)) if path.exists() else {'tasks':{}}
    for name,command in commands(manifest):
        task=ledger['tasks'].setdefault(name,{'status':'pending','command':command})
        if task['status']=='completed':continue
        if task['status']=='failed' and not retry_failed:
            raise SystemExit(f'Analysis {name} failed; inspect {task["log"]}, fix the cause, then use --retry-failed.')
        log=Path('reports/analysis_logs')/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        task.update(status='running',log=str(log));write_json(path,ledger)
        with open(log,'a') as stream:
            rc=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT).returncode
        task.update(status='completed' if rc==0 else 'failed',exit_code=rc);write_json(path,ledger)
        if rc:raise SystemExit(f'Analysis {name} failed with exit {rc}; see {log}.')
    write_progress()
    subprocess.run(['python','-m','scripts.preserve'],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--retry-failed',action='store_true')
    main(parser.parse_args().retry_failed)
