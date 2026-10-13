"""Read-only progress accounting; never mutate the scheduler's job ledger."""
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def snapshot(root=Path('.')):
    root = Path(root)
    manifest = json.loads((root / 'experiments/manifest.json').read_text())
    jobs = []
    for job in manifest['jobs']:
        path = root / 'results/raw' / job['run_id'] / 'status.json'
        saved = json.loads(path.read_text()) if path.exists() else {}
        jobs.append({**job, **saved})
    counts = dict(Counter(j['status'] for j in jobs))
    live = {}
    for name in ['scheduler', 'preservation']:
        pidfile = root / f'experiments/{name}.pid'
        pid = int(pidfile.read_text()) if pidfile.exists() else None
        try:
            stat = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
            command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
            expected = 'scripts.schedule' if name == 'scheduler' else 'scripts.preserve'
            live[name] = {'pid': pid, 'alive': stat != 'Z' and expected in command}
        except (FileNotFoundError, ProcessLookupError):
            live[name] = {'pid': pid, 'alive': False}
    completed = counts.get('completed', 0)
    return {
        'updated_utc': datetime.now(timezone.utc).isoformat(),
        'total_slots': len(jobs), 'completed': completed,
        'remaining': len(jobs) - completed, 'counts': counts, 'processes': live,
        'running_jobs': [{k: j.get(k) for k in ['run_id', 'last_completed_epoch', 'best_checkpoint', 'resume_command']} for j in jobs if j['status'] == 'running'],
        'phase_counts': {str(p): dict(Counter(j['status'] for j in jobs if j['phase'] == p)) for p in [5, 6, 7, 8]},
        'reused_slots': sum(j['phase'] == 7 and j['seed'] == 42 or j.get('variant') == 'no_regularization' for j in jobs),
        'all_training_slots_completed': completed == len(jobs),
        'research_verified_complete': completed==len(jobs) and (root / 'reports/research_verification.json').exists() and
            json.loads((root / 'reports/research_verification.json').read_text()).get('verified', False),
    }


def write_progress(root=Path('.')):
    root = Path(root); progress = snapshot(root)
    path = root / 'reports/progress.json'
    # A single preservation worker refreshes this heartbeat; it does not own the manifest.
    tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(progress, indent=2) + '\n');tmp.replace(path)
    text = '# Experiment status\n\n'
    text += f"Updated UTC: {progress['updated_utc']}\n\n"
    text += f"Completed: **{progress['completed']}**. Remaining: **{progress['remaining']}**. Total schedule slots: {progress['total_slots']}.\n\n"
    text += '```json\n' + json.dumps(progress['counts'], indent=2) + '\n```\n\n'
    for name, state in progress['processes'].items():
        text += f"{name.capitalize()} process: {'running' if state['alive'] else 'not running'} (PID {state['pid']}).\n\n"
    for job in progress['running_jobs']:
        text += f"Current run: `{job['run_id']}`; last completed epoch: {job['last_completed_epoch']}.\n\n"
    text += f"The {progress['reused_slots']} reuse/alias slots are not independent new training experiments. Default ablation reuses final LightSAKT evidence. The separate smoke run is excluded from schedule counts.\n\n"
    text += 'The 108 LightSAKT grid configurations, equal baseline tuning budgets, and five matched final seeds use the frozen student splits. See experiments/manifest.json and per-run status files.\n\n'
    text += 'Research completion requires every scheduled slot and all dependent analyses, plots, checkpoints, and traceability checks to pass. Raw-data provenance limitations and the historical simplified GKT label remain disclosed.\n'
    path=root / 'reports/EXPERIMENT_STATUS.md';tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(text);tmp.replace(path)
    return progress


if __name__ == '__main__':
    print(json.dumps(write_progress(), indent=2))
