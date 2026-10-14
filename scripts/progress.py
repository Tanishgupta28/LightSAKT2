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
    for name in ['scheduler', 'preservation', 'eta', 'offsite', 'coordinator', 'progressive', 'active_recovery']:
        pidfile = root / f'experiments/{name}.pid'
        pid = int(pidfile.read_text()) if pidfile.exists() else None
        try:
            stat = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
            command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
            expected = {'scheduler':'scripts.schedule','preservation':'scripts.preserve',
                        'eta':'scripts.eta','offsite':'scripts.offsite','coordinator':'scripts.coordinator',
                        'progressive':'scripts.progressive','active_recovery':'scripts.active_recovery'}[name]
            live[name] = {'pid': pid, 'alive': stat != 'Z' and expected in command}
        except (FileNotFoundError, ProcessLookupError):
            live[name] = {'pid': pid, 'alive': False}
    completed = counts.get('completed', 0)
    reuse_path=root/'experiments/reuse_plan.json'
    extra_reuses=len(json.loads(reuse_path.read_text())['jobs']) if reuse_path.exists() else 0
    return {
        'updated_utc': datetime.now(timezone.utc).isoformat(),
        'total_slots': len(jobs), 'completed': completed,
        'remaining': len(jobs) - completed, 'counts': counts, 'processes': live,
        'running_jobs': [{k: j.get(k) for k in ['run_id', 'last_completed_epoch', 'best_checkpoint', 'resume_command']} for j in jobs if j['status'] == 'running'],
        'phase_counts': {str(p): dict(Counter(j['status'] for j in jobs if j['phase'] == p)) for p in [5, 6, 7, 8]},
        'reused_slots': sum(j['phase'] == 7 and j['seed'] == 42 or j.get('variant') == 'no_regularization' for j in jobs)+extra_reuses,
        'additional_primary_grid_reuses':extra_reuses,
        'distinct_training_runs_planned':len(jobs)-sum(j['phase'] == 7 and j['seed'] == 42 or j.get('variant') == 'no_regularization' for j in jobs)-extra_reuses,
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
    text += f"Distinct planned trainings: {progress['distinct_training_runs_planned']}. The live manifest plus experiments/reuse_plan.json form the effective schedule; {progress['additional_primary_grid_reuses']} exact primary/grid duplicates reuse verified evidence without removing grid rows. See reports/SCHEDULE_AUDIT.md.\n\n"
    eta_path=root/'reports/ETA.json'
    if eta_path.exists():
        eta=json.loads(eta_path.read_text());hours=eta['remaining_training_gpu_reserved_hours']
        text+=f"Measured-reference serial equivalent: {hours['central']:,.1f} training hours; scenario envelope {hours['low']:,.1f}–{hours['high']:,.1f} hours. "
        occupied=eta.get('two_worker_training_wall_hours')
        if occupied:
            text+=f"Current execution forecast: {occupied['central']:,.1f} occupied MIG wall hours; scenario envelope {occupied['low']:,.1f}–{occupied['high']:,.1f} hours, using independently verified two-worker GKT-MHA throughput only. "
        text+=f"Estimate updated {eta['updated_utc']}; startup, test inference and final verification add time. This is a planning scenario, not a confidence interval or guarantee. See reports/ETA_REPORT.md.\n\n"
    text+='Process existence alone is not a health guarantee. Last independently sampled CPU/checkpoint progress and checkpoint integrity are recorded in reports/EXECUTION_HEALTH.md. A detached scheduler survives a terminal disconnect, but no automatic restart after environment shutdown is configured.\n\n'
    offsite=root/'reports/offsite_backup.json'
    if offsite.exists():
        backup=json.loads(offsite.read_text())
        text+=f"Private draft recovery assets independently download/checksum verified: {len(backup['assets'])}. All available completed archives synchronized: {backup.get('all_available_completed_archives_synced',False)}. See reports/OFFSITE_RECOVERY.md and reports/offsite_backup.json.\n\n"
    text += 'The 108 LightSAKT grid configurations, equal baseline tuning budgets, and five matched final seeds use the frozen student splits. See experiments/manifest.json and per-run status files.\n\n'
    efficiency=root/'reports/efficiency_verification.json'
    if efficiency.exists():
        evidence=json.loads(efficiency.read_text())
        if evidence.get('verified'):
            text+=f"Verified existing efficiency evidence: {evidence['cases']} cases and {evidence['timed_repetitions_verified']} raw timed repetitions. LightSAKT was slower in {evidence['higher_light_latency']}/{evidence['matched_comparisons']} matched latency comparisons and used lower peak allocated memory in {evidence['lower_light_peak_allocated_memory']}/{evidence['matched_comparisons']}. Original measurements are preserved; theoretical scaling, latency, memory and parameters are discussed separately in paper/CORRECTED_RESULTS.md and paper/EFFICIENCY_RESULTS.md.\n\n"
    text += 'Research completion requires every scheduled slot and all dependent analyses, plots, checkpoints, and traceability checks to pass. Raw-data provenance limitations and the historical simplified GKT label remain disclosed.\n'
    path=root / 'reports/EXPERIMENT_STATUS.md';tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(text);tmp.replace(path)
    return progress


if __name__ == '__main__':
    print(json.dumps(write_progress(), indent=2))
