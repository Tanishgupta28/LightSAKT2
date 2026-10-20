"""Read-only scientific-artifact inventory for a user-requested training hold.

Only a fresh output directory is written. No imports of Torch, training,
schedulers, signals, GPU APIs or external services. File hashing is streamed.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


def digest(path):
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Artifact changed while hashing: {path}')
    return dict(sha256=h.hexdigest(), bytes=after.st_size)


def process_snapshot(root):
    project, gpu, inaccessible = [], [], 0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            status = dict(row.split(':', 1) for row in (proc/'status').read_text().splitlines() if ':' in row)
            state = status['State'].strip()
            try:
                cwd = os.readlink(proc/'cwd')
            except OSError:
                cwd = None
            argv = (proc/'cmdline').read_bytes().split(b'\0')
            module = None
            if b'-m' in argv:
                candidate = argv[argv.index(b'-m')+1].decode(errors='replace')
                if re.fullmatch('[A-Za-z0-9_.]+', candidate):
                    module = candidate
            base = dict(pid=int(proc.name), uid=status['Uid'].strip(),
                        parent_pid=int(status['PPid']), state=state,
                        comm=(proc/'comm').read_text().strip(), module=module)
            if cwd == str(root) and module and module.startswith('scripts.'):
                project.append({**base, 'cwd': cwd})
            try:
                fds = list((proc/'fd').iterdir())
            except PermissionError:
                inaccessible += 1
                continue
            handles = []
            for fd in fds:
                try:
                    link = os.readlink(fd)
                    if link.startswith('/dev/nvidia'):
                        handles.append(link)
                except OSError:
                    pass
            if handles:
                gpu.append({**base, 'inside_lightsakt2': cwd == str(root),
                            'inside_hitskt': cwd == '/workspace/capstone',
                            'nvidia_handles': sorted(set(handles))})
        except (OSError, ValueError, IndexError):
            pass
    return dict(project_module_processes=project, visible_gpu_handle_owners=gpu,
                inaccessible_fd_directories=inaccessible,
                visibility_limit='Only processes visible in this PID namespace; no external scheduler reservation evidence')


def capture(root, output):
    root = root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protected = {}
    for directory in ('checkpoints', 'results/raw', 'results/predictions', 'results/tables',
                      'configs', 'reports/run_logs', 'src'):
        for path in sorted((root/directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                protected[str(path.relative_to(root))] = digest(path)
    for relative in ('experiments/manifest.json', 'experiments/optimization_request.json'):
        protected[relative] = digest(root/relative)
    manifest = json.loads((root/'experiments/manifest.json').read_text())
    counts, unfinished = Counter(), []
    for job in manifest['jobs']:
        status_path = root/'results/raw'/job['run_id']/'status.json'
        saved = json.loads(status_path.read_text()) if status_path.exists() else {}
        actual = {**job, **saved}
        counts[actual['status']] += 1
        if actual['status'] not in ('completed', 'pending'):
            unfinished.append({k:actual.get(k) for k in ('run_id', 'status', 'last_completed_epoch', 'best_checkpoint', 'waiting_for_gpu_admission')})
    failures, references, unique = [], [], set()
    for path in sorted((root/'results/raw').glob('*/metrics.json')):
        result = json.loads(path.read_text())
        checkpoint = root/result['checkpoint']
        key = str(checkpoint.relative_to(root))
        if key not in protected or protected[key]['sha256'] != result['checkpoint_sha256']:
            failures.append(dict(run_id=path.parent.name, issue='missing or mismatched selected checkpoint'))
        else:
            unique.add(key)
        config = path.parent/'config.json'
        if not config.exists():
            failures.append(dict(run_id=path.parent.name, issue='missing configuration'))
        prediction = result.get('prediction_path')
        if prediction:
            if prediction not in protected or protected[prediction]['sha256'] != result['prediction_sha256']:
                failures.append(dict(run_id=path.parent.name, issue='missing or mismatched stored prediction'))
        references.append(dict(run_id=path.parent.name, checkpoint=result['checkpoint'],
            expected_sha256=result['checkpoint_sha256'], config_present=config.exists(),
            prediction_path=prediction))
    copies = output/'saved_metadata'; copies.mkdir()
    # Scientific metadata only; no credentials or unrelated process argv/env.
    for relative in ('experiments/manifest.json', 'experiments/optimization_request.json',
                     'reports/progress.json', 'reports/execution_health.json',
                     'reports/optimization/coordinator_status.json', 'reports/recovery/supervisor.json',
                     'configs/recovery.json'):
        source = root/relative
        if source.exists():
            target = copies/relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    summary = dict(observed_utc=datetime.now(timezone.utc).isoformat(), root=str(root),
        counts=dict(counts), total_slots=len(manifest['jobs']), unfinished=unfinished,
        selected_checkpoint_references=len(references), unique_selected_checkpoint_paths=len(unique),
        protected_files=len(protected), protected_bytes=sum(v['bytes'] for v in protected.values()),
        failures=failures, preservation_status='PASS' if not failures else 'FAIL',
        processes=process_snapshot(root), signals_sent=0,
        scope='presence/hash preservation, not fresh numerical replay or proof of previous backup assets')
    with gzip.open(output/'artifact_inventory.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump(dict(files=protected, selected_checkpoint_references=references), handle, sort_keys=True)
    (output/'snapshot.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    capture(Path(__file__).resolve().parents[1], args.output)
