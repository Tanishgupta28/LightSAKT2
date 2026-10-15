"""Preserve AUC-verified SAKT outputs while original-device checks wait for admission.

This archive explicitly remains preliminary. It never changes experiment results,
acquires GPU capacity, or marks full original-device verification complete.
"""
import fcntl
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts import offsite
from scripts.preserve import make_archive
from scripts.sakt_matched_grid import AUDIT, AUC_INFERENCE, fingerprint
from src.datasets.preprocess import sha256
from src.training.engine import write_json


def preserve_preview():
    assert not torch.cuda.is_available(), 'This preservation step must run on CPU'
    audit = json.load(open(AUDIT))
    proof = json.load(open(AUC_INFERENCE))
    assert proof['verified'] and len(proof['runs']) == 108
    protected = json.load(open('reports/sakt_grid_protected_artifacts.json'))
    assert fingerprint() == protected['model_source_sha256'] == audit['training_code_sha256']
    for record in protected['files']:
        assert sha256(record['path']) == record['sha256']
    archives = {r['run_id']: r for r in json.load(open('reports/completed_run_backups.json'))}
    remote_assets = json.load(open('reports/offsite_backup.json'))['assets']
    run_ids = {r['SAKT_run_id'] for r in audit['rows']}
    run_ids |= {r['result']['run_id'] for r in audit['finals'] if r['result']['model'] == 'SAKT'}
    references = []
    for run_id in sorted(run_ids):
        archive = archives[run_id]
        for component in [archive] + ([archive['result_archive']] if archive.get('result_archive') else []):
            asset = remote_assets[Path(component['path']).name]
            assert asset['download_sha256_verified'] and asset['sha256'] == component['sha256']
            references.append({'run_id': run_id, 'archive': component['path'],
                               'sha256': component['sha256'], 'asset_id': asset['asset_id']})
    write_json('reports/sakt_grid_existing_offsite_references.json', references)
    members = [Path(p) for p in [
        'paper/sakt_grid', 'paper/SAKT_GRID_RESULTS.md', 'results/predictions/sakt_matched_grid',
        str(AUDIT), str(AUC_INFERENCE), 'reports/SAKT_GRID_SEARCH_REPORT.md',
        'reports/SAKT_LIGHTSAKT_COMPARISON.md', 'reports/SAKT_GRID_CONTINUATION.md',
        'reports/sakt_grid_cpu_validation_inference.json', 'reports/sakt_grid_cpu_validation_failure.json',
        'reports/sakt_grid_correctness_tests.json', 'reports/sakt_grid_existing_offsite_references.json',
        'reports/sakt_grid_protected_artifacts.json', 'reports/selected_final_inference_verification.json',
        'reports/light_grid_inference_verification.json', 'reports/training_code_equivalence.json',
        'scripts/sakt_matched_grid.py', 'scripts/publish_sakt_grid.py', 'scripts/preserve_sakt_preview.py',
        'scripts/manuscript.py', 'scripts/report.py', 'tests/test_sakt_matched_grid.py', 'configs/protocol.json']]
    members += [p for p in Path('results/tables').glob('*')
                if 'SAKT_' in p.name or 'LightSAKT_vs_SAKT_grid_' in p.name]
    members += [p for p in Path('results/figures').glob('*') if 'matched_sakt_' in p.name]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    archive = Path('backups/publication') / f'sakt-grid-auc-verified-gpu-check-pending-{stamp}.tar.gz'
    with open('experiments/sakt_matched_grid_render.lock', 'a') as render_lock:
        fcntl.flock(render_lock, fcntl.LOCK_EX)
        record = make_archive(archive, members)
    assert archive.stat().st_size < 2 * 1024**3
    offsite.TAG = 'research-sakt-grid-' + datetime.now(timezone.utc).strftime('%Y%m%d')
    release = offsite.release()
    offsite.command(['gh', 'release', 'upload', offsite.TAG, str(archive), '--repo', offsite.REPO])
    asset = offsite.load_assets(release['id'])[archive.name]
    downloaded = Path('backups/remote_verification') / archive.name
    downloaded.parent.mkdir(parents=True, exist_ok=True)
    with downloaded.open('wb') as output:
        subprocess.run(['gh', 'api', '-H', 'Accept: application/octet-stream',
                        f'repos/{offsite.REPO}/releases/assets/{asset["id"]}'],
                       stdout=output, stderr=subprocess.PIPE, check=True)
    assert sha256(downloaded) == record['sha256']
    write_json('reports/sakt_grid_pending_offsite_backup.json', {
        **record, 'release_tag': offsite.TAG, 'asset_id': asset['id'], 'private': True, 'draft': True,
        'download_sha256_verified': True, 'auc_configurations_verified': 108,
        'original_device_full_metrics_verified': False,
        'status': 'AUC outputs preserved; original-device full-metric verification remains pending',
        'existing_checkpoint_result_archive_references': len(references),
        'verified_utc': datetime.now(timezone.utc).isoformat()})
    print('Private preliminary archive uploaded and downloaded SHA256 verified:', archive, flush=True)


if __name__ == '__main__':
    preserve_preview()
