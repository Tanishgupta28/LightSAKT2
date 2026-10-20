"""One-shot CPU verification and local archive; does not resume training."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

import torch


def forbidden(*args, **kwargs):
    raise AssertionError('No CUDA API is allowed in this preservation check')


for name in ('is_available', 'init', '_lazy_init', 'manual_seed', 'manual_seed_all',
             'get_device_properties'):
    setattr(torch.cuda, name, forbidden)

from scripts.verify_checkpoint_states import verify

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
RUN = 'ASSIST2015-GKT-baseline_tuning-42-e01b589e46cc'
WAITING = 'ASSIST2015-GKT-baseline_tuning-42-e1c69a92d6b5'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


assert Path.cwd() == ROOT
config = json.loads((ROOT/f'results/raw/{RUN}/config.json').read_text())
stats = json.loads((ROOT/'reports/ASSIST2015_statistics.json').read_text())
verified = [verify(ROOT/f'checkpoints/{RUN}/{kind}.pt', config, stats)
            for kind in ('best', 'latest')]
latest = torch.load(ROOT/f'checkpoints/{RUN}/latest.pt', map_location='cpu', weights_only=False)
assert latest['config'] == config
assert latest['epoch'] == latest['best_epoch'] == 2
assert [row['epoch'] for row in latest['history']] == [1, 2]
assert latest['history'] == json.loads((ROOT/f'results/raw/{RUN}/history.json').read_text())
# Stream the processed data hash once; do not preprocess or load the dataset.
assert sha(ROOT/'data/processed/ASSIST2015/interactions.csv') == stats['processed_sha256']
assert sha(ROOT/'data/splits/ASSIST2015.json') == stats['split_sha256']

with gzip.open(OUTPUT/'before_hold/artifact_inventory.json.gz', 'rt') as stream:
    inventory = json.load(stream)['files']
selected = {path for path in inventory if RUN in path or WAITING in path or
            path.startswith('configs/') or path.startswith('src/') or
            path in ('experiments/manifest.json', 'experiments/optimization_request.json')}
archive = ROOT/'backups/priority_handoff_20261020/interrupted_gkt_epoch2.tar.gz'
archive.parent.mkdir(parents=True, exist_ok=True)
assert not archive.exists(), 'Preserve any earlier recovery archive'
temporary = archive.with_suffix('.tmp')
with tarfile.open(temporary, 'w:gz') as bundle:
    for relative in sorted(selected):
        bundle.add(ROOT/relative, arcname=relative, recursive=False)
with tarfile.open(temporary, 'r:gz') as bundle:
    members = bundle.getmembers()
    assert {item.name for item in members} == selected
    for member in members:
        assert member.isfile()
        h = hashlib.sha256()
        with bundle.extractfile(member) as stream:
            for block in iter(lambda: stream.read(4*1024*1024), b''):
                h.update(block)
        assert h.hexdigest() == inventory[member.name]['sha256']
temporary.replace(archive)
# Existing backup inventory is recorded separately; do not claim new offsite
# verification or initiate any upload from this script.
record = dict(observed_utc=datetime.now(timezone.utc).isoformat(), status='PASS',
    verified_checkpoints=verified, epoch=latest['epoch'], best_epoch=latest['best_epoch'],
    best_validation_auc=latest['best_auc'], bad_epochs=latest['bad_epochs'],
    history_epochs=[row['epoch'] for row in latest['history']],
    model_optimizer_rng_loader_state_verified_on_cpu=True,
    processed_and_split_hashes_verified=True, cuda_calls=0, optimizer_updates=0,
    local_archive=str(archive.relative_to(ROOT)), local_archive_sha256=sha(archive),
    local_archive_bytes=archive.stat().st_size, local_archive_members=len(selected),
    local_archive_content_hashes_verified=True,
    scope='Epoch-boundary state preserved; incomplete epoch3 work is not a resumable checkpoint; no training or inference',
    offsite_verification='Not rerun; local recovery archive only')
(OUTPUT/'interrupted_checkpoint_and_archive.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
