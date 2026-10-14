import json
import tarfile
import torch
from tests.test_training import fixture
from src.training.engine import run
from scripts.active_recovery import capture,snapshot_due


def test_first_gkt_epoch_is_backed_up_and_duplicate_epochs_are_not_uploaded():
    policy={'gkt_interval_seconds':300}
    epochs=[{'epoch':1,'epoch_seconds':168}]
    assert snapshot_due('GKT',epochs,[],1000,policy)
    assert snapshot_due('GKT_MHA',epochs,[],1000,policy)
    assert not snapshot_due('DKVMN',epochs,[],1000,policy)
    prior=[{'epoch':1,'captured_time':1000}]
    assert not snapshot_due('GKT',epochs,prior,2000,policy)
    epochs.append({'epoch':2,'epoch_seconds':168})
    assert not snapshot_due('GKT',epochs,prior,1200,policy)
    assert snapshot_due('GKT',epochs,prior,1300,policy)


def test_active_snapshot_contains_consistent_resumable_state(tmp_path,monkeypatch):
    config=fixture(tmp_path);monkeypatch.chdir(tmp_path)
    run(config,tmp_path,stop_after_epoch=1)
    result=capture(config['run_id'])
    assert result['epoch']==1 and result['verified_contents'] and not result['live_run_interrupted']
    with tarfile.open(result['path']) as archive:
        latest=torch.load(archive.extractfile(f"checkpoints/{config['run_id']}/latest.pt"),map_location='cpu',weights_only=False)
        best=torch.load(archive.extractfile(f"checkpoints/{config['run_id']}/best.pt"),map_location='cpu',weights_only=False)
        history=json.load(archive.extractfile(f"results/raw/{config['run_id']}/history.json"))
        assert latest['best_epoch']==best['epoch'] and latest['history']==history
        assert all(k in latest for k in ['optimizer','rng','loader_rng','config'])
