import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from scripts import recover
from src.training.engine import run,write_json
from tests.test_training import fixture


def test_recovery_never_replaces_healthy_controller_or_orphan_worker():
    for module in ['scripts.schedule','scripts.coordinator','scripts.run','scripts.profile_runtime','scripts.finalize']:
        decision=recover.plan([{'module':module}],10,957)
        assert not decision['start_scheduler'] and not decision['start_finalizer']
        assert decision['leave_healthy_training_untouched']
    assert recover.plan([],10,957)['start_scheduler']
    assert recover.plan([],957,957)['start_finalizer']


def test_existing_gpu_lock_is_never_waited_on_or_interrupted(tmp_path):
    path=tmp_path/'lock'
    with recover.available_lock(path) as owner:
        assert owner
        with recover.available_lock(path) as peer:assert not peer
    with recover.available_lock(path) as peer:assert peer


def test_gpu_mask_restored_without_enabling_cuda_in_cpu_helpers(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    assert 'CUDA_VISIBLE_DEVICES' not in recover.gpu_environment({'cuda_visible_devices':None})
    assert recover.gpu_environment({'cuda_visible_devices':'0'})['CUDA_VISIBLE_DEVICES']=='0'
    assert recover.cpu_environment()['CUDA_VISIBLE_DEVICES']==''


def test_healthy_training_does_not_launch_cuda_probe_or_touch_manifest(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    observed=[{'module':m,'pid':i} for i,m in enumerate(['scripts.schedule']+[v[0] for v in recover.HELPERS.values()])]
    monkeypatch.setattr(recover,'processes',lambda:observed)
    progress={'completed':10,'remaining':947,'total_slots':957,'research_verified_complete':False,'all_training_slots_completed':False}
    monkeypatch.setattr('scripts.progress.snapshot',lambda:progress)
    def forbidden(*args,**kwargs):raise AssertionError('Healthy training was touched')
    monkeypatch.setattr(recover,'launch',forbidden);monkeypatch.setattr(recover.subprocess,'run',forbidden)
    result=recover.ensure({'max_attempts_without_progress':3,'retry_cooldown_seconds':120},{})
    assert result['launched']==[] and result['signals_sent']==0 and not result['completed_training_restarted']


def test_abrupt_worker_death_restores_exact_updates_and_retains_completed_evidence(tmp_path,monkeypatch):
    reference,resumed=tmp_path/'reference',tmp_path/'resumed'
    a,b=fixture(reference),fixture(resumed)
    for root,config in [(reference,a),(resumed,b)]:write_json(root/'config.json',config)
    # Exercise a real OS-level crash after an atomic epoch state was committed.
    code='''import json,time
from pathlib import Path
from src.training import engine
original=engine.save_state
def saved(path,state):
    original(path,state)
    if Path(path).name=="latest.pt" and state["epoch"]==1:
        Path("ready").write_text("checkpoint committed")
        time.sleep(30)
engine.save_state=saved
engine.run(json.load(open("config.json")),root=".")
'''
    env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='';env['PYTHONPATH']=str(recover.PROJECT)
    child=subprocess.Popen([sys.executable,'-c',code],cwd=resumed,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+20
        while not (resumed/'ready').exists():
            assert child.poll() is None, 'Toy interrupted worker exited before its checkpoint'
            if time.monotonic()>deadline:raise TimeoutError('Toy checkpoint readiness')
            time.sleep(.05)
        child.kill();assert child.wait(timeout=5)<0
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=5)
    # Recover the same data, config, Adam and RNG state through the public
    # preflight before using the unchanged training engine to finish the toy run.
    from scripts import coordinator
    with monkeypatch.context() as patch:
        patch.chdir(resumed)
        write_json('experiments/manifest.json',{'jobs':[{'run_id':'toy','status':'running','phase':5,'config':'config.json'}]})
        write_json('experiments/runtime_jobs/toy.json',{'run_id':'toy','status':'running','phase':5,'config':'config.json'})
        policy={'training_code_sha256':recover.code_fingerprint()}
        checked=recover.validate_resume_states(policy)
        assert len(checked)==2 and checked[0]['optimizer_rng_verified']
        resumed_result=run(b,'.')
        before={p.name:p.read_bytes() for p in Path('results/raw/toy').glob('*')}
        assert run(b,'.')==resumed_result
        assert before=={p.name:p.read_bytes() for p in Path('results/raw/toy').glob('*')}
    with monkeypatch.context() as patch:
        patch.chdir(reference);full_result=run(a,'.')
    full=torch.load(reference/'checkpoints/toy/latest.pt',weights_only=False)
    after=torch.load(resumed/'checkpoints/toy/latest.pt',weights_only=False)
    assert full_result['validation']==resumed_result['validation']
    assert full_result['best_epoch']==resumed_result['best_epoch']
    for name,value in full['model'].items():assert torch.equal(value,after['model'][name])
    assert torch.equal(full['loader_rng'],after['loader_rng']) and torch.equal(full['rng']['torch'],after['rng']['torch'])
    for index,state in full['optimizer']['state'].items():
        for name,value in state.items():
            other=after['optimizer']['state'][index][name]
            assert torch.equal(value,other) if isinstance(value,torch.Tensor) else value==other
