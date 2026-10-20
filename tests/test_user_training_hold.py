"""CPU-only restart prevention; no training or CUDA device initialization."""
import json
import subprocess
import sys

import pytest

from scripts import user_training_hold as hold


def test_absent_marker_allows_metadata_gate_and_malformed_marker_blocks(tmp_path):
    path=tmp_path/'hold.json'
    assert not hold.training_is_held(path)
    hold.require_training_enabled(path)
    path.write_text('partial non-JSON marker')
    assert hold.training_is_held(path)
    with pytest.raises(SystemExit,match='explicit user hold'):
        hold.require_training_enabled(path)


def test_recovery_hold_returns_before_snapshot_or_any_launch(tmp_path,monkeypatch):
    from scripts import recover
    path=tmp_path/'hold.json';path.write_text('{}');monkeypatch.setattr(hold,'HOLD',path)
    def forbidden(*args,**kwargs):raise AssertionError('no inspection/launch after hold gate')
    monkeypatch.setattr(recover,'launch',forbidden)
    result=recover.ensure({}, {})
    assert result['launched']==[] and result['signals_sent']==0 and result['blocked']


@pytest.mark.parametrize('module,args',[
    ('scripts.schedule',[]),('scripts.coordinator',[]),('scripts.recover',['--start']),
    ('scripts.recover',['--watch']),('scripts.run',['--config','DO_NOT_READ.json']),
    ('scripts.finalize',[]),
])
def test_actual_entrypoints_refuse_before_cuda_or_child_creation(module,args,tmp_path):
    # A private marker keeps the tests valid even after an authorized resume.
    # GPU APIs and
    # Import dependencies under the CUDA trap, then forbid child creation for
    # actual entrypoint execution. Python's platform discovery may run uname
    # during dependency import; that is not a training launch.
    code='''
import importlib,runpy,sys,torch,subprocess
def forbidden(*a,**k):raise AssertionError('held entrypoint cannot probe CUDA or spawn a child')
for name in ['is_available','init','_lazy_init','manual_seed','manual_seed_all','get_device_properties']:
 setattr(torch.cuda,name,forbidden)
from pathlib import Path
from scripts import user_training_hold as hold
hold.HOLD=Path(sys.argv[1])
importlib.import_module(sys.argv[2])
subprocess.Popen=forbidden
sys.argv=sys.argv[2:]
runpy.run_module(sys.argv[0],run_name='__main__')
'''
    marker=tmp_path/'hold.json';marker.write_text('{}')
    result=subprocess.run([sys.executable,'-c',code,str(marker),module,*args],text=True,capture_output=True,timeout=30)
    assert result.returncode!=0
    assert 'explicit user hold' in result.stderr
    assert 'AssertionError' not in result.stderr


def test_gpu_admission_and_post_run_hooks_refuse_without_probes(tmp_path,monkeypatch):
    from scripts import gpu_admission,execution_hooks
    path=tmp_path/'hold.json';path.write_text('{}');monkeypatch.setattr(hold,'HOLD',path)
    with pytest.raises(SystemExit,match='explicit user hold'):
        with gpu_admission.admission('GKT',str(tmp_path/'must_not_create.lock')):
            raise AssertionError('cannot admit held GPU work')
    assert not (tmp_path/'must_not_create.lock').exists()
    monkeypatch.setattr(execution_hooks,'required_efficiency',lambda:(_ for _ in ()).throw(AssertionError('no GPU hook')))
    assert execution_hooks.after_run() is None
