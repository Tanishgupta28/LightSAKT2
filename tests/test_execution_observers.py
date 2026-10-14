import json
import os
from pathlib import Path
from types import SimpleNamespace
from src.training.engine import write_json


def test_first_long_epoch_is_not_flagged_as_stalled(tmp_path,monkeypatch):
    from scripts import health
    monkeypatch.chdir(tmp_path)
    write_json('results/raw/active/status.json',{'status':'running'})
    Path('reports').mkdir()
    Path('reports/ETA_REMAINING_RUNS.csv').write_text('run_id,central_hours,central_expected_total_epochs\nactive,2,12\n')
    state={'processes':{'scheduler':{'pid':os.getpid(),'alive':True}},'counts':{'running':1},
        'running_jobs':[{'run_id':'active','last_completed_epoch':0}]}
    monkeypatch.setattr(health,'snapshot',lambda:state)
    monkeypatch.setattr(health.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout='874MiB / 40320MiB',stderr=''))
    began=Path('results/raw/active/status.json').stat().st_mtime
    monkeypatch.setattr(health.time,'time',lambda:began+120)
    result=health.inspect()['active_runs'][0]
    assert result['recent_mean_epoch_seconds'] is None
    assert result['startup_epoch_seconds_extrapolated']==600
    assert result['stall_threshold_seconds']==1800 and not result['checkpoint_stalled']
    monkeypatch.setattr(health.time,'time',lambda:began+1801)
    assert health.inspect()['active_runs'][0]['checkpoint_stalled']


def test_fast_first_epoch_allows_process_initialization(tmp_path,monkeypatch):
    from scripts import health
    monkeypatch.chdir(tmp_path)
    write_json('results/raw/active/status.json',{'status':'running'})
    (tmp_path/'reports').mkdir()
    (tmp_path/'reports/ETA_REMAINING_RUNS.csv').write_text('run_id,central_hours,central_expected_total_epochs\nactive,0.001,36\n')
    state={'processes':{'scheduler':{'pid':os.getpid(),'alive':True}},'counts':{'running':1},
           'running_jobs':[{'run_id':'active','last_completed_epoch':0}]}
    monkeypatch.setattr(health,'snapshot',lambda:state)
    monkeypatch.setattr(health.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout='',stderr=''))
    began=(tmp_path/'results/raw/active/status.json').stat().st_mtime
    monkeypatch.setattr(health.time,'time',lambda:began+2)
    sample=health.inspect()['active_runs'][0]
    assert sample['startup_epoch_seconds_extrapolated']==.1
    assert sample['stall_threshold_seconds']==30 and not sample['checkpoint_stalled']
    monkeypatch.setattr(health.time,'time',lambda:began+31)
    assert health.inspect()['active_runs'][0]['checkpoint_stalled']


def test_eta_handles_all_completed_slots_without_empty_table_failure(tmp_path,monkeypatch):
    from scripts import eta
    monkeypatch.chdir(tmp_path)
    config={'run_id':'done','dataset':'ASSIST2009','model':'LightSAKT','group':'primary'}
    write_json('results/raw/done/config.json',config)
    write_json('results/raw/done/history.json',[{'epoch':1,'epoch_seconds':10}])
    write_json('results/raw/done/metrics.json',{'epochs_trained':1,'training_seconds':10})
    write_json('results/raw/done/status.json',{'status':'completed'})
    write_json('configs/done.json',config)
    write_json('experiments/manifest.json',{'jobs':[{'run_id':'done','phase':5,'status':'completed','config':'configs/done.json'}]})
    Path('reports').mkdir()
    monkeypatch.setattr(eta,'snapshot',lambda:{'counts':{'completed':1},'remaining':0})
    monkeypatch.setattr(eta,'work_counts',lambda:{})
    result=eta.estimate()
    assert result['remaining_training_gpu_reserved_hours']=={'low':0.,'central':0.,'high':0.}
    assert result['longest_expected_remaining']==[]
    assert json.load(open('reports/ETA.json'))['remaining_slots']==0
