import json
from pathlib import Path
import pytest
from scripts.progress import snapshot
from scripts.finalize import main as finalize
from src.training.engine import run,write_json


def test_live_status_overrides_stale_scheduler_manifest(tmp_path):
    write_json(tmp_path/'experiments/manifest.json',{'jobs':[
        {'run_id':'finished','phase':5,'status':'running'},
        {'run_id':'active','phase':5,'status':'pending'},
        {'run_id':'reuse','phase':7,'seed':42,'status':'pending'},
        {'run_id':'alias','phase':8,'variant':'no_regularization','status':'pending'}]})
    write_json(tmp_path/'results/raw/finished/status.json',{'status':'completed'})
    write_json(tmp_path/'results/raw/active/status.json',{'status':'running','last_completed_epoch':7})
    state=snapshot(tmp_path)
    assert state['completed']==1 and state['remaining']==3
    assert state['counts']=={'completed':1,'running':1,'pending':2}
    assert state['running_jobs'][0]['last_completed_epoch']==7 and state['reused_slots']==2
    assert not state['all_training_slots_completed'] and not state['research_verified_complete']
    assert json.load(open(tmp_path/'experiments/manifest.json'))['jobs'][0]['status']=='running'


def test_finalization_rejects_missing_training_even_with_prior_verification(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_json('experiments/manifest.json',{'jobs':[{'run_id':'missing','phase':5,'status':'pending'}]})
    write_json('reports/research_verification.json',{'verified':True})
    assert not snapshot()['research_verified_complete']
    with pytest.raises(SystemExit,match='1 experiment slots remain'):finalize()
    assert not Path('experiments/analyses.json').exists()


def test_completed_experiment_returns_without_initialization_or_training(tmp_path,monkeypatch):
    config={'run_id':'already_done'};directory=tmp_path/'results/raw/already_done'
    write_json(directory/'config.json',config);write_json(directory/'status.json',{'status':'completed'})
    result={'run_id':'already_done','validation':{'auc':.75}}
    write_json(directory/'metrics.json',result)
    def forbidden(*args,**kwargs):raise AssertionError('Completed experiment attempted to initialize or train again')
    monkeypatch.setattr('src.training.engine.seed_everything',forbidden)
    assert run(config,root=tmp_path)==result
    assert json.load(open(directory/'metrics.json'))==result


def test_final_analysis_dependency_order():
    from scripts.finalize import commands
    manifest={'jobs':[{'phase':7,'dataset':d,'model':'LightSAKT','seed':42,'run_id':d+'-selected'}
                      for d in ['ASSIST2009','ASSIST2015','Algebra2005']]}
    names=[name for name,_ in commands(manifest)]
    assert names.index('checkpoint_and_predictions')>names.index('full_tests')
    assert names[-1]=='research_outputs'
    assert sum(name.startswith('visualization_') for name in names)==3
