import json
from pathlib import Path
from src.training.engine import write_json
from scripts.schedule import materialize
from scripts.analyze import holm


def test_grid_count_and_equal_tuning_budget():
    manifest=json.load(open('experiments/manifest.json'))
    grid=[j for j in manifest['jobs'] if j['phase']==6 and 'LightSAKT-grid' in j['run_id']]
    assert len(grid)==108
    for dataset in ['ASSIST2009','ASSIST2015','Algebra2005']:
        for model in ['DKT','DKT+','DKVMN','GKT','SAKT','LightSAKT','GKT_MHA']:
            jobs=[j for j in manifest['jobs'] if j['phase']==6 and j['run_id'].startswith(dataset+'-'+model.replace('+','plus')+'-')]
            assert len(jobs)==36
            configs=[json.load(open(j['config'])) for j in jobs]
            assert all(not c['evaluate_test'] for c in configs)


def test_selection_uses_validation_and_freezes(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);jobs=[]
    for i in range(36):
        run_id=f'ASSIST2009-LightSAKT-grid-{i}'
        c={'model_config':{'embed_dim':32,'num_heads':2,'dropout':.1,'max_seq_len':100},'learning_rate':.001}
        config_path=f'configs/{i}.json';write_json(config_path,c)
        write_json(f'results/raw/{run_id}/metrics.json',{'validation':{'auc':.6+i*.001},'test':{'auc':1-i*.01}})
        jobs.append({'phase':6,'run_id':run_id,'config':config_path,'status':'completed'})
    job={'phase':7,'run_id':'ASSIST2009-LightSAKT-final-42','model':'LightSAKT','dataset':'ASSIST2009','seed':42}
    assert materialize(job,{'jobs':jobs}) and job['reuse_selected_run'].endswith('-35')
    assert json.load(open(job['selection_path']))['test_access_for_selection'] is False


def test_no_regularization_alias_requires_completed_no_dropout():
    source={'phase':8,'dataset':'ASSIST2009','seed':42,'variant':'no_dropout','run_id':'drop','status':'pending'}
    job={'phase':8,'dataset':'ASSIST2009','seed':42,'variant':'no_regularization'}
    assert not materialize(job,{'jobs':[source]})
    source['status']='completed';assert materialize(job,{'jobs':[source]}) and job['reuse_alias_run']=='drop'


def test_holm_correction():
    values=holm([.01,.03,.04]);assert values.tolist()==[.03,.06,.06]
