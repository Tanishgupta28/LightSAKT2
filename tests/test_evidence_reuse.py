import hashlib
import json
from pathlib import Path
import pytest
from scripts.audit_schedule import training_identity,identity_hash
from scripts.reuse_training import reuse_if_available,physical_source,training_code_compatible
from src.training.engine import write_json


@pytest.fixture
def reusable(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    source={'run_id':'primary','group':'primary','evaluate_test':True,'dataset':'ASSIST2009',
            'model':'LightSAKT','seed':42,'batch_size':64,'seq_len':100,'learning_rate':.001,
            'model_config':{'embed_dim':32,'num_heads':2,'dropout':.1},'max_epochs':50,'patience':5}
    grid={**source,'run_id':'grid','group':'grid','evaluate_test':False}
    result={'run_id':'primary','group':'primary','epochs_trained':10,'checkpoint':'checkpoint.pt',
            'training_code_sha256':hashlib.sha256(b'').hexdigest(),
            'validation':{'auc':.7},'test':{'auc':.8},'prediction_path':'test.npz','prediction_sha256':'original'}
    write_json('results/raw/primary/status.json',{'status':'completed'})
    write_json('results/raw/primary/metrics.json',result)
    write_json('results/raw/primary/config.json',source)
    write_json('results/raw/primary/history.json',[{'epoch':10}])
    write_json('results/raw/primary/environment.json',{'hardware':'test fixture'})
    write_json('experiments/reuse_plan.json',{'jobs':{'grid':{'source_run_id':'primary','identity_sha256':identity_hash(grid)}}})
    monkeypatch.setattr('scripts.reuse_training.verify_stored_result',lambda path:(dict(result),dict(source),{}))
    return source,grid,result


@pytest.mark.parametrize('key,value',[
    ('seed',123),('learning_rate',.005),('batch_size',8),('seq_len',200),
    ('patience',6),('max_epochs',49),('future_training_option',True)])
def test_identity_preserves_every_training_option(key,value):
    original={'run_id':'a','group':'primary','evaluate_test':True,'seed':42,'learning_rate':.001,
              'batch_size':64,'seq_len':100,'patience':5,'max_epochs':50}
    assert identity_hash(original)!=identity_hash({**original,key:value})
    assert training_identity(original)==training_identity({**original,'run_id':'b','group':'grid',
        'evaluate_test':False,'selection_path':'frozen.json'})


def test_grid_reuse_exposes_only_validation_and_never_mutates_source(reusable,monkeypatch):
    _,grid,result=reusable
    originals={str(p):p.read_bytes() for p in Path('results/raw/primary').glob('*')}
    assert reuse_if_available(grid)
    copied=json.load(open('results/raw/grid/metrics.json'))
    assert copied['validation']==result['validation'] and copied['test'] is None
    assert 'prediction_path' not in copied and 'prediction_sha256' not in copied
    assert not copied['training_executed_here'] and copied['reused_training_run']=='primary'
    assert json.load(open('results/raw/grid/config.json'))==grid
    assert json.load(open('results/raw/grid/status.json'))['status']=='completed'
    assert {p:Path(p).read_bytes() for p in originals}==originals
    def forbidden(*args):raise AssertionError('Completed reuse attempted verification or training again')
    monkeypatch.setattr('scripts.reuse_training.verify_stored_result',forbidden)
    assert reuse_if_available(grid)


def test_incomplete_source_is_never_published(reusable):
    _,grid,_=reusable
    write_json('results/raw/primary/status.json',{'status':'running'})
    assert not reuse_if_available(grid)
    assert not Path('results/raw/grid/status.json').exists()


def test_mismatched_training_cannot_be_reused(reusable):
    _,grid,_=reusable
    with pytest.raises(AssertionError):reuse_if_available({**grid,'seed':123})
    assert not Path('results/raw/grid/status.json').exists()


def test_code_reuse_rejects_unaudited_changes_and_other_datasets(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    result={'training_code_sha256':'old','dataset':'ASSIST2009'}
    assert training_code_compatible(result,'old')
    assert not training_code_compatible(result,'new')
    write_json('reports/training_code_equivalence.json',[{'source_sha256':'old','current_sha256':'new',
        'datasets':['ASSIST2009'],'verified':True}])
    assert training_code_compatible(result,'new')
    assert not training_code_compatible(result,'other')
    assert not training_code_compatible({**result,'dataset':'Algebra2005'},'new')


def test_physical_source_resolves_aliases_and_rejects_cycles(reusable):
    _,grid,result=reusable
    assert reuse_if_available(grid)
    write_json('results/raw/final/metrics.json',{'reused_training_run':'grid'})
    assert physical_source('final')==('primary',result)
    write_json('results/raw/primary/metrics.json',{'reused_training_run':'final'})
    with pytest.raises(ValueError,match='Cyclic'):physical_source('final')


def test_final_evaluation_reuses_primary_test_only_after_frozen_selection(reusable,monkeypatch):
    import numpy as np
    from scripts import evaluate_selected
    from src.datasets.preprocess import sha256
    from src.evaluation.metrics import metrics
    source,grid,result=reusable
    Path('checkpoint.pt').write_bytes(b'weights')
    path=Path('data/processed/ASSIST2009/interactions.csv');path.parent.mkdir(parents=True);path.write_text('fixture')
    write_json('data/splits/ASSIST2009.json',{'fixture':True})
    y=np.array([0,1,0,1]);p=np.array([.2,.8,.4,.6]);np.savez('test.npz',target=y,probability=p)
    result.update(checkpoint_sha256=sha256('checkpoint.pt'),processed_sha256=sha256(path),
        split_sha256=sha256('data/splits/ASSIST2009.json'),prediction_sha256=sha256('test.npz'),
        test=metrics(y,p),dataset='ASSIST2009')
    write_json('reports/ASSIST2009_statistics.json',{'processed_sha256':result['processed_sha256'],
        'split_sha256':result['split_sha256'],'num_q':2})
    write_json('results/raw/primary/metrics.json',result)
    write_json('results/raw/primary/prediction_identity.json',[{'student':'heldout'}])
    assert reuse_if_available(grid)
    job={'run_id':'final','reuse_selected_run':'grid','selection_path':'selection.json'}
    write_json('experiments/manifest.json',{'jobs':[job]})
    write_json('selection.json',{'selected_run_id':'different','selection_metric':'validation_auc'})
    with pytest.raises(AssertionError):evaluate_selected.main('final')
    assert not Path('results/raw/final/metrics.json').exists()
    write_json('selection.json',{'selected_run_id':'grid','selection_metric':'validation_auc'})
    def forbidden(*args,**kwargs):raise AssertionError('An identical heldout inference was repeated')
    monkeypatch.setattr(evaluate_selected,'seed_everything',forbidden)
    monkeypatch.setattr(evaluate_selected,'build_model',forbidden)
    original=Path('results/raw/primary/metrics.json').read_bytes()
    evaluate_selected.main('final')
    copied=json.load(open('results/raw/final/metrics.json'))
    assert copied['test']==metrics(y,p) and copied['reused_test_evaluation']=='primary'
    assert copied['reused_training_run']=='primary' and copied['selected_logical_run']=='grid'
    final_config=json.load(open('results/raw/final/config.json'))
    assert final_config['run_id']=='final' and final_config['group']=='final' and final_config['evaluate_test']
    assert Path('results/raw/primary/metrics.json').read_bytes()==original
    evaluate_selected.main('final')


def test_logical_alias_backup_includes_new_test_predictions(reusable,monkeypatch):
    from scripts import preserve
    from src.datasets.preprocess import sha256
    _,grid,result=reusable
    assert reuse_if_available(grid)
    Path('checkpoint.pt').write_bytes(b'weights')
    Path('test.npz').write_bytes(b'primary predictions')
    Path('new_test.npz').write_bytes(b'new selected predictions')
    result['checkpoint']='checkpoints/primary/best.pt'
    Path(result['checkpoint']).parent.mkdir(parents=True)
    Path(result['checkpoint']).write_bytes(b'weights')
    write_json('results/raw/primary/metrics.json',result)
    write_json('results/raw/final/metrics.json',{**result,'run_id':'final',
        'reused_training_run':'grid','prediction_path':'new_test.npz'})
    monkeypatch.setattr(preserve,'verify_stored_result',lambda path,*args:(json.load(open(path)),{},{}))
    preserve.archive_completed()
    index={r['run_id']:r for r in json.load(open('reports/completed_run_backups.json'))}
    assert index['final']['source_run_id']=='primary'
    assert index['final']['path']==index['primary']['path']
    import tarfile
    supplemental=index['final']['result_archive']
    assert sha256(supplemental['path'])==supplemental['sha256']
    with tarfile.open(supplemental['path']) as archive:
        assert 'new_test.npz' in archive.getnames()
        assert archive.extractfile('new_test.npz').read()==b'new selected predictions'
        assert not any(name.startswith('checkpoints/') for name in archive.getnames())


def test_cached_checkpoint_verification_still_checks_each_logical_metric(tmp_path,monkeypatch):
    from scripts import verify_artifacts as verify
    monkeypatch.chdir(tmp_path)
    result={'checkpoint':'same.pt','checkpoint_sha256':'weights','processed_sha256':'data',
            'split_sha256':'split','dataset':'ASSIST2009','validation':{'auc':.7,'accuracy':.6,'f1':.5,'mse':.2,'bce':.6}}
    config={'seed':42,'model':'LightSAKT','model_config':{},'seq_len':100,'batch_size':64}
    for run in ['one','two']:write_json(f'results/raw/{run}/metrics.json',{**result,'run_id':run})
    monkeypatch.setattr(verify,'verify_stored_result',lambda path,*args:(json.load(open(path)),config,{'num_q':2}))
    class FakeModel:
        def to(self,device):return self
        def load_state_dict(self,state):pass
    calls=[]
    monkeypatch.setattr(verify,'seed_everything',lambda seed:None)
    monkeypatch.setattr(verify,'build_model',lambda *args:FakeModel())
    monkeypatch.setattr(verify.torch,'load',lambda *args,**kwargs:{'model':{}})
    monkeypatch.setattr(verify.torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(verify,'Sequences',lambda *args:[])
    def evaluate(*args):calls.append(1);return result['validation'],[],[]
    monkeypatch.setattr(verify,'evaluate',evaluate)
    verify.main()
    assert len(calls)==1
    records=json.load(open('reports/artifact_verification.json'))
    assert not records[0]['reused_identical_checkpoint_inference'] and records[1]['reused_identical_checkpoint_inference']
    write_json('results/raw/two/metrics.json',{**result,'run_id':'two','validation':{**result['validation'],'auc':.8}})
    with pytest.raises(AssertionError):verify.main()
