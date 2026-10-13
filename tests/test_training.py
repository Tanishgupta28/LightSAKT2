import copy
import json
import pandas as pd
import pytest
import torch
from src.datasets.preprocess import encode_records, sha256
from src.training.engine import run, evaluate, training_loss, update_early_stop
from src.training.reproducibility import seed_everything
from src.models.factory import build_model, predict
from torch.utils.data import DataLoader
from src.datasets.sequences import Sequences


def fixture(root, flipped=False):
    (root/'reports').mkdir(parents=True);directory=root/'data/processed/toy';directory.mkdir(parents=True)
    (root/'data/splits').mkdir(parents=True)
    records=[]
    for student in range(20):
        split='train' if student<14 else 'validation' if student<16 else 'test'
        for t in range(7):
            records.append({'student':str(student),'q':t%3,'label':(t+student)%2,'split':split})
    frame=pd.DataFrame(records)
    if flipped:frame.loc[frame.split!='train','label']=1-frame.loc[frame.split!='train','label']
    frame.to_csv(directory/'interactions.csv',index=False)
    (root/'data/splits/toy.json').write_text('{}')
    stats={'num_q':3,'processed_sha256':sha256(directory/'interactions.csv'),'split_sha256':sha256(root/'data/splits/toy.json')}
    (root/'reports/toy_statistics.json').write_text(json.dumps(stats))
    return {'run_id':'toy','dataset':'toy','model':'LightSAKT','seed':42,'group':'test',
            'seq_len':5,'batch_size':4,'max_epochs':3,'patience':5,'learning_rate':.001,
            'betas':[.9,.999],'epsilon':1e-8,'weight_decay':0.,'device':'cpu','evaluate_test':False,
            'model_config':{'embed_dim':8,'num_heads':2,'dropout':.1,'max_seq_len':5}}


def test_exact_resume_and_best_restoration(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';ca,cb=fixture(a),fixture(b)
    full=run(ca,a);run(cb,b,stop_after_epoch=1);resumed=run(cb,b)
    s1=torch.load(a/'checkpoints/toy/latest.pt',weights_only=False);s2=torch.load(b/'checkpoints/toy/latest.pt',weights_only=False)
    for k in s1['model']:torch.testing.assert_close(s1['model'][k],s2['model'][k],atol=0,rtol=0)
    assert full['validation']==resumed['validation']
    h=json.load(open(a/'results/raw/toy/history.json'))
    best=max(h,key=lambda x:x['validation']['auc'])
    assert full['best_epoch']==best['epoch'] and full['validation']==best['validation']


def test_heldout_labels_do_not_change_training_updates(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';ca,cb=fixture(a),fixture(b,True)
    run(ca,a,stop_after_epoch=1);run(cb,b,stop_after_epoch=1)
    s1=torch.load(a/'checkpoints/toy/latest.pt',weights_only=False);s2=torch.load(b/'checkpoints/toy/latest.pt',weights_only=False)
    for k in s1['model']:torch.testing.assert_close(s1['model'][k],s2['model'][k],atol=0,rtol=0)


def test_evaluation_does_not_update_parameters(tmp_path):
    config=fixture(tmp_path);seed_everything(42);model=build_model('LightSAKT',3,config['model_config'])
    before=copy.deepcopy(model.state_dict());ds=Sequences(tmp_path/'data/processed/toy/interactions.csv','test',5)
    result,y,p=evaluate(model,'LightSAKT',DataLoader(ds,batch_size=3),'cpu')
    assert result['n_targets']==ds.n_targets
    for k,v in before.items():torch.testing.assert_close(v,model.state_dict()[k],atol=0,rtol=0)


def test_early_stopping_validation_only_and_nan():
    best,bad=-float('inf'),0
    for auc in [.7,.8,.79,.78,.77,.76,.75]:
        best,bad,improved,stop=update_early_stop(auc,best,bad,5)
    assert best==.8 and bad==5 and stop
    with pytest.raises(ValueError):update_early_stop(None,best,bad)


def test_dktplus_short_history_penalty_is_finite():
    config={'emb_size':8,'hidden_size':8,'lambda_r':.01,'lambda_w1':.003,'lambda_w2':3.}
    model=build_model('DKT+',3,config);batch=[torch.zeros(1,1,dtype=torch.long) for _ in range(4)]+[torch.ones(1,1,dtype=torch.bool)]
    p,all_p=predict(model,'DKT+',batch);loss=training_loss(model,'DKT+',batch,p,all_p)
    assert torch.isfinite(loss);loss.backward()


def test_real_splits_chronology_and_reproducible_checksums():
    for dataset in ['ASSIST2009','ASSIST2015','Algebra2005']:
        from pathlib import Path
        path=Path(f'data/processed/{dataset}/interactions.csv')
        if not path.exists():pytest.skip('Raw datasets not installed')
        frame=pd.read_csv(path,dtype={'student':str});manifest=json.load(open(f'data/splits/{dataset}.json'))
        sets=[set(x) for x in manifest.values()]
        assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
        assert frame.groupby('student').split.nunique().max()==1
        assert frame.groupby('student').order.apply(lambda x:x.is_monotonic_increasing).all()
        stats=json.load(open(f'reports/{dataset}_statistics.json'));assert sha256(path)==stats['processed_sha256']
        for split in manifest:
            ds=Sequences(path,split,100)
            sub=frame[frame.split==split]
            assert ds.n_targets==len(sub)-sub.student.nunique()
