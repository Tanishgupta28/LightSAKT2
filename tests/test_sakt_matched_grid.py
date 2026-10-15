import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.sakt_matched_grid import GRID, matched_protocol, mask_supplement, publication_table, winner
from scripts.make_schedule import config
from src.datasets.preprocess import sha256


def pair():
    return (config('ASSIST2009','LightSAKT','grid',{'max_seq_len':100,'embed_dim':64,'num_heads':4,'dropout':.2}),
            config('ASSIST2009','SAKT','baseline_tuning',{'n':100,'d':64,'num_attn_heads':4,'dropout':.2}))


def test_exact_pair_protocol_rejects_training_and_architecture_differences():
    light,sakt=pair();assert matched_protocol(light,sakt)['seed']==42
    assert len(GRID)==len(set(GRID))==36
    for field,value in [('seed',123),('patience',6),('weight_decay',.01),('batch_size',32)]:
        changed=copy.deepcopy(sakt);changed[field]=value
        with pytest.raises(AssertionError):matched_protocol(light,changed)
    changed=copy.deepcopy(light);changed['model_config']['variant']='no_dropout'
    with pytest.raises(AssertionError):matched_protocol(changed,sakt)
    changed=copy.deepcopy(sakt);changed['model_config']['extra_component']=True
    with pytest.raises(AssertionError):matched_protocol(light,changed)


def test_full_precision_maxima_and_deterministic_tie_break(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    data=pd.DataFrame([{'Best Validation AUC':score,'SAKT_run_id':rid,'SAKT_checkpoint':'checkpoints/'+rid+'/best.pt',
                        'SAKT_best_epoch':2} for rid,score in [('b',.8123456789),('a',.8123456789),('c',.812345678899)]])
    publication_table('sample',data,['Best Validation AUC'],['Best Validation AUC'],caption='Example')
    md=(tmp_path/'results/tables/sample.md').read_text()
    assert md.count('**0.81234568**')==2
    assert 'Selected SAKT run: `a`' in md
    assert winner([{'run_id':r['SAKT_run_id'],'score':r['Best Validation AUC']} for r in data.to_dict('records')],'score')['run_id']=='a'
    assert (tmp_path/'paper/sakt_grid/sample.tex').read_text().count('\\textbf')==2


def test_explicit_mask_reconstructs_valid_predictions_without_overwriting_source(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);target=np.array([1,0,1]);probability=np.array([.8,.2,.7],dtype=np.float32)
    np.savez_compressed('original.npz',target=target,probability=probability)
    digest=sha256('original.npz');mask=np.array([[True,True,False],[True,False,False]])
    dataset=SimpleNamespace(arrays=[None]*4+[mask],n_targets=3)
    r={'run_id':'toy','prediction_path':'original.npz','prediction_sha256':digest}
    first=mask_supplement(r,dataset);second=mask_supplement(r,dataset)
    assert first==second and sha256('original.npz')==digest
    with np.load(first['path']) as arrays:
        np.testing.assert_array_equal(arrays['target_padded'][arrays['valid_mask']],target)
        np.testing.assert_array_equal(arrays['probability_padded'][arrays['valid_mask']],probability)
        assert np.isnan(arrays['probability_padded'][~arrays['valid_mask']]).all()
