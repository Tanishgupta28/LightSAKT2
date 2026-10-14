import json
from pathlib import Path

import numpy as np
import pytest

from scripts.locked_test_evidence import LockedTestEvidence
from src.datasets.preprocess import sha256
from src.training.engine import write_json


def fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path=Path('data/processed/toy/interactions.csv');path.parent.mkdir(parents=True)
    path.write_text('student,split,q,label\na,train,1,0\na,train,2,1\nb,validation,1,0\nb,validation,2,1\nc,test,1,0\nc,test,2,1\nc,test,1,0\n')
    split=Path('data/splits/toy.json');write_json(split,{'train':['a'],'validation':['b'],'test':['c']})
    identities=Path('results/raw/run/prediction_identity.json')
    write_json(identities,[{'student':'c','start':0,'n_targets':2}])
    Path('results/predictions').mkdir(parents=True)
    predictions='results/predictions/run.npz';np.savez(predictions,target=np.array([1,0]),probability=np.array([.7,.2]))
    result={'run_id':'run','dataset':'toy','processed_sha256':sha256(path),'split_sha256':sha256(split),
            'prediction_path':predictions,'test':{'n_targets':2}}
    return result,{'seq_len':2},identities


def test_locked_test_rejects_changed_target_labels(tmp_path,monkeypatch):
    result,config,_=fixture(tmp_path,monkeypatch)
    checked=LockedTestEvidence();assert checked.check(result,config)['locked_test_labels_verified']
    np.savez(result['prediction_path'],target=np.array([0,1]),probability=np.array([.7,.2]))
    with pytest.raises(AssertionError,match='locked test targets'):checked.check(result,config)


def test_locked_test_rejects_inference_from_other_students(tmp_path,monkeypatch):
    result,config,path=fixture(tmp_path,monkeypatch)
    write_json(path,[{'student':'a','start':0,'n_targets':2}])
    with pytest.raises(AssertionError,match='identity mismatch'):LockedTestEvidence().check(result,config)


def test_locked_test_rejects_overlap_even_with_updated_manifest_hash(tmp_path,monkeypatch):
    result,config,_=fixture(tmp_path,monkeypatch)
    path=Path('data/splits/toy.json');write_json(path,{'train':['a','c'],'validation':['b'],'test':['c']})
    result['split_sha256']=sha256(path)
    with pytest.raises(AssertionError,match='Student split overlap'):LockedTestEvidence().check(result,config)
