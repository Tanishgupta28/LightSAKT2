import copy
import json
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader
from scripts.optimization_audit import coarse_gkt
from scripts.sequence_cache import build,CachedSequences
from src.datasets.sequences import Sequences
from tests.test_training import fixture
from src.training.engine import run


def test_gkt_coarse_proposal_is_prespecified_not_metric_selected():
    assert coarse_gkt({'model':'GKT','model_config':{'hidden_size':30}})
    assert not coarse_gkt({'model':'GKT','model_config':{'hidden_size':104}})
    assert coarse_gkt({'model':'GKT_MHA','model_config':{'hidden_size':64,'num_attn_heads':2}})
    assert not coarse_gkt({'model':'GKT_MHA','model_config':{'hidden_size':64,'num_attn_heads':8}})
    assert coarse_gkt({'model':'LightSAKT','model_config':{}})


def test_lossless_cache_keeps_exact_inputs_identities_and_shuffle(tmp_path):
    c=fixture(tmp_path);path=tmp_path/'data/processed/toy/interactions.csv'
    for split in ['train','validation','test']:
        build(path,split,c['seq_len']);a=Sequences(path,split,c['seq_len']);b=CachedSequences(path,split,c['seq_len'])
        assert a.identities==b.identities and a.n_targets==b.n_targets and len(a)==len(b)
        for original,cached in zip(a,b):
            for x,y in zip(original,cached):assert torch.equal(x,y) and x.dtype==y.dtype
        gen1=torch.Generator().manual_seed(42);gen2=torch.Generator().manual_seed(42)
        for x,y in zip(DataLoader(a,batch_size=4,shuffle=True,generator=gen1),DataLoader(b,batch_size=4,shuffle=True,generator=gen2)):
            assert all(torch.equal(u,v) for u,v in zip(x,y))
    changed=b[0];changed[0].zero_();assert torch.equal(b[0][0],a[0][0])


def test_cache_preserves_complete_training_and_resume_updates(tmp_path,monkeypatch):
    a,b=tmp_path/'reference',tmp_path/'cached';ca,cb=fixture(a),fixture(b)
    original=run(ca,a)
    for split in ['train','validation']:build(b/'data/processed/toy/interactions.csv',split,cb['seq_len'])
    monkeypatch.setattr('src.training.engine.Sequences',CachedSequences)
    run(cb,b,stop_after_epoch=1);cached=run(cb,b)
    before=torch.load(a/'checkpoints/toy/latest.pt',weights_only=False)
    after=torch.load(b/'checkpoints/toy/latest.pt',weights_only=False)
    assert original['validation']==cached['validation']
    for name,value in before['model'].items():assert torch.equal(value,after['model'][name])
    assert torch.equal(before['rng']['torch'],after['rng']['torch'])
    assert torch.equal(before['loader_rng'],after['loader_rng'])


def test_cache_corruption_is_rejected(tmp_path):
    c=fixture(tmp_path);path=tmp_path/'data/processed/toy/interactions.csv';directory=Path(build(path,'train',c['seq_len']))
    (directory/'q.npy').write_bytes(b'corrupted')
    with pytest.raises(AssertionError,match='corruption'):CachedSequences(path,'train',c['seq_len'])


def test_every_light_grid_is_retained_in_both_schedules():
    complete=json.load(open('experiments/complete_schedule.json'));proposal=json.load(open('experiments/accelerated_schedule_proposal.json'))
    a={j['run_id'] for j in complete['jobs'] if j['phase']==6 and '-LightSAKT-grid-' in j['run_id']}
    b={j['run_id'] for j in proposal['jobs'] if j['phase']==6 and '-LightSAKT-grid-' in j['run_id']}
    assert len(a)==108 and a==b
    assert {j['run_id'] for j in complete['jobs'] if j['phase'] in [5,7,8]}=={j['run_id'] for j in proposal['jobs'] if j['phase'] in [5,7,8]}
    assert proposal['approval'].startswith('NOT APPROVED')
