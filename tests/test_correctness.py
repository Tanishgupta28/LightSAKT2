import importlib.util
import json
import numpy as np
import pandas as pd
import pytest
import torch
from src.models.light_sakt import LightSAKT
from src.models.factory import build_model, predict
from src.datasets.preprocess import split_students, encode_records
from src.datasets.sequences import Sequences
from src.evaluation.metrics import masked_bce, metrics
from src.training.reproducibility import seed_everything


def original_model():
    spec = importlib.util.spec_from_file_location("historical_light", "original_snapshot/train/models/light_sakt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Light_SAKT


def batch(b=2, length=8):
    g = torch.Generator().manual_seed(31)
    return (torch.randint(0, 7, (b, length), generator=g), torch.randint(0, 2, (b, length), generator=g),
            torch.randint(0, 7, (b, length), generator=g), torch.randint(0, 2, (b, length), generator=g),
            torch.ones(b, length, dtype=torch.bool))


def test_architecture_parameter_names_and_shapes():
    old, new = original_model()(7, 16, 2, .1, 20), LightSAKT(7, 16, 2, .1, 20)
    assert {k: tuple(v.shape) for k,v in old.state_dict().items()} == {k: tuple(v.shape) for k,v in new.state_dict().items()}
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())
    assert new.FFN[0].out_features == 8


def test_original_future_leakage_regression():
    seed_everything(42)
    old = original_model()(7, 16, 2, 0, 8).eval()
    new = LightSAKT(7, 16, 2, 0, 8).eval()
    new.load_state_dict(old.state_dict())
    q, r, qry, _, valid = batch()
    r2 = r.clone(); r2[:, 4:] = 1-r2[:, 4:]
    assert (old(q,r,qry)[0][:,:4]-old(q,r2,qry)[0][:,:4]).abs().max() > 1e-6
    torch.testing.assert_close(new(q,r,qry,valid)[0][:,:4], new(q,r2,qry,valid)[0][:,:4], atol=1e-7, rtol=1e-6)


def test_kernel_dense_reference_and_gradients():
    from src.attention.linear import LinearAttention
    seed_everything(3)
    a = LinearAttention(8, 2, 0).double()
    inputs = [torch.randn(2,6,8,dtype=torch.double,requires_grad=True) for _ in range(3)]
    valid = torch.tensor([[1,1,0,1,1,0],[1,1,1,1,0,0]],dtype=torch.bool)
    out, w = a(*inputs, valid, True)
    q,k,v = [proj(x).reshape(2,6,2,4).transpose(1,2) for proj,x in zip([a.q_proj,a.k_proj,a.v_proj],inputs)]
    q,k = torch.nn.functional.elu(q)+1, torch.nn.functional.elu(k)+1
    scores = q @ k.transpose(-1,-2)
    allowed = torch.ones(6,6,dtype=torch.bool).tril()[None,None] & valid[:,None,None,:] & valid[:,None,:,None]
    scores = scores.masked_fill(~allowed,0)
    weights = scores/(scores.sum(-1,keepdim=True)+1e-8)
    ref = a.out_proj((weights @ v).transpose(1,2).reshape(2,6,8)).masked_fill(~valid[...,None],0)
    torch.testing.assert_close(out,ref,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(w,weights,atol=1e-10,rtol=1e-10)
    for x,y in zip(torch.autograd.grad(out.sum(),inputs,retain_graph=True),torch.autograd.grad(ref.sum(),inputs)):
        torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(w.sum(-1)[valid[:,None,:].expand(-1,2,-1)],torch.ones(int(valid.sum())*2,dtype=torch.double),atol=1e-8,rtol=1e-8)
    assert not w.masked_select(~allowed).any()


CONFIGS = {"LightSAKT": {"embed_dim":16,"num_heads":2,"max_seq_len":110},
           "SAKT": {"d":20,"num_attn_heads":2,"n":110,"dropout":0.2},
           "DKT": {"emb_size":10,"hidden_size":10},
           "DKT+": {"emb_size":10,"hidden_size":10,"lambda_r":.01,"lambda_w1":.003,"lambda_w2":3.},
           "DKVMN": {"dim_s":8,"size_m":5},
           "GKT": {"hidden_size":6,"num_attn_heads":2,"method":"PAM"},
           "GKT_MHA": {"hidden_size":6,"num_attn_heads":2,"method":"MHA"}}


@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("b,length", [(1,1),(2,8),(1,100)])
def test_all_models_forward_backward_and_causality(name,b,length):
    seed_everything(42)
    m = build_model(name,7,CONFIGS[name]).eval()
    data = batch(b,length)
    p,_ = predict(m,name,data)
    assert p.shape == (b,length) and torch.isfinite(p).all()
    masked_bce(p,data[3],data[4]).backward()
    used = [x.grad for x in m.parameters() if x.grad is not None]
    assert used and all(torch.isfinite(x).all() for x in used)
    if length>1:
        altered=list(data); altered[1]=1-data[1]; altered[1][:,:length//2]=data[1][:,:length//2]
        altered[0]=data[0].clone();altered[0][:,length//2:]=(data[0][:,length//2:]+1)%7
        torch.testing.assert_close(p[:,:length//2],predict(m,name,altered)[0][:,:length//2],atol=1e-7,rtol=1e-6)


@pytest.mark.parametrize("name", CONFIGS)
def test_padding_replacement_extension_batch_independence(name):
    seed_everything(42)
    m=build_model(name,7,CONFIGS[name]).eval()
    data=list(batch());data[4][:,5:]=False
    p=predict(m,name,data)[0]
    changed=[x.clone() for x in data];changed[0][~data[4]]=5000;changed[1][~data[4]]=-9;changed[2][~data[4]]=7000
    torch.testing.assert_close(p[:,:5],predict(m,name,changed)[0][:,:5],atol=1e-7,rtol=1e-6)
    extended=[torch.cat([x,torch.zeros(2,4,dtype=x.dtype)],1) for x in data]
    torch.testing.assert_close(p[:,:5],predict(m,name,extended)[0][:,:5],atol=1e-7,rtol=1e-6)
    alone=[x[:1] for x in data]
    torch.testing.assert_close(p[:1,:5],predict(m,name,alone)[0][:,:5],atol=1e-7,rtol=1e-6)


def test_attention_visualization_does_not_change_prediction():
    m=LightSAKT(7,16,2,0,8).eval();q,r,qry,_,valid=batch();valid[:,6:]=False
    p,_=m(q,r,qry,valid);p2,w=m(q,r,qry,valid,get_attn_weights=True)
    torch.testing.assert_close(p,p2,atol=0,rtol=0)
    assert not w[...,6:].any() and not w[:,:,6:,:].any()


def test_checkpoint_equivalence(tmp_path):
    m=LightSAKT(7,16,2,0,8).eval();data=batch();p=predict(m,"LightSAKT",data)[0]
    torch.save(m.state_dict(),tmp_path/"model.pt")
    restored=LightSAKT(7,16,2,0,8).eval();restored.load_state_dict(torch.load(tmp_path/"model.pt",weights_only=True))
    torch.testing.assert_close(p,predict(restored,"LightSAKT",data)[0],atol=0,rtol=0)


def test_split_chronology_alignment_and_full_coverage(tmp_path):
    assignments=split_students(range(20));assert assignments==split_students(range(20))
    sets=list(map(set,assignments.values()));assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    frame=pd.DataFrame({"student":["a"]*13,"q":np.arange(13)%7,"label":np.arange(13)%2,"split":["train"]*13})
    path=tmp_path/"data.csv";frame.to_csv(path,index=False);dataset=Sequences(path,"train",5)
    assert dataset.n_targets==12
    for sample,ident in zip(dataset.samples,dataset.identities):
        q,r,qry,t,valid=sample;start=ident['start'];n=ident['n_targets']
        assert q[:n].tolist()==frame.q[start:start+n].tolist()
        assert qry[:n].tolist()==frame.q[start+1:start+n+1].tolist()
        assert t[:n].tolist()==frame.label[start+1:start+n+1].tolist()
        # Chronological indices, not values, establish no corresponding future target response input.
        assert all(start+j < start+j+1 for j in range(n))


def test_preprocessing_fits_training_only():
    frame=pd.DataFrame({"student":["a","a","b","c"],"skill":["x","z","heldout","test"],"label":[1,0,0,1]})
    splits={"train":["a"],"validation":["b"],"test":["c"]}
    encoded,vocab=encode_records(frame,splits)
    changed=frame.copy();changed.loc[2:,'skill']=['other','another'];changed.loc[2:,'label']=1-changed.loc[2:,'label']
    encoded2,vocab2=encode_records(changed,splits)
    assert vocab==vocab2 and encoded.q[:2].equals(encoded2.q[:2])
    assert encoded.q[2:].tolist()==[0,0]


def test_masked_bce_metrics_and_undefined_auc():
    p=torch.tensor([[.8,.2,float('nan')]],requires_grad=True);t=torch.tensor([[1,0,-9]]);v=torch.tensor([[True,True,False]])
    loss=masked_bce(p,t,v);assert torch.isfinite(loss);loss.backward();assert p.grad[0,2]==0
    result=metrics([1,0],[.8,.2]);assert result['auc']==1 and result['mse']==pytest.approx(.04)
    assert metrics([1,1],[.8,.2])['auc'] is None
    with pytest.raises(ValueError): metrics([1],[float('nan')])


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_cuda_cpu_kernel_agreement():
    seed_everything(42)
    m=LightSAKT(7,16,2,0,8).eval();data=batch();p=predict(m,"LightSAKT",data)[0]
    gpu=predict(m.cuda(),"LightSAKT",[x.cuda() for x in data])[0].cpu()
    torch.testing.assert_close(p,gpu,atol=2e-6,rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
@pytest.mark.parametrize("name", CONFIGS)
def test_cuda_baselines_backward_repeatable(name):
    results=[]
    for _ in range(2):
        seed_everything(42)
        m=build_model(name,7,CONFIGS[name]).cuda().train()
        data=[x.cuda() for x in batch(1,8)]
        p,_=predict(m,name,data);masked_bce(p,data[3],data[4]).backward()
        assert all(torch.isfinite(x.grad).all() for x in m.parameters() if x.grad is not None)
        results.append(p.detach())
    torch.testing.assert_close(*results,atol=0,rtol=0)
