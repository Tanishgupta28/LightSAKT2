import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from src.datasets.preprocess import sha256
from src.datasets.sequences import Sequences
from src.models.factory import build_model
from src.training.reproducibility import seed_everything
from scripts.analyze import save_figure
from src.training.engine import write_json


def visualize(run_id):
    directory=Path('results/raw')/run_id;c=json.load(open(directory/'config.json'));result=json.load(open(directory/'metrics.json'))
    if c['model']!='LightSAKT':raise ValueError('LightSAKT checkpoint required')
    stats=json.load(open(f"reports/{c['dataset']}_statistics.json"));path=Path('data/processed')/c['dataset']/'interactions.csv'
    assert sha256(path)==result['processed_sha256'] and sha256(result['checkpoint'])==result['checkpoint_sha256']
    seed_everything(c['seed']);model=build_model('LightSAKT',stats['num_q'],c['model_config']).eval()
    state=torch.load(result['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(state['model'])
    ds=Sequences(path,'test',c['seq_len'])
    # Longest available chunk, then lexical student and start; never select using scores or response labels.
    index=min(range(len(ds)),key=lambda i:(-ds.identities[i]['n_targets'],ds.identities[i]['student'],ds.identities[i]['start']))
    q,r,qry,target,valid=[x[None] for x in ds[index]];length=int(valid.sum())
    with torch.no_grad():
        p,_=model(q,r,qry,valid);p2,weights=model(q,r,qry,valid,get_attn_weights=True)
    torch.testing.assert_close(p,p2,atol=0,rtol=0)
    w=weights[0,:,:length,:length].mean(0).numpy()
    assert np.allclose(w.sum(-1),1,atol=2e-6) and not np.triu(w,1).any()
    prefix=f"{c['dataset']}_{c['group']}_{c['seed']}"
    Path('experiments/case_studies').mkdir(parents=True,exist_ok=True)
    metadata={'run_id':run_id,'dataset':c['dataset'],'checkpoint':result['checkpoint'],'checkpoint_sha256':result['checkpoint_sha256'],
              'sample':ds.identities[index],'sample_index':index,'selection':'longest heldout chunk, lexical student and start tie break; no label or score criterion',
              'attention':'head-averaged normalized ELU+1 causal kernel; no softmax',
              'probability_interpretation':'predicted correctness conditional on observed history; not directly observed mastery',
              'context':'within chunk only; state resets at chunk boundary'}
    np.savez_compressed(f'experiments/case_studies/{prefix}_attention.npz',weights=w,q=q.numpy(),r=r.numpy(),query=qry.numpy(),target=target.numpy(),probability=p.numpy(),valid=valid.numpy())
    pd.DataFrame(w).to_csv(f'results/figures/Figure_9_{prefix}.csv',index_label='target_query_index')
    plt.figure(figsize=(6,5));shown=min(25,length);plt.imshow(w[:shown,:shown],cmap='viridis',vmin=0);plt.colorbar(label='Normalized kernel attention');plt.xlabel('History interaction j');plt.ylabel('Target query after interaction t');plt.title('LightSAKT causal attention — '+c['dataset']);save_figure('Figure_9_'+prefix)
    if c['dataset']=='ASSIST2009':
        vocab=json.load(open('data/processed/ASSIST2009/vocabulary.json'));reverse={v:k for k,v in vocab.items()}
        raw=pd.read_csv('data/raw/ASSIST2009.csv',encoding='latin1',usecols=['skill_id','skill_name'])
        requested={}
        for original_id in [98,55,45,33,32]:
            names=sorted(raw.loc[pd.to_numeric(raw.skill_id,errors='coerce')==original_id,'skill_name'].dropna().unique().tolist())
            matches={name:vocab[name] for name in names if name in vocab}
            requested[str(original_id)]={'original_names':names,'direct_training_vocabulary_matches':matches}
        observed=sorted(set(q[0,:length].tolist())-{0},key=lambda x:(-int((q[0,:length]==x).sum()),reverse[x]))[:5]
        trajectories=[]
        with torch.no_grad():
            for skill in observed:
                probe=torch.full_like(q,skill)
                predictions=model(q,r,probe,valid)[0][0,:length].numpy()
                trajectories.append(predictions)
        matrix=np.stack(trajectories)
        pd.DataFrame(matrix.T,columns=[reverse[x] for x in observed]).to_csv(f'results/figures/{prefix}_skill_probabilities.csv',index_label='history_index')
        pd.DataFrame(matrix.T,columns=[reverse[x] for x in observed]).to_csv(f'results/figures/Figure_10_{prefix}.csv',index_label='history_index')
        actual=np.full_like(matrix,np.nan)
        for i,skill in enumerate(observed):
            selection=(qry[0,:length].numpy()==skill);actual[i,selection]=target[0,:length].numpy()[selection]
        np.savez_compressed(f'experiments/case_studies/{prefix}_skill_probabilities.npz',probabilities=matrix,skills=np.array(observed),actual_next_response=actual)
        trajectory_rows=[{'history_index':t,'encoded_skill':skill,'skill_name':reverse[skill],
                          'predicted_correctness':float(matrix[i,t]),'actual_next_response':float(actual[i,t]) if np.isfinite(actual[i,t]) else None}
                         for i,skill in enumerate(observed) for t in range(length)]
        pd.DataFrame(trajectory_rows).to_csv(f'results/figures/Figure_11_{prefix}.csv',index=False)
        metadata.update(requested_original_skill_ids=requested,selected_encoded_skills={str(x):reverse[x] for x in observed},
                        skill_selection='up to five most frequent observed encoded categories in the selected heldout chunk, lexical category tie break; requested historical IDs audited separately')
        plt.figure(figsize=(10,4));plt.imshow(matrix,aspect='auto',vmin=0,vmax=1,cmap='viridis');plt.yticks(range(len(observed)),[reverse[x][:45] for x in observed],fontsize=8);plt.xlabel('Observed interactions completed (chunk index)');plt.colorbar(label='Predicted next-response correctness');plt.title('LightSAKT conditional correctness — heldout ASSIST2009');save_figure('Figure_10_'+prefix)
        plt.figure(figsize=(10,4))
        for i,skill in enumerate(observed):
            plt.plot(np.arange(length),matrix[i],label=reverse[skill][:40]);positions=np.flatnonzero(np.isfinite(actual[i]));plt.scatter(positions,actual[i,positions],s=12,alpha=.5)
        plt.ylim(0,1);plt.xlabel('Observed interactions completed (chunk index)');plt.ylabel('Predicted correctness probability');plt.title('LightSAKT skill queries; dots = corresponding actual next responses');plt.legend(fontsize=7);save_figure('Figure_11_'+prefix)
    write_json(f'experiments/case_studies/{prefix}_metadata.json',metadata)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);visualize(parser.parse_args().run_id)
