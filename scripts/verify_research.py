"""Reject missing or internally inconsistent final research outputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.progress import snapshot
from src.datasets.preprocess import sha256
from src.models.factory import build_model
from src.training.engine import write_json
from src.training.reproducibility import seed_everything
from src.datasets.sequences import Sequences


def require_table(path, rows=None):
    frame=pd.read_csv(path)
    if rows is not None:assert len(frame)==rows, f'{path}: expected {rows} rows, found {len(frame)}'
    return frame


def verify():
    progress=snapshot();assert progress['all_training_slots_completed'], 'Training schedule incomplete'
    tasks=json.load(open('experiments/analyses.json'))['tasks']
    for name in ['tables','statistics_and_performance_figures','efficiency','visualization_ASSIST2009',
                 'visualization_ASSIST2015','visualization_Algebra2005','recovery_archives','manuscript_outputs','full_tests','checkpoint_and_predictions']:
        assert tasks[name]['status']=='completed', f'Analysis not verified: {name}'
    all_runs=require_table('results/tables/all_verified_runs.csv')
    manifest=json.load(open('experiments/manifest.json'))
    assert set(j['run_id'] for j in manifest['jobs']) <= set(all_runs.run_id)
    require_table('results/tables/Table_2.csv',3);require_table('results/tables/Table_3.csv',21)
    for number in [4,5,6]:
        grid=require_table(f'results/tables/Table_{number}.csv',36)
        assert (grid.status=='completed').all() and grid.validation_auc.notna().all()
    final=require_table('results/tables/Table_7.csv',21)
    assert (final.completed_seeds==5).all()
    for metric in ['auc','accuracy','f1','mse','bce']:
        assert final[f'{metric}_mean'].notna().all() and final[f'{metric}_std'].notna().all()
    ablation=require_table('results/tables/Table_8.csv',18)
    assert (ablation.completed_seeds==5).all()
    for dataset,data in ablation.groupby('dataset'):
        indexed=data.set_index('variant')
        for metric in ['auc','accuracy','f1','mse','bce']:
            assert indexed.loc['no_regularization',f'{metric}_mean']==indexed.loc['no_dropout',f'{metric}_mean']
    significance=require_table('results/tables/significance.csv',18)
    assert (significance.n_pairs==5).all() and (significance.holm_p>=significance.p_value).all()
    timings=require_table('results/tables/efficiency.csv',120)
    assert (timings.latency_ms_mean>0).all() and (timings.throughput_tokens_per_second>0).all()
    require_table('results/tables/parameters.csv',21)
    require_table('results/tables/model_complexity.csv',2)
    require_table('results/tables/epoch_runtime.csv')
    figures=['Figure_1_conceptual','Figure_2_conceptual','Figure_3_conceptual','Figure_4_architecture',
             'Figure_5','Figure_6','Figure_7']
    for dataset in ['ASSIST2009','ASSIST2015','Algebra2005']:
        figures.extend([f'Figure_8_{dataset}',f'Figure_9_{dataset}_final_42'])
        metadata=json.load(open(f'experiments/case_studies/{dataset}_final_42_metadata.json'))
        result=json.load(open(Path('results/raw')/metadata['run_id']/'metrics.json'))
        config=json.load(open(Path('results/raw')/metadata['run_id']/'config.json'))
        stats=json.load(open(f'reports/{dataset}_statistics.json'))
        assert metadata['checkpoint_sha256']==sha256(metadata['checkpoint'])==result['checkpoint_sha256']
        seed_everything(config['seed']);model=build_model('LightSAKT',stats['num_q'],config['model_config']).eval()
        state=torch.load(metadata['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(state['model'])
        with np.load(f'experiments/case_studies/{dataset}_final_42_attention.npz') as arrays:
            weights=arrays['weights'];length=weights.shape[0]
            assert not np.triu(weights,1).any() and np.allclose(weights.sum(-1),1,atol=2e-6)
            tensors=[torch.from_numpy(arrays[key]) for key in ['q','r','query','valid']]
            ds=Sequences(f'data/processed/{dataset}/interactions.csv','test',config['seq_len'])
            assert ds.identities[metadata['sample_index']]==metadata['sample']
            for name,value in zip(['q','r','query','target','valid'],ds[metadata['sample_index']]):
                np.testing.assert_array_equal(arrays[name],value[None].numpy())
            with torch.no_grad():predictions,actual_weights=model(*tensors,get_attn_weights=True)
            np.testing.assert_array_equal(predictions.numpy(),arrays['probability'])
            assert length==int(arrays['valid'].sum())
            np.testing.assert_array_equal(actual_weights[0,:,:length,:length].mean(0).numpy(),weights)
            if dataset=='ASSIST2009':
                with np.load(f'experiments/case_studies/{dataset}_final_42_skill_probabilities.npz') as case:
                    for i,skill in enumerate(case['skills']):
                        with torch.no_grad():actual=model(tensors[0],tensors[1],torch.full_like(tensors[2],int(skill)),tensors[3])[0]
                        np.testing.assert_array_equal(actual[0,:length].numpy(),case['probabilities'][i])
    figures += ['Figure_10_ASSIST2009_final_42','Figure_11_ASSIST2009_final_42']
    for name in figures:
        for extension in ['png','svg','pdf']:
            path=Path('results/figures')/(name+'.'+extension)
            assert path.exists() and path.stat().st_size>0, f'Missing figure: {path}'
        if name.startswith(('Figure_5','Figure_6','Figure_7','Figure_8','Figure_9','Figure_10','Figure_11')):
            require_table(Path('results/figures')/(name+'.csv'))
    backups=json.load(open('reports/completed_run_backups.json'))
    assert set(j['run_id'] for j in manifest['jobs']) <= {r['run_id'] for r in backups}
    for entry in backups:
        assert sha256(entry['path'])==entry['sha256'], f'Backup checksum: {entry["path"]}'
        if entry.get('result_archive'):
            record=entry['result_archive'];assert sha256(record['path'])==record['sha256']
    verified=json.load(open('reports/artifact_verification.json'))
    assert set(j['run_id'] for j in manifest['jobs']) <= {r['run_id'] for r in verified if r['validation_recomputed']}
    assert Path('paper/CORRECTED_RESULTS.md').exists()
    for entry in json.load(open('paper/RESULT_ARTIFACTS.json')):
        assert sha256(entry['path'])==entry['sha256'], f'Paper artifact checksum: {entry["path"]}'
    write_json('reports/research_verification.json',{'verified':True,'scheduled_slots':len(manifest['jobs']),
               'verified_result_rows':len(all_runs),'verified_paper_figures':figures,
               'limits':'Local dataset provenance and historical simplified GKT limitations remain disclosed; seed intervals are conditional on one fixed split.'})
    print('All scheduled experiments and required final output checks passed.')


if __name__=='__main__':verify()
