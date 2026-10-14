"""Verify finished LightSAKT tables/plots against immutable experiment evidence."""
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.light_grid_results import generate as verify_grid
from scripts.verify_artifacts import verify_stored_result
from scripts.locked_test_evidence import LockedTestEvidence
from scripts.light_final_report import SEEDS,METRICS,seed_statistics,table
from src.datasets.preprocess import sha256
from src.training.engine import write_json


def main():
    grid=verify_grid();assert grid['validation_checkpoint_inference_recomputed']
    manifest=json.load(open('experiments/manifest.json'));checked=set();locked=LockedTestEvidence();records=[]
    inference={r['run_id']:r for r in json.load(open('reports/selected_final_inference_verification.json'))['runs']}
    jobs=[j for j in manifest['jobs'] if j['phase']==8 or j['phase']==7 and j['model']=='LightSAKT']
    assert len(jobs)==90
    for job in jobs:
        path=Path('results/raw')/job['run_id'];assert json.load(open(path/'status.json'))['status']=='completed'
        result,config,_=verify_stored_result(path/'metrics.json',checked);identities=locked.check(result,config)
        proof=inference[job['run_id']]
        assert proof['verified'] and proof['checkpoint_sha256']==result['checkpoint_sha256']
        assert proof['metrics_sha256']==sha256(path/'metrics.json') and proof['prediction_sha256']==result['prediction_sha256']
        variant='default' if job['phase']==7 else config.get('scientific_variant',config['model_config']['variant'])
        records.append({'dataset':config['dataset'],'variant':variant,'seed':result['seed'],'run_id':job['run_id'],
                        'test':result['test'],'checkpoint':result['checkpoint'],'checkpoint_sha256':result['checkpoint_sha256'],
                        'training_git_commit':result['git_commit'],'metrics_sha256':sha256(path/'metrics.json'),**identities})
    summary=pd.read_csv('results/tables/Table_8.csv');assert len(summary)==18 and (summary.completed_seeds==5).all()
    for row in summary.to_dict('records'):
        source=[r for r in records if r['dataset']==row['dataset'] and r['variant']==row['variant']]
        actual=seed_statistics(source)
        for metric in METRICS:
            np.testing.assert_allclose([row[metric+'_mean'],row[metric+'_std']],
                                       [actual[metric+'_mean'],actual[metric+'_std']],rtol=1e-12,atol=1e-14)
    artifacts=[]
    for dataset in ['ASSIST2009','ASSIST2015','Algebra2005']:
        group=summary[summary.dataset==dataset].set_index('variant')
        for metric in METRICS:
            assert group.loc['no_dropout',metric+'_mean']==group.loc['no_regularization',metric+'_mean']
            name=dataset+'_ablation_'+metric;plot=pd.read_csv('results/figures/'+name+'.csv').set_index('variant')
            assert (plot['count']==5).all()
            for variant in plot.index:
                np.testing.assert_allclose([plot.loc[variant,'mean'],plot.loc[variant,'std']],
                    [group.loc[variant,metric+'_mean'],group.loc[variant,metric+'_std']],rtol=1e-12,atol=1e-14)
            for ext in ['csv','png','svg','pdf']:
                path=Path('results/figures')/(name+'.'+ext);assert path.exists() and path.stat().st_size
                artifacts.append({'path':str(path),'sha256':sha256(path)})
        grid_frame=pd.read_csv('results/tables/'+{'Algebra2005':'Table_4.csv','ASSIST2009':'Table_5.csv','ASSIST2015':'Table_6.csv'}[dataset])
        expected=grid_frame.pivot_table(index='embed_dim',columns='learning_rate',values='validation_auc',aggfunc='max')
        observed=pd.read_csv('results/figures/'+dataset+'_grid_heatmap.csv',index_col=0)
        np.testing.assert_allclose(observed.values,expected.values,rtol=1e-12,atol=1e-14)
        for ext in ['csv','png','svg','pdf']:
            path=Path('results/figures')/(dataset+'_grid_heatmap.'+ext);assert path.exists() and path.stat().st_size
            artifacts.append({'path':str(path),'sha256':sha256(path)})
    now=datetime.now(timezone.utc).isoformat()
    write_json('reports/light_outputs_verification.json',{'verified':True,'verified_utc':now,'grid_rows':108,
               'ablation_slots':75,'fresh_component_ablations':60,'equivalent_aliases':15,'default_reference_final_seeds':15,
               'summary_rows':18,'numerical_plot_sets':18,'plots_and_source_data':artifacts,'source_runs':records,
               'architecture_changed':False,'experimental_metrics_changed':False})
    shown=summary[['dataset','variant']].copy()
    for metric in METRICS:shown[metric.upper()]=summary.apply(lambda r:f'{r[metric+"_mean"]:.6f} ± {r[metric+"_std"]:.6f}',axis=1)
    lines=['# Verified LightSAKT grids and ablations','',f'Generated UTC: {now}','',
           'All 108 unique validation-grid rows and their checkpoint inference are verified. Selection remains validation-only. Tables 4–6 retain all 36 candidates per dataset; no test result selected a configuration.','',
           'All 75 logical ablation slots passed original-GPU validation/test inference. Every saved test probability reproduced bit-for-bit. There are 60 fresh component-removal trainings and 15 equivalent no-regularization aliases. The default reference uses the 15 selected final seeds; it is not retrained. Split seed 42 and the five prescribed training seeds are unchanged.','',
           table(shown,['dataset','variant']+[m.upper() for m in METRICS]),'',
           'Mean ± sample SD. All individual values and checkpoint/Git references are in results/tables/Table_8_partial.csv and reports/light_outputs_verification.json. Table_8.csv contains the complete 18-row summary and relative changes of means. paired_ablations.csv instead records per-seed relative changes; their averaged ratios need not equal the ratio of means.','',
           'No regularization equals no dropout by author instruction. Their weights, probabilities and metrics are the same evidence; these rows are not independent samples. Their locked inference identities are resolved through checksum-bound physical source artifacts.','',
           'The ablations do not universally favor the default. Single-head attention has higher mean AUC on ASSIST2015 and Algebra2005; removing positions has higher mean AUC on Algebra2005. These five-seed descriptive effects do not establish conventional statistical significance.','',
           'The three grid heatmaps show the maximum validation AUC over heads/dropout for each learning-rate/width cell, with the full 36-row data retained in each grid table. All five ablation metrics have PNG/SVG/PDF plots and source CSVs for all three datasets. Source values, means, sample SDs and counts were checked against immutable metrics.','',
           'The CPU no-position validation-AUC discrepancy was retained as a failed strict CPU metric comparison. Original-GPU inference reproduced the saved metrics; the measured cross-device probability/rank-bound diagnosis is in reports/ablation_cpu_gpu_ranking_diagnostic.json. No tolerance was loosened, checkpoint rewritten or experiment retrained.','',
           'Source evidence: reports/light_grid_inference_verification.json; selected_final_inference_verification.json; light_outputs_verification.json; results/tables/Table_4.csv–Table_6.csv and Table_8.csv. Full project completion and remaining baseline analyses are still pending.']
    Path('reports/LIGHTSAKT_GRID_ABLATION_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('Verified grids, ablations and plots:',108,75,len(artifacts))


if __name__=='__main__':main()
