"""Paper tables from verified extracts; no experiment results hardcoded here."""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from common import ASSETS, DATASETS, MODELS, SEEDS, SHORT, DISPLAY, csv, write, table_exports


def formatted(row,key):
    if row['seed_count']!=5:return 'PENDING'
    return f'{row[key+"_mean"]:.6f} ± {row[key+"_std"]:.6f}'


def main():
    import json
    meta=json.loads((ASSETS/'source_data/dataset_provenance/dataset_metadata.json').read_text())
    raw_ednet=json.loads((ASSETS/'source_data/dataset_provenance/ednet_statistics.json').read_text())
    rows=[]
    for r in meta:
        rows.append({'Dataset':DISPLAY.get(r['dataset'],r['dataset']),
            'Raw rows':r.get('raw_rows',raw_ednet['filter_counts']['raw_rows']),
            'Raw scope':'selected 5k files' if r['dataset']==DATASETS[-1] else 'local raw release',
            'Processed interactions':r['clean_interactions'],'Students':sum(r['split_students'].values()),
            'Train/val/test students':'/'.join(str(r['split_students'][s]) for s in ['train','validation','test']),
            'Input skill vocabulary incl.UNK':r['num_q'],
            'Train/val/test targets':'/'.join(str(r['split_interactions'][s]-r['split_students'][s]) for s in ['train','validation','test'])})
    table_exports(pd.DataFrame(rows),'tables/dataset_overview/Table_2_dataset_overview','Table2. Actual dataset scope; targets=interactions−students')
    configs=pd.read_csv(ASSETS/'source_data/verified_metric_references/training_configurations.csv')
    rows=[]
    for (d,m),g in configs.groupby(['dataset','model'],sort=False):
        assert set(g.seed).issubset(SEEDS)
        cfg=json.loads(g.model_configuration.iloc[0])
        embedding=cfg.get('embed_dim',cfg.get('emb_size',cfg.get('d',cfg.get('dim_s',cfg.get('hidden_size')))))
        hidden=cfg.get('hidden_size','N/A')
        native_dropout={'DKT':'0.5 output logits','DKT+':'0.5 output logits','DKVMN':'none',
                        'GKT':'MLP0.5','GKT-MHA (implemented variant)':'MLP0.5; MHA0'}
        rows.append({'Dataset':DISPLAY.get(d,d),'Model':SHORT.get(m,m),'LR':g.learning_rate.iloc[0],
             'Batch':int(g.batch_size.iloc[0]),'Length':int(g.seq_len.iloc[0]),
             'Embedding / hidden':f'{embedding} / {hidden}',
             'Heads':cfg.get('num_heads',cfg.get('num_attn_heads','N/A')),
             'Dropout':cfg.get('dropout',native_dropout.get(m,'see exact configuration')),
             'Epochs (actual range)':f'{g.epochs_trained.min()}–{g.epochs_trained.max()}',
             'Ceiling/patience':f'{int(g.max_epochs.iloc[0])}/{int(g.patience.iloc[0])}',
             'Measured seeds':f'{len(g)}/5'})
    table_exports(pd.DataFrame(rows),'tables/training_configurations/Table_3_training_configurations',
                  'Table3. Measured configurations (GKT1/5 separately marked); Adam; exact per-seed settings in source_data')
    for d,num in [('Algebra2005',4),('ASSIST2009',5)]:
        grid=pd.read_csv(ASSETS/f'source_data/verified_metric_references/LightSAKT_grid_{d}.csv')
        grid=grid[['learning_rate','embedding_dim','heads','dropout','LightSAKT_validation_auc','LightSAKT_best_epoch']]
        table_exports(grid,f'tables/hyperparameter_tuning/Table_{num}_{d}_LightSAKT_grid',
                      f'Table{num}. {d}: genuine36 seed42 validation-only candidates, not independent repetitions')
    selection=pd.DataFrame([{'Dataset':DISPLAY.get(d,d),'Model':SHORT.get(m,m),
        'Selection protocol':'Prespecified common d64,heads2,dropout0.1,LR0.001; no EdNet grid',
        'Checkpoint criterion':'maximum validation AUC','Epoch ceiling/patience':'50/5','Seeds':5}
        for d,m in [(DATASETS[-1],m) for m in ['SAKT','LightSAKT','LightSAKT + Question Difficulty']]])
    table_exports(selection,'tables/hyperparameter_tuning/Table_6_EdNet_actual_selection_protocol',
                  'Table6 replacement: actual EdNet selection protocol; not a fabricated hyperparameter grid')
    master=pd.read_csv(ASSETS/'source_data/verified_metric_references/master_metrics.csv')
    csv(ASSETS/'tables/full_model_comparison/master_comparison_all_metrics.csv',master)
    columns={'validation_auc':'Validation AUC','test_auc':'Test AUC','test_accuracy':'Accuracy','test_f1':'F1',
             'test_mse':'MSE','test_bce':'BCE','test_precision':'Precision','test_recall':'Recall'}
    display=[]
    for r in master.to_dict('records'):
        display.append({'Dataset':DISPLAY.get(r['dataset'],r['dataset']),'Model':SHORT.get(r['model'],r['model']),
                       **{label:formatted(r,k) for k,label in columns.items()},'Seeds/status':f'{r["seed_count"]}/5 '+r['status']})
    table_exports(pd.DataFrame(display),'tables/full_model_comparison/Table_7_full_model_comparison',
       'Table7. Actual mean ± sample SD; fixed split, five initializations; PENDING has no aggregate',max_rows=14)
    wide=[]
    for d in DATASETS:
        group=master[master.dataset==d].set_index('model')
        for key,label in columns.items():
            wide.append({'Dataset':DISPLAY.get(d,d),'Metric':label,**{SHORT.get(m,m):formatted(group.loc[m],key) for m in MODELS}})
    table_exports(pd.DataFrame(wide),'tables/full_model_comparison/Table_7_original_structure',
                  'Table7 alternate layout: dataset/metric rows and model columns',max_rows=8)
    ab=pd.read_csv(ASSETS/'source_data/verified_metric_references/ablation_summary.csv')
    abrows=[]
    for variant in ab.variant.unique():
        row={'Variant':variant}
        for d in DATASETS:
            r=ab[(ab.dataset==d)&(ab.variant==variant)].iloc[0]
            row[DISPLAY.get(d,d)]=formatted(r,'test_auc') if r.seed_count==5 else 'UNAVAILABLE (not run)'
        abrows.append(row)
    # Original no-layer-normalization row has no corresponding corrected experiment.
    abrows.append({'Variant':'no_layer_normalization',**{DISPLAY.get(d,d):'UNAVAILABLE (not run)' for d in DATASETS}})
    table_exports(pd.DataFrame(abrows),'tables/ablation_study/Table_8_corrected_ablation',
       'Table8. Test AUC; item-difficulty addition distinct from removals; no_regularization aliases no_dropout')
    seeds=pd.read_csv(ASSETS/'source_data/verified_metric_references/individual_seed_metrics.csv')
    csv(ASSETS/'tables/full_model_comparison/individual_seed_results.csv',seeds)
    tests=[];pairs=[]
    comparisons=[('LightSAKT','SAKT'),('LightSAKT + Question Difficulty','LightSAKT'),
                 ('LightSAKT + Question Difficulty','SAKT')]
    for d in DATASETS:
        for a,b in comparisons:
            x=seeds[(seeds.dataset==d)&(seeds.model==a)].set_index('seed')
            y=seeds[(seeds.dataset==d)&(seeds.model==b)].set_index('seed')
            assert set(x.index)==set(y.index)==set(SEEDS)
            delta=np.array([x.loc[s,'test_auc']-y.loc[s,'test_auc'] for s in SEEDS])
            assert np.count_nonzero(delta)==5  # exact no-zero small-n test
            result=wilcoxon(delta,alternative='two-sided',method='exact')
            tests.append({'dataset':d,'model_A':a,'model_B':b,'metric':'test_auc','n_matched_seeds':5,
                'W':float(result.statistic),'p_two_sided_exact':float(result.pvalue),
                'mean_paired_difference':float(delta.mean()),'sample_SD_difference':float(delta.std(ddof=1)),
                'positive_seeds':int((delta>0).sum()),'negative_seeds':int((delta<0).sum()),
                'design':'same fixed student split; independent initialization seeds; not cohort replication',
                'family':'9 prelisted seed-matched AUC comparisons; Holm correction'})
            for s,z in zip(SEEDS,delta):pairs.append({'dataset':d,'model_A':a,'model_B':b,'seed':s,'test_auc_difference':z})
    order=np.argsort([r['p_two_sided_exact'] for r in tests]);running=0
    for rank,i in enumerate(order):
        running=max(running,min(1,(len(tests)-rank)*tests[i]['p_two_sided_exact']))
        tests[i]['p_Holm']=running;tests[i]['reject_Holm_0_05']=running<.05
    csv(ASSETS/'tables/statistical_tests/matched_seed_AUC_observations.csv',pairs)
    csv(ASSETS/'tables/statistical_tests/Wilcoxon_matched_seed_AUC.csv',tests)
    small=pd.DataFrame(tests)[['dataset','model_A','model_B','n_matched_seeds','W','p_two_sided_exact','p_Holm','mean_paired_difference','positive_seeds','negative_seeds']]
    table_exports(small,'tables/statistical_tests/Wilcoxon_summary',
        'Exact two-sided paired Wilcoxon, n5; Holm9; minimum attainable no-tie p=0.0625')
    # Available tests against all complete comparators, descriptive only, separate from the declared9-test family.
    allgains=[]
    for d in DATASETS:
        a=seeds[(seeds.dataset==d)&(seeds.model=='LightSAKT + Question Difficulty')].set_index('seed')
        for m in MODELS:
            if m=='LightSAKT + Question Difficulty':continue
            b=seeds[(seeds.dataset==d)&(seeds.model==m)].set_index('seed')
            if len(b)!=5:continue
            diff=a.test_auc-b.test_auc
            allgains.append({'dataset':d,'baseline':m,'absolute_test_AUC_gain':diff.mean(),
                'sample_SD_paired_gain':diff.std(ddof=1),'positive_seeds':int((diff>0).sum()),'negative_seeds':int((diff<0).sum()),'n':5})
    csv(ASSETS/'tables/full_model_comparison/difficulty_absolute_AUC_gains.csv',allgains)
    write(ASSETS/'verification/metrics_verification/statistics_scope.json',{
        'unit':'initialization seed, conditional on one fixed split per dataset','n':5,
        'epochs_and_grid_candidates_not_repetitions':True,'two_sided_exact_minimum_p_n5':.0625,
        'Holm_family_size':9,'all_Holm_p_ge_0_05':all(t['p_Holm']>=.05 for t in tests),
        'old_manuscript_W78_p0_00048':'Not reproduced; underlying independent matched-seed observations unavailable. Not carried forward.',
        'interpretation':'Descriptive consistent differences do not establish population or attention-architecture superiority.'})
    print('Generated Tables2–8, individual seeds, genuine grids and exact matched-seed statistics.')


if __name__=='__main__':main()
