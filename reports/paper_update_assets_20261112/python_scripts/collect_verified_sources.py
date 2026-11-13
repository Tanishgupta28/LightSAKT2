"""Read-only original-artifact audit. Never trains, performs inference or writes source artifacts.

Run from the research checkout. Portable plotting/table generation uses the
resulting compact extracts and does not require datasets or checkpoint binaries.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import json
import shutil
from statistics import mean, stdev
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from common import ASSETS, ROOT, DATASETS, MODELS, SEEDS, csv, sha, write
from scripts.build_master_comparison_20261110 import checkpoint_check, model_label
from src.models.factory import build_model

SOURCE = ROOT.parent / 'LightSAKT2'
MASTER = ROOT / 'reports/master_comparison_20261110'
METRICS = ['validation_auc', 'test_auc', 'test_accuracy', 'test_f1', 'test_precision', 'test_recall', 'test_mse', 'test_bce']
SOURCES = {}


def track(path):
    path = Path(path).resolve(); digest = sha(path)
    if str(path) in SOURCES: assert SOURCES[str(path)] == digest, ('source changed', path)
    SOURCES[str(path)] = digest
    return path


def read(path):
    return json.loads(track(path).read_text())


def metrics(y, p):
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    assert y.ndim == p.ndim == 1 and y.shape == p.shape
    assert np.isin(y, [0, 1]).all() and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    t, pred = y == 1, p >= .5
    tp, tn = int((t & pred).sum()), int((~t & ~pred).sum())
    fp, fn = int((~t & pred).sum()), int((t & ~pred).sum())
    n1, n0 = int(t.sum()), int((~t).sum()); assert n1 and n0
    pc = np.clip(p, 1e-7, 1-1e-7)
    out = {'auc': float((rankdata(p)[t].sum() - n1*(n1+1)/2)/(n1*n0)),
        'accuracy': (tp+tn)/len(t), 'f1': 2*tp/(2*tp+fp+fn) if tp else 0.,
        'precision': tp/(tp+fp) if tp+fp else 0., 'recall': tp/(tp+fn) if tp+fn else 0.,
        'mse': float(np.square(p-y).mean()), 'bce': float((-y*np.log(pc)-(1-y)*np.log(1-pc)).mean()),
        'n_targets': len(t), 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    for k, v in [('auc',roc_auc_score(y,p)),('accuracy',accuracy_score(y,pred)),
                  ('f1',f1_score(y,pred)),('precision',precision_score(y,pred,zero_division=0)),
                  ('recall',recall_score(y,pred,zero_division=0))]: assert abs(out[k]-v) < 1e-12
    return out


def predictions(artifact, expected):
    path = track(artifact['path']); assert sha(path) == artifact['sha256']
    with np.load(path) as a:
        y, p = a['target'], a['prediction'] if 'prediction' in a else a['probability']
    assert np.array_equal(y, expected), ('wrong target mask/order', path)
    return metrics(y,p), y, p


def aggregate(rows):
    out=[]
    assert len({(r['dataset'],r['model'],r['seed']) for r in rows}) == len(rows)
    for d in DATASETS:
        for model in MODELS:
            group=[r for r in rows if r['dataset']==d and r['model']==model]
            complete={r['seed'] for r in group} == set(SEEDS)
            record={'dataset':d,'model':model,'seed_count':len(group),'status':'VERIFIED_5_SEEDS' if complete else 'PENDING'}
            for m in METRICS:
                record[m+'_mean']=mean(r[m] for r in group) if complete else None
                record[m+'_std']=stdev(r[m] for r in group) if complete else None
            record['checkpoint_references']=';'.join(r['checkpoint'] for r in group)
            record['source_references']=';'.join(r['source_metrics'] for r in group)
            out.append(record)
    return out


def main():
    torch.set_num_threads(1)
    for folder in ['tables/dataset_overview','tables/training_configurations','tables/hyperparameter_tuning',
                   'tables/full_model_comparison','tables/ablation_study','tables/statistical_tests',
                   'graphs/ASSIST2009','graphs/Algebra2005','graphs/EdNet_KT1_5k','graphs/model_comparisons',
                   'graphs/ablations','graphs/parameters','graphs/student_knowledge','source_data/verified_metric_references',
                   'source_data/epoch_records','source_data/dataset_provenance','verification/checkpoint_verification',
                   'verification/metrics_verification','verification/resource_checks','verification/graph_reproducibility','documentation']:
        (ASSETS/folder).mkdir(parents=True,exist_ok=True)
    paper=track(ROOT/'research_paper/original_research_paper.pdf')
    provenance=read(MASTER/'experiment_provenance.json')
    new_replays=ASSETS/'verification/checkpoint_verification/new_GKT_CPU_replays.json'
    additional=read(new_replays) if new_replays.exists() else []
    completed_records=provenance['completed_records']+additional
    metadata=read(MASTER/'dataset_provenance.json'); stats={s['dataset']:s for s in metadata}
    frames,expected={},{}
    for d in DATASETS:
        work=ROOT/'.tools/ednet_replacement_20261104/workspace' if d==DATASETS[-1] else SOURCE
        data=track(work/f'data/processed/{d}/interactions.csv'); split=track(work/f'data/splits/{d}.json')
        assert sha(data)==stats[d]['processed_sha256'] and sha(split)==stats[d]['split_sha256']
        frame=pd.read_csv(data,dtype={'student':str}); frames[d]=frame
        sets=[set(frame.loc[frame.split==s,'student']) for s in ['train','validation','test']]
        assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
        for s in ['validation','test']:
            expected[d,s]=np.concatenate([g.label.to_numpy()[1:] for _,g in frame[frame.split==s].groupby('student',sort=True)])
    rows,epoch_rows,audits,configs,student_rows,prior_differences=[],[],[],[],[],[]
    old_individual=pd.read_csv(track(MASTER/'individual_seed_results.csv'))
    verification=provenance['verification']
    for r in completed_records:
        d=r['dataset']; model=model_label(r['model']); cfg=r['config']
        for key in ['checkpoint','config_file','recorded_metrics_path']:track(r[key])
        assert read(Path(r['recorded_metrics_path']).parent/'status.json')['status']=='completed'
        assert sha(r['config_file'])==r['config_sha256'] and sha(r['recorded_metrics_path'])==r['recorded_metrics_sha256']
        audit=checkpoint_check(r,stats); history=read(audit['history'])
        assert [h['epoch'] for h in history]==list(range(1,r['epochs_trained']+1))
        assert max(history,key=lambda h:h['validation']['auc'])['epoch']==r['best_epoch']
        tests,y,p=predictions(r['prediction_artifacts']['test'],expected[d,'test'])
        # Thirty old baseline arrays are the original GPU predictions; their
        # earlier CPU replay differs only in floating-point digits. Bind the
        # array to its actual reference, retain both origins, never rewrite it.
        array_reference=r.get('native_recorded_test',r['cpu_replay']['test'])
        for k in ['auc','accuracy','f1','bce']:assert abs(tests[k]-array_reference[k])<1e-12,(d,model,r['seed'],k)
        for k in ['auc','accuracy','f1','bce']:
            assert abs(tests[k]-r['test'][k])<2e-6
            prior_differences.append({'dataset':d,'model':model,'seed':r['seed'],'metric':'test_'+k,
                'existing_published_CPU_replay':r['test'][k],'current_saved_array_recomputation':tests[k],
                'difference':tests[k]-r['test'][k],'reason':'original GPU array versus prior CPU replay' if 'native_recorded_test' in r else 'same saved CPU array'})
        old_group=old_individual[(old_individual.dataset==d)&(old_individual.model==model)&(old_individual.seed==r['seed'])]
        old=old_group.iloc[0]
        if old.experiment_status=='VERIFIED':
            for k in ['auc','accuracy','f1']:assert abs(tests[k]-old['test_'+k])<2e-6
        if 'validation' in r['prediction_artifacts']:
            v,_,_=predictions(r['prediction_artifacts']['validation'],expected[d,'validation'])
            assert abs(v['auc']-r['validation']['auc'])<2e-6
        row={'dataset':d,'model':model,'seed':r['seed'],'status':'VERIFIED','validation_auc':r['validation']['auc'],
             **{'test_'+k:v for k,v in tests.items()},'parameters':r['parameters'],'best_epoch':r['best_epoch'],
             'epochs_trained':r['epochs_trained'],'training_seconds':r['training_seconds'],
             'checkpoint':r['checkpoint'],'checkpoint_sha256':r['checkpoint_sha256'],
             'run_id':r['run_id'],'source_metrics':r['recorded_metrics_path'],'metrics_sha256':r['recorded_metrics_sha256'],
             'test_predictions':r['prediction_artifacts']['test']['path'],
             'test_predictions_sha256':r['prediction_artifacts']['test']['sha256'],
             'processed_sha256':r['processed_sha256'],'split_sha256':r['split_sha256'],
             'training_commit':r.get('selected_git_commit',r.get('git_commit'))}
        row['recorded_training_code_sha256']=r.get('recorded_training_code_sha256',r.get('training_code_sha256'))
        rows.append(row);audits.append({'dataset':d,'model':model,'seed':r['seed'],**audit})
        config_path=ASSETS/f'source_data/verified_metric_references/configurations/{d}-{cfg["model"]}-{r["seed"]}.json'
        write(config_path,cfg)
        configs.append({'dataset':d,'model':model,'seed':r['seed'],'parameters':r['parameters'],
            'learning_rate':cfg['learning_rate'],'batch_size':cfg['batch_size'],'seq_len':cfg['seq_len'],
            'max_epochs':cfg['max_epochs'],'patience':cfg['patience'],'best_epoch':r['best_epoch'],
            'epochs_trained':r['epochs_trained'],'training_seconds':r['training_seconds'],
            'peak_GPU_allocated_bytes':r.get('peak_allocated_bytes'),'optimizer':'Adam',
            'model_configuration':json.dumps(cfg['model_config'],sort_keys=True),
            'optimizer_other':json.dumps({k:cfg[k] for k in ['betas','epsilon','weight_decay']}),
            'configuration_extract':str(config_path.relative_to(ASSETS)),
            'selection':'strict maximum validation AUC; patience5; test excluded'})
        for h in history:
            assert np.isfinite(h['train_objective']) and np.isfinite(h['validation']['auc'])
            epoch_rows.append({'dataset':d,'model':model,'seed':r['seed'],'run_id':r['run_id'],'epoch':h['epoch'],
                'training_objective':h['train_objective'],'objective_semantics':'BCE plus reconstruction/waviness regularization' if model=='DKT+' else 'masked BCE',
                **{'validation_'+k:v for k,v in h['validation'].items() if k in ['auc','accuracy','f1','mse','bce','n_targets']},
                'epoch_seconds':h.get('epoch_seconds'),'source_history':audit['history'],'source_sha256':sha(audit['history'])})
        if r['seed']==42 and model in ['SAKT','LightSAKT','LightSAKT + Question Difficulty']:
            offset=0
            for student,g in frames[d][frames[d].split=='test'].groupby('student',sort=True):
                n=max(0,len(g)-1)
                if n>=20:
                    token=__import__('hashlib').sha256(('paper-example-v1:'+d+':'+student).encode()).hexdigest()[:12]
                    for i in range(min(n,50)):
                        student_rows.append({'dataset':d,'anonymous_student':token,'model':model,'seed':42,
                           'target_interaction_index':i+2,'probability':float(p[offset+i]),'observed_correct':int(y[offset+i]),
                           'saved_prediction_sha256':row['test_predictions_sha256']})
                    break
                offset+=n
    assert len(rows)==75+len(additional)
    csv(ASSETS/'source_data/verified_metric_references/individual_seed_metrics.csv',rows)
    write(ASSETS/'source_data/verified_metric_references/experiment_provenance.json',{
        'completed_records':completed_records,'new_training_runs':0,'new_CPU_checkpoint_evaluations':len(additional),
        'historical_master_source':str(MASTER/'experiment_provenance.json'),
        'historical_master_sha256':sha(MASTER/'experiment_provenance.json'),
        'weights_and_full_probability_arrays':'separately preserved; not redistributed in assets'})
    csv(ASSETS/'verification/metrics_verification/prior_CPU_vs_saved_array_differences.csv',prior_differences)
    csv(ASSETS/'source_data/verified_metric_references/training_configurations.csv',configs)
    csv(ASSETS/'source_data/verified_metric_references/anonymized_prediction_examples.csv',student_rows)
    write(ASSETS/'verification/checkpoint_verification/completed_checkpoints.json',audits)
    csv(ASSETS/'source_data/verified_metric_references/master_metrics.csv',aggregate(rows))
    pending=read(ROOT/'experiments/master_comparison_20261110/plan.json')['entries']
    matrix=[{'dataset':r['dataset'],'model':r['model'],'seed':r['seed'],'state':'VERIFIED_COMPLETED',
             'checkpoint':r['checkpoint'],'config_status':'frozen','next_action':'none'} for r in rows]
    for entry in pending:
        if any((r['dataset'],r['model'],r['seed'])==(entry['dataset'],entry['model'],entry['seed']) for r in rows):continue
        action=entry['action'];state='REUSABLE_TRAINING_PENDING_EVALUATION' if 'REUSE' in action else ('MISSING_HYPERPARAMETER_SELECTION' if 'SELECTION' in action else 'PENDING_TRAINING')
        matrix.append({'dataset':entry['dataset'],'model':entry['model'],'seed':entry['seed'],'state':state,
            'checkpoint':entry.get('reusable_checkpoint',''),'config_status':entry.get('selection_protocol',''),
            'next_action':action,'resource_status':'BLOCKED_CURRENT_ALLOCATION_AND_TOTAL_BUDGET'})
    csv(ASSETS/'source_data/verified_metric_references/completion_matrix.csv',matrix)
    write(ASSETS/'source_data/verified_metric_references/missing_experiment_plan.json',{'entries':pending,'automatic_execution':False})
    # Recheck selected GKT weights; do not mistake these two validation-only runs for completed five-seed test groups.
    reusable=read(MASTER/'selected_reusable_gkt_checkpoints.json')
    for r in reusable:
        cp=track(r['checkpoint']);assert sha(cp)==r['checkpoint_sha256']
        state=torch.load(cp,map_location='cpu',weights_only=False)
        assert all(torch.isfinite(v).all() for v in state['model'].values())
        model=build_model(r['config']['model'],stats['ASSIST2009']['num_q'],r['config']['model_config'])
        model.load_state_dict(state['model'],strict=True)
        assert abs(state['validation']['auc']-r['best_validation_auc'])<1e-12
        history_path=Path(r['source']).parent/'history.json'
        for h in read(history_path):
            if any(x['dataset']=='ASSIST2009' and x['config']['model']==r['config']['model'] and x['seed']==42 for x in additional):continue
            epoch_rows.append({'dataset':'ASSIST2009','model':model_label(r['config']['model']),
                'seed':42,'run_id':r['config']['run_id'],'epoch':h['epoch'],
                'training_objective':h['train_objective'],'objective_semantics':'masked BCE; selected training only, test pending',
                **{'validation_'+k:v for k,v in h['validation'].items() if k in ['auc','accuracy','f1','mse','bce','n_targets']},
                'epoch_seconds':h.get('epoch_seconds'),'source_history':str(history_path),'source_sha256':sha(history_path)})
    csv(ASSETS/'source_data/epoch_records/completed_final_epochs.csv',epoch_rows)
    write(ASSETS/'verification/checkpoint_verification/reusable_GKT_training.json',reusable)
    # Corrected, independently inferred historical component removals: no aliases double counted.
    ablations=[]
    for d in DATASETS[:2]:
        for variant in ['no_attention','no_dropout','no_position','single_head']:
            for seed in SEEDS:
                raw=SOURCE/f'results/raw/{d}-LightSAKT-{variant}-{seed}'
                m,cfg=read(raw/'metrics.json'),read(raw/'config.json')
                assert read(raw/'status.json')['status']=='completed'
                cp=track(SOURCE/m['checkpoint']); assert sha(cp)==m['checkpoint_sha256']
                state=torch.load(cp,map_location='cpu',weights_only=False)
                assert all(torch.isfinite(v).all() for v in state['model'].values())
                assert state['config']==cfg
                model=build_model('LightSAKT',stats[d]['num_q'],cfg['model_config']);model.load_state_dict(state['model'],strict=True)
                assert state['processed_sha256']==stats[d]['processed_sha256'] and state['split_sha256']==stats[d]['split_sha256']
                history=read(raw/'history.json');assert max(history,key=lambda h:h['validation']['auc'])['epoch']==state['epoch']==m['best_epoch']
                test,_,_=predictions({'path':str(SOURCE/m['prediction_path']),'sha256':m['prediction_sha256']},expected[d,'test'])
                for k in ['auc','accuracy','f1','bce']:assert abs(test[k]-m['test'][k])<1e-12
                ablations.append({'dataset':d,'variant':variant,'seed':seed,'validation_auc':m['validation']['auc'],
                   **{'test_'+k:v for k,v in test.items()},'checkpoint':str(cp),'checkpoint_sha256':sha(cp),
                   'source_metrics':str(raw/'metrics.json'),'source_sha256':sha(raw/'metrics.json')})
    csv(ASSETS/'source_data/verified_metric_references/corrected_ablation_seeds.csv',ablations)
    # Tables 4/5 are actual validation-only 36-point LightSAKT grids, not the old manuscript numbers.
    for d in DATASETS[:2]:
        grid=pd.read_csv(track(ROOT/f'results/tables/LightSAKT_vs_SAKT_grid_{d}.csv'))
        assert len(grid)==36
        for row in grid.to_dict('records'):
            m=read(SOURCE/row['LightSAKT_metrics_path']);cp=track(SOURCE/row['LightSAKT_checkpoint'])
            cfg=read(SOURCE/Path(row['LightSAKT_metrics_path']).parent/'config.json')
            assert sha(cp)==row['LightSAKT_checkpoint_sha256'] and sha(SOURCE/row['LightSAKT_metrics_path'])==row['LightSAKT_metrics_sha256']
            assert cfg['evaluate_test'] is False and m['test'] is None and m['validation']['auc']==row['LightSAKT_validation_auc']
            history=read(SOURCE/Path(row['LightSAKT_metrics_path']).parent/'history.json')
            assert max(h['validation']['auc'] for h in history)==m['validation']['auc']
        csv(ASSETS/f'source_data/verified_metric_references/LightSAKT_grid_{d}.csv',grid)
    write(ASSETS/'source_data/dataset_provenance/dataset_metadata.json',metadata)
    # Prior successful leakage tests remain provenance; original programs are not executed in their writing mode.
    for name,source in {'old_datasets_leakage':'reports/professor_release_20261031/real_data_leakage_audit.json',
        'ednet_leakage':'reports/ednet_replacement_20261104/real_data_feature_audit.json',
        'ednet_temporal':'reports/ednet_replacement_20261104/offline_temporal_availability_audit.json',
        'ednet_statistics':'reports/ednet_replacement_20261104/dataset_statistics.json',
        'ednet_feature_spec':'reports/ednet_replacement_20261104/feature_specification.json',
        'previous_preservation':'reports/master_comparison_20261110/preservation_audit.json'}.items():
        write(ASSETS/f'source_data/dataset_provenance/{name}.json',read(ROOT/source))
    for rel in ['src/models/light_sakt.py','src/models/light_sakt_question.py','src/attention/linear.py',
                'src/datasets/sequences.py','scripts/question_difficulty_inputs.py','scripts/professor_data_20261030.py']:
        track(ROOT/rel)
    # Every source was captured before using it and is checked again after extraction.
    assert all(sha(p)==h for p,h in SOURCES.items())
    write(ASSETS/'verification/source_fingerprints.json',{'files':SOURCES,'unchanged_after_extraction':True,
          'paper_sha256':sha(paper),'manuscript_modified':False})
    write(ASSETS/'verification/metrics_verification/verification_summary.json',{
        'status':'PASS_AVAILABLE_EVIDENCE_WITH_PENDING_GROUPS','completed_seed_records':len(rows),
        'five_seed_groups':15,'pending_groups':12,'missing_seed_results':135-len(rows),'new_training_runs':0,
        'best_checkpoints_strictly_loaded':len(rows),'reusable_GKT_checkpoints_strictly_loaded':2,
        'corrected_component_ablation_checkpoints_verified':len(ablations),'real_grid_records_verified':72,
        'saved_prediction_arrays_recomputed':120+2*len(additional)+len(ablations),'metric_precision':'float64',
        'independent_AUC':'average ranks; crosscheck sklearn ROC AUC',
        'threshold':.5,'BCE_probability_clamp':[1e-7,1-1e-7],'SD_ddof':1,
        'validation_metric_origin':'native selected epoch; 45 saved validation arrays crosschecked',
        'test_metric_origin':'fresh arithmetic on genuine saved probabilities; two newly CPU-evaluated frozen GKT checkpoints if present',
        'new_CPU_checkpoints_evaluated':len(additional),'new_actual_data_inference_split_passes':2*len(additional),
        'GPU_training_authorization':'UNVERIFIED_CURRENT_ALLOCATION_AND_BUDGET',
        'HiTSKT':'not accessed or modified'})
    print(f'PASS:{len(rows)} final seeds,40 ablation seeds,72 validation-grid records; zero training.')


if __name__=='__main__':main()
