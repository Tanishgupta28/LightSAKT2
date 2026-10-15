"""Reuse and verify the already scheduled matched SAKT/LightSAKT experiments.

No scheduler mutation, training, CUDA allocation, or test-based selection occurs here.
"""
import argparse
import hashlib
import itertools
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.audit_schedule import training_identity
from scripts.light_final_report import SEEDS, METRICS, seed_statistics
from scripts.locked_test_evidence import LockedTestEvidence
from scripts.reuse_training import training_code_compatible
from scripts.sequence_cache import CachedSequences
from scripts.verify_artifacts import verify_stored_result
from scripts.verify_checkpoint_states import verify as verify_state
from src.datasets.preprocess import sha256
from src.models.factory import build_model
from src.training.engine import evaluate, write_json
from src.training.reproducibility import seed_everything

DATASETS = [('Algebra2005', 'A', 4), ('ASSIST2009', 'B', 5), ('ASSIST2015', 'C', 6)]
GRID = list(itertools.product([.0005, .001, .005], [32, 64, 128], [2, 4], [.1, .2]))
AUDIT = Path('experiments/sakt_matched_grid_manifest.json')
INFERENCE = Path('reports/sakt_grid_validation_inference.json')


def fingerprint():
    return hashlib.sha256(b''.join(str(p).encode()+p.read_bytes() for p in sorted(Path('src').rglob('*.py'))
                                   if 'visualization' not in p.parts)).hexdigest()


def key(config):
    m = config['model_config']
    return (config['learning_rate'], m['d'] if config['model']=='SAKT' else m['embed_dim'],
            m['num_attn_heads'] if config['model']=='SAKT' else m['num_heads'], m['dropout'])


def matched_protocol(light, sakt):
    assert light['model']=='LightSAKT' and sakt['model']=='SAKT' and key(light)==key(sakt)
    assert set(sakt['model_config'])=={'n','d','num_attn_heads','dropout'}
    assert set(light['model_config']) <= {'max_seq_len','embed_dim','num_heads','dropout','variant'}
    assert light['model_config'].get('variant','default')=='default'
    common = lambda c: {k:v for k,v in training_identity(c).items() if k not in ['model','model_config']}
    assert common(light)==common(sakt), 'Training protocol differs'
    assert light['model_config']['max_seq_len']==sakt['model_config']['n']==light['seq_len']
    return common(light)


def winner(records, field):
    return sorted(records, key=lambda r:(-r[field], r['run_id']))[0]


def text_file(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists() or path.read_text()!=value:
        tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(value);tmp.replace(path)


def csv_file(path, frame):
    text_file(path, frame.to_csv(index=False))


def completed(run_id, checked, current):
    root=Path('results/raw')/run_id
    assert json.load(open(root/'status.json'))['status']=='completed', run_id
    result, config, stats=verify_stored_result(root/'metrics.json', checked)
    assert training_code_compatible(result, current)
    if config['model']=='SAKT':assert result['training_code_sha256']==current
    state=torch.load(result['checkpoint'],map_location='cpu',weights_only=False)
    model=build_model(config['model'],stats['num_q'],config['model_config'])
    model.load_state_dict(state['model'],strict=True)
    assert all(torch.isfinite(x).all() for x in state['model'].values() if x.is_floating_point())
    assert state['epoch']==result['best_epoch'] and training_identity(state['config'])==training_identity(config)
    assert state['split_sha256']==result['split_sha256'] and state['processed_sha256']==result['processed_sha256']
    history=json.load(open(root/'history.json')) if (root/'history.json').exists() else state.get('history')
    if history:
        assert len(history)==result['epochs_trained']<=config['max_epochs']
        best=max(history,key=lambda h:h['validation']['auc'])
        assert best['epoch']==result['best_epoch'] and best['validation']==result['validation']
        bad=0;score=-np.inf
        for i,h in enumerate(history):
            if h['validation']['auc']>score:score=h['validation']['auc'];bad=0
            else:bad+=1
            assert bad<config['patience'] or i==len(history)-1
        assert len(history)==config['max_epochs'] or bad==config['patience']
    if config['model']=='SAKT':
        assert isinstance(model.attn,torch.nn.MultiheadAttention)
        verify_state(Path(result['checkpoint']).parent/'latest.pt',config,stats)
    return result,config,stats


def mask_supplement(result, dataset):
    mask=np.asarray(dataset.arrays[4]).copy()
    with np.load(result['prediction_path']) as saved:
        target=saved['target'].copy();probability=saved['probability'].copy()
    assert mask.dtype==bool and int(mask.sum())==len(target)==dataset.n_targets
    padded_target=np.full(mask.shape,-1,dtype=target.dtype);padded_target[mask]=target
    padded_probability=np.full(mask.shape,np.nan,dtype=probability.dtype);padded_probability[mask]=probability
    arrays={'target':target,'probability':probability,'valid_mask':mask,
            'target_padded':padded_target,'probability_padded':padded_probability}
    path=Path('results/predictions/sakt_matched_grid')/(result['run_id']+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path) as saved:
            assert set(saved.files)==set(arrays)
            for k,v in arrays.items():np.testing.assert_array_equal(saved[k],v)
    else:
        tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
        tmp.replace(path)
    return {'path':str(path),'sha256':sha256(path),'shape':list(mask.shape),
            'valid_targets':int(mask.sum()),'original_prediction_sha256':result['prediction_sha256'],
            'semantics':'Flat arrays contain valid targets only; padded arrays align with the explicit boolean mask.'}


def audit():
    assert not torch.cuda.is_available(), 'Run this workflow CPU-only during live training'
    current=fingerprint();checked=set();manifest=json.load(open('experiments/manifest.json'))
    original_ids=[j['run_id'] for j in manifest['jobs']]
    rows=[];finals=[];inferred={r['run_id']:r for r in json.load(open('reports/selected_final_inference_verification.json'))['runs']}
    locked=LockedTestEvidence();runtime=[];selections=[];mask_datasets={}
    for dataset,letter,table in DATASETS:
        candidates={};scheduled_counts={'LightSAKT':0,'SAKT':0}
        for j in manifest['jobs']:
            if j['phase']!=6 or 'config' not in j:continue
            c=json.load(open(j['config']))
            if c['dataset']==dataset and c['model'] in ['LightSAKT','SAKT']:
                scheduled_counts[c['model']]+=1
                candidates.setdefault(c['model'],{})[key(c)]=j
        assert set(scheduled_counts.values())=={36}, 'Duplicate or missing scheduled grid slots'
        assert all(len(candidates[m])==36 and set(candidates[m])==set(GRID) for m in ['LightSAKT','SAKT'])
        order=pd.read_csv(f'results/tables/Table_{table}.csv')
        order_keys=list(zip(order.learning_rate,order.embed_dim,order.num_heads,order.dropout))
        assert order_keys==GRID, 'Corrected LightSAKT table order differs from the paper grid'
        dataset_rows=[];seconds=[];epoch_seconds=[]
        for values in order_keys:
            records={}
            for model in ['LightSAKT','SAKT']:
                j=candidates[model][values];r,c,s=completed(j['run_id'],checked,current)
                assert json.load(open(j['config']))==c and c['evaluate_test'] is False and r['test'] is None
                records[model]=(r,c,s)
            lr,lc,ls=records['LightSAKT'];sr,sc,ss=records['SAKT'];protocol=matched_protocol(lc,sc)
            assert lr['processed_sha256']==sr['processed_sha256'] and lr['split_sha256']==sr['split_sha256']
            assert ss['split_seed']==42
            row={'dataset':dataset,'learning_rate':values[0],'embedding_dim':int(values[1]),'heads':int(values[2]),
                 'dropout':values[3],'seed':lc['seed'],'split_seed':ss['split_seed'],'seq_len':sc['seq_len'],
                 'batch_size':sc['batch_size'],'LightSAKT_validation_auc':lr['validation']['auc'],
                 'SAKT_validation_auc':sr['validation']['auc'],'difference':lr['validation']['auc']-sr['validation']['auc'],
                 'training_protocol':protocol}
            for model,(r,c,s) in records.items():
                for name in ['run_id','checkpoint','checkpoint_sha256','best_epoch','git_commit','training_code_sha256','processed_sha256','split_sha256']:
                    row[model+'_'+name]=r[name]
                row[model+'_metrics_path']=f"results/raw/{r['run_id']}/metrics.json"
                row[model+'_metrics_sha256']=sha256(row[model+'_metrics_path'])
                row[model+'_config_sha256']=sha256(Path('results/raw')/r['run_id']/'config.json')
            dataset_rows.append(row);seconds.append(sr['training_seconds'])
            epoch_seconds += [h['epoch_seconds'] for h in json.load(open(Path('results/raw')/sr['run_id']/'history.json'))]
        rows+=dataset_rows
        selection=json.load(open(f'experiments/selections/{dataset}-SAKT.json'))
        best=winner([{'run_id':r['SAKT_run_id'],'auc':r['SAKT_validation_auc']} for r in dataset_rows],'auc')
        assert selection['selected_run_id']==best['run_id'] and selection['selection_metric']=='validation_auc'
        assert selection['test_access_for_selection'] is False and set(selection['all_candidates'])=={r['SAKT_run_id'] for r in dataset_rows}
        light_selection=json.load(open(f'experiments/selections/{dataset}-LightSAKT.json'))
        light_best=winner([{'run_id':r['LightSAKT_run_id'],'auc':r['LightSAKT_validation_auc']} for r in dataset_rows],'auc')
        assert light_selection['selected_run_id']==light_best['run_id'] and light_selection['test_access_for_selection'] is False
        assert set(light_selection['all_candidates'])=={r['LightSAKT_run_id'] for r in dataset_rows}
        selections += [{'dataset':dataset,'model':model,**value} for model,value in [('SAKT',selection),('LightSAKT',light_selection)]]
        for model in ['SAKT','LightSAKT']:
            for seed in SEEDS:
                rid=f'{dataset}-{model}-final-{seed}';r,c,s=completed(rid,checked,current)
                own=json.load(open(f'experiments/selections/{dataset}-{model}.json'))
                chosen=json.load(open(Path('results/raw')/own['selected_run_id']/'config.json'))
                assert training_identity(c)==training_identity({**chosen,'seed':seed})
                assert own['test_access_for_selection'] is False and c['seed']==r['seed']==seed
                identity=locked.check(r,c);proof=inferred[rid]
                for k,v in {'metrics_sha256':sha256(Path('results/raw')/rid/'metrics.json'),
                            'checkpoint_sha256':r['checkpoint_sha256'],'prediction_sha256':r['prediction_sha256'],
                            'processed_sha256':r['processed_sha256'],'split_sha256':r['split_sha256'],
                            'config_sha256':sha256(Path('results/raw')/rid/'config.json')}.items():assert proof[k]==v
                assert proof['verified'] and proof['validation_recomputed'] and proof['test_probabilities_recomputed']
                extra=None
                if model=='SAKT':
                    if dataset not in mask_datasets:
                        mask_datasets[dataset]=CachedSequences(f'data/processed/{dataset}/interactions.csv','test',c['seq_len'])
                    extra=mask_supplement(r,mask_datasets[dataset])
                finals.append({'result':r,'config':c,'locked_test':identity,'explicit_mask':extra})
        runtime.append({'dataset':dataset,'grid_runs':36,'measured_grid_training_seconds':sum(seconds),
                        'mean_run_seconds':float(np.mean(seconds)),'mean_epoch_seconds':float(np.mean(epoch_seconds)),
                        'new_grid_runs':0,'new_final_seed_runs':0,'additional_training_gpu_hours':0.})
    assert len(rows)==108 and len(finals)==30 and {r['seed'] for r in rows}=={42}
    assert fingerprint()==current and [j['run_id'] for j in json.load(open('experiments/manifest.json'))['jobs']]==original_ids
    value={'verified':True,'verified_utc':datetime.now(timezone.utc).isoformat(),'training_code_sha256':current,
           'original_manifest_run_ids_sha256':hashlib.sha256(json.dumps(original_ids).encode()).hexdigest(),
           'original_manifest_slots':len(original_ids),'new_manifest_slots':0,'reusable_grid_slots':108,
           'reusable_final_seed_slots':15,'new_grid_trainings':0,'new_final_trainings':0,
           'training_seed':42,'split_seed':42,'rows':rows,'finals':finals,'runtime':runtime,'selections':selections,
           'model_architecture_changed':False,'healthy_training_signalled':False,
           'source_equivalence_note':'One ASSIST2009 LightSAKT reuse has an audited global fingerprint transition for an Algebra-only preprocessing correction; model mathematics and ASSIST2009 records are unchanged. See reports/training_code_equivalence.json.'}
    write_json(AUDIT,value);print('Audited reusable grid/final slots: 108 / 15; fresh training required: 0',flush=True)
    return value


def verify_grid():
    assert not torch.cuda.is_available(), 'CPU-only validation verification required'
    audit_data=json.load(open(AUDIT));assert audit_data['training_code_sha256']==fingerprint()
    old=json.load(open(INFERENCE)) if INFERENCE.exists() else {'runs':[]}
    records={r['run_id']:r for r in old['runs']};datasets={};started=time.perf_counter()
    for row in audit_data['rows']:
        rid=row['SAKT_run_id'];root=Path('results/raw')/rid
        identity={'metrics_sha256':sha256(root/'metrics.json'),'config_sha256':sha256(root/'config.json'),
                  'checkpoint_sha256':sha256(row['SAKT_checkpoint'])}
        previous=records.get(rid,{})
        if previous.get('verified') and all(previous.get(k)==v for k,v in identity.items()):continue
        c=json.load(open(root/'config.json'));r=json.load(open(root/'metrics.json'));s=json.load(open(f"reports/{row['dataset']}_statistics.json"))
        seed_everything(c['seed']);torch.set_num_threads(1)
        model=build_model('SAKT',s['num_q'],c['model_config']).eval()
        state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True)
        dkey=(c['dataset'],c['seq_len'])
        if dkey not in datasets:datasets[dkey]=CachedSequences(f"data/processed/{c['dataset']}/interactions.csv",'validation',c['seq_len'])
        measured,y,_=evaluate(model,'SAKT',DataLoader(datasets[dkey],batch_size=c['batch_size'],shuffle=False),'cpu')
        assert len(y)==r['validation']['n_targets']==datasets[dkey].n_targets
        for m in METRICS:np.testing.assert_allclose(measured[m],r['validation'][m],atol=1e-7,rtol=1e-6,err_msg=rid+' '+m)
        records[rid]={'run_id':rid,'verified':True,'device':'cpu','test_evaluated':False,
                      'regenerated_validation':measured,'training_git_commit':r['git_commit'],**identity}
        write_json(INFERENCE,{'verified':len(records)==108,'runs':list(records.values()),
                             'verified_utc':datetime.now(timezone.utc).isoformat(),'retrained_runs':0,
                             'metric_atol':1e-7,'metric_rtol':1e-6,'verification_cpu_threads':1})
        print('Validation verified',rid,flush=True)
    assert len(records)==108 and all(r['verified'] for r in records.values())
    value={'verified':True,'runs':list(records.values()),'verified_utc':datetime.now(timezone.utc).isoformat(),
           'retrained_runs':0,'metric_atol':1e-7,'metric_rtol':1e-6,'verification_cpu_threads':1,
           'last_invocation_seconds':time.perf_counter()-started,'test_evaluated':False}
    write_json(INFERENCE,value);return value


def publication_table(name, frame, columns, bold=(), caption=''):
    maxima={c:frame[c].max() for c in bold}
    markdown=['# '+caption,'','|'+'|'.join(columns)+'|','|'+'|'.join(['---']*len(columns))+'|']
    latex=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{'+caption+'}',r'\label{tab:'+name+'}',r'\resizebox{\textwidth}{!}{%',
           r'\begin{tabular}{'+'r'*len(columns)+'}',r'\hline',' & '.join(c.replace('_',r'\_') for c in columns)+r' \\',r'\hline']
    for row in frame.to_dict('records'):
        md=[];tex=[]
        for c in columns:
            v=row[c];s=f'{v:.8f}' if isinstance(v,float) else str(v)
            mark=c in maxima and v==maxima[c]
            md.append('**'+s+'**' if mark else s);tex.append(r'\textbf{'+s+'}' if mark else s)
        markdown.append('|'+ '|'.join(md)+'|');latex.append(' & '.join(tex)+r' \\')
    latex += [r'\hline',r'\end{tabular}',r'}',r'\end{table}']
    footer=[]
    for c in bold:
        model='SAKT' if c in ['Best Validation AUC','SAKT_validation_auc'] else 'LightSAKT'
        if model+'_run_id' in frame:
            selected=sorted(frame.to_dict('records'),key=lambda r:(-r[c],r[model+'_run_id']))[0]
            footer += [f"Selected {model} run: `{selected[model+'_run_id']}`; checkpoint `{selected[model+'_checkpoint']}`; validation epoch {selected[model+'_best_epoch']}."]
    for folder in [Path('results/tables'),Path('paper/sakt_grid')]:
        csv_file(folder/(name+'.csv'),frame)
        text_file(folder/(name+'.md'),'\n'.join(markdown)+'\n\nSeed 42; locked split seed 42. Bold marks exact maxima before rounding; exact ties select the lexically smallest run ID. CSV retains full precision, run IDs, selected epochs and checkpoint paths.\n\n'+'\n\n'.join(footer)+'\n')
        text_file(folder/(name+'.tex'),'\n'.join(latex)+'\n')


def render(plots=False):
    a=json.load(open(AUDIT));proof=json.load(open(INFERENCE));assert a['verified'] and proof['verified']
    assert a['training_code_sha256']==fingerprint() and len(proof['runs'])==108
    inferred={r['run_id']:r for r in proof['runs']};rows=pd.DataFrame(a['rows']);counts=[];selected=[]
    for dataset,letter,_ in DATASETS:
        frame=rows[rows.dataset==dataset].copy()
        for r in frame.to_dict('records'):
            p=inferred[r['SAKT_run_id']]
            assert p['verified'] and p['metrics_sha256']==sha256(r['SAKT_metrics_path']) and p['checkpoint_sha256']==sha256(r['SAKT_checkpoint'])
            assert sha256(r['LightSAKT_metrics_path'])==r['LightSAKT_metrics_sha256']
        for col in ['LightSAKT_validation_auc','SAKT_validation_auc']:
            model=col.split('_')[0];index=sorted(frame.index,key=lambda i:(-frame.loc[i,col],frame.loc[i,model+'_run_id']))[0]
            r=frame.loc[index].to_dict();selected.append({'dataset':dataset,'model':model,'learning_rate':r['learning_rate'],
                'embedding_dim':r['embedding_dim'],'heads':r['heads'],'dropout':r['dropout'],'validation_auc':r[col],
                'best_epoch':r[model+'_best_epoch'],'seed':r['seed'],'run_id':r[model+'_run_id'],
                'checkpoint':r[model+'_checkpoint'],'checkpoint_sha256':r[model+'_checkpoint_sha256']})
        lightwins=int((frame.difference>0).sum());saktwins=int((frame.difference<0).sum())
        counts.append({'dataset':dataset,'configurations':len(frame),'LightSAKT_higher':lightwins,'SAKT_higher':saktwins,
                       'exact_ties':int((frame.difference==0).sum()),'mean_difference':float(frame.difference.mean()),
                       'median_difference':float(frame.difference.median()),'minimum_difference':float(frame.difference.min()),
                       'maximum_difference':float(frame.difference.max()),'descriptive_sd':float(frame.difference.std(ddof=1))})
        simple=frame.drop(columns=['training_protocol']).rename(columns={'SAKT_validation_auc':'Best Validation AUC'})
        publication_table(f'SAKT_Table_{letter}_{dataset}',simple,
                          ['learning_rate','embedding_dim','heads','dropout','Best Validation AUC'],['Best Validation AUC'],
                          f'SAKT Table {letter}: {dataset} validation grid')
        publication_table(f'LightSAKT_vs_SAKT_grid_{dataset}',frame.drop(columns=['training_protocol']),
                          ['learning_rate','embedding_dim','heads','dropout','LightSAKT_validation_auc','SAKT_validation_auc','difference'],
                          ['LightSAKT_validation_auc','SAKT_validation_auc'],f'Matched LightSAKT and SAKT validation grid: {dataset}')
    selected=pd.DataFrame(selected);counts=pd.DataFrame(counts);individual=[];summaries=[];comparisons=[];sensitivity=[]
    for dataset,_,_ in DATASETS:
        group=rows[rows.dataset==dataset]
        for model in ['SAKT','LightSAKT']:
            for parameter in ['learning_rate','embedding_dim','heads','dropout']:
                for value,part in group.groupby(parameter):
                    values=part[model+'_validation_auc']
                    sensitivity.append({'dataset':dataset,'model':model,'parameter':parameter,'level':value,
                        'configurations':len(values),'mean_validation_auc':float(values.mean()),
                        'minimum_validation_auc':float(values.min()),'maximum_validation_auc':float(values.max())})
    for dataset,_,_ in DATASETS:
        models={}
        for model in ['LightSAKT','SAKT']:
            rs=[f['result'] for f in a['finals'] if f['result']['dataset']==dataset and f['result']['model']==model]
            models[model]=seed_statistics(rs);summaries.append({'dataset':dataset,'model':model,**models[model]})
            for f in a['finals']:
                r=f['result']
                if r['dataset']==dataset and r['model']==model:
                    individual.append({'dataset':dataset,'model':model,'seed':r['seed'],'run_id':r['run_id'],
                        'best_epoch':r['best_epoch'],'checkpoint':r['checkpoint'],'checkpoint_sha256':r['checkpoint_sha256'],
                        'git_commit':r['git_commit'],'split_sha256':r['split_sha256'],
                        'original_predictions':r['prediction_path'],'explicit_mask':(f['explicit_mask'] or {}).get('path'),**r['test']})
        for metric in METRICS:comparisons.append({'dataset':dataset,'metric':metric,
             'LightSAKT_mean':models['LightSAKT'][metric+'_mean'],'LightSAKT_sd':models['LightSAKT'][metric+'_std'],
             'SAKT_mean':models['SAKT'][metric+'_mean'],'SAKT_sd':models['SAKT'][metric+'_std'],
             'difference':models['LightSAKT'][metric+'_mean']-models['SAKT'][metric+'_mean']})
    for name,frame in [('SAKT_grid_win_counts',counts),('SAKT_matched_grid_selected_configurations',selected),
                       ('SAKT_matched_grid_individual_seeds',pd.DataFrame(individual)),
                       ('SAKT_matched_grid_final_summary',pd.DataFrame(summaries)),
                       ('LightSAKT_SAKT_final_comparison',pd.DataFrame(comparisons)),
                       ('SAKT_grid_hyperparameter_sensitivity',pd.DataFrame(sensitivity)),
                       ('SAKT_matched_grid_runtime',pd.DataFrame(a['runtime']))]:
        csv_file('results/tables/'+name+'.csv',frame);csv_file('paper/sakt_grid/'+name+'.csv',frame)
    from scripts.light_final_report import table
    report=['# SAKT grid search: verified reuse and complete tables','',
        'All 108 requested configurations and 15 independently selected five-seed SAKT slots already existed in the approved schedule. Their checkpoints, configurations, histories, split/data hashes, code compatibility and validation-only selections were checked. No duplicate jobs were added, no completed training was repeated and no healthy worker was signalled. The main manifest is unchanged; experiments/sakt_matched_grid_manifest.json is a reference view of its existing jobs.','',
        'Paper Tables 4–6 specify the same 3 learning rates × 3 widths × 2 head counts × 2 dropout rates. Table order is preserved. Grid training seed and split seed are both 42. Context lengths are 100/100/200 for ASSIST2009/ASSIST2015/Algebra; batch size 64; Adam betas (0.9,0.999), epsilon 1e-8, weight decay 0, maximum 50 epochs and validation-AUC patience 5 match between models. Standard SAKT retains PyTorch multi-head attention; LightSAKT retains ELU+1 causal prefix attention and its reduced-width FFN.','',
        '## Complete independently selected validation configurations','',table(selected,list(selected.columns)),'',
        'Selection maximizes full-precision validation AUC among all 36 candidates for each model/dataset. Exact ties select the lexically smallest run ID. Bold table cells include all exact maxima. Test results never select these configurations. Identical search spaces do not require identical selected configurations.','',
        'Tables A/B/C and their paired comparison tables are in results/tables/SAKT_Table_*.{csv,md,tex}, LightSAKT_vs_SAKT_grid_*.{csv,md,tex}, and paper/sakt_grid/. CSVs retain every source run ID, checkpoint and metrics-file hash. Original LightSAKT Tables 4–6 and the original paper are preserved.','',
        '## Measured execution budget','',table(pd.DataFrame(a['runtime']),list(a['runtime'][0])),'',
        'Genuinely new grid trainings: 0. Reused grid configurations: 108. Additional final-seed trainings: 0; all 15 final slots are verified, including three selected seed-42 checkpoint reuses. Additional training GPU hours: 0. CPU validation verification has been executed without allocating GPU memory. There is no new GPU workload to overlap with GKT; existing scheduling priorities and recovery remain in force. Current overall forecast remains in reports/ETA_REPORT.md.','',
        'Proof: reports/sakt_grid_validation_inference.json; experiments/sakt_matched_grid_manifest.json; reports/selected_final_inference_verification.json. One historical global LightSAKT fingerprint differs only through the audited Algebra preprocessing correction; the unchanged ASSIST2009 model/data equivalence is documented in reports/training_code_equivalence.json. SAKT code fingerprints match the current frozen source exactly. Explicit non-padding test masks and original valid-target predictions are preserved as additional artifacts under results/predictions/sakt_matched_grid/.']
    comparison=['# LightSAKT versus SAKT: two distinct comparisons','',
        '## A. Identical hyperparameters — validation only','',table(counts,list(counts.columns)),'',
        'Difference is LightSAKT validation AUC minus SAKT validation AUC. The 36 rows are a deterministic hyperparameter sweep at one seed, not 36 independent random trials; no configuration-level significance test is performed. Mean/median/range/SD describe that specified grid. This comparison covers the complete architectures, including their attention and FFN differences, and does not isolate the attention kernel alone.','',
        '## B. Independently tuned configurations — held-out test','',table(selected,['dataset','model','learning_rate','embedding_dim','heads','dropout','validation_auc']),'',
        'Both models independently selected their own highest-validation-AUC configuration from the same complete search space, then used seeds 42,123,2026,3407,7777 on the same frozen split. Seed 42 reuses its selected grid checkpoint. All five seeds contribute; SD is sample SD (ddof=1).','']
    display=pd.DataFrame(comparisons)
    display['LightSAKT']=display.apply(lambda r:f'{r.LightSAKT_mean:.6f} ± {r.LightSAKT_sd:.6f}',axis=1)
    display['SAKT']=display.apply(lambda r:f'{r.SAKT_mean:.6f} ± {r.SAKT_sd:.6f}',axis=1)
    comparison += [table(display,['dataset','metric','LightSAKT','SAKT','difference']),'',
        'Positive differences favor LightSAKT for AUC, accuracy and F1; negative differences favor LightSAKT for MSE and BCE. These are descriptive differences conditional on one student split. Five seeds cannot support a two-sided exact signed-rank p<0.05; grid rows are not used to inflate that sample size. Individual metrics, selected epochs and checkpoint references are in SAKT_matched_grid_individual_seeds.csv.','',
        'The existing efficiency evidence remains unchanged: LightSAKT was slower in all 60 matched latency comparisons on the recorded H100 MIG partition; lower peak allocated memory occurred in 7/60. Theoretical O(L) attention work, measured latency, memory and configuration-specific parameter counts remain separate. This completed additional analysis does not imply the remaining GKT research schedule is complete.']
    sensitivity_frame=pd.DataFrame(sensitivity)
    comparison += ['', '## Hyperparameter effects within the specified grid', '',
        'SAKT_grid_hyperparameter_sensitivity.csv reports balanced marginal means over all other coordinates, with counts and observed minima/maxima. These are descriptive averages of the fixed grid, not independent replications or confidence intervals. They do not replace joint validation-based selection.']
    for dataset,_,_ in DATASETS:
        best_levels=[]
        for parameter in ['learning_rate','embedding_dim','heads','dropout']:
            sub=sensitivity_frame[(sensitivity_frame.dataset==dataset)&(sensitivity_frame.model=='SAKT')&(sensitivity_frame.parameter==parameter)]
            r=sub.sort_values(['mean_validation_auc','level'],ascending=[False,True]).iloc[0]
            best_levels.append(f'{parameter}={r.level:g} (marginal mean validation AUC {r.mean_validation_auc:.6f})')
        comparison.append(dataset+': highest SAKT marginal means occur at '+ '; '.join(best_levels)+'. Interactions can make the jointly selected configuration differ.')
    text_file('reports/SAKT_GRID_SEARCH_REPORT.md','\n'.join(report)+'\n')
    text_file('reports/SAKT_LIGHTSAKT_COMPARISON.md','\n'.join(comparison)+'\n')
    text_file('paper/SAKT_GRID_RESULTS.md','\n'.join(report+['']+comparison)+'\n')
    names=[f'SAKT_Table_{letter}_{dataset}' for dataset,letter,_ in DATASETS]+[f'LightSAKT_vs_SAKT_grid_{dataset}' for dataset,_,_ in DATASETS]
    text_file('paper/sakt_grid/README.md','# Matched SAKT grid appendix\n\n'+
        '\n'.join(f'- [{name}]({name}.md): [CSV]({name}.csv), [LaTeX]({name}.tex).' for name in names)+
        '\n\nCompile the standalone appendix with `pdflatex sakt_grid_appendix.tex` from this directory when LaTeX is installed. The snippets require graphicx. LaTeX compilation was not executed in the current environment because pdflatex is unavailable; the CSV, Markdown and LaTeX sources and all numerical checks are complete.\n')
    text_file('paper/sakt_grid/sakt_grid_appendix.tex',r'\documentclass{article}'+'\n'+r'\usepackage[margin=1.5cm]{geometry}'+'\n'+r'\usepackage{graphicx}'+'\n'+r'\begin{document}'+'\n'+
        '\n'.join(r'\input{'+name+'.tex}'+r'\clearpage' for name in names)+'\n'+r'\end{document}'+'\n')
    if plots:plot(rows,counts,pd.DataFrame(summaries))
    artifacts=[p for folder in ['results/tables','paper/sakt_grid'] for p in Path(folder).glob('*') if p.is_file() and ('SAKT_' in p.name or 'LightSAKT_vs_SAKT_grid_' in p.name)]
    write_json('reports/sakt_matched_grid_verification.json',{'verified':True,'grid_configurations':108,
               'paired_configurations':108,'final_seed_slots':15,'new_trainings':0,'additional_gpu_training_hours':0,
               'verified_utc':datetime.now(timezone.utc).isoformat(),
               'artifacts':[{'path':str(p),'sha256':sha256(p)} for p in sorted(artifacts)],
               'counts':counts.to_dict('records'),'model_source_fingerprint':fingerprint()})
    print(counts.to_string(index=False),flush=True)


def plot(rows,counts,summary):
    import matplotlib.pyplot as plt
    from scripts.analyze import save_figure
    for dataset,_,_ in DATASETS:
        frame=rows[rows.dataset==dataset].drop(columns=['training_protocol'])
        for field,title in [('SAKT_validation_auc','SAKT validation AUC'),('difference','LightSAKT minus SAKT validation AUC')]:
            name=dataset+'_matched_sakt_'+field
            csv_file('results/figures/'+name+'.csv',frame)
            fig,axes=plt.subplots(2,2,figsize=(10,8));limit=max(float(frame.difference.abs().max()),1e-9)
            for ax,(heads,dropout) in zip(axes.flat,itertools.product([2,4],[.1,.2])):
                sub=frame[(frame.heads==heads)&(frame.dropout==dropout)]
                matrix=sub.pivot(index='learning_rate',columns='embedding_dim',values=field).loc[[.0005,.001,.005],[32,64,128]]
                im=ax.imshow(matrix,interpolation='nearest',cmap='RdBu' if field=='difference' else 'viridis',
                             vmin=-limit if field=='difference' else float(frame[field].min()),
                             vmax=limit if field=='difference' else float(frame[field].max()))
                ax.set_xticks(range(3),[32,64,128]);ax.set_yticks(range(3),['0.0005','0.001','0.005'])
                ax.set_xlabel('Embedding dimension');ax.set_ylabel('Learning rate');ax.set_title(f'Heads {heads}; dropout {dropout}')
                for (i,j),v in np.ndenumerate(matrix.to_numpy()):
                    white=abs(v)>limit*.6 if field=='difference' else v<(frame[field].min()+frame[field].max())/2
                    ax.text(j,i,f'{v:.4f}',ha='center',va='center',color='white' if white else 'black')
                fig.colorbar(im,ax=ax,label='AUC difference' if field=='difference' else 'Validation AUC')
            fig.suptitle(dataset+' — '+title);fig.tight_layout();save_figure(name)
        data=summary[summary.dataset==dataset];name=dataset+'_matched_sakt_final_auc'
        csv_file('results/figures/'+name+'.csv',data);plt.figure(figsize=(5,4))
        plt.bar(data.model,data.auc_mean,yerr=data.auc_std,capsize=5);plt.ylabel('Test AUC (mean ± sample SD, 5 seeds)');plt.title(dataset)
        plt.ylim(0,1);save_figure(name)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audit','verify','render','all']);parser.add_argument('--plots',action='store_true');args=parser.parse_args()
    assert not torch.cuda.is_available(),'Use CUDA_VISIBLE_DEVICES empty; healthy training must stay untouched'
    import fcntl
    lock=open('experiments/sakt_matched_grid.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.stage in ['audit','all']:audit()
    if args.stage in ['verify','all']:verify_grid()
    if args.stage in ['render','all']:render(args.plots)


if __name__=='__main__':main()
