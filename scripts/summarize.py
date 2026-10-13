import csv
import fcntl
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from src.training.engine import write_json


def summarize():
    rows=[]
    for path in sorted(Path('results/raw').glob('*/metrics.json')):
        result=json.load(open(path));config_path=path.parent/'config.json';config=json.load(open(config_path)) if config_path.exists() else {}
        row={k:result.get(k) for k in ['run_id','dataset','model','group','seed','best_epoch','epochs_trained','training_seconds','parameters','checkpoint','checkpoint_sha256','split_sha256','git_commit','peak_allocated_bytes','peak_reserved_bytes']}
        row.update(learning_rate=config.get('learning_rate'),**config.get('model_config',{}))
        row['variant']=config.get('scientific_variant',config.get('model_config',{}).get('variant','default'))
        row['equivalent_to']=config.get('equivalent_to')
        for split in ['validation','test']:
            for metric in ['auc','accuracy','f1','mse','bce','n_targets']:row[f'{split}_{metric}']=(result.get(split) or {}).get(metric)
        rows.append(row)
    frame=pd.DataFrame(rows)
    Path('results/tables').mkdir(parents=True,exist_ok=True)
    frame.to_csv('results/tables/all_verified_runs.csv',index=False)
    stats=[json.load(open(f'reports/{d}_statistics.json')) for d in ['ASSIST2009','ASSIST2015','Algebra2005']]
    pd.DataFrame([{k:s[k] for k in ['dataset','raw_rows','clean_interactions','students','unique_skill_identifiers','num_q','raw_sha256','processed_sha256','split_sha256']} for s in stats]).to_csv('results/tables/Table_2.csv',index=False)
    manifest=json.load(open('experiments/manifest.json'))
    primary=[json.load(open(j['config'])) for j in manifest['jobs'] if j['phase']==5]
    pd.json_normalize(primary).to_csv('results/tables/Table_3.csv',index=False)
    for dataset,table in [('Algebra2005',4),('ASSIST2009',5),('ASSIST2015',6)]:
        jobs=[j for j in manifest['jobs'] if j['phase']==6 and j['run_id'].startswith(dataset+'-LightSAKT-grid')]
        output=[]
        for job in jobs:
            c=json.load(open(job['config']));r={'run_id':job['run_id'],'dataset':dataset,'learning_rate':c['learning_rate'],**c['model_config'],'seed':c['seed'],'status':job['status']}
            found=next((x for x in rows if x['run_id']==job['run_id']),None)
            if found:r.update(found)
            output.append(r)
        pd.DataFrame(output).to_csv(f'results/tables/Table_{table}.csv',index=False)
    if frame.empty:return
    primary_frame=frame[frame.group=='primary'];primary_frame.to_csv('results/tables/Table_7_primary_partial.csv',index=False)
    frame[frame.group=='ablation'].to_csv('results/tables/Table_8_partial.csv',index=False)
    summary=[]
    final=frame[frame.group=='final']
    for (dataset,model),group in final.groupby(['dataset','model']):
        row={'dataset':dataset,'model':model,'completed_seeds':len(group),'required_seeds':5}
        for metric in ['auc','accuracy','f1','mse','bce']:
            values=group[f'test_{metric}'].dropna().to_numpy()
            row[f'{metric}_mean']=float(values.mean()) if len(values) else None
            row[f'{metric}_std']=float(values.std(ddof=1)) if len(values)>1 else None
            interval=t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)) if len(values)>1 else None
            row[f'{metric}_ci95_halfwidth']=float(interval) if interval is not None else None
        summary.append(row)
    pd.DataFrame(summary).to_csv('results/tables/repeated_seed_summary.csv',index=False)
    pd.DataFrame(summary).to_csv('results/tables/Table_7.csv',index=False)
    ablation_summary=[]
    ablations=frame[frame.group=='ablation']
    for (dataset,variant),group in ablations.groupby(['dataset','variant']):
        # Aliases retain their explicit scientific labels in all_verified_runs.
        row={'dataset':dataset,'variant':variant,'completed_seeds':len(group),'required_seeds':5}
        for metric in ['auc','accuracy','f1','mse','bce']:
            values=group[f'test_{metric}'].dropna().to_numpy()
            row[f'{metric}_mean']=float(values.mean()) if len(values) else None
            row[f'{metric}_std']=float(values.std(ddof=1)) if len(values)>1 else None
        ablation_summary.append(row)
    for row in summary:
        if row['model']=='LightSAKT':ablation_summary.append({**row,'variant':'default','evidence_reused_from':'final LightSAKT'})
    for row in ablation_summary:
        default=next((r for r in summary if r['dataset']==row['dataset'] and r['model']=='LightSAKT'),None)
        row['equivalent_to']='no_dropout' if row['variant']=='no_regularization' else None
        for metric in ['auc','accuracy','f1','mse','bce']:
            base=default.get(f'{metric}_mean') if default else None;value=row.get(f'{metric}_mean')
            row[f'{metric}_relative_change']=(value-base)/base if base and value is not None else None
    pd.DataFrame(ablation_summary).to_csv('results/tables/Table_8.csv',index=False)
    pd.json_normalize(stats).to_csv('results/tables/dataset_preprocessing_audit.csv',index=False)
    write_json('results/summary/traceability.json',[{'run_id':r['run_id'],'source':f"results/raw/{r['run_id']}/metrics.json",'checkpoint':r['checkpoint'],'split_sha256':r['split_sha256'],'git_commit':r['git_commit']} for r in rows])

if __name__=='__main__':
    with open('experiments/reporting.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        summarize()
