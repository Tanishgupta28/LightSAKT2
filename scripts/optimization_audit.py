"""Inventory two reviewable schedules without modifying the live scheduler ledger."""
import json
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from src.training.engine import write_json


def coarse_gkt(config):
    mc=config['model_config']
    if config['model']=='GKT':return mc['hidden_size'] in [16,30,64]
    if config['model']=='GKT_MHA':return mc['hidden_size'] in [16,32,64] and mc['num_attn_heads']==2
    return True


def audit():
    manifest=json.load(open('experiments/manifest.json'));reuse=json.load(open('experiments/reuse_plan.json'))['jobs']
    timings=pd.read_csv('reports/ETA_REMAINING_RUNS.csv').set_index('run_id');rows=[];accelerated=[]
    for job in manifest['jobs']:
        runtime=Path('experiments/runtime_jobs')/(job['run_id']+'.json')
        if runtime.exists():job.update(json.load(open(runtime)))
        c=json.load(open(job['config'])) if 'config' in job else job
        status_path=Path('results/raw')/job['run_id']/'status.json'
        state=json.load(open(status_path))['status'] if status_path.exists() else job['status']
        alias=job['run_id'] in reuse or job['phase']==7 and job['seed']==42 or job.get('variant')=='no_regularization'
        category='primary' if job['phase']==5 else 'LightSAKT_grid' if job['phase']==6 and c['model']=='LightSAKT' else 'baseline_tuning' if job['phase']==6 else 'repeated_selected' if job['phase']==7 else 'ablations'
        keep=state in ['completed','running'] or job['phase']!=6 or coarse_gkt(c)
        row={'run_id':job['run_id'],'dataset':c['dataset'],'model':c['model'],'category':category,
             'seed':c['seed'],'status':state,'fresh_remaining':state!='completed' and not alias,
             'training_reuse':alias,
             'reuse_remaining':state!='completed' and alias,'accelerated_retained':keep,
             'scientific_role':'Required paper grid' if category=='LightSAKT_grid' else
                'Required matched five-seed comparison' if category=='repeated_selected' else
                'Required component removal (regularization alias)' if category=='ablations' else
                'Completed baseline evidence' if category=='primary' else
                'Expanded supplementary baseline tuning; full budget preserved unless approved',
             'change_reason':None if keep else 'PROPOSAL ONLY: restrict GKT tuning to 3 learning rates x 3 capacities; MHA heads fixed at 2; preserve both variants, all seeds, and all primary evidence.'}
        for k in ['low','central','high']:row[k+'_hours']=float(timings.loc[job['run_id'],k+'_hours']) if job['run_id'] in timings.index else 0.
        rows.append(row)
        if keep:accelerated.append({**job,'observed_status':state})
    frame=pd.DataFrame(rows);Path('reports/optimization').mkdir(exist_ok=True)
    frame.to_csv('reports/optimization/SCHEDULE_INVENTORY.csv',index=False)
    frame[~frame.accelerated_retained].to_csv('reports/optimization/PROPOSED_DIFFERENCES.csv',index=False)
    grouped=frame.groupby(['category','dataset']).agg(completed=('status',lambda s:(s=='completed').sum()),
        fresh_remaining=('fresh_remaining','sum'),reuse_remaining=('reuse_remaining','sum')).reset_index()
    grouped.to_csv('reports/optimization/REMAINING_BY_CATEGORY.csv',index=False)
    write_json('experiments/complete_schedule.json',{'approval':'Existing authorized complete protocol','jobs':manifest['jobs'],'reuse_overlay':'experiments/reuse_plan.json'})
    write_json('experiments/accelerated_schedule_proposal.json',{'approval':'NOT APPROVED; not used by training',
        'scope_difference':'GKT supplementary tuning 36 -> 9 per variant/dataset; other grids and all selected five-seed/ablation runs unchanged',
        'selection_rule':'Validation AUC among retained candidates plus the completed historical primary GKT-MHA config; previously completed/running evidence retained.',
        'jobs':accelerated,'deferred_run_ids':frame.loc[~frame.accelerated_retained,'run_id'].tolist()})
    full={k:float(frame[k+'_hours'].sum()) for k in ['low','central','high']}
    fast={k:float(frame.loc[frame.accelerated_retained,k+'_hours'].sum()) for k in ['low','central','high']}
    eta=json.load(open('reports/ETA.json'));factor=eta.get('two_worker_measured_throughput_speedup',1.)
    def occupied(subset):
        gkt=subset.model=='GKT_MHA'
        return {k:float(subset.loc[~gkt,k+'_hours'].sum()+subset.loc[gkt,k+'_hours'].sum()/(factor if k!='high' else 1.)) for k in ['low','central','high']}
    # Final GKT config is not known. This conservative comparison retains the same
    # final-training cost envelope instead of assuming a favorable selected width.
    summary={'updated_utc':datetime.now(timezone.utc).isoformat(),'logical_total':len(frame),
        'completed':int((frame.status=='completed').sum()),'fresh_remaining':int(frame.fresh_remaining.sum()),
        'reuse_remaining':int(frame.reuse_remaining.sum()),'distinct_fresh_total':len(frame)-int(frame.training_reuse.sum()),
        'proposed_deferred_fresh':int((~frame.accelerated_retained & frame.fresh_remaining).sum()),
        'accelerated_fresh_remaining':int((frame.accelerated_retained & frame.fresh_remaining).sum()),
        'complete_training_hours':full,'accelerated_proposal_training_hours':fast,
        'complete_two_worker_wall_hours':occupied(frame),'proposal_two_worker_wall_hours':occupied(frame[frame.accelerated_retained]),
        'measured_gkt_concurrency_speedup':factor,
        'hour_scope':'Same measured-reference model as current ETA; no unverified implementation speedup applied; additional test inference/final benchmarking excluded.',
        'grid_seeds':sorted(set(frame.loc[frame.category.isin(['LightSAKT_grid','baseline_tuning']),'seed'])),
        'repeated_grid_configurations':0,'required_light_grid':int((frame.category=='LightSAKT_grid').sum()),
        'scope_approval_required':True,'proposal_active':False}
    write_json('reports/optimization/SCHEDULE_COMPARISON.json',summary)
    text='# Schedule and scientific-scope audit\n\n'
    text+=f"Observed completion: {summary['completed']}/{len(frame)}; remaining fresh trainings {summary['fresh_remaining']}, reuse/evaluation slots {summary['reuse_remaining']}. The 912 figure was the planned total of distinct trainings, not the remaining count.\n\n"
    text+='|Category|Dataset|Completed|Fresh remaining|Reuse remaining|\n|---|---|---|---|---|\n'
    for r in grouped.itertuples():text+=f'|{r.category}|{r.dataset}|{r.completed}|{r.fresh_remaining}|{r.reuse_remaining}|\n'
    text+='\nEvery one of the 756 grid rows uses seed 42 only. Five matched seeds are scheduled only for validation-selected final configurations and controlled ablations. There is no grid-seed replication to remove. Splits remain frozen and identical across model seeds. No-regularization remains a no-dropout alias; default ablations reuse final LightSAKT evidence.\n\n'
    text+='Primary comparisons, all 108 LightSAKT candidates, matched five-seed final comparisons and four distinct ablations per dataset remain essential to the requested paper. The equal 36-candidate baseline tuning budget was an expanded fairness/reproducibility design; it is not an exact search space stated in Tables 4–6. Those tables specify LightSAKT only. Both historical simplified GKT variants remain present. Removing supplementary tuning still changes the established comparison protocol and requires approval.\n\n'
    text+=f"Complete remaining training forecast: {full['central']:.1f} hours ({full['low']:.1f}–{full['high']:.1f} scenarios). Proposal: {fast['central']:.1f} hours ({fast['low']:.1f}–{fast['high']:.1f}). Proposed deferred fresh trainings: {summary['proposed_deferred_fresh']}; remaining fresh trainings under proposal: {summary['accelerated_fresh_remaining']}. This is a training-only forecast, not a confidence interval. Selected-configuration uncertainty remains.\n\n"
    text+=f"Using measured independent GKT-worker throughput {factor:.3f}: complete schedule {summary['complete_two_worker_wall_hours']['central']:.1f} occupied MIG wall hours; proposal {summary['proposal_two_worker_wall_hours']['central']:.1f} hours. These forecasts retain identical final-training envelopes. The short pilot may not predict sustained full-epoch throughput; upper scenarios assume no concurrent speedup. Serial worker-hours and occupied MIG hours are distinct. See ETA_REPORT.md.\n\n"
    text+='Proposed GKT-PAM grid: learning rates [.0005,.001,.005] × hidden widths [16,30,64]. Proposed GKT-MHA grid: the same learning rates × widths [16,32,64], heads fixed at 2; historical primary width 30/head 2 remains an additional validation candidate using completed evidence. Every other baseline keeps all 36 configurations. Previously completed/running entries are retained regardless of membership. All five final seeds and all ablations remain.\n\n'
    text+='Paper impact: LightSAKT grid tables and ablation design remain intact. Table 7 and significance comparisons would describe GKT chosen using a smaller, disclosed tuning budget, not an exhaustive 36-configuration GKT search. Equal validation access is preserved; tuning-count equality is lost, while GKT still receives substantially more measured compute than LightSAKT. No test score or response label determines the retained configurations.\n\n'
    text+='Full run-level before/after membership is in PROPOSED_DIFFERENCES.csv and the two schedule JSON files. The proposal is not active. No completed result, architecture, batch size, optimizer, split, precision, maximum epoch or early-stopping rule was changed.\n'
    Path('reports/optimization/SCHEDULE_AUDIT.md').write_text(text)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':audit()
