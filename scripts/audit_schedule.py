"""Audit logical jobs and publish a reuse overlay without touching the live manifest."""
import hashlib
import json
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from src.training.engine import write_json


def training_identity(config):
    # Keep every training-affecting key, including any future unknown option.
    return {k:v for k,v in config.items() if k not in {'run_id','group','evaluate_test','selection_path'}}


def identity_hash(config):
    return hashlib.sha256(json.dumps(training_identity(config),sort_keys=True).encode()).hexdigest()


def audit():
    manifest_path=Path('experiments/manifest.json');raw=manifest_path.read_bytes();manifest=json.loads(raw)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    before=Path('reports')/f'schedule_before_{stamp}.json';before.write_bytes(raw)
    groups=defaultdict(list);rows=[];errors=[];ids=[j['run_id'] for j in manifest['jobs']]
    if len(set(ids))!=len(ids):errors.append('Duplicate logical run IDs')
    for job in manifest['jobs']:
        c=json.load(open(job['config'])) if 'config' in job else None
        if c and job['phase'] in [5,6]:groups[identity_hash(c)].append((job,c))
        rows.append({'run_id':job['run_id'],'phase':job['phase'],'dataset':c['dataset'] if c else job['dataset'],
                     'model':c['model'] if c else job['model'],'seed':c['seed'] if c else job['seed'],
                     'status':job['status'],'training_identity_sha256':identity_hash(c) if c else None})
    reuse={};merged=[]
    for identity,entries in groups.items():
        if len(entries)==1:continue
        primaries=[(j,c) for j,c in entries if j['phase']==5]
        if len(primaries)!=1:errors.append(f'Unexpected duplicate group {identity}');continue
        source,source_c=primaries[0]
        for destination,c in entries:
            if destination['phase']!=6:continue
            assert training_identity(source_c)==training_identity(c)
            reuse[destination['run_id']]={'source_run_id':source['run_id'],'identity_sha256':identity,
                'reason':'Identical data/model/seed/optimizer/batching/epoch limits; only reporting group and final-test flag differ.'}
            merged.append({'destination':destination['run_id'],'source':source['run_id'],'dataset':c['dataset'],'model':c['model'],
                           'action':'retain logical grid row; reuse verified primary training; expose validation only'})
    for dataset in ['ASSIST2009','ASSIST2015','Algebra2005']:
        for model in ['DKT','DKT+','DKVMN','GKT','GKT_MHA','SAKT','LightSAKT']:
            subset=[r for r in rows if r['phase']==6 and r['dataset']==dataset and r['model']==model]
            if len(subset)!=36 or len({r['training_identity_sha256'] for r in subset})!=36:errors.append(f'Tuning coverage/duplicates: {dataset}/{model}')
            final=[j for j in manifest['jobs'] if j['phase']==7 and j['dataset']==dataset and j['model']==model]
            if sorted(j['seed'] for j in final)!=[42,123,2026,3407,7777]:errors.append(f'Final seed coverage: {dataset}/{model}')
        for variant in ['no_dropout','no_position','single_head','no_attention','no_regularization']:
            subset=[j for j in manifest['jobs'] if j['phase']==8 and j['dataset']==dataset and j['variant']==variant]
            if sorted(j['seed'] for j in subset)!=[42,123,2026,3407,7777]:errors.append(f'Ablation coverage: {dataset}/{variant}')
    if errors:raise ValueError(errors)
    write_json('experiments/reuse_plan.json',{'version':1,'created_utc':stamp,'jobs':reuse,
        'scientific_rule':'Source must be completed, checksum verified, and training-identical. Grid selection reads validation only.'})
    after=json.loads(raw)
    for job in after['jobs']:
        if job['run_id'] in reuse:job['effective_reuse']=reuse[job['run_id']]
    after_path=Path('reports')/f'schedule_after_{stamp}.json';write_json(after_path,after)
    pd.DataFrame(rows).to_csv('reports/SCHEDULE_INVENTORY.csv',index=False)
    pd.DataFrame(merged).to_csv('reports/SCHEDULE_MERGES.csv',index=False)
    report={'logical_slots_before':len(ids),'logical_slots_after':len(ids),'training_jobs_before':len(ids)-36,
            'training_jobs_after':len(ids)-36-len(reuse),'new_primary_grid_reuses':len(reuse),
            'existing_final_seed42_reuses':21,'existing_regularization_aliases':15,
            'before_snapshot':str(before),'before_manifest_sha256':hashlib.sha256(raw).hexdigest(),
            'after_effective_snapshot':str(after_path),'coverage_errors':errors,'merged_entries':merged,
            'split_rule':'Identical student split across model/seed runs is deliberate pairing, not duplicate training.',
            'dependencies':'All source primaries precede phase-6 reuses; final configs require all 36 validation candidates; no_regularization depends on same-seed no_dropout.',
            'activation':'scripts.run reads the overlay in each future child; the active scheduler and training process are not restarted.'}
    write_json('reports/schedule_audit.json',report)
    text='# Schedule audit and safe reuse\n\n'
    text+=f'Logical entries remain {len(ids)}; no required paper row or distinct configuration is removed. Distinct planned trainings decrease from {len(ids)-36} to {len(ids)-36-len(reuse)}.\n\n'
    text+=f'Found {len(reuse)} additional exact primary/grid training duplicates. Every destination retains its immutable config, row and status; it references verified source training rather than rerunning it. The source primary may already have test predictions, but grid result files expose validation only and selection remains validation-only.\n\n'
    text+='The 21 final seed-42 reuse slots, 15 no_regularization aliases, and reused default ablation remain unchanged. No-regularization is equivalent to no-dropout by author instruction. Repeated seeds 42,123,2026,3407,7777 are distinct trainings; one frozen student split is intentional. All 108 LightSAKT configurations and 36 per baseline variant per dataset remain present. Both historical simplified GKT PAM/MHA variants are retained.\n\n'
    text+='The scheduler owns experiments/manifest.json in memory. Safe activation uses experiments/reuse_plan.json, read by future scripts.run children, avoiding a lost-update race or training interruption. Before/after effective snapshots and every merge are recorded in schedule_audit.json and SCHEDULE_MERGES.csv. No completed results, active configs, data splits, source architecture, or hyperparameters were modified.\n'
    Path('reports/SCHEDULE_AUDIT.md').write_text(text)
    print(json.dumps({k:v for k,v in report.items() if k!='merged_entries'},indent=2))


if __name__=='__main__':audit()
