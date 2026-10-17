"""CPU-only continuation of the matched-grid verification/publication workflow."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime,timezone

import numpy as np
import pandas as pd
import torch

from scripts import offsite
from scripts.preserve import make_archive,preserve
from scripts.sakt_matched_grid import DATASETS,AUDIT,INFERENCE,fingerprint
from src.datasets.preprocess import sha256
from src.training.engine import write_json


def validate():
    audit=json.load(open(AUDIT));proof=json.load(open(INFERENCE))
    assert proof['verified'] and len(proof['runs'])==108 and all(r['device']=='cuda' for r in proof['runs'])
    protected=json.load(open('reports/sakt_grid_protected_artifacts.json'))
    for r in protected['files']:assert sha256(r['path'])==r['sha256'],r['path']
    assert fingerprint()==protected['model_source_sha256']==audit['training_code_sha256']
    for dataset,letter,_ in DATASETS:
        table=pd.read_csv(f'results/tables/SAKT_Table_{letter}_{dataset}.csv')
        pair=pd.read_csv(f'results/tables/LightSAKT_vs_SAKT_grid_{dataset}.csv')
        rows=[r for r in audit['rows'] if r['dataset']==dataset]
        assert len(table)==len(pair)==len(rows)==36
        assert table.SAKT_run_id.tolist()==[r['SAKT_run_id'] for r in rows]
        np.testing.assert_allclose(table['Best Validation AUC'],[r['SAKT_validation_auc'] for r in rows],atol=1e-15,rtol=0)
        np.testing.assert_allclose(pair.difference,pair.LightSAKT_validation_auc-pair.SAKT_validation_auc,atol=1e-15,rtol=0)
        for r in rows:
            for model in ['LightSAKT','SAKT']:
                assert sha256(r[model+'_metrics_path'])==r[model+'_metrics_sha256']
                assert sha256(r[model+'_checkpoint'])==r[model+'_checkpoint_sha256']
    for f in audit['finals']:
        if f['explicit_mask']:
            r=f['explicit_mask'];assert sha256(r['path'])==r['sha256']
            with np.load(r['path']) as arrays:
                np.testing.assert_array_equal(arrays['probability_padded'][arrays['valid_mask']],arrays['probability'])
                np.testing.assert_array_equal(arrays['target_padded'][arrays['valid_mask']],arrays['target'])
    return audit


def publish():
    assert not torch.cuda.is_available(),'Publication continuation must stay on CPU'
    lock=open('experiments/sakt_matched_grid_publish.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=Path('reports/sakt_grid_publication_execution.json')
    prior=json.load(open(status)) if status.exists() else {}
    if prior.get('status')=='completed':
        validate();print('Completed publication retained; nothing repeated',flush=True);return
    while True:
        gpu=Path('reports/sakt_grid_verification_execution.json')
        if gpu.exists() and json.load(open(gpu)).get('status')=='failed':raise RuntimeError('Original-device verification failed; publication deferred')
        if (INFERENCE.exists() and json.load(open(INFERENCE)).get('verified') and gpu.exists()
            and json.load(open(gpu)).get('status')=='completed'):
            from scripts.recover import available_lock
            with available_lock('experiments/sakt_matched_grid.lock') as free:
                if free:break
        write_json(status,{'status':'waiting_for_original_device_verification','pid':os.getpid(),
            'checked_utc':datetime.now(timezone.utc).isoformat(),'healthy_training_interrupted':False})
        time.sleep(20)
    for command in [['python','-m','scripts.sakt_matched_grid','render','--plots'],
                    ['python','-m','scripts.manuscript'],['python','-m','scripts.report']]:
        write_json(status,{'status':'running','pid':os.getpid(),'step':command,'healthy_training_interrupted':False})
        subprocess.run(command,check=True)
    audit=validate();offsite_record=json.load(open('reports/offsite_backup.json'))
    archive_index={r['run_id']:r for r in json.load(open('reports/completed_run_backups.json'))}
    references=[]
    ids={r['SAKT_run_id'] for r in audit['rows']}|{f['result']['run_id'] for f in audit['finals'] if f['result']['model']=='SAKT'}
    for rid in sorted(ids):
        r=archive_index[rid];asset=offsite_record['assets'][Path(r['path']).name]
        assert asset['sha256']==r['sha256'] and asset['download_sha256_verified']
        references.append({'run_id':rid,'checkpoint_archive':r['path'],'sha256':r['sha256'],'asset_id':asset['asset_id']})
    write_json('reports/sakt_grid_checkpoint_offsite_references.json',references)
    members=[Path('paper/sakt_grid'),Path('paper/SAKT_GRID_RESULTS.md'),Path('results/predictions/sakt_matched_grid'),
        AUDIT,INFERENCE,Path('reports/SAKT_GRID_SEARCH_REPORT.md'),Path('reports/SAKT_LIGHTSAKT_COMPARISON.md'),
        Path('reports/sakt_matched_grid_verification.json'),Path('reports/sakt_grid_verification_execution.json'),
        Path('reports/sakt_grid_cpu_validation_failure.json'),Path('reports/sakt_grid_checkpoint_offsite_references.json'),
        Path('scripts/sakt_matched_grid.py'),Path('scripts/publish_sakt_grid.py'),Path('tests/test_sakt_matched_grid.py')]
    members += [p for p in Path('results/tables').glob('*') if ('SAKT_' in p.name or 'LightSAKT_vs_SAKT_grid_' in p.name)]
    members += [p for p in Path('results/figures').glob('*') if 'matched_sakt_' in p.name]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    members += [Path(p) for p in ['reports/training_code_equivalence.json',
        'reports/selected_final_inference_verification.json','reports/light_grid_inference_verification.json',
        'reports/sakt_grid_protected_artifacts.json','configs/protocol.json']]
    archive=Path('backups/publication')/f'sakt-matched-grid-{stamp}.tar.gz'
    with open('experiments/sakt_matched_grid_render.lock','a') as render_lock:
        fcntl.flock(render_lock,fcntl.LOCK_EX)
        record=make_archive(archive,members)
    assert archive.stat().st_size<2*1024**3
    offsite.TAG='research-sakt-grid-'+datetime.now(timezone.utc).strftime('%Y%m%d')
    release=offsite.release();assets=offsite.load_assets(release['id'])
    if archive.name not in assets:
        offsite.upload_asset(release,archive);assets=offsite.load_assets(release['id'])
    asset=assets[archive.name];download=Path('backups/remote_verification')/archive.name
    with download.open('wb') as output:
        subprocess.run(['gh','api','-H','Accept: application/octet-stream',f'repos/{offsite.REPO}/releases/assets/{asset["id"]}'],stdout=output,stderr=subprocess.PIPE,check=True)
    assert sha256(download)==record['sha256']
    write_json('reports/sakt_grid_offsite_backup.json',{**record,'asset_id':asset['id'],'release_tag':offsite.TAG,
        'private':True,'draft':True,'download_sha256_verified':True,'original_checkpoint_references':len(references)})
    write_json(status,{'status':'completed','pid':os.getpid(),'completed_utc':datetime.now(timezone.utc).isoformat(),
        'grid_slots_verified':108,'selected_final_seed_slots':15,'new_trainings':0,
        'offsite_download_verified':True,'healthy_training_interrupted':False})
    commit=preserve('Publish complete matched SAKT grids, five-seed comparisons and verified offsite appendix')
    print('Verified SAKT publication completed; Git milestone',commit,flush=True)


if __name__=='__main__':
    try:publish()
    except BaseException as error:
        write_json('reports/sakt_grid_publication_failure.json',{'failure_type':type(error).__name__,
            'status':'pending','original_training_untouched':True,'resume_command':"CUDA_VISIBLE_DEVICES='' python -m scripts.publish_sakt_grid"})
        raise
