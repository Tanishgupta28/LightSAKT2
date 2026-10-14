"""Reconstruct locked records in staging; install only absent data paths."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

from scripts.recover import PROJECT,processes,GPU_MODULES
from src.datasets.preprocess import prepare,sha256
from src.training.engine import write_json


def reconstruct(dataset,stage):
    (Path(stage)/'reports').mkdir(parents=True,exist_ok=True)
    reference=json.load(open(f'reports/{dataset}_statistics.json'))
    raw=Path('data/raw')/(dataset+('.tsv' if dataset=='Algebra2005' else '.csv'))
    assert sha256(raw)==reference['raw_sha256'], 'Raw dataset differs from locked protocol'
    restored=prepare(dataset,raw,root=stage,seed=reference['split_seed'])
    for name in ['raw_sha256','processed_sha256','split_sha256','num_q','clean_interactions','students']:
        assert restored[name]==reference[name], 'Reconstruction differs from locked data: '+name
    return {'dataset':dataset,'staging_root':str(stage),'raw_sha256':reference['raw_sha256'],
            'processed_sha256':restored['processed_sha256'],'split_sha256':restored['split_sha256'],
            'split_seed':reference['split_seed'],'verified':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--install-missing',action='store_true');args=parser.parse_args()
    assert Path.cwd().resolve()==PROJECT, 'Use the existing project root'
    if args.install_missing:
        assert not any(p['module'] in GPU_MODULES | {'scripts.schedule','scripts.coordinator'} for p in processes()), \
            'Live training is protected; only staging verification is allowed'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    stage=Path('backups/recovery_checks')/('preprocessing-'+stamp);records=[]
    for dataset in ['ASSIST2009','ASSIST2015','Algebra2005']:
        records.append(reconstruct(dataset,stage))
        if args.install_missing:
            for source in list((stage/'data/processed'/dataset).glob('*'))+[stage/'data/splits'/(dataset+'.json')]:
                target=Path(source.relative_to(stage))
                if target.exists():assert sha256(target)==sha256(source), 'Existing locked data differs; never overwrite'
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
    result={'verified_utc':datetime.now(timezone.utc).isoformat(),'verified':True,'datasets':records,
            'existing_files_overwritten':0,'installed_missing':args.install_missing,
            'real_dataset_reconstruction':True,'training_restarted':False}
    write_json('reports/recovery/preprocessing_reconstruction.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
