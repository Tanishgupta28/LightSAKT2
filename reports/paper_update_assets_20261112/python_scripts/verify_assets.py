"""Independently check exports, trace graph inputs and byte-reproduce all graphs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
import numpy as np
import pandas as pd
from PIL import Image
from common import ASSETS, DATASETS, MODELS, SEEDS, sha, write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reproduce',action='store_true')
    parser.add_argument('--sources',action='store_true');args=parser.parse_args()
    singles=pd.read_csv(ASSETS/'source_data/verified_metric_references/individual_seed_metrics.csv',float_precision='round_trip')
    master=pd.read_csv(ASSETS/'source_data/verified_metric_references/master_metrics.csv',float_precision='round_trip')
    matrix=pd.read_csv(ASSETS/'source_data/verified_metric_references/completion_matrix.csv')
    assert len(matrix)==135 and not matrix.duplicated(['dataset','model','seed']).any()
    assert not singles.duplicated(['dataset','model','seed']).any()
    assert len(master)==27 and not master.duplicated(['dataset','model']).any()
    metrics=['validation_auc','test_auc','test_accuracy','test_f1','test_precision','test_recall','test_mse','test_bce']
    counts={}
    for row in master.to_dict('records'):
        g=singles[(singles.dataset==row['dataset'])&(singles.model==row['model'])]
        assert len(g)==row['seed_count']
        complete=set(g.seed)==set(SEEDS)
        if len(g):assert g.processed_sha256.nunique()==g.split_sha256.nunique()==1
        for m in metrics:
            if complete:
                assert abs(row[m+'_mean']-mean(g[m]))<1e-14
                assert abs(row[m+'_std']-stdev(g[m]))<1e-14
            else:assert np.isnan(row[m+'_mean']) and np.isnan(row[m+'_std'])
    # Independent confusion reconstruction including zero-denominator conventions.
    for r in singles.to_dict('records'):
        tp,tn,fp,fn=[r['test_'+k] for k in ['tp','tn','fp','fn']]
        assert abs(r['test_accuracy']-(tp+tn)/(tp+tn+fp+fn))<1e-14
        assert abs(r['test_f1']-2*tp/(2*tp+fp+fn))<1e-14
        assert abs(r['test_precision']-tp/(tp+fp))<1e-14
        assert abs(r['test_recall']-tp/(tp+fn))<1e-14
    epochs=pd.read_csv(ASSETS/'source_data/epoch_records/completed_final_epochs.csv',float_precision='round_trip')
    for _,g in epochs.groupby(['dataset','model','seed']):
        assert sorted(g.epoch)==list(range(1,len(g)+1))
    for r in singles.to_dict('records'):
        g=epochs[(epochs.dataset==r['dataset'])&(epochs.model==r['model'])&(epochs.seed==r['seed'])]
        assert int(g.loc[g.validation_auc.idxmax(),'epoch'])==r['best_epoch']
        assert len(g)==r['epochs_trained']
    graph_files=[]
    for p in sorted((ASSETS/'graphs').rglob('*.provenance.json')):
        r=json.loads(p.read_text())
        assert r['generator'].endswith('.py')
        for entry in r['input_files']:assert sha(ASSETS/entry['path'])==entry['sha256']
        for entry in r['outputs']:
            f=ASSETS/entry['path'];assert sha(f)==entry['sha256'];graph_files.append(f)
            if f.suffix=='.png':
                with Image.open(f) as im:assert min(im.info['dpi'])>=300
    before={str(p.relative_to(ASSETS)):sha(p) for p in graph_files}
    if args.reproduce:
        subprocess.run([sys.executable,'-B',str(ASSETS/'python_scripts/generate_all_paper_graphs.py')],check=True)
        after={str(p.relative_to(ASSETS)):sha(p) for p in graph_files}
        assert before==after, 'Graph regeneration changed bytes'
    if args.sources:
        source=json.loads((ASSETS/'verification/source_fingerprints.json').read_text())
        assert all(sha(p)==digest for p,digest in source['files'].items())
    write(ASSETS/'verification/graph_reproducibility/verification.json',{
        'status':'PASS','aggregate_rows':27,'verified_seed_records':len(singles),
        'complete_five_seed_groups':int((master.seed_count==5).sum()),'pending_groups':int((master.seed_count!=5).sum()),
        'mean_sample_SD_recomputed':True,'incomplete_aggregate_numeric_cells_blank':True,
        'threshold_confusion_counts_rechecked':True,'epoch_selection_consistent':True,
        'graph_sets':len(graph_files)//3,'graph_files':len(graph_files),'PNG_min_DPI':300,
        'all_graph_inputs_SHA256_matched':True,'byte_identical_regeneration':bool(args.reproduce),
        'original_source_files_unchanged':bool(args.sources),
        'graph_SHA256':before,'portable_scope':'Plot/table reproduction from compact extracts; full checkpoint/model replay requires separately preserved datasets/checkpoints.'})
    print(f'PASS:27groups,{len(singles)}seeds,{len(graph_files)}graph files; reproduce={args.reproduce}; source preservation={args.sources}')


if __name__=='__main__':main()
