"""Repeat real-data label-exclusion and frozen-feature audits; CPU only, new outputs only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import json
import numpy as np
import pandas as pd
from common import ASSETS,ROOT,write,sha
from scripts import audit_professor_campaign_20261030 as old
from scripts.question_difficulty_inputs import difficulty_features
from scripts.professor_data_20261030 import heldout_difficulty


def main():
    plan=json.loads((ROOT/'experiments/professor_campaign_20261030/plan.json').read_text())
    old.OUT=ASSETS/'verification/metrics_verification/current_leakage'
    old.leakage_audit(plan)  # Writes only to explicitly replaced new output root.
    work=ROOT/'.tools/ednet_replacement_20261104/workspace';d='EdNetKT1_5k_first'
    data=work/f'data/processed/{d}/interactions.csv'
    frame=pd.read_csv(data,dtype={'student':str})
    spec=json.loads((ROOT/'reports/ednet_replacement_20261104/feature_specification.json').read_text())
    assert sha(data)==spec['processed_sha256']
    assert sha(work/f'data/splits/{d}.json')==spec['split_sha256']
    tv=frame[frame.split.isin(['train','validation'])].copy()
    original=difficulty_features(tv,smoothing=10.)
    changed=tv.copy();mask=changed.split=='validation';changed.loc[mask,'label']=1-changed.loc[mask,'label']
    np.testing.assert_allclose(difficulty_features(changed),original,atol=1e-12,rtol=0)
    own_id=tv[tv.split=='train'].student.value_counts().index[0];own=(tv.student==own_id).to_numpy()
    changed=tv.copy();changed.loc[own,'label']=1-changed.loc[own,'label']
    np.testing.assert_allclose(difficulty_features(changed)[own],original[own],atol=1e-12,rtol=0)
    changed=tv.copy();other=(changed.split=='train')&~own;changed.loc[other,'label']=1-changed.loc[other,'label']
    positive=float(np.max(np.abs(difficulty_features(changed)[own]-original[own])));assert positive>0
    training=frame[frame.split=='train']
    hashes={}
    for split in ['train_validation','test']:
        path=Path(spec[split+'_feature_path']);assert sha(path)==spec[split+'_feature_sha256'];hashes[str(path)]=sha(path)
        frozen=json.loads(path.read_text())['students']
        if split=='train_validation':rows=tv.copy();rows['difficulty']=original
        else:
            rows=frame[frame.split=='test'].copy()
            rows['difficulty']=heldout_difficulty(training,rows[['q','problem_id']],smoothing=10.)
        for identity,g in rows.groupby('student',sort=True):
            np.testing.assert_allclose(g.difficulty,frozen[identity]['values'],rtol=0,atol=1e-12)
        if split=='test':
            unseen=~rows.problem_id.isin(training.problem_id.unique());assert np.max(np.abs(rows.loc[unseen,'difficulty']))<1e-12
    # Same frozen train/test feature construction confirmed for old datasets by old.leakage_audit.
    hashes[str(data)]=sha(data)
    write(ASSETS/'verification/metrics_verification/current_leakage/EdNet_real_data_audit.json',{
       'status':'PASS_OFFLINE_PROTOCOL','training_rows':len(training),'own_student_rows_perturbed':int(own.sum()),
       'whole_own_student_label_flip_invariant':True,'all_validation_label_flip_invariant':True,
       'heldout_transform_interface':'q and authentic problem_id only; no heldout labels',
       'other_student_positive_control_maximum_change':positive,'frozen_train_validation_test_features_reproduced':True,
       'unseen_question_correction_zero':True,'source_hashes':hashes,
       'scope':'Offline fitted-training-corpus statistics; later OTHER training-student observations permitted. Does not validate forward-time availability.'})
    print('PASS: actual-data leakage and frozen sidecars rechecked on allthree datasets; offline limitation remains.')


if __name__=='__main__':main()
