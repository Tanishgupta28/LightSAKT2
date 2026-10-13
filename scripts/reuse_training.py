"""Publish validation evidence for training-identical logical grid entries."""
import hashlib
import json
from pathlib import Path
from scripts.audit_schedule import training_identity,identity_hash
from scripts.verify_artifacts import verify_stored_result
from src.training.engine import write_json


def training_code_compatible(result,current_code):
    if result['training_code_sha256']==current_code:return True
    path=Path('reports/training_code_equivalence.json')
    if not path.exists():return False
    # Only an explicit audited source transition for this dataset is allowed.
    return any(entry['source_sha256']==result['training_code_sha256'] and
               entry['current_sha256']==current_code and result['dataset'] in entry['datasets'] and
               entry['verified'] for entry in json.load(open(path)))


def physical_source(run_id):
    visited=set()
    while True:
        if run_id in visited:raise ValueError('Cyclic evidence reuse')
        visited.add(run_id)
        result=json.load(open(Path('results/raw')/run_id/'metrics.json'))
        if not result.get('reused_training_run'):return run_id,result
        run_id=result['reused_training_run']


def reuse_if_available(config):
    plan_path=Path('experiments/reuse_plan.json')
    if not plan_path.exists():return False
    entry=json.load(open(plan_path))['jobs'].get(config['run_id'])
    if entry is None:return False
    destination=Path('results/raw')/config['run_id']
    if (destination/'status.json').exists() and json.load(open(destination/'status.json'))['status']=='completed':return True
    source=Path('results/raw')/entry['source_run_id']
    if not (source/'status.json').exists() or json.load(open(source/'status.json'))['status']!='completed':return False
    result,source_config,_=verify_stored_result(source/'metrics.json')
    assert training_identity(config)==training_identity(source_config) and identity_hash(config)==entry['identity_sha256']
    current_code=hashlib.sha256(b''.join(str(p).encode()+p.read_bytes() for p in sorted(Path('src').rglob('*.py')) if 'visualization' not in p.parts)).hexdigest()
    assert training_code_compatible(result,current_code),'Reuse source training implementation differs'
    if (destination/'config.json').exists():assert json.load(open(destination/'config.json'))==config
    source_id=result['run_id'];result.update(run_id=config['run_id'],group=config['group'],
        reused_training_run=source_id,training_executed_here=False,test=None)
    for key in ['prediction_path','prediction_sha256']:result.pop(key,None)
    write_json(destination/'config.json',config);write_json(destination/'metrics.json',result)
    for name in ['history.json','environment.json']:
        write_json(destination/name,json.load(open(source/name)))
    write_json(destination/'reuse_provenance.json',{'source_run_id':source_id,'plan':str(plan_path),
        'training_identity_sha256':entry['identity_sha256'],'test_used_for_selection':False,
        'source_primary_test_access':'Source may have previously reported its prespecified primary test; it is not copied into grid selection.'})
    write_json(destination/'status.json',{'run_id':config['run_id'],'status':'completed','last_completed_epoch':result['epochs_trained'],
        'reused_training_run':source_id,'best_checkpoint':result['checkpoint'],'failure_reason':None})
    print('Reused verified training:',source_id,'->',config['run_id'],flush=True)
    return True
