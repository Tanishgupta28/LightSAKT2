"""Hourly resumable active-run recovery snapshots; independent private draft assets."""
import io
import json
import os
import subprocess
import tarfile
import time
from datetime import datetime,timezone
from pathlib import Path
import torch
from src.datasets.preprocess import sha256
from src.training.engine import write_json
from scripts.progress import snapshot
from scripts import offsite


def capture(run_id):
    source=Path('checkpoints')/run_id
    for _ in range(5):
        latest_bytes=(source/'latest.pt').read_bytes();best_bytes=(source/'best.pt').read_bytes()
        latest=torch.load(io.BytesIO(latest_bytes),map_location='cpu',weights_only=False)
        best=torch.load(io.BytesIO(best_bytes),map_location='cpu',weights_only=False)
        if best['epoch']==latest['best_epoch']:break
        time.sleep(.2)
    else:raise RuntimeError('Checkpoint boundary changed during snapshot; retry next cycle')
    epoch=latest['epoch'];dest=Path('backups/active_runs')/f'{run_id}-epoch{epoch}.tar.gz';dest.parent.mkdir(parents=True,exist_ok=True)
    payloads={f'checkpoints/{run_id}/latest.pt':latest_bytes,f'checkpoints/{run_id}/best.pt':best_bytes,
        f'results/raw/{run_id}/config.json':json.dumps(latest['config'],indent=2).encode(),
        f'results/raw/{run_id}/history.json':json.dumps(latest['history'],indent=2).encode(),
        f'results/raw/{run_id}/status.json':json.dumps({'run_id':run_id,'status':'interrupted','last_completed_epoch':epoch,
            'resume_command':f'python -m scripts.run --config results/raw/{run_id}/config.json',
            'failure_reason':'Restored active-run snapshot; original live run was not interrupted'}).encode()}
    if not dest.exists():
        temporary=dest.with_suffix('.tmp')
        with tarfile.open(temporary,'w:gz') as archive:
            for name,data in payloads.items():
                info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
        with tarfile.open(temporary,'r:gz') as archive:
            for member in archive:assert archive.extractfile(member).read()==payloads[member.name]
        temporary.replace(dest)
    return {'run_id':run_id,'epoch':epoch,'path':str(dest),'sha256':sha256(dest),'captured_time':time.time(),
            'verified_contents':True,'live_run_interrupted':False,'config':latest['config']}


def upload(record):
    # Separate daily draft containers keep the complete-result release below its asset limit.
    offsite.TAG='research-active-recovery-'+datetime.now(timezone.utc).strftime('%Y%m%d')
    release=offsite.release();assets=offsite.load_assets(release['id']);path=Path(record['path'])
    if path.name not in assets:
        offsite.command(['gh','release','upload',offsite.TAG,str(path),'--repo',offsite.REPO]);assets=offsite.load_assets(release['id'])
    asset=assets[path.name];dest=Path('backups/remote_verification')/path.name
    with dest.open('wb') as stream:
        subprocess.run(['gh','api','-H','Accept: application/octet-stream',f'repos/{offsite.REPO}/releases/assets/{asset["id"]}'],stdout=stream,stderr=subprocess.PIPE,check=True)
    assert sha256(dest)==record['sha256'],'Offsite active snapshot differs'
    return {**record,'asset_id':asset['id'],'release_tag':offsite.TAG,'private':True,'draft':True,'download_sha256_verified':True,
            'verified_utc':datetime.now(timezone.utc).isoformat()}


def cycle():
    index=Path('reports/active_recovery_index.json');records=json.load(open(index)) if index.exists() else []
    for job in snapshot()['running_jobs']:
        run_id=job['run_id'];history=Path('results/raw')/run_id/'history.json'
        if not history.exists():continue
        epochs=json.load(open(history))
        # Fast jobs receive completed-run backups; long jobs additionally get resumable snapshots.
        if not epochs or sum(r['epoch_seconds'] for r in epochs)<900:continue
        prior=[r for r in records if r['run_id']==run_id]
        if prior and time.time()-max(r['captured_time'] for r in prior)<3600:continue
        record=capture(run_id);record=upload(record);records.append(record);write_json(index,records)
        print('Verified private active-run snapshot',run_id,'epoch',record['epoch'],flush=True)


if __name__=='__main__':
    assert not torch.cuda.is_available(),'Active recovery is CPU-only'
    import fcntl
    lock=open('experiments/active_recovery.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    Path('experiments/active_recovery.pid').write_text(str(os.getpid())+'\n')
    while True:
        try:cycle()
        except Exception as error:
            write_json('reports/active_recovery_failure.json',{'failure_type':type(error).__name__,'local_checkpoints_retained':True,
                'updated_utc':datetime.now(timezone.utc).isoformat()})
            print('Active recovery pending:',type(error).__name__,flush=True)
        time.sleep(300)
