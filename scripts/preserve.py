"""Preserve completed milestones while training runs in a separate process."""
import argparse
import fcntl
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path
from src.datasets.preprocess import sha256
from src.training.engine import write_json
from scripts.progress import write_progress
from scripts.verify_artifacts import verify_stored_result
from scripts.reuse_training import physical_source


def make_archive(archive,members):
    import hashlib
    archive.parent.mkdir(parents=True,exist_ok=True);tmp=archive.with_suffix('.tmp')
    with tarfile.open(tmp,'w:gz') as bundle:
        for member in members:
            if member.exists():bundle.add(member,arcname=str(member))
    with tarfile.open(tmp,'r:gz') as bundle:
        for member in bundle:
            if member.isfile():
                h=hashlib.sha256()
                with bundle.extractfile(member) as stream:
                    while block:=stream.read(1024*1024):h.update(block)
                assert h.hexdigest()==sha256(member.name), f'Backup content mismatch: {member.name}'
    tmp.replace(archive)
    return {'path':str(archive),'sha256':sha256(archive),'verified_contents':True}


def archive_completed():
    """Back up each unique completed training once; aliases reuse its archive."""
    archives=Path('backups/completed_runs');archives.mkdir(parents=True,exist_ok=True)
    index=Path('reports/completed_run_backups.json')
    previous={r['run_id']:r for r in json.load(open(index))} if index.exists() else {}
    references=[];data_verified=set()
    for path in sorted(Path('results/raw').glob('*/metrics.json')):
        digest=sha256(path);old=previous.get(path.parent.name)
        if old and old.get('archive_schema')==2 and old.get('metrics_sha256')==digest and Path(old['path']).exists() and (not old.get('result_archive') or Path(old['result_archive']['path']).exists()):
            references.append(old);continue
        result,_,_=verify_stored_result(path,data_verified)
        source,evidence=physical_source(result['run_id'])
        archive=archives/(source+'.tar.gz');record=archives/(source+'.json')
        if not archive.exists() or not record.exists():
            members=[Path('results/raw')/source,Path(evidence['checkpoint']).parent]
            if evidence.get('prediction_path'):members.append(Path(evidence['prediction_path']))
            members += [Path('configs/runs')/(source+'.json')]
            write_json(record,{'source_run_id':source,**make_archive(archive,members)})
        reference={'run_id':result['run_id'],'archive_schema':2,'metrics_sha256':digest,**json.load(open(record))}
        if result['run_id']!=source:
            # Reused training may acquire a new test evaluation after its original archive.
            # Save logical metrics/configs/predictions without duplicating the weights.
            supplemental=Path('backups/reused_results')/(result['run_id']+'.tar.gz')
            members=[path.parent,Path('configs/runs')/(result['run_id']+'.json')]
            if result.get('prediction_path') and result['prediction_path']!=evidence.get('prediction_path'):
                members.append(Path(result['prediction_path']))
            reference['result_archive']=make_archive(supplemental,members)
        references.append(reference)
    write_json('reports/completed_run_backups.json',references)


def _preserve():
    archive_completed()
    subprocess.run(['python','-m','scripts.eta'],check=True)
    subprocess.run(['python','-m','scripts.summarize'],check=True)
    subprocess.run(['python','-m','scripts.report'],check=True)
    subprocess.run(['git','add','.'],check=True)
    changed=subprocess.run(['git','diff','--cached','--quiet']).returncode!=0
    if changed:subprocess.run(['git','commit','-qm','Preserve verified experiment progress and status'],check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    backup=Path('backups')/(commit+'.bundle')
    if not backup.exists():subprocess.run(['git','bundle','create',str(backup),'--all'],check=True)
    subprocess.run(['git','bundle','verify',str(backup)],check=True,stdout=subprocess.DEVNULL)
    write_json('reports/git_backup.json',{'path':str(backup),'sha256':sha256(backup),'git_commit':commit,'verified':True,'storage':'durable /workspace mount; not independent offsite artifact storage'})
    push=subprocess.run(['git','push'],capture_output=True,text=True)
    remote=subprocess.run(['git','ls-remote','origin','refs/heads/research-rebuild'],capture_output=True,text=True)
    remote_commit=remote.stdout.split()[0] if remote.returncode==0 and remote.stdout.strip() else None
    status='synchronized' if push.returncode==0 and remote_commit==commit else 'push pending'
    write_json('reports/github_sync.json',{'commit':commit,'remote_commit':remote_commit,'push_exit_code':push.returncode,'verified':status=='synchronized'})
    Path('GITHUB_SYNC_PENDING.md').write_text(f'# GitHub synchronization\n\nStatus: {status}\n\nBranch: research-rebuild\n\nLatest pushed or pending milestone commit: {commit}\n\nVerified recovery bundle: {backup}\n\nRemote: https://github.com/Tanishgupta28/LightSAKT2 (private).\n\nPending operation: '+('none for the milestone above; the live scheduler may generate newer uncommitted results.' if status=='synchronized' else 'git push origin research-rebuild; inspect reports/preservation.log for failure details.')+'\n\nRestore: git clone '+str(backup)+' recovered-history (choose an authorized recovery location). Raw files and checkpoints are in the durable workspace archive, not Git bundles.\n')
    return commit


def preserve():
    # The watcher and explicit milestone saves must never race over Git's index.
    with open('experiments/preservation.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _preserve()


def main(watch=False):
    if watch:
        watch_lock=open('experiments/preservation_watch.lock','w')
        try:fcntl.flock(watch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('A preservation watcher is already running')
        Path('experiments/preservation.pid').write_text(str(os.getpid())+'\n')
    previous=-1
    while True:
        write_progress()
        completed=sum(json.load(open(p)).get('status')=='completed' for p in Path('results/raw').glob('*/status.json'))
        if not watch or completed>previous or previous<0:
            try:
                preserve();previous=completed
            except Exception as error:
                print(f'Preservation failed: {type(error).__name__}: {error}',flush=True)
                if not watch:raise
        if not watch:return
        time.sleep(30)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');main(parser.parse_args().watch)
