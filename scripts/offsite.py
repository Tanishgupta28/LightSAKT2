"""Immutable private release assets; never publish, replace, or delete evidence."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlencode
from src.datasets.preprocess import sha256
from src.training.engine import write_json

REPO='Tanishgupta28/LightSAKT2'
TAG='research-recovery-20261013'


def command(args):
    # Provider failures can contain signed asset URLs; do not emit credentials.
    return subprocess.check_output(args,text=True,stderr=subprocess.PIPE).strip()


def release():
    repository=json.loads(command(['gh','api',f'repos/{REPO}']))
    assert repository['private'],'Artifact backups require the existing private repository'
    lookup=['gh','api',f'repos/{REPO}/releases','--paginate','--jq',f'.[] | select(.tag_name=="{TAG}") | .id']
    identities={int(value) for value in command(lookup).splitlines() if value.strip()}
    if not identities:
        notes=Path('reports/OFFSITE_RECOVERY.md')
        command(['gh','release','create',TAG,'--repo',REPO,'--draft','--target','research-rebuild',
                 '--title','Verified research recovery artifacts — experiments incomplete','--notes-file',str(notes)])
        for _ in range(5):
            identities={int(value) for value in command(lookup).splitlines() if value.strip()}
            if identities:break
            time.sleep(1)
    assert identities,'Recovery release lookup returned no IDs'
    # GitHub permits duplicate unpublished drafts with the same tag. Preserve
    # every draft, but continue the container bound to the verified asset ledger.
    ledger=Path('reports/offsite_backup.json')
    prior=json.load(open(ledger)) if ledger.exists() else {}
    pinned=prior.get('release_id') if prior.get('repository')==REPO and prior.get('tag')==TAG else None
    identity=pinned if pinned in identities else min(identities)
    result=json.loads(command(['gh','api',f'repos/{REPO}/releases/{identity}']))
    assert result['draft'],'Recovery artifact container must remain a draft'
    assert result['tag_name']==TAG,'Recovery release tag differs from requested container'
    if len(identities)>1:
        write_json('reports/offsite_release_resolution.json',{
            'updated_utc':datetime.now(timezone.utc).isoformat(),'repository':REPO,'tag':TAG,
            'matching_release_ids':sorted(identities),'selected_release_id':identity,
            'selection':'verified asset ledger' if identity==pinned else 'oldest release ID',
            'existing_releases_modified_or_deleted':False,'uploads_address_release_id':True})
    return result


def upload_asset(remote,path):
    """Address the selected private draft by ID, avoiding ambiguous tag uploads."""
    assert remote['draft'],'Recovery artifact container must remain a draft'
    query=urlencode({'name':Path(path).name})
    endpoint=f'https://uploads.github.com/repos/{REPO}/releases/{remote["id"]}/assets?{query}'
    return json.loads(command(['gh','api','--method','POST','-H',
        'Content-Type: application/octet-stream','--input',str(path),endpoint]))


def sync():
    remote=release();assets=load_assets(remote['id'])
    record_path=Path('reports/offsite_backup.json')
    previous=json.load(open(record_path)) if record_path.exists() else {'assets':{}}
    files=[json.load(open('reports/original_archive.json'))]
    for entry in json.load(open('reports/completed_run_backups.json')):
        files.append(entry)
        if entry.get('result_archive'):files.append(entry['result_archive'])
    # One independent full history seed; subsequent source/metric commits stay on
    # the pushed branch and all local milestone bundles remain on NFS.
    if not any(name.endswith('.bundle') for name in assets):files.append(json.load(open('reports/git_backup.json')))
    files={str(Path(f['path'])):f for f in files}
    assert len(assets)+sum(Path(p).name not in assets for p in files)<=1000
    result={**previous,'repository':REPO,'tag':TAG,'release_id':remote['id'],'url':remote['html_url'],
            'private':True,'draft':True,'all_available_completed_archives_synced':False,
            'updated_utc':datetime.now(timezone.utc).isoformat()}
    write_json(record_path,result)
    for path,entry in files.items():
        source=Path(path);name=source.name;expected=entry['sha256']
        assert source.stat().st_size<2*1024**3 and sha256(source)==expected
        asset=assets.get(name)
        if asset is None:
            print('Uploading private recovery asset:',name,flush=True)
            upload_asset(remote,source)
            assets=load_assets(remote['id']);asset=assets[name]
        assert asset['state']=='uploaded' and asset['size']==source.stat().st_size
        if asset.get('digest'):assert asset['digest']=='sha256:'+expected
        old=previous['assets'].get(name)
        if old and old['asset_id']==asset['id'] and old['sha256']==expected and old['download_sha256_verified']:
            result['assets'][name]=old;continue
        destination=Path('backups/remote_verification')/name;destination.parent.mkdir(parents=True,exist_ok=True)
        for attempt in range(3):
            try:
                with destination.open('wb') as stream:
                    subprocess.run(['gh','api','-H','Accept: application/octet-stream',
                                    f'repos/{REPO}/releases/assets/{asset["id"]}'],stdout=stream,stderr=subprocess.PIPE,check=True)
                break
            except subprocess.CalledProcessError:
                if attempt==2:raise
                print('Retrying interrupted artifact download:',name,flush=True);time.sleep(1)
        assert sha256(destination)==expected,'Downloaded recovery artifact differs'
        result['assets'][name]={'asset_id':asset['id'],'path':path,'sha256':expected,
            'bytes':asset['size'],'download_sha256_verified':True,'verified_utc':datetime.now(timezone.utc).isoformat()}
        write_json(record_path,result)
    result['all_available_completed_archives_synced']=True
    result['updated_utc']=datetime.now(timezone.utc).isoformat()
    write_json(record_path,result)
    print('Verified private offsite assets:',len(result['assets']),flush=True)
    return result


def load_assets(release_id):
    rows=command(['gh','api',f'repos/{REPO}/releases/{release_id}/assets?per_page=100',
                  '--paginate','--jq','.[] | @json'])
    return {a['name']:a for a in (json.loads(row) for row in rows.splitlines())}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    lock=open('experiments/offsite.lock','w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('An offsite backup worker is already running')
    if args.watch:Path('experiments/offsite.pid').write_text(str(os.getpid())+'\n')
    while True:
        try:sync()
        except Exception as error:
            write_json('reports/offsite_backup_failure.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),
                'reason':f'{type(error).__name__}: {error}','local_backups_retained':True})
            print('Offsite synchronization pending:',type(error).__name__,error,flush=True)
            if not args.watch:raise
        if not args.watch:break
        time.sleep(60)
