"""Restore verified archive bytes into missing paths only, inside this project."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

from scripts.recover import PROJECT,processes,GPU_MODULES
from src.datasets.preprocess import sha256


def restore(archive,expected_sha256,root=PROJECT,prefix=None):
    archive=Path(archive);root=Path(root).resolve()
    assert sha256(archive)==expected_sha256, 'Archive checksum mismatch'
    plan=[];unchanged=0
    with tarfile.open(archive,'r:*') as bundle:
        for member in bundle:
            relative=PurePosixPath(member.name)
            assert not relative.is_absolute() and '..' not in relative.parts, 'Unsafe archive path'
            assert not member.issym() and not member.islnk() and not member.isdev(), 'Unsupported archive entry'
            if prefix and not str(relative).startswith(prefix):continue
            if not member.isfile():continue
            target=root.joinpath(*relative.parts)
            assert target.resolve().is_relative_to(root), 'Archive escapes project root'
            data=bundle.extractfile(member).read();digest=hashlib.sha256(data).hexdigest()
            if target.exists():
                assert target.is_file() and sha256(target)==digest, f'Existing evidence differs; never overwrite: {relative}'
                unchanged+=1;continue
            plan.append((target,data))
    # All conflicts are detected before writing any member. O_EXCL protects
    # concurrently appearing files; this procedure never deletes existing data.
    for target,data in plan:
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(data)
        assert sha256(target)==hashlib.sha256(data).hexdigest()
    return {'archive':str(archive),'restored_missing_files':len(plan),'identical_existing_files':unchanged,
            'existing_files_overwritten':0,'checksum_verified':True}


def asset_record(name):
    report=json.load(open('reports/offsite_backup.json'))
    record=report['assets'].get(name)
    if record is None:
        for row in json.load(open('reports/active_recovery_index.json')) if Path('reports/active_recovery_index.json').exists() else []:
            if Path(row['path']).name==name:record=row;break
    assert record and record.get('download_sha256_verified'), 'Asset has no verified checksum reference'
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--asset');parser.add_argument('--archive',type=Path)
    parser.add_argument('--sha256');parser.add_argument('--prefix')
    parser.add_argument('--all-completed',action='store_true');parser.add_argument('--latest-active',action='store_true');args=parser.parse_args()
    assert Path.cwd().resolve()==PROJECT, 'Restore only inside the existing project root'
    assert not any(p['module'] in GPU_MODULES | {'scripts.schedule','scripts.coordinator'} for p in processes()), \
        'Live project training/controller exists; restoration is deferred without signals'
    if args.all_completed:
        names=set()
        for row in json.load(open('reports/completed_run_backups.json')):
            names.add(Path(row['path']).name)
            if row.get('result_archive'):names.add(Path(row['result_archive']['path']).name)
        for name in sorted(names):download_and_restore(name)
        print('Verified completed-result archives restored:',len(names));return
    if args.latest_active:
        from scripts.coordinator import refreshed_jobs
        done={j['run_id'] for j in refreshed_jobs() if j['status']=='completed'};latest={}
        for row in json.load(open('reports/active_recovery_index.json')):
            if row['run_id'] in done:continue
            old=latest.get(row['run_id'])
            if old is None or row['epoch']>old['epoch']:latest[row['run_id']]=row
        for row in latest.values():
            directory=Path('checkpoints')/row['run_id']
            if (directory/'latest.pt').exists() and (directory/'best.pt').exists():continue
            # Preserve any newer Git-carried history before resuming from the
            # most recent offsite optimizer/RNG state; never restore old status
            # or history bytes over existing logs.
            source=Path('results/raw')/row['run_id']
            for name in ['history.json','status.json']:
                p=source/name
                if p.exists():
                    target=Path('reports/recovery/historical_logs')/row['run_id']/(sha256(p)+'-'+name)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if not target.exists():target.write_bytes(p.read_bytes())
            download_and_restore(Path(row['path']).name,prefix='checkpoints/')
        print('Incomplete runs with verified active snapshot references:',len(latest));return
    if args.asset:print(json.dumps(download_and_restore(args.asset,args.prefix),indent=2));return
    assert args.archive and args.sha256, 'Supply --asset, --all-completed, --latest-active, or --archive and --sha256'
    print(json.dumps(restore(args.archive,args.sha256,prefix=args.prefix),indent=2))


def download_and_restore(name,prefix=None):
        record=asset_record(name);archive=Path('backups/downloaded_recovery')/name
        archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():
            with archive.open('xb') as output:
                subprocess.run(['gh','api','-H','Accept: application/octet-stream',
                    f'repos/Tanishgupta28/LightSAKT2/releases/assets/{record["asset_id"]}'],
                    stdout=output,stderr=subprocess.PIPE,check=True)
        digest=record['sha256']
        return restore(archive,digest,prefix=prefix)


if __name__=='__main__':main()
