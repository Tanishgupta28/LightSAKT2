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
    parser.add_argument('--sha256');parser.add_argument('--prefix');args=parser.parse_args()
    assert Path.cwd().resolve()==PROJECT, 'Restore only inside the existing project root'
    assert not any(p['module'] in GPU_MODULES | {'scripts.schedule','scripts.coordinator'} for p in processes()), \
        'Live project training/controller exists; restoration is deferred without signals'
    if args.asset:
        record=asset_record(args.asset);archive=Path('backups/downloaded_recovery')/args.asset
        archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():
            with archive.open('xb') as output:
                subprocess.run(['gh','api','-H','Accept: application/octet-stream',
                    f'repos/Tanishgupta28/LightSAKT2/releases/assets/{record["asset_id"]}'],
                    stdout=output,stderr=subprocess.PIPE,check=True)
        digest=record['sha256']
    else:
        assert args.archive and args.sha256, 'Supply --asset or --archive and --sha256'
        archive,digest=args.archive,args.sha256
    print(json.dumps(restore(archive,digest,prefix=args.prefix),indent=2))


if __name__=='__main__':main()
