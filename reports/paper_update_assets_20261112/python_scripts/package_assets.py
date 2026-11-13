"""Deterministic ZIP, payload hashes and CRC verification. No checkpoints/raw datasets."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from common import ASSETS,ROOT,sha,write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--destination',type=Path);args=parser.parse_args()
    destination=args.destination or ROOT.parent/'LightSAKT2/backups/paper_update_assets_20261112'
    destination.mkdir(parents=True,exist_ok=True)
    files=[p for p in sorted(ASSETS.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='PAYLOAD_SHA256.json']
    assert not any(p.suffix in ['.pt','.npz','.pkl'] for p in files)
    manifest={str(p.relative_to(ASSETS)):sha(p) for p in files}
    write(ASSETS/'PAYLOAD_SHA256.json',manifest);files.append(ASSETS/'PAYLOAD_SHA256.json')
    archive=destination/'LightSAKT_paper_update_assets_20261112.zip'
    temp=archive.with_suffix('.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(files):
            info=zipfile.ZipInfo('paper_update_assets_20261112/'+str(p.relative_to(ASSETS)),(2026,11,12,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            z.writestr(info,p.read_bytes())
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
        for relative,digest in manifest.items():
            assert hashlib.sha256(z.read('paper_update_assets_20261112/'+relative)).hexdigest()==digest
    temp.replace(archive)
    write(destination/'archive_verification.json',{'status':'PASS','archive':str(archive),'sha256':sha(archive),
        'bytes':archive.stat().st_size,'files':len(files),'CRC_and_all_payload_SHA256_verified':True,
        'raw_datasets_checkpoints_original_manuscript_included':False})
    print(archive,sha(archive),len(files),'files, CRC/SHA256 PASS')


if __name__=='__main__':main()
