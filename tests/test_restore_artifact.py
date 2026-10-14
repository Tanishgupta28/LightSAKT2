import io
from pathlib import Path
import tarfile

import pytest

from scripts.restore_artifact import restore
from src.datasets.preprocess import sha256


def archive(path,entries):
    with tarfile.open(path,'w:gz') as bundle:
        for name,data in entries:
            entry=tarfile.TarInfo(name);entry.size=len(data);bundle.addfile(entry,io.BytesIO(data))


def test_verified_restore_is_idempotent_and_never_overwrites_existing_evidence(tmp_path):
    source=tmp_path/'source.tar.gz';root=tmp_path/'restored'
    archive(source,[('checkpoints/run/latest.pt',b'optimizer and RNG'),('results/raw/run/status.json',b'completed')])
    first=restore(source,sha256(source),root);assert first['restored_missing_files']==2
    before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert restore(source,sha256(source),root)['restored_missing_files']==0
    assert before=={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    (root/'results/raw/run/status.json').write_bytes(b'newer evidence')
    with pytest.raises(AssertionError,match='never overwrite'):restore(source,sha256(source),root)


def test_restore_rejects_checksums_and_path_traversal_before_writes(tmp_path):
    source=tmp_path/'source.tar.gz';root=tmp_path/'restored'
    archive(source,[('safe.txt',b'safe'),('../outside.txt',b'unsafe')])
    with pytest.raises(AssertionError,match='checksum'):restore(source,'incorrect',root)
    with pytest.raises(AssertionError,match='Unsafe archive path'):restore(source,sha256(source),root)
    assert not root.exists() and not (tmp_path/'outside.txt').exists()
