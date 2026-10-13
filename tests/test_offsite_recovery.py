import json
from pathlib import Path
import pytest
from scripts import offsite
from src.datasets.preprocess import sha256
from src.training.engine import write_json


def test_offsite_refuses_public_repository(monkeypatch):
    calls=[]
    def command(args):calls.append(args);return json.dumps({'private':False})
    monkeypatch.setattr(offsite,'command',command)
    with pytest.raises(AssertionError,match='private'):offsite.release()
    assert len(calls)==1


@pytest.mark.parametrize('corrupt',[False,True])
def test_remote_download_is_required_before_recovery_verification(tmp_path,monkeypatch,corrupt):
    monkeypatch.chdir(tmp_path)
    file=Path('backups/original.tar.gz');file.parent.mkdir();file.write_bytes(b'genuine fixture archive')
    write_json('reports/original_archive.json',{'path':str(file),'sha256':sha256(file)})
    write_json('reports/completed_run_backups.json',[])
    assets={'seed.bundle':{'name':'seed.bundle'},file.name:{'id':1,'name':file.name,'state':'uploaded','size':file.stat().st_size}}
    monkeypatch.setattr(offsite,'release',lambda:{'id':1,'html_url':'private release'})
    monkeypatch.setattr(offsite,'load_assets',lambda identity:assets)
    def command(args):raise AssertionError('Existing immutable artifact must not be uploaded again')
    monkeypatch.setattr(offsite,'command',command)
    calls=[]
    def download(args,stdout,**kwargs):
        calls.append(args);stdout.write(b'corrupt' if corrupt else file.read_bytes())
    monkeypatch.setattr(offsite.subprocess,'run',download)
    if corrupt:
        with pytest.raises(AssertionError,match='Downloaded'):offsite.sync()
        record=json.load(open('reports/offsite_backup.json'))
        assert not record['all_available_completed_archives_synced'] and file.name not in record['assets']
    else:
        result=offsite.sync()
        assert result['assets'][file.name]['download_sha256_verified']
        assert result['assets'][file.name]['sha256']==sha256(file)
        offsite.sync()
        assert len(calls)==1,'Already verified immutable remote asset should not be downloaded twice'
