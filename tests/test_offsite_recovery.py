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


@pytest.mark.parametrize('pinned,expected',[(11,11),(22,22),(None,11)])
def test_duplicate_drafts_reuse_verified_container_without_creation_or_deletion(tmp_path,monkeypatch,pinned,expected):
    monkeypatch.chdir(tmp_path)
    if pinned is not None:
        write_json('reports/offsite_backup.json',{'repository':offsite.REPO,'tag':offsite.TAG,'release_id':pinned})
    def command(args):
        if args==['gh','api',f'repos/{offsite.REPO}']:return json.dumps({'private':True})
        if '--paginate' in args:return '22\n11'
        assert args==['gh','api',f'repos/{offsite.REPO}/releases/{expected}']
        return json.dumps({'id':expected,'draft':True,'tag_name':offsite.TAG})
    monkeypatch.setattr(offsite,'command',command)
    assert offsite.release()['id']==expected
    record=json.load(open('reports/offsite_release_resolution.json'))
    assert record['matching_release_ids']==[11,22]
    assert not record['existing_releases_modified_or_deleted']


def test_recovery_upload_uses_selected_release_id_not_ambiguous_tag(tmp_path,monkeypatch):
    path=tmp_path/'checkpoint evidence.tar.gz';path.write_bytes(b'checkpoint fixture')
    calls=[]
    def command(args):calls.append(args);return json.dumps({'id':345,'state':'uploaded'})
    monkeypatch.setattr(offsite,'command',command)
    assert offsite.upload_asset({'id':11,'draft':True},path)['id']==345
    assert calls[0][-1]==f'https://uploads.github.com/repos/{offsite.REPO}/releases/11/assets?name=checkpoint+evidence.tar.gz'
    assert calls[0][calls[0].index('--input')+1]==str(path)
    assert 'release' not in calls[0]


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
