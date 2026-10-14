import fcntl
from scripts.gpu_admission import admission


def test_pam_and_light_runs_exclude_other_workers(tmp_path):
    path=tmp_path/'gpu.lock'
    for model in ['GKT','LightSAKT','DKVMN']:
        with admission(model,path):
            with open(path,'a') as peer:
                try:fcntl.flock(peer,fcntl.LOCK_SH|fcntl.LOCK_NB)
                except BlockingIOError:pass
                else:raise AssertionError('Exclusive model incorrectly shared GPU')


def test_mha_shared_admission_excludes_pam(tmp_path):
    path=tmp_path/'gpu.lock'
    with admission('GKT_MHA',path),admission('GKT_MHA',path):
        with open(path,'a') as peer:
            try:fcntl.flock(peer,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:raise AssertionError('PAM overlapped admitted MHA workers')
