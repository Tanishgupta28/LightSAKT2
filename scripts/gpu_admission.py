"""MHA workers may share the GPU; PAM and all other model runs are exclusive."""
from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def admission(model,path='experiments/gpu_admission.lock'):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH if model=='GKT_MHA' else fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)
