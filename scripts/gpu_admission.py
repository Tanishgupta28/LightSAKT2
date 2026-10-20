"""MHA workers may share the GPU; PAM and all other model runs are exclusive."""
from contextlib import contextmanager
import fcntl
import time
from pathlib import Path


@contextmanager
def admission(model,path='experiments/gpu_admission.lock'):
    from scripts.user_training_hold import require_training_enabled
    require_training_enabled()
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH if model=='GKT_MHA' else fcntl.LOCK_EX)
        require_training_enabled()
        slot=None
        try:
            if model=='GKT_MHA':
                # Kernel-owned locks bound concurrency even after a controller
                # crash leaves an independently healthy training child alive.
                while slot is None:
                    for i in range(2):
                        candidate=open(str(path)+f'.mha{i}','a')
                        try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:candidate.close();continue
                        slot=candidate;break
                    if slot is None:time.sleep(.1)
            yield
        finally:
            if slot is not None:slot.close()
            fcntl.flock(lock,fcntl.LOCK_UN)
