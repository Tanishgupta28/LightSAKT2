import argparse
import json
from src.training.engine import run
from scripts.reuse_training import reuse_if_available

if __name__ == "__main__":
    from scripts.user_training_hold import require_training_enabled
    require_training_enabled()
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True)
    args=parser.parse_args();config=json.load(open(args.config))
    from scripts.runtime_policy import install
    from scripts.gpu_admission import admission
    # Queued workers allocate no CUDA tensors before admission. The controller
    # can be upgraded without signaling or interrupting a healthy training job.
    from pathlib import Path
    from src.training.engine import write_json
    status=Path('results/raw')/config['run_id']/'status.json'
    prior=json.load(open(status)) if status.exists() else {}
    if prior.get('status')!='completed':
        write_json(status,{**prior,'run_id':config['run_id'],'status':'pending','waiting_for_gpu_admission':True})
    with admission(config['model']):
        require_training_enabled()
        install(config)
        if not reuse_if_available(config):run(config)
    from scripts.execution_hooks import after_run
    after_run()
