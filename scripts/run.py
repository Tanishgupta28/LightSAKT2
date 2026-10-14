import argparse
import json
from src.training.engine import run
from scripts.reuse_training import reuse_if_available

if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True)
    args=parser.parse_args();config=json.load(open(args.config))
    if not reuse_if_available(config):run(config)
    from scripts.execution_hooks import after_run
    after_run()
