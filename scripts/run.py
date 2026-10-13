import argparse
import json
from src.training.engine import run

if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True)
    args=parser.parse_args();run(json.load(open(args.config)))
