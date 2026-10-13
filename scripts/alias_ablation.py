import argparse
import json
from pathlib import Path
from src.training.engine import write_json


def main(job_id):
    job=next(j for j in json.load(open('experiments/manifest.json'))['jobs'] if j['run_id']==job_id)
    source=Path('results/raw')/job['reuse_alias_run'];destination=Path('results/raw')/job_id
    result=json.load(open(source/'metrics.json'));config=json.load(open(source/'config.json'))
    result.update(run_id=job_id,reused_training_run=job['reuse_alias_run'],scientific_variant='no_regularization',
                  equivalent_to='no_dropout',independent_observation=False)
    config.update(run_id=job_id,scientific_variant='no_regularization',equivalent_to='no_dropout')
    write_json(destination/'metrics.json',result);write_json(destination/'config.json',config)
    write_json(destination/'status.json',{'run_id':job_id,'status':'completed','reused_training_run':job['reuse_alias_run'],'independent_observation':False})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);main(parser.parse_args().job)
