"""Preserve completed milestones while training runs in a separate process."""
import argparse
import json
import subprocess
import time
from pathlib import Path
from src.datasets.preprocess import sha256
from src.training.engine import write_json


def preserve():
    subprocess.run(['python','-m','scripts.summarize'],check=True)
    subprocess.run(['python','-m','scripts.report'],check=True)
    subprocess.run(['git','add','.'],check=True)
    changed=subprocess.run(['git','diff','--cached','--quiet']).returncode!=0
    if changed:subprocess.run(['git','commit','-qm','Preserve verified experiment progress and status'],check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    backup=Path('backups')/(commit+'.bundle')
    if not backup.exists():subprocess.run(['git','bundle','create',str(backup),'--all'],check=True)
    subprocess.run(['git','bundle','verify',str(backup)],check=True,stdout=subprocess.DEVNULL)
    write_json('reports/git_backup.json',{'path':str(backup),'sha256':sha256(backup),'git_commit':commit,'verified':True,'storage':'durable /workspace mount; not independent offsite artifact storage'})
    push=subprocess.run(['git','push'],capture_output=True,text=True)
    status='synchronized' if push.returncode==0 else 'push pending'
    Path('GITHUB_SYNC_PENDING.md').write_text(f'# GitHub synchronization\n\nStatus: {status}\n\nBranch: research-rebuild\n\nLatest pushed or pending milestone commit: {commit}\n\nVerified recovery bundle: {backup}\n\nRemote: https://github.com/Tanishgupta28/LightSAKT2 (private).\n\nPending operation: '+('none for the milestone above; the live scheduler may generate newer uncommitted results.' if push.returncode==0 else 'git push origin research-rebuild; inspect reports/preservation.log for failure details.')+'\n\nRestore: git clone '+str(backup)+' recovered-history (choose an authorized recovery location). Raw files and checkpoints are in the durable workspace archive, not Git bundles.\n')
    return commit


def main(watch=False):
    previous=-1
    while True:
        completed=sum(json.load(open(p)).get('status')=='completed' for p in Path('results/raw').glob('*/status.json'))
        if not watch or completed>previous or previous<0:
            preserve();previous=completed
        if not watch:return
        time.sleep(30)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');main(parser.parse_args().watch)
