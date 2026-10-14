"""Lossless, checksummed caches of the established sequence-loader outputs."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from src.datasets.sequences import Sequences
from src.datasets.preprocess import sha256
from src.training.engine import write_json


def location(path,split,length):
    path=Path(path);identity=hashlib.sha256((sha256(path)+split+str(length)+'exact-sequences-v1').encode()).hexdigest()
    return path.parent.parent.parent/'cache'/identity


def build(path,split,length):
    directory=location(path,split,length)
    if (directory/'manifest.json').exists():return str(directory)
    ds=Sequences(path,split,length);tmp=directory.with_name(directory.name+f'.{os.getpid()}.tmp');tmp.mkdir(parents=True,exist_ok=True)
    files={}
    for index,name in enumerate(['q','r','query','target','valid']):
        array=np.stack([sample[index].numpy() for sample in ds.samples]) if ds.samples else np.empty((0,length),dtype=bool if name=='valid' else np.int64)
        file=tmp/(name+'.npy');np.save(file,array);files[name+'.npy']=sha256(file)
    write_json(tmp/'identities.json',ds.identities);files['identities.json']=sha256(tmp/'identities.json')
    write_json(tmp/'manifest.json',{'version':'exact-sequences-v1','processed_sha256':sha256(path),
        'split':split,'seq_len':length,'samples':len(ds),'n_targets':ds.n_targets,'files':files})
    if directory.exists():raise RuntimeError('Cache directory exists without a complete manifest')
    tmp.rename(directory);return str(directory)


class CachedSequences(Dataset):
    def __init__(self,path,split,seq_len):
        directory=location(path,split,seq_len)
        if not (directory/'manifest.json').exists():raise FileNotFoundError('Verified cache not prepared')
        m=json.load(open(directory/'manifest.json'))
        assert m['processed_sha256']==sha256(path) and m['split']==split and m['seq_len']==seq_len
        for name,digest in m['files'].items():assert sha256(directory/name)==digest,'Cache corruption'
        self.seq_len=seq_len;self.n_targets=m['n_targets'];self.identities=json.load(open(directory/'identities.json'))
        self.arrays=[np.load(directory/(name+'.npy'),mmap_mode='r') for name in ['q','r','query','target','valid']]
        self.manifest_path=str(directory/'manifest.json')

    def __len__(self):return len(self.identities)

    def __getitem__(self,index):
        # Every returned tensor owns mutable memory; consumers cannot alter the cache.
        return tuple(torch.from_numpy(array[index].copy()) for array in self.arrays)


def build_all():
    tasks=[(f'data/processed/{d}/interactions.csv',split,200 if d=='Algebra2005' else 100)
           for d in ['ASSIST2009','ASSIST2015','Algebra2005'] for split in ['train','validation','test']]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(build,*task) for task in tasks]
        paths=[f.result() for f in futures]
    write_json('reports/optimization/sequence_cache.json',{'version':'exact-sequences-v1','paths':paths,
        'label_fitting':'none; byte-identical previously processed sequences, no normalization or vocabulary changes'})
    print('Prepared',len(paths),'sequence caches')


if __name__=='__main__':build_all()
