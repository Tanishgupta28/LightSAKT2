"""Audited execution metadata and exact sequence-cache substitution."""
import hashlib
import os
from pathlib import Path


def install(config):
    from src.training import engine
    from src.datasets.sequences import Sequences
    from scripts.sequence_cache import CachedSequences,location
    original_environment=engine.environment
    used=[]
    def sequences(path,split,length):
        directory=location(path,split,length)
        if (directory/'manifest.json').exists():
            result=CachedSequences(path,split,length);used.append(result.manifest_path);return result
        return Sequences(path,split,length)
    def environment():
        record=original_environment()
        record['execution']={'sequence_cache':'exact-sequences-v1','cache_manifests':list(used),
            'runtime_policy_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'sequence_cache_code_sha256':hashlib.sha256(Path('scripts/sequence_cache.py').read_bytes()).hexdigest(),
            'mixed_precision':False,'compiled_model':False,'batch_order_changed':False,
            'configuration_unchanged':True}
        record['execution']['maximum_concurrent_gkt_workers']=int(os.environ.get('LIGHTSAKT_MAX_CONCURRENT_GKT','1'))
        record['execution']['coordinated_child']=bool(os.environ.get('LIGHTSAKT_COORDINATED_CHILD'))
        return record
    engine.Sequences=sequences;engine.environment=environment

