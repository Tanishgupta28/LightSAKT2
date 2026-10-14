"""Check saved inference identities and labels against the frozen test records."""
import json
from pathlib import Path

import numpy as np

from src.datasets.preprocess import sha256
from src.datasets.sequences import Sequences
from scripts.sequence_cache import CachedSequences, location


class LockedTestEvidence:
    def __init__(self):
        self.datasets = {}

    def check(self, result, config):
        key = (result['dataset'], config['seq_len'])
        split_path = Path('data/splits') / (result['dataset'] + '.json')
        processed_path = Path('data/processed') / result['dataset'] / 'interactions.csv'
        if key not in self.datasets:
            assert sha256(split_path) == result['split_sha256'], 'Locked split checksum mismatch'
            assert sha256(processed_path) == result['processed_sha256'], 'Processed records checksum mismatch'
            assignments = json.load(open(split_path))
            members = {name: set(assignments[name]) for name in ['train', 'validation', 'test']}
            assert not (members['train'] & members['validation'] or members['train'] & members['test']
                        or members['validation'] & members['test']), 'Student split overlap'
            cached = location(processed_path, 'test', config['seq_len'])
            ds = CachedSequences(processed_path, 'test', config['seq_len']) if (cached / 'manifest.json').exists() else Sequences(processed_path, 'test', config['seq_len'])
            assert {row['student'] for row in ds.identities} <= members['test'], 'Inference includes a held-in student'
            targets = np.concatenate([sample[3][sample[4]].numpy() for sample in ds])
            assert len(targets) == ds.n_targets and np.isin(targets, [0, 1]).all()
            self.datasets[key] = (ds.identities, targets, result['split_sha256'], result['processed_sha256'])
        identities, targets, split_sha, processed_sha = self.datasets[key]
        assert result['split_sha256'] == split_sha and result['processed_sha256'] == processed_sha
        identity_path = Path('results/raw') / result['run_id'] / 'prediction_identity.json'
        assert json.load(open(identity_path)) == identities, 'Inference identity mismatch with locked test sequences'
        with np.load(result['prediction_path']) as saved:
            np.testing.assert_array_equal(saved['target'], targets, err_msg='Saved labels differ from locked test targets')
            assert saved['probability'].shape == targets.shape and np.isfinite(saved['probability']).all()
            assert ((saved['probability'] >= 0) & (saved['probability'] <= 1)).all()
        assert result['test']['n_targets'] == len(targets)
        return {'locked_test_identity_verified': True, 'locked_test_labels_verified': True,
                'locked_test_targets': len(targets), 'prediction_identity_sha256': sha256(identity_path),
                'split_manifest': str(split_path), 'processed_data': str(processed_path)}
