"""Validate and summarize saved measurements without executing another benchmark."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.preprocess import sha256
from src.training.engine import write_json


PAIR_KEYS = ['scope', 'length', 'batch', 'operation']
MEASUREMENTS = ['latency_ms_mean', 'peak_allocated_bytes', 'parameters']


def paired_measurements(frame):
    required = PAIR_KEYS + ['model'] + MEASUREMENTS
    if not set(required) <= set(frame.columns):
        raise ValueError('Missing efficiency measurement columns')
    if set(frame.model) != {'LightSAKT', 'SAKT'}:
        raise ValueError('Expected separately labelled LightSAKT and SAKT measurements')
    if frame[required].isna().any().any():
        raise ValueError('Missing efficiency measurement or comparison identity')
    if frame.duplicated(PAIR_KEYS + ['model']).any():
        raise ValueError('Duplicate efficiency comparison identity')
    values = frame[MEASUREMENTS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError('Efficiency measurements must be finite and positive')
    light = frame[frame.model == 'LightSAKT'].set_index(PAIR_KEYS)
    sakt = frame[frame.model == 'SAKT'].set_index(PAIR_KEYS)
    if set(light.index) != set(sakt.index):
        raise ValueError('Unmatched efficiency measurements; no comparison can be dropped')
    pairs = light[MEASUREMENTS].join(sakt[MEASUREMENTS], lsuffix='_light', rsuffix='_sakt', validate='one_to_one').sort_index()
    for label, column in [('latency', 'latency_ms_mean'), ('peak_allocated', 'peak_allocated_bytes'), ('parameter', 'parameters')]:
        pairs[label + '_ratio_light_over_sakt'] = pairs[column + '_light'] / pairs[column + '_sakt']
    return pairs


def manuscript_findings(frame, parameters):
    pairs = paired_measurements(frame)
    slower = int((pairs.latency_ratio_light_over_sakt > 1).sum())
    faster = int((pairs.latency_ratio_light_over_sakt < 1).sum())
    lower_memory = int((pairs.peak_allocated_ratio_light_over_sakt < 1).sum())
    lines = [
        f'**Measured runtime:** LightSAKT had higher mean latency in {slower} of {len(pairs)} matched comparisons and lower mean latency in {faster}. These genuine findings contradict an unconditional speed advantage. The saved {len(frame)} FP32 cases cover lengths 50, 100, 200, 500 and 1000, batch sizes 1 and 8, inference and forward/backward, and three separately labelled comparison scopes on the H100 MIG partition. Each case used three warm-ups and ten synchronized timed repetitions. We preserved the original architecture and these measurements.',
        '',
        f'**Measured memory:** LightSAKT had lower peak allocated memory in {lower_memory} of {len(pairs)} matched comparisons. Allocated peaks, reserved peaks and baseline allocations are reported separately; these process measurements include model, inputs, workspace and warm-up state. They do not establish an unconditional memory advantage or the asymptotic storage behavior of a different attention backend.',
        '',
        '|Comparison scope|Matched cases|LightSAKT slower|LightSAKT lower peak allocated memory|',
        '|---|---|---|---|',
    ]
    for scope, data in pairs.groupby(level='scope'):
        lines.append(f'|{scope}|{len(data)}|{int((data.latency_ratio_light_over_sakt > 1).sum())}|{int((data.peak_allocated_ratio_light_over_sakt < 1).sum())}|')
    lines += ['', '**Parameter efficiency:** Counts for the documented primary configurations are:', '',
              '|Dataset|LightSAKT parameters|SAKT parameters|LightSAKT/SAKT|', '|---|---|---|---|']
    for dataset, data in parameters.groupby('dataset'):
        selected = data[data.model.isin(['LightSAKT', 'SAKT'])]
        if len(selected) != 2 or set(selected.model) != {'LightSAKT', 'SAKT'}:
            raise ValueError(f'Incomplete or duplicated parameter comparison: {dataset}')
        counts = selected.set_index('model').parameters
        a, b = int(counts.LightSAKT), int(counts.SAKT)
        lines.append(f'|{dataset}|{a}|{b}|{a / b:.6f}|')
    lines += ['',
              'Primary model widths differ, so these counts cannot be attributed solely to the kernel or feed-forward reduction. Equal-width complete-model comparisons isolate parameter differences more closely. Benchmark positional capacity grows with tested length; parameter counts and serialized sizes are recorded for each case.',
              '',
              '**Theoretical scaling and interpretation:** At fixed head dimensions, the causal ELU+1 kernel performs O(B H L d_k d_v) attention work. This vectorized training implementation also stores sequence-length key/value prefixes; its dimension-dependent constants matter. This theoretical O(L) work does not imply lower measured latency or lower memory throughout the tested range. The isolated dense SAKT reference requests attention weights, while the existing complete SAKT uses PyTorch MHA without requesting them. Backend and comparison scope remain explicit.',
              '',
              'Benchmark weights were newly initialized, and forward/backward used a squared-output surrogate. These timings are distinct from dataset epoch times, which include training and validation. Ten timing repetitions measure engineering variability on one recorded device, not independent student or training-seed uncertainty. Scope-specific numerical results, complete raw repetitions and hardware metadata are preserved in results/tables/efficiency.csv, results/tables/efficiency_paired_comparisons.csv, results/tables/parameters.csv, experiments/efficiency/raw_timings.json and experiments/efficiency/environment.json. The full measured discussion is in paper/EFFICIENCY_RESULTS.md.']
    return '\n'.join(lines)


def verify_saved_measurements(root=Path('.')):
    root = Path(root)
    table_path = root / 'results/tables/efficiency.csv'
    raw_path = root / 'experiments/efficiency/raw_timings.json'
    frame = pd.read_csv(table_path)
    raw = pd.DataFrame(json.loads(raw_path.read_text()))
    pairs = paired_measurements(frame)
    identities = PAIR_KEYS + ['model']
    expected = set(itertools.product(
        ['equal_attention', 'equal_complete', 'documented_complete'],
        [50, 100, 200, 500, 1000], [1, 8], ['inference_forward', 'forward_backward'], ['LightSAKT', 'SAKT']))
    assert set(frame[identities].itertuples(index=False, name=None)) == expected, 'Missing or extra benchmark cases'
    paired_measurements(raw)  # Reject duplicate/unmatched raw records too.
    assert set(raw[identities].itertuples(index=False, name=None)) == expected, 'Raw benchmark coverage mismatch'
    indexed = frame.set_index(identities)
    for row in raw.to_dict('records'):
        saved = indexed.loc[tuple(row[key] for key in identities)]
        times = np.asarray(row['timings_ms'], dtype=float)
        assert times.shape == (10,) and np.isfinite(times).all() and (times > 0).all(), 'Invalid timed repetitions'
        # The raw JSON stores ten decimal places; allow only that serialization error.
        np.testing.assert_allclose([saved.latency_ms_mean, saved.latency_ms_std],
                                   [times.mean(), times.std(ddof=1)], rtol=1e-9, atol=2e-9)
        np.testing.assert_allclose(saved.throughput_tokens_per_second,
                                   row['batch'] * row['length'] * 1000 / times.mean(), rtol=1e-9)
        for field in ['baseline_allocated_bytes', 'baseline_reserved_bytes', 'peak_allocated_bytes',
                      'peak_reserved_bytes', 'parameters', 'serialized_bytes', 'width', 'heads', 'dropout']:
            assert saved[field] == row[field], f'Raw/table measurement mismatch: {field}'
        assert saved.peak_allocated_bytes >= saved.baseline_allocated_bytes
        assert saved.peak_reserved_bytes >= saved.baseline_reserved_bytes
    artifact_paths = ['results/tables/efficiency.csv', 'experiments/efficiency/raw_timings.json',
                      'experiments/efficiency/environment.json', 'results/tables/parameters.csv']
    record = {'verified': True, 'cases': len(frame), 'matched_comparisons': len(pairs),
              'timed_repetitions_verified': sum(len(t) for t in raw.timings_ms),
              'higher_light_latency': int((pairs.latency_ratio_light_over_sakt > 1).sum()),
              'lower_light_latency': int((pairs.latency_ratio_light_over_sakt < 1).sum()),
              'lower_light_peak_allocated_memory': int((pairs.peak_allocated_ratio_light_over_sakt < 1).sum()),
              'artifacts': [{'path': path, 'sha256': sha256(root / path)} for path in artifact_paths],
              'scope': 'CPU-only rederivation of saved measurements; no new benchmark or model change'}
    write_json(root / 'reports/efficiency_verification.json', record)
    return record


if __name__ == '__main__':
    print(json.dumps(verify_saved_measurements(), indent=2))
