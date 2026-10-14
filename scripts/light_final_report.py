"""Verified five-seed results, completed-only comparisons and a measured ETA."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.datasets.preprocess import sha256
from src.models.factory import build_model
from src.training.engine import write_json
from scripts.progress import snapshot
from scripts.verify_artifacts import verify_stored_result


DATASETS = ['ASSIST2009', 'ASSIST2015', 'Algebra2005']
SEEDS = [42, 123, 2026, 3407, 7777]
METRICS = ['auc', 'accuracy', 'f1', 'mse', 'bce']


def eligible_result(result):
    return result['group'] == 'final' or result['group'] == 'primary' and result['model'] != 'LightSAKT'


def seed_statistics(records):
    seeds = [r['seed'] for r in records]
    if len(seeds) != 5 or sorted(seeds) != SEEDS:
        raise ValueError('Exactly one observation per prespecified seed is required')
    summary = {'completed_seeds': 5}
    for metric in METRICS:
        values = np.asarray([r['test'][metric] for r in records], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f'Undefined five-seed metric: {metric}')
        summary[metric + '_mean'] = float(values.mean())
        summary[metric + '_std'] = float(values.std(ddof=1))
    return summary


def save_csv(frame, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    frame.to_csv(tmp, index=False); tmp.replace(path)


def table(frame, columns):
    lines = ['|' + '|'.join(columns) + '|', '|' + '|'.join(['---'] * len(columns)) + '|']
    for row in frame.to_dict('records'):
        cells = [f'{row[c]:.6f}' if isinstance(row[c], float) else str(row[c]) for c in columns]
        lines.append('|' + '|'.join(cells) + '|')
    return '\n'.join(lines)


def generate(eta_dir=Path('reports'), refresh_eta=False):
    eta_dir = Path(eta_dir)
    if refresh_eta:
        assert eta_dir.resolve() != Path('reports').resolve(), 'Use an isolated ETA output directory during live training'
        from scripts.eta import estimate
        estimate(eta_dir)
    selections = {}; checks = []; data_checked = set(); results = []
    code_hash = hashlib.sha256(b''.join(str(p).encode() + p.read_bytes() for p in sorted(Path('src').rglob('*.py'))
                                      if 'visualization' not in p.parts)).hexdigest()
    for path in sorted(Path('results/raw').glob('*/metrics.json')):
        result = json.load(open(path))
        if not eligible_result(result):
            continue
        status = path.parent / 'status.json'
        if not status.exists() or json.load(open(status)).get('status') != 'completed':
            continue
        result, config, stats = verify_stored_result(path, data_checked)
        assert result['training_code_sha256'] == code_hash, 'Training architecture/code version mismatch'
        assert config['seed'] == result['seed'] and config['model'] == result['model']
        state = torch.load(result['checkpoint'], map_location='cpu', weights_only=False)
        model = build_model(config['model'], stats['num_q'], config['model_config'])
        model.load_state_dict(state['model'], strict=True)
        assert all(torch.isfinite(t).all() for t in state['model'].values() if t.is_floating_point())
        assert state['epoch'] == result['best_epoch']
        assert state['processed_sha256'] == result['processed_sha256'] and state['split_sha256'] == result['split_sha256']
        for metric in METRICS:
            np.testing.assert_allclose(state['validation'][metric], result['validation'][metric], rtol=1e-6, atol=1e-7)
            assert np.isfinite(result['test'][metric])
        if result['group'] == 'final':
            selection_path = Path(config['selection_path']); selection = json.load(open(selection_path))
            assert selection['selection_metric'] == 'validation_auc' and selection['test_access_for_selection'] is False
            assert len(selection['all_candidates']) == len(set(selection['all_candidates'])) == 36
            selected = json.load(open(Path('results/raw') / selection['selected_run_id'] / 'config.json'))
            assert config['model_config'] == selected['model_config'] and config['learning_rate'] == selected['learning_rate']
            best = sorted(((-json.load(open(Path('results/raw') / run_id / 'metrics.json'))['validation']['auc'], run_id)
                           for run_id in selection['all_candidates']))[0][1]
            assert best == selection['selected_run_id']
            if result['model'] == 'LightSAKT':
                selections[result['dataset']] = {'config': config, 'selected_run_id': best, 'path': str(selection_path)}
        results.append(result)
        checks.append({'run_id': result['run_id'], 'group': result['group'], 'dataset': result['dataset'],
                       'model': result['model'], 'seed': result['seed'], 'checkpoint_sha256': result['checkpoint_sha256'],
                       'prediction_sha256': result['prediction_sha256'], 'metrics_sha256': sha256(path),
                       'split_sha256': result['split_sha256'], 'strict_checkpoint_load': True,
                       'test_metrics_recomputed_from_saved_predictions': True})
    light = []; individual = []; comparisons = []; partial = []
    for dataset in DATASETS:
        completed = [r for r in results if r['dataset'] == dataset and r['model'] == 'LightSAKT' and r['group'] == 'final']
        row = {'dataset': dataset, 'model': 'LightSAKT', **seed_statistics(completed)}
        c = selections[dataset]['config']
        row.update(learning_rate=c['learning_rate'], embed_dim=c['model_config']['embed_dim'],
                   num_heads=c['model_config']['num_heads'], dropout=c['model_config']['dropout'],
                   seq_len=c['seq_len'], batch_size=c['batch_size'], selected_grid_run=selections[dataset]['selected_run_id'])
        light.append(row)
        seed42 = next(r for r in completed if r['seed'] == 42)
        for r in completed:
            individual.append({'dataset': dataset, 'seed': r['seed'], 'run_id': r['run_id'], **r['test'],
                               'best_epoch': r['best_epoch'], 'epochs_trained': r['epochs_trained'],
                               'checkpoint': r['checkpoint'], 'checkpoint_sha256': r['checkpoint_sha256'],
                               'prediction_path': r['prediction_path'], 'prediction_sha256': r['prediction_sha256']})
        primary = [r for r in results if r['dataset'] == dataset and r['model'] != 'LightSAKT' and r['group'] == 'primary']
        baseline_models = sorted({r['model'] for r in results if r['dataset'] == dataset and r['model'] != 'LightSAKT'})
        for model_name in baseline_models:
            final = [r for r in results if r['dataset'] == dataset and r['model'] == model_name and r['group'] == 'final']
            if len(final) == 5:
                baseline = seed_statistics(final)
                comparisons.append({'dataset': dataset, 'baseline': model_name, 'evidence_scope': 'selected five-seed final',
                                    'baseline_seed_count': 5, **{f'baseline_{k}': v for k, v in baseline.items() if k != 'completed_seeds'},
                                    **{f'light_minus_baseline_{m}_mean': row[m + '_mean'] - baseline[m + '_mean'] for m in METRICS}})
            elif final:
                partial.append({'dataset': dataset, 'model': model_name, 'completed_seeds': len(final),
                                'scope': 'Excluded from final comparison until all five seeds finish'})
        for r in primary:
            assert r['split_sha256'] == seed42['split_sha256'], 'Comparison split mismatch'
            comparisons.append({'dataset': dataset, 'baseline': r['model'], 'evidence_scope': 'fresh primary seed 42; descriptive only',
                                'baseline_seed_count': 1, 'baseline_run_id': r['run_id'],
                                **{f'baseline_{m}_mean': r['test'][m] for m in METRICS},
                                **{f'light_minus_baseline_{m}_mean': row[m + '_mean'] - r['test'][m] for m in METRICS},
                                'light_seed42_minus_baseline_auc': seed42['test']['auc'] - r['test']['auc']})
    summary = pd.DataFrame(light); comparison = pd.DataFrame(comparisons)
    paths = ['results/tables/LightSAKT_final_summary.csv', 'results/tables/LightSAKT_final_individual_seeds.csv',
             'results/tables/LightSAKT_completed_baseline_comparisons.csv']
    for frame, path in zip([summary, pd.DataFrame(individual), comparison], paths): save_csv(frame, path)
    eta = json.load(open(eta_dir / 'ETA.json')); remaining = pd.read_csv(eta_dir / 'ETA_REMAINING_RUNS.csv')
    factor = eta['two_worker_measured_throughput_speedup']
    remaining['occupied_mig_hours'] = remaining.central_hours.where(remaining.model != 'GKT_MHA', remaining.central_hours / factor)
    costs = remaining.groupby('model').agg(remaining_entries=('run_id', 'count'), fresh_training_runs=('action', lambda x: int((x == 'training').sum())),
                                         serial_equivalent_hours=('central_hours', 'sum'), occupied_mig_hours=('occupied_mig_hours', 'sum')).reset_index()
    cost_path = 'reports/LightSAKT_summary_ETA_by_model.csv'; save_csv(costs, cost_path)
    wall = float(remaining.occupied_mig_hours.sum()); gkt = float(costs.loc[costs.model.str.startswith('GKT'), 'occupied_mig_hours'].sum())
    np.testing.assert_allclose(wall, eta['two_worker_training_wall_hours']['central'], rtol=1e-12)
    progress = snapshot(); now = datetime.now(timezone.utc).isoformat()
    formatted = summary[['dataset']].copy()
    for metric in METRICS: formatted[metric.upper()] = summary.apply(lambda r: f'{r[metric + "_mean"]:.6f} ± {r[metric + "_std"]:.6f}', axis=1)
    lines = ['# Verified final LightSAKT results and ongoing comparisons', '', 'Generated UTC: ' + now, '',
             'All numbers below are generated from fresh completed experiment files. Five selected seeds per dataset are complete; the overall research schedule remains incomplete. The three fixed-configuration LightSAKT primary runs are outside this selected-final summary; baseline primary runs are labelled separately.', '',
             '## Selected configurations', '', table(summary, ['dataset', 'learning_rate', 'embed_dim', 'num_heads', 'dropout', 'seq_len', 'batch_size']), '',
             'Each dataset selection used all 36 prespecified validation candidates and a lexical run-ID tie-break. Adam betas (0.9, 0.999), epsilon 1e-8, weight decay zero, maximum 50 epochs and validation-AUC patience five are unchanged. Student split seed 42 remains fixed at 70/10/20 before chronological chunking.', '',
             '## Five-seed test results', '', table(formatted, ['dataset'] + [m.upper() for m in METRICS]), '',
             'Mean ± sample standard deviation (ddof=1), with seeds 42, 123, 2026, 3407 and 7777. Seed 42 reuses the selected genuine tuning checkpoint. These quantities describe initialization variability on one fixed student split; they do not estimate uncertainty across new student samples. They never report the best seed.', '',
             'Verification recalculated every saved test-prediction metric, checked checkpoint and prediction hashes, strictly loaded finite checkpoint states, verified the best validation epoch, confirmed unchanged training-source fingerprints and independently reproduced validation-only configuration selection. Independent full validation-checkpoint inference remains in the final verification schedule.', '',
             '## Completed baseline evidence', '']
    final_rows = comparison[comparison.evidence_scope == 'selected five-seed final']
    if final_rows.empty: lines += ['No baseline has a complete selected five-seed final comparison at this report time. No pending baseline score or historical manuscript score is included as final evidence.', '']
    else: lines += [table(final_rows, ['dataset', 'baseline', 'baseline_seed_count', 'baseline_auc_mean', 'baseline_auc_std', 'light_minus_baseline_auc_mean']), '']
    primary_rows = comparison[comparison.evidence_scope.str.startswith('fresh primary')]
    lines += ['The following fresh completed baseline primary runs are seed-42, fixed-configuration measurements on the same split. Differences against the five-seed LightSAKT mean are descriptive comparisons with unequal replication and tuning stages, not a matched final ranking or statistical significance result. Both GKT labels refer to the corrected original simplified PAM/MHA implementations, not canonical GKT reproduction.', '',
              table(primary_rows, ['dataset', 'baseline', 'baseline_auc_mean', 'light_minus_baseline_auc_mean', 'light_seed42_minus_baseline_auc']), '',
              'All five baseline metrics, deltas and evidence scope are in LightSAKT_completed_baseline_comparisons.csv. Partial selected baseline seed sets are excluded until all five finish. Paired statistical analysis proceeds only from genuinely matched completed observations.', '',
              '## Recalculated remaining training time', '',
              f'ETA measurements refreshed: {eta["updated_utc"]}. Serial isolated-equivalent remaining training: {eta["remaining_training_gpu_reserved_hours"]["central"]:.3f} hours. Current execution forecast: {wall:.3f} occupied MIG wall hours ({wall / 24:.2f} continuous days), applying the measured {factor:.3f} two-worker throughput factor only to GKT-MHA.', '',
              table(costs, ['model', 'remaining_entries', 'fresh_training_runs', 'serial_equivalent_hours', 'occupied_mig_hours']), '',
              f'GKT-PAM and GKT-MHA account for {gkt:.3f} of {wall:.3f} remaining occupied MIG hours ({100 * gkt / wall:.2f}%). Consequently, completion count is a poor proxy for compute progress. The priority order completed short LightSAKT grids, selected seeds, ablations and fast baseline tuning first. Reuse/alias entries add no training. The expensive GKT tuning and selected final-seed runs still remain.', '',
              'The GKT forecast is calibrated by six completed primary model/dataset runs plus real primary/wide engineering profiles, rather than multiplying the global mean runtime by remaining entries. Recent fast-model completions refine those strata but add little evidence about the still-untrained slow GKT configurations. One GKT epoch has measured durations of minutes, whereas many DKT/SAKT/LightSAKT epochs take seconds or less; planned configuration widths and stopping counts also differ.', '',
              f'The occupied-MIG planning envelope is {eta["two_worker_training_wall_hours"]["low"]:.1f}–{eta["two_worker_training_wall_hours"]["high"]:.1f} hours. It is an early-stop/maximum-epoch scenario envelope, not a confidence interval or guarantee. Most future GKT stopping counts and sustained concurrency performance are unknown. Startup, selected test inference and final artifact verification are additional; the completed controlled efficiency benchmark is reused.', '',
              '## Continuing execution and preserved findings', '',
              f'Live schedule snapshot: {progress["completed"]}/{progress["total_slots"]} entries complete; {progress["remaining"]} remain. Phase counts: {json.dumps(progress["phase_counts"])}.', '',
              'Scheduler and independent CPU reporting, Git preservation, checkpoint recovery and offsite workers remain active. No completed training was restarted and no experimental scope was removed. Detached execution survives client disconnect while the environment stays up; no automatic restart after environment shutdown is configured.', '',
              'The original LightSAKT architecture, causal/padding controls and dataset splits remain unchanged. The preserved efficiency evidence shows LightSAKT slower in all 60 matched latency comparisons. Theoretical O(L) work, runtime, memory and parameter efficiency remain separate in paper/CORRECTED_RESULTS.md and paper/EFFICIENCY_RESULTS.md.', '',
              'Local data and split hashes are verified. Independent publisher byte identity and explicit redistribution licensing remain unresolved, Algebra uses the available labelled training subset, and simplified GKT fidelity is disclosed.', '',
              '## Reproduce this summary', '', '```bash', 'cd /workspace/LightSAKT2',
              "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.light_final_report --refresh-eta --eta-dir reports/summary_eta/latest", '```', '',
              'This command performs CPU-only evidence checks and regenerates reports/tables. It does not start, stop or reset GPU training. Source metric/checkpoint/prediction hashes and per-run verification are recorded in reports/light_final_summary_verification.json.']
    Path('reports/LIGHTSAKT_FINAL_RESULTS.md').write_text('\n'.join(lines) + '\n')
    corrections=Path('PAPER_CORRECTIONS.md')
    begin='<!-- selected-light-results:start -->'; end='<!-- selected-light-results:end -->'
    wording=[begin,'','## Verified selected LightSAKT results (Tables 4–7; pages 14–19)','',
             'Replace historical LightSAKT scores with the following fresh five-seed results under the corrected causal, student-disjoint protocol. The original PDF is preserved. Hyperparameters were selected using all 36 validation candidates per dataset; historical numbers and pending baseline runs are excluded from these final LightSAKT measurements.','',
             table(formatted,['dataset']+[m.upper() for m in METRICS]),'',
             'Numbers are mean ± sample SD on one fixed split. A complete baseline superiority statement still requires the matched selected five-seed comparisons. Fresh primary seed-42 comparisons remain descriptive, with different replication and tuning stages. Source: results/tables/LightSAKT_final_summary.csv, LightSAKT_final_individual_seeds.csv, LightSAKT_completed_baseline_comparisons.csv and reports/light_final_summary_verification.json. All original efficiency findings remain preserved.','',end]
    original=corrections.read_text(); replacement='\n'.join(wording)
    if begin in original:
        first,tail=original.split(begin,1); assert end in tail
        _,last=tail.split(end,1); corrections.write_text(first+replacement+last)
    else:corrections.write_text(original+'\n'+replacement+'\n')
    write_json('reports/light_final_summary_verification.json', {'verified': True, 'verified_utc': now,
               'light_final_seed_slots': len(individual), 'summary_method': 'Five prespecified seeds; arithmetic mean and sample SD; saved metrics independently recomputed',
               'completed_five_seed_baseline_comparisons': len(final_rows), 'fresh_primary_baseline_comparisons': len(primary_rows),
               'partial_final_baselines_excluded': partial, 'runs': checks,
               'training_code_sha256': code_hash, 'eta_source': str(eta_dir / 'ETA.json'),
               'eta_source_sha256': sha256(eta_dir / 'ETA.json'), 'remaining_occupied_mig_hours': wall,
               'gkt_remaining_occupied_mig_hours': gkt,
               'artifacts': [{'path': p, 'sha256': sha256(p)} for p in paths + [cost_path]]})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--refresh-eta', action='store_true')
    parser.add_argument('--eta-dir', type=Path, default=Path('reports')); args = parser.parse_args()
    print(generate(args.eta_dir, args.refresh_eta).to_string(index=False))
