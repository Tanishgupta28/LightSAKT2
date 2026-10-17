"""Create supervisor table images from checksum-bound, completed research evidence.

This is a CPU-only publication workflow. It never invokes training, inference,
preprocessing, a scheduler, or a checkpoint writer. Frozen input copies allow
the sharing pack to remain reproducible while the research scheduler continues.
"""
import argparse
import csv
import hashlib
import html
import io
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import textwrap
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DATASETS = ['ASSIST2009', 'ASSIST2015', 'Algebra2005']
LABEL = {'ASSIST2009': 'ASSIST2009', 'ASSIST2015': 'ASSIST2015', 'Algebra2005': 'Algebra 2005–2006'}
SEEDS = [42, 123, 2026, 3407, 7777]
MODELS = ['DKT', 'DKT+', 'DKVMN', 'SAKT', 'LightSAKT']
METRICS = ['auc', 'f1', 'accuracy', 'mse', 'bce']
ORIGINALS = {2: '234727', 3: '235129', 4: '234747', 5: '234803',
             6: '234809', 7: '234818', 8: '234828'}
NAMES = {2: 'Table_02_Dataset_Statistics', 3: 'Table_03_Training_Configurations',
         4: 'Table_04_LightSAKT_Grid_Algebra2005', 5: 'Table_05_LightSAKT_Grid_ASSIST2009',
         6: 'Table_06_LightSAKT_Grid_ASSIST2015', 7: 'Table_07_Verified_Model_Comparison',
         8: 'Table_08_LightSAKT_Ablation_Study'}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding='utf-8')


def save_json(path, value):
    write(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def fingerprint(root):
    return hashlib.sha256(b''.join(str(p.relative_to(root)).encode() + p.read_bytes()
        for p in sorted((root / 'src').rglob('*.py')) if 'visualization' not in p.parts)).hexdigest()


class Evidence:
    def __init__(self, root, out):
        self.root, self.out = root, out
        self.files, self.verified_runs, self.hashed = {}, {}, {}

    def read(self, path):
        path = str(path)
        target = self.out / 'source' / 'project_snapshot' / path
        if not target.exists():
            original = self.root / path
            assert original.is_relative_to(self.root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(original.read_bytes())
        data = target.read_bytes()
        self.files[path] = {'project_path': path, 'snapshot_path': str(target.relative_to(self.out)),
                            'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        return data

    def json(self, path):
        return json.loads(self.read(path))

    def frame(self, path):
        return pd.read_csv(io.BytesIO(self.read(path)))

    def check_hash(self, path, expected):
        path = str(path)
        if path not in self.hashed:
            self.hashed[path] = digest(self.root / path)
        assert self.hashed[path] == expected, path

    def run(self, run_id, proof, test=False):
        if run_id in self.verified_runs:
            return self.verified_runs[run_id]
        metrics_path = f'results/raw/{run_id}/metrics.json'
        config_path = f'results/raw/{run_id}/config.json'
        r, c = self.json(metrics_path), self.json(config_path)
        assert r['run_id'] == run_id and proof['run_id'] == run_id and proof['verified']
        assert self.files[metrics_path]['sha256'] == proof['metrics_sha256']
        assert self.files[config_path]['sha256'] == proof['config_sha256']
        assert json.load(open(self.root / f'results/raw/{run_id}/status.json'))['status'] == 'completed'
        assert r['checkpoint_sha256'] == proof['checkpoint_sha256']
        self.check_hash(r['checkpoint'], r['checkpoint_sha256'])
        stats = self.json(f'reports/{r["dataset"]}_statistics.json')
        for key in ['processed_sha256', 'split_sha256']:
            assert r[key] == proof[key] == stats[key]
        if test:
            assert proof['validation_recomputed'] and proof['test_probabilities_recomputed']
            assert proof['locked_test_identity_verified'] and proof['locked_test_labels_verified']
            assert r['test'] is not None and c['seed'] in SEEDS
            assert r['prediction_sha256'] == proof['prediction_sha256']
            self.check_hash(r['prediction_path'], r['prediction_sha256'])
            assert r['test']['n_targets'] == proof['locked_test_targets']
        else:
            assert proof['validation_recomputed'] and not proof['test_evaluated']
            assert c['evaluate_test'] is False and r['test'] is None
        self.verified_runs[run_id] = (r, c)
        return r, c


def spec(number, title, columns, body, evidence, notes, **kwargs):
    return {'number': number, 'name': NAMES[number], 'title': title,
            'original_filename': f'allTables/Screenshot 2026-10-17 {ORIGINALS[number]}.png',
            'columns': columns, 'body': body, 'row_evidence': evidence, 'notes': notes,
            'bold_cells': [], 'bold_rows': [], 'groups': [], 'grid': False,
            'gkt': 'not present in original table', 'partially_unavailable': [], **kwargs}


def build(root, out):
    e = Evidence(root, out)
    protocol = e.json('configs/protocol.json')
    selected_proof = e.json('reports/selected_final_inference_verification.json')
    grid_proof = e.json('reports/light_grid_inference_verification.json')
    assert selected_proof['verified'] and grid_proof['verified']
    final_proofs = {p['run_id']: p for p in selected_proof['runs']}
    grid_proofs = {p['run_id']: p for p in grid_proof['runs']}
    statistics, final, tables, configurations = {}, {}, [], {}
    table2 = e.frame('results/tables/Table_2.csv').set_index('dataset')
    assert protocol['training_seeds'] == SEEDS and protocol['split_seed'] == 42
    for ds in DATASETS:
        s = e.json(f'reports/{ds}_statistics.json')
        statistics[ds] = s
        for key in ['students', 'clean_interactions', 'unique_skill_identifiers', 'raw_rows', 'num_q']:
            assert int(table2.loc[ds, key]) == s[key]
        e.check_hash(s['raw_path'], s['raw_sha256'])
        e.check_hash(f'data/processed/{ds}/interactions.csv', s['processed_sha256'])
        e.check_hash(f'data/splits/{ds}.json', s['split_sha256'])
        splits = json.load(open(root / f'data/splits/{ds}.json'))
        assert all(len(splits[a]) == s['split_students'][a] for a in splits)
        assert sum(map(len, splits.values())) == s['students']
        assert not any(set(splits[a]) & set(splits[b]) for a, b in itertools.combinations(splits, 2))
        raw_columns = pd.read_csv(root / s['raw_path'], sep='\t' if ds == 'Algebra2005' else ',',
                                  encoding='latin1', nrows=0).columns.tolist()
        assert raw_columns == s['columns']
        vocabulary = e.json(f'data/processed/{ds}/vocabulary.json')
        assert len(vocabulary) + 1 == s['num_q']
        for model in MODELS:
            for seed in SEEDS:
                rid = f'{ds}-{model}-final-{seed}'
                r, c = e.run(rid, final_proofs[rid], test=True)
                assert r['dataset'] == ds and r['model'] == model and r['seed'] == seed
                final[ds, model, seed] = r
                if model == 'LightSAKT' and seed == 42:
                    configurations[ds] = c

    tables.append(spec(2, 'Table 2: Overview of datasets',
        ['Dataset', 'Students', 'Skill identifiers', 'Records', 'Raw columns'],
        [[LABEL[ds], f'{statistics[ds]["students"]:,}', f'{statistics[ds]["unique_skill_identifiers"]:,}',
          f'{statistics[ds]["clean_interactions"]:,}', str(len(statistics[ds]['columns']))] for ds in DATASETS],
        [{'statistics': f'reports/{ds}_statistics.json',
          'fields': ['students', 'unique_skill_identifiers', 'clean_interactions', 'len(columns)']} for ds in DATASETS],
        ['Records, students and skill identifiers are counted after cleaning; raw columns count source-file columns.',
         'Skill identifiers are composite tags (ASSIST2009/Algebra) or sequence IDs (ASSIST2015), not uniformly atomic skills.',
         f'Algebra: labelled training subset; {statistics["Algebra2005"]["unique_skill_identifiers"]} observed composite identifiers, '
         f'with {statistics["Algebra2005"]["num_q"] - 1} training IDs plus UNK.'],
        size=[11.2, 3.6], font=13.4, widths=[2.5, 1.4, 2.0, 1.8, 1.6],
        subtitle='Verified corrected dataset variants', table_top=.79, table_bottom=.41))

    rows, row_sources = [], []
    for label, field, fmt in [
        ('train_ratio', 'train_ratio', '.1f'), ('validation_ratio', 'validation_ratio', '.1f'),
        ('test_ratio', 'test_ratio', '.1f'), ('learning_rate', 'learning_rate', '.4f'),
        ('seq_len', 'seq_len', 'd'), ('optimizer', 'optimizer', ''),
        ('embedding_dimension', 'embed_dim', 'd'), ('num_heads', 'num_heads', 'd'),
        ('dropout', 'dropout', '.1f'), ('batch_size', 'batch_size', 'd'),
        ('max_epochs', 'max_epochs', 'd'), ('early_stopping_patience', 'patience', 'd')]:
        values = []
        for ds in DATASETS:
            c = configurations[ds]
            value = {'train_ratio': protocol['split_ratios'][0], 'validation_ratio': protocol['split_ratios'][1],
                     'test_ratio': protocol['split_ratios'][2], 'optimizer': protocol['optimizer']}.get(field)
            if value is None:
                value = c['model_config'][field] if field in c['model_config'] else c[field]
            values.append(format(value, fmt))
        rows.append([label] + values)
        row_sources.append({'field': field, 'protocol': 'configs/protocol.json',
                            'selected_final_configs': [f'results/raw/{ds}-LightSAKT-final-42/config.json' for ds in DATASETS]})
    tables.append(spec(3, 'Table 3: Training network configurations for different datasets',
        ['Setting'] + [LABEL[ds] for ds in DATASETS], rows, row_sources,
        ['LightSAKT configurations selected independently by validation AUC; target student-disjoint ratios are 70/10/20.',
         'Adam: betas (0.9, 0.999), epsilon 1e−8, weight decay 0. Early stopping monitors validation AUC.',
         'Split/grid seed 42; final training seeds: 42, 123, 2026, 3407, 7777. Test data do not select configurations.'],
        size=[11.5, 6.3], font=12.5, widths=[2.6, 2.0, 2.0, 2.4], grid=True,
        subtitle='Validation-selected LightSAKT settings', table_top=.85, table_bottom=.23))

    grid = list(itertools.product([.0005, .001, .005], [32, 64, 128], [2, 4], [.1, .2]))
    for number, ds in [(4, 'Algebra2005'), (5, 'ASSIST2009'), (6, 'ASSIST2015')]:
        frame = e.frame(f'results/tables/Table_{number}.csv')
        assert len(frame) == 36
        assert list(zip(frame.learning_rate, frame.embed_dim, frame.num_heads, frame.dropout)) == grid
        rows, row_sources, scores = [], [], []
        for row in frame.to_dict('records'):
            r, c = e.run(row['run_id'], grid_proofs[row['run_id']])
            assert c['seed'] == 42 and c['dataset'] == ds and c['model'] == 'LightSAKT'
            assert row['validation_auc'] == r['validation']['auc'] or abs(row['validation_auc'] - r['validation']['auc']) < 1e-15
            m = c['model_config']; score = r['validation']['auc']; scores.append(score)
            assert (c['learning_rate'], m['embed_dim'], m['num_heads'], m['dropout']) == (
                row['learning_rate'], row['embed_dim'], row['num_heads'], row['dropout'])
            rows.append([f'{c["learning_rate"]:.6f}', str(m['embed_dim']), str(m['num_heads']),
                         f'{m["dropout"]:.1f}', f'{score:.6f}'])
            row_sources.append({'run_id': r['run_id'], 'metrics': f'results/raw/{r["run_id"]}/metrics.json',
                                'field': 'validation.auc', 'value': score, 'checkpoint': r['checkpoint'],
                                'checkpoint_sha256': r['checkpoint_sha256'], 'best_epoch': r['best_epoch']})
        best = sorted(range(36), key=lambda i: (-scores[i], row_sources[i]['run_id']))[0]
        selection = e.json(f'experiments/selections/{ds}-LightSAKT.json')
        assert selection['selected_run_id'] == row_sources[best]['run_id'] and not selection['test_access_for_selection']
        c = configurations[ds]
        tables.append(spec(number, f'Table {number}: Hyperparameter tuning results for {LABEL[ds]}',
            ['Learning Rate', 'Embedding', 'Heads', 'Dropout', 'Best validation AUC'], rows, row_sources,
            ['LightSAKT; all 36 configurations. Training seed 42, split seed 42; validation AUC only.',
             'Bold marks the highest full-precision validation AUC. Exact ties select the lexically smallest run ID.',
             f'Selected: LR {c["learning_rate"]:g}, dimension {c["model_config"]["embed_dim"]}, '
             f'heads {c["model_config"]["num_heads"]}, dropout {c["model_config"]["dropout"]:g}; epoch {row_sources[best]["best_epoch"]}.'],
            size=[8.7, 12.5], font=13.2, widths=[1.6, 1.4, .8, 1.0, 2.3],
            subtitle='Corrected causal LightSAKT · validation results', bold_rows=[i for i, s in enumerate(scores) if s == max(scores)],
            table_top=.90, table_bottom=.12, selected_run_id=row_sources[best]['run_id']))

    current_summary = e.frame('results/tables/Table_7.csv').set_index(['dataset', 'model'])
    rows, row_sources, bold = [], [], []
    for ds in DATASETS:
        for metric in METRICS:
            aggregates, sources = {}, {}
            for model in MODELS:
                rs = [final[ds, model, seed] for seed in SEEDS]
                values = np.array([r['test'][metric] for r in rs], dtype=float)
                mean, sd = float(values.mean()), float(values.std(ddof=1))
                original = current_summary.loc[ds, model]
                assert int(original.completed_seeds) == int(original.required_seeds) == 5
                np.testing.assert_allclose([mean, sd], [original[metric + '_mean'], original[metric + '_std']], atol=1e-15, rtol=0)
                aggregates[model] = (mean, sd)
                sources[model] = {'run_ids': [r['run_id'] for r in rs], 'metric': 'test.' + metric,
                                  'mean': mean, 'sample_sd': sd, 'seeds': SEEDS}
            score = min(v[0] for v in aggregates.values()) if metric in ['mse', 'bce'] else max(v[0] for v in aggregates.values())
            values = [f'{aggregates[m][0]:.6f} ± {aggregates[m][1]:.6f}' if m != 'GKT' else 'Pending'
                      for m in ['DKT', 'DKT+', 'GKT', 'DKVMN', 'SAKT', 'LightSAKT']]
            for col, model in enumerate(['DKT', 'DKT+', 'GKT', 'DKVMN', 'SAKT', 'LightSAKT'], 2):
                if model != 'GKT' and aggregates[model][0] == score:
                    bold.append([len(rows), col])
            metric_label = {'auc': 'AUC ↑', 'f1': 'F1 ↑', 'accuracy': 'Accuracy ↑', 'mse': 'MSE ↓', 'bce': 'BCE ↓'}[metric]
            rows.append([LABEL[ds] if metric == 'accuracy' else '', metric_label] + values)
            row_sources.append({'dataset': ds, 'field': 'test.' + metric, 'models': sources,
                                'aggregation': 'Arithmetic mean and sample SD (ddof=1) across all five seeds',
                                'GKT': 'Pending; final results deliberately excluded'})
    tables.append(spec(7, 'Table 7: Performance comparison of knowledge-tracing models across datasets',
        ['Dataset', 'Metric', 'DKT', 'DKT+', 'GKT', 'DKVMN', 'SAKT', 'LightSAKT'], rows, row_sources,
        ['Final held-out test results: mean ± sample SD over seeds 42, 123, 2026, 3407, 7777, on the same locked split.',
         'Bold marks the best completed mean per metric: higher AUC/F1/accuracy; lower MSE/BCE. Differences are descriptive.',
         'GKT final five-seed results are pending; no partial, primary or historical GKT scores are shown. Loss_mean is labelled BCE.'],
        size=[16.4, 8.0], font=11.4, widths=[1.75, 1.2, 2.1, 2.1, 1.15, 2.1, 2.1, 2.1],
        subtitle='Verified final five-seed results · GKT pending', bold_cells=bold, groups=[5, 10],
        gkt='Pending in all 15 metric rows; neither PAM nor MHA final values are inserted',
        table_top=.85, table_bottom=.24))

    ablation_summary = e.frame('results/tables/Table_8.csv').set_index(['dataset', 'variant'])
    rows, row_sources, bold, column_means = [], [], [], {ds: [] for ds in DATASETS}
    variants = [('Default', 'default'), ('No Dropout', 'no_dropout'), ('No Position Encoding', 'no_position'),
                ('Single Head', 'single_head'), ('Without attention', 'no_attention'),
                ('No layer normalization*', None), ('Without regularization†', 'no_regularization')]
    for label, variant in variants:
        cells, sources = [label], {}
        for ds in DATASETS:
            if variant is None:
                cells.append('Not run');column_means[ds].append(None)
                continue
            rs = []
            for seed in SEEDS:
                rid = f'{ds}-LightSAKT-final-{seed}' if variant == 'default' else f'{ds}-LightSAKT-{variant}-{seed}'
                r, _ = e.run(rid, final_proofs[rid], test=True)
                rs.append(r)
                if variant == 'no_regularization':
                    original, _ = e.verified_runs[f'{ds}-LightSAKT-no_dropout-{seed}']
                    assert r['test'] == original['test'] and r['checkpoint_sha256'] == original['checkpoint_sha256']
                    assert r['prediction_sha256'] == original['prediction_sha256'] and r['equivalent_to'] == 'no_dropout'
            values = np.array([r['test']['auc'] for r in rs])
            mean, sd = float(values.mean()), float(values.std(ddof=1))
            old = ablation_summary.loc[ds, variant]
            assert int(old.completed_seeds) == int(old.required_seeds) == 5
            np.testing.assert_allclose([mean, sd], [old.auc_mean, old.auc_std], rtol=0, atol=1e-15)
            cells.append(f'{mean:.6f} ± {sd:.6f}');column_means[ds].append(mean)
            sources[ds] = {'run_ids': [r['run_id'] for r in rs], 'metric': 'test.auc', 'mean': mean,
                           'sample_sd': sd, 'equivalent_to': 'no_dropout' if variant == 'no_regularization' else None}
        rows.append(cells);row_sources.append({'variant': variant, 'models': sources,
            'status': 'Not run; no number available' if variant is None else 'Five verified seeds'})
    for col, ds in enumerate(DATASETS, 1):
        best = max(v for v in column_means[ds] if v is not None)
        bold += [[i, col] for i, v in enumerate(column_means[ds]) if v == best]
    tables.append(spec(8, 'Table 8: Ablation study', ['Architecture'] + [LABEL[ds] for ds in DATASETS],
        rows, row_sources,
        ['Held-out test AUC: mean ± sample SD over five seeds. Bold marks the highest measured mean in each dataset.',
         '* Not run: the original image includes layer-normalization removal, but no verified experiment exists for it.',
         '† Without regularization equals No Dropout (same evidence); it is not an independent ablation.'],
        size=[12.1, 4.8], font=12.1, widths=[3.4, 2.5, 2.5, 2.6],
        subtitle='Corrected LightSAKT · five-seed test AUC', bold_cells=bold,
        partially_unavailable=['No layer normalization: not run; all three cells explicitly labelled Not run'],
        table_top=.82, table_bottom=.32))

    saved_runs = []
    for rid, (r, c) in sorted(e.verified_runs.items()):
        saved_runs.append({'run_id': rid, 'model': r['model'], 'dataset': r['dataset'], 'seed': r['seed'],
            'checkpoint': r['checkpoint'], 'checkpoint_sha256': r['checkpoint_sha256'],
            'training_git_commit': r['git_commit'], 'metrics': f'results/raw/{rid}/metrics.json',
            'metrics_sha256': e.files[f'results/raw/{rid}/metrics.json']['sha256'],
            'config': f'results/raw/{rid}/config.json', 'config_sha256': e.files[f'results/raw/{rid}/config.json']['sha256'],
            'split_sha256': r['split_sha256'], 'processed_sha256': r['processed_sha256'],
            'best_validation_epoch': r['best_epoch'], 'prediction_path': r.get('prediction_path'),
            'prediction_sha256': r.get('prediction_sha256')})
    save_json(out / 'source' / 'verified_runs.json', saved_runs)
    save_json(out / 'source' / 'evidence_manifest.json', {'files': list(e.files.values()),
        'numeric_source_policy': 'Only current checksum-bound completed artifacts and independent inference proofs; no manuscript numbers copied',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'root_git_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()})
    save_json(out / 'source' / 'tables.json', tables)
    return tables


def latex_escape(value):
    return ''.join({'&': r'\&', '%': r'\%', '_': r'\_', '#': r'\#', '*': r'\textasteriskcentered{}',
                    '±': r'$\pm$', '↑': r'$\uparrow$', '↓': r'$\downarrow$',
                    '†': r'$\dagger$', '−': '-', '–': '--'}.get(c, c) for c in str(value))


def export_sources(t, out):
    source = out / 'source'; name = t['name']
    with (source / (name + '.csv')).open('w', newline='') as stream:
        writer = csv.writer(stream);writer.writerow(t['columns']);writer.writerows(t['body'])
    marked = lambda i, j: i in t['bold_rows'] or [i, j] in t['bold_cells']
    md = ['# ' + t['title'], '', t.get('subtitle', ''), '',
          '|' + '|'.join(t['columns']) + '|', '|' + '|'.join(['---'] * len(t['columns'])) + '|']
    tex = [r'\begin{table}[htbp]', r'\centering', r'\setcounter{table}{' + str(t['number'] - 1) + '}',
           r'\caption{' + latex_escape(t['title'].split(': ', 1)[1]) + '}',
           r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{' + 'l' * len(t['columns']) + '}',
           r'\toprule', ' & '.join(map(latex_escape, t['columns'])) + r' \\', r'\midrule']
    headers = ''.join('<th>' + html.escape(c) + '</th>' for c in t['columns'])
    h = ['<!doctype html><meta charset="utf-8"><title>' + html.escape(t['title']) + '</title>',
         '<style>body{font-family:"Times New Roman",serif;margin:40px;color:#111}h2,header{text-align:center}table{border-collapse:collapse;width:100%;margin:24px 0}th,td{padding:7px 12px;text-align:center;white-space:nowrap}thead{border-top:2px solid;border-bottom:1px solid}tbody{border-bottom:2px solid}.group{border-top:1px solid}.notes{font-size:13px;line-height:1.5}td:first-child{text-align:left}</style>',
         '<h2>' + html.escape(t['title']) + '</h2><header>' + html.escape(t.get('subtitle', '')) + '</header>',
         '<table><thead><tr>' + headers + '</tr></thead><tbody>']
    for i, row in enumerate(t['body']):
        md.append('|' + '|'.join('**' + c + '**' if marked(i, j) else c for j, c in enumerate(row)) + '|')
        if i in t['groups']:
            tex.append(r'\midrule')
        tex.append(' & '.join(r'\textbf{' + latex_escape(c) + '}' if marked(i, j) else latex_escape(c)
                              for j, c in enumerate(row)) + r' \\')
        h.append('<tr' + (' class="group"' if i in t['groups'] else '') + '>' + ''.join(
            '<td>' + ('<strong>' + html.escape(c) + '</strong>' if marked(i, j) else html.escape(c)) + '</td>'
            for j, c in enumerate(row)) + '</tr>')
    md += ['', *t['notes']]
    tex += [r'\bottomrule', r'\end{tabular}}', r'\begin{minipage}{\textwidth}\footnotesize',
            r'\par '.join(map(latex_escape, t['notes'])), r'\end{minipage}', r'\end{table}']
    h += ['</tbody></table><div class="notes">' + '<br>'.join(map(html.escape, t['notes'])) + '</div>']
    write(source / (name + '.md'), '\n'.join(md) + '\n')
    write(source / (name + '.tex'), '\n'.join(tex) + '\n')
    write(source / (name + '.html'), '\n'.join(h) + '\n')


def draw(t, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=t['size'], dpi=150, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1]);ax.set_axis_off()
    title = ax.text(.5, .96, t['title'], ha='center', va='top', fontsize=16)
    subtitle = ax.text(.5, .905 if t['number'] not in [2, 8] else .88, t.get('subtitle', ''),
                       ha='center', va='top', fontsize=12.5)
    if t['number'] in [4, 5, 6]:subtitle.set_y(.925)
    if t['number'] == 3:subtitle.set_y(.905)
    widths = np.array(t['widths'], dtype=float);widths /= widths.sum()
    box = [.045, t['table_bottom'], .91, t['table_top'] - t['table_bottom']]
    table = ax.table(cellText=t['body'], colLabels=t['columns'], colWidths=widths.tolist(),
                     loc='center', cellLoc='center', bbox=box)
    table.auto_set_font_size(False);table.set_fontsize(t['font'])
    for (i, j), cell in table.get_celld().items():
        cell.set_facecolor('white');cell.set_edgecolor('black');cell.set_linewidth(.7)
        cell.visible_edges = 'closed' if t['grid'] else ''
        cell.PAD = .055
        text = cell.get_text()
        if i == 0:
            text.set_fontweight('bold')
        elif (i - 1 in t['bold_rows']) or [i - 1, j] in t['bold_cells']:
            text.set_fontweight('bold')
        if i > 0 and j == 0 and t['number'] not in [4, 5, 6]:
            text.set_ha('left')
        if text.get_text() in ['Pending', 'Not run']:
            text.set_fontstyle('italic');text.set_color('#444444')
    fig.canvas.draw()
    if not t['grid']:
        ytop = box[1] + box[3];step = box[3] / (len(t['body']) + 1)
        for y, weight in [(ytop, 1.1), (ytop - step, .8), (box[1], 1.1)] + [
                (ytop - step * (g + 1), .7) for g in t['groups']]:
            ax.add_line(Line2D([box[0], box[0] + box[2]], [y, y], color='black', linewidth=weight))
    # Fit notes to the available lower margin; retain explicit source warnings.
    max_chars = int(t['size'][0] * 13.3)
    note_lines = [part for note in t['notes'] for part in textwrap.wrap(note, max_chars, break_long_words=False)]
    note_font = 10.2 if t['number'] in [4, 5, 6] else 10.7
    note_step = note_font * 1.34 / 72 / t['size'][1]
    first_note = box[1] - .035
    note_artists = [ax.text(.045, first_note - i * note_step, line, ha='left', va='top', fontsize=note_font)
                    for i, line in enumerate(note_lines)]
    fig.canvas.draw();renderer = fig.canvas.get_renderer()
    for (i, j), cell in table.get_celld().items():
        text = cell.get_text();tb = text.get_window_extent(renderer);cb = cell.get_window_extent(renderer)
        assert tb.width <= cb.width - 1, f'{t["name"]} text overflow at {(i,j)}: {text.get_text()}'
        assert tb.height <= cb.height - 1, f'{t["name"]} row overflow at {(i,j)}'
    for item in [title, subtitle] + note_artists:
        b = item.get_window_extent(renderer)
        assert b.x0 >= 0 and b.y0 >= 0 and b.x1 <= fig.bbox.width and b.y1 <= fig.bbox.height, (t['name'], item.get_text())
    fig.savefig(out / (t['name'] + '.png'), dpi=300, facecolor='white')
    fig.savefig(out / (t['name'] + '.pdf'), facecolor='white',
                metadata={'Title': t['title'], 'Subject': t.get('subtitle', ''), 'Author': 'LightSAKT2 corrected research artifacts'})
    plt.close(fig)


def normalize(value):
    return ''.join(str(value).split()).replace('−', '-').replace('–', '-').replace('—', '-')


def verify_outputs(out, tables):
    import pymupdf as fitz
    from PIL import Image
    records = []
    for t in tables:
        with (out / 'source' / (t['name'] + '.csv')).open() as stream:
            rows = list(csv.reader(stream))
        assert rows == [t['columns']] + t['body']
        with Image.open(out / (t['name'] + '.png')) as image:
            assert image.size == tuple(round(x * 300) for x in t['size'])
            image.verify()
        document = fitz.open(out / (t['name'] + '.pdf'))
        assert len(document) == 1
        text = normalize(document[0].get_text())
        for row in [t['columns']] + t['body']:
            for cell in row:
                assert normalize(cell) in text, (t['name'], cell, 'not found in generated vector PDF')
        for note in t['notes']:
            assert normalize(note) in text, (t['name'], note)
        # GKT is always a placeholder; no model-number inference from active jobs.
        if t['number'] == 7:
            gkt = t['columns'].index('GKT')
            assert all(row[gkt] == 'Pending' for row in t['body'])
            assert all('GKT' not in row['models'] for row in t['row_evidence'])
        records.append({'table': t['number'], 'name': t['name'], 'rows': len(t['body']),
                        'columns': len(t['columns']), 'pdf_cells_and_notes_checked': True,
                        'png_sha256': digest(out / (t['name'] + '.png')),
                        'pdf_sha256': digest(out / (t['name'] + '.pdf')),
                        'gkt_status': t['gkt'], 'unavailable': t['partially_unavailable']})
    document = fitz.open(out / 'All_Updated_Tables.pdf')
    assert len(document) == len(tables) == 7
    for i, t in enumerate(tables):
        assert normalize(t['title']) in normalize(document[i].get_text())
    return records


def index(out, tables, records):
    lines = ['# Updated paper tables for the supervisor', '',
        'Seven original table images were inspected and recreated from verified corrected artifacts. '
        'The combined PDF is in paper order: Tables 2, 3, 4, 5, 6, 7, 8. '
        'No training or checkpoint was started, stopped or changed.', '',
        '[Download the combined PDF](All_Updated_Tables.pdf)', '',
        'Untouched extracted reference images are kept separately in `paper/table_image_inputs/`. '
        'The PNGs in this sharing folder are the updated versions.', '',
        '|Table|Original image inside allTables.zip|PNG / individual PDF|Meaning and verified source|GKT handling|',
        '|---|---|---|---|---|']
    for t in tables:
        n = t['number']
        sources = {2: 'reports/*_statistics.json; results/tables/Table_2.csv; raw headers and locked split checks',
                   3: 'configs/protocol.json; selected LightSAKT final configs; experiments/selections/',
                   4: 'results/tables/Table_4.csv; completed grid metrics/checkpoints; reports/light_grid_inference_verification.json',
                   5: 'results/tables/Table_5.csv; completed grid metrics/checkpoints; reports/light_grid_inference_verification.json',
                   6: 'results/tables/Table_6.csv; completed grid metrics/checkpoints; reports/light_grid_inference_verification.json',
                   7: 'results/tables/Table_7.csv; 75 final-seed metrics; reports/selected_final_inference_verification.json; reports/BASELINE_COMPARISON.md',
                   8: 'results/tables/Table_8.csv; 90 logical default/ablation seed records and independent inference proofs'}[n]
        lines.append(f'|{n}|`{t["original_filename"]}`|[PNG]({t["name"]}.png) · [PDF]({t["name"]}.pdf)|{t["title"]}. {sources}|'
                     + ('Pending in all metric rows' if n == 7 else 'Not present') + '|')
    lines += ['', '## Explicit limitations and corrections', '',
        '- Table 8 retains the original “No layer normalization” row as **Not run**. It has no verified experiment and no inferred numbers. '
        'The approved “Without regularization” row is added and explicitly reuses No Dropout evidence.',
        '- Table 7 marks all GKT final results **Pending**. Historical scores, primary runs, incomplete seeds and both unfinished variants are excluded.',
        '- Table 7 shows five-seed held-out test means ± sample SD. Table 8 shows five-seed test AUC. '
        'Tables 4–6 show validation AUC, never test AUC. Best values use full precision; lower MSE/BCE is better.',
        '- Table 2 labels students, cleaned records, observed composite/sequence skill identifiers and raw column counts explicitly. '
        'Algebra is the labelled training subset, with training-only vocabulary and an unknown-ID bucket.',
        '- The original screenshot of Table 5 spans two manuscript pages; the updated table presents all 36 configurations together.',
        '- Formatting changes do not establish statistical significance or alter the preserved efficiency findings. '
        'There are no efficiency/significance tables in this particular zip.', '',
        '## Editable sources and verification', '',
        '`source/` contains one CSV, Markdown, LaTeX and HTML per table, `tables.json` with rendered cells, '
        'full-precision values and row-level provenance, and the generator script. '
        'The LaTeX snippets require `booktabs` and `graphicx`. PDFs were generated directly as vector tables, not with LaTeX.', '',
        '`source/project_snapshot/` freezes every consumed config, metric, summary and independent proof. '
        '`source/evidence_manifest.json` binds these copies to project paths and SHA256 checksums. '
        '`source/verified_runs.json` records run IDs, seeds, training Git commits, checkpoints and hashes. '
        '`source/verification.json` records source checks, image dimensions, searchable PDF cell checks and output checksums.', '',
        'Rebuild from current verified project inputs (uses retained input snapshots when present):', '',
        '```bash', 'cd /workspace/LightSAKT2',
        "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.supervisor_tables", '```', '',
        'Render the frozen editable table specifications without accessing datasets or checkpoints:', '',
        '```bash', "CUDA_VISIBLE_DEVICES='' python paper/updated_tables_for_sir/source/generate_tables.py --render-only", '```', '',
        'All seven images and the combined PDF are ready to share **with the visible Pending/Not run disclosures**. '
        'The remaining GKT research schedule continues independently.']
    write(out / 'README.md', '\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--out', type=Path, default=Path('paper/updated_tables_for_sir'))
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve();out = (root / args.out).resolve()
    assert out.is_relative_to(root) and root.name == 'LightSAKT2'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Run CPU-only during healthy GPU training'
    out.mkdir(parents=True, exist_ok=True);(out / 'source').mkdir(exist_ok=True)
    inventory = json.load(open(out / 'source' / 'original_inventory.json'))
    assert len(inventory['images']) == 7
    assert {r['original_filename'] for r in inventory['images']} == {
        f'allTables/Screenshot 2026-10-17 {suffix}.png' for suffix in ORIGINALS.values()}
    for image in inventory['images']:
        assert digest(root / image['extracted_path']) == image['sha256']
    zip_present = (root / inventory['zip_path']).exists()
    if zip_present:
        assert digest(root / inventory['zip_path']) == inventory['zip_sha256']
    original_fingerprint = fingerprint(root)
    manifest_ids = [j['run_id'] for j in json.load(open(root / 'experiments/manifest.json'))['jobs']]
    if args.render_only:
        tables = json.load(open(out / 'source' / 'tables.json'))
    else:
        tables = build(root, out)
    from pypdf import PdfWriter
    writer = PdfWriter()
    for t in tables:
        export_sources(t, out);draw(t, out)
        writer.append(out / (t['name'] + '.pdf'))
    writer.add_metadata({'/Title': 'LightSAKT2 — corrected paper Tables 2–8', '/Subject': 'Verified results; GKT pending'})
    with (out / 'All_Updated_Tables.pdf').open('wb') as stream:
        writer.write(stream)
    records = verify_outputs(out, tables)
    assert fingerprint(root) == original_fingerprint
    assert [j['run_id'] for j in json.load(open(root / 'experiments/manifest.json'))['jobs']] == manifest_ids
    for r in json.load(open(out / 'source' / 'evidence_manifest.json'))['files']:
        assert digest(out / r['snapshot_path']) == r['sha256']
    save_json(out / 'source' / 'verification.json', {'verified': True,
        'verified_utc': datetime.now(timezone.utc).isoformat(), 'original_images': 7, 'updated_tables': 7,
        'combined_pdf': 'All_Updated_Tables.pdf', 'combined_pdf_sha256': digest(out / 'All_Updated_Tables.pdf'),
        'tables': records, 'model_source_fingerprint_unchanged': original_fingerprint,
        'schedule_run_ids_unchanged': True, 'training_invoked': False, 'inference_invoked': False,
        'processes_signalled': False, 'gkt_final_values_excluded': True,
        'all_original_image_checksums_verified': True, 'original_zip_checksum_checked': zip_present,
        'original_zip_sha256': inventory['zip_sha256']})
    index(out, tables, records)
    script = Path(__file__).resolve();copy = out / 'source' / 'generate_tables.py'
    if script != copy:
        shutil.copyfile(script, copy)
    print(f'Verified 7 updated table PNGs, 7 individual PDFs and combined PDF: {out}', flush=True)


if __name__ == '__main__':
    main()
