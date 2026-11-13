"""Shared, deterministic presentation settings; no model-performance constants."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ASSETS = Path(__file__).resolve().parents[1]
ROOT = ASSETS.parents[1]
DATASETS = ['ASSIST2009', 'Algebra2005', 'EdNetKT1_5k_first']
MODELS = ['DKT', 'DKT+', 'DKVMN', 'GKT', 'GKT-MHA (implemented variant)', 'SAKT',
          'LightSAKT', 'LightSAKT + Question Difficulty', 'SAKT + Question Difficulty']
SEEDS = [42, 123, 2026, 3407, 7777]
SHORT = {'LightSAKT + Question Difficulty': 'LightSAKT + QD', 'SAKT + Question Difficulty': 'SAKT + QD',
         'GKT-MHA (implemented variant)': 'GKT-MHA (impl.)'}
DISPLAY = {'EdNetKT1_5k_first': 'EdNet-KT1 (5k, first/session)'}
FOLDERS = {'EdNetKT1_5k_first': 'EdNet_KT1_5k', 'ASSIST2009': 'ASSIST2009', 'Algebra2005': 'Algebra2005'}
COLORS = dict(zip(MODELS, ['#1f77b4', '#9467bd', '#d62728', '#8c564b', '#7f7f7f',
                          '#2ca02c', '#ff7f0e', '#17becf', '#e377c2']))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9,
    'axes.labelsize': 8, 'legend.fontsize': 7, 'axes.linewidth': .7, 'lines.linewidth': 1.15,
    'grid.alpha': .25, 'svg.hashsalt': 'lightsakt-paper-assets-20261112',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 320})


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csv(path, records):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    frame.to_csv(path, index=False, lineterminator='\n', float_format='%.17g')


def save_figure(fig, relative, inputs, caption):
    """Write genuine plots plus the numerical-input hashes and honest captions."""
    base = ASSETS / relative; base.parent.mkdir(parents=True, exist_ok=True)
    fixed = datetime(2026, 11, 12, tzinfo=timezone.utc)
    for ext in ['png', 'pdf', 'svg']:
        metadata = {'Creator': 'LightSAKT reproducible Python plotting'}
        if ext == 'pdf': metadata.update(CreationDate=fixed, ModDate=fixed)
        if ext == 'svg': metadata = {'Date': fixed.isoformat(), 'Creator': metadata['Creator']}
        fig.savefig(base.with_suffix('.' + ext), dpi=320, bbox_inches='tight', metadata=metadata)
        if ext == 'svg':
            target = base.with_suffix('.svg')
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)
    paths = [ASSETS / p for p in inputs]
    write(base.with_suffix('.provenance.json'), {'caption': caption, 'generator': os.environ.get('PAPER_GENERATOR', ''),
        'input_files': [{'path': str(p.relative_to(ASSETS)), 'sha256': sha(p)} for p in paths],
        'outputs': [{'path': str(base.with_suffix('.' + e).relative_to(ASSETS)),
                     'sha256': sha(base.with_suffix('.' + e))} for e in ['png', 'pdf', 'svg']],
        'PNG_dpi': 320, 'performance_data': 'genuine saved experiments only; no interpolation or imputation'})


def table_exports(frame, relative, title, max_rows=22):
    """CSV, escaped LaTeX snippet, and paginated vector PDF, without a TeX install."""
    base = ASSETS / relative; base.parent.mkdir(parents=True, exist_ok=True)
    csv(base.with_suffix('.csv'), frame)
    frame.to_latex(base.with_suffix('.tex'), index=False, escape=True, longtable=len(frame) > max_rows,
                   caption=title, float_format=lambda x: f'{x:.6f}')
    from matplotlib.backends.backend_pdf import PdfPages
    fixed = datetime(2026, 11, 12, tzinfo=timezone.utc)
    shown = frame.fillna('PENDING').copy().astype(str)
    with PdfPages(base.with_suffix('.pdf'), metadata={'Title': title, 'CreationDate': fixed, 'ModDate': fixed}) as pdf:
        for start in range(0, max(1, len(shown)), max_rows):
            part = shown.iloc[start:start + max_rows]
            fig, ax = plt.subplots(figsize=(max(11.7, len(part.columns)*1.5), max(3, .28*(len(part)+3))))
            ax.axis('off'); ax.set_title(title, pad=16)
            table = ax.table(cellText=part.values, colLabels=part.columns, cellLoc='center', loc='center')
            table.auto_set_font_size(False); table.set_fontsize(6); table.scale(1, 1.45)
            for (r, c), cell in table.get_celld().items():
                cell.set_edgecolor('#cccccc'); cell.set_linewidth(.35)
                if r == 0: cell.set_facecolor('#eef1f4'); cell.set_text_props(weight='bold')
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
