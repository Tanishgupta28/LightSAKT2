import json
import subprocess
from pathlib import Path
import pandas as pd
from scripts.progress import write_progress


def main():
    manifest=json.load(open('experiments/manifest.json'));progress=write_progress();counts=progress['counts']
    metrics=[]
    for path in Path('results/raw').glob('*/metrics.json'):
        result=json.load(open(path));metrics.append(result)
    primary=[r for r in metrics if r['group']=='primary'];smoke=[r for r in metrics if r['group']=='smoke']
    lines=['# LightSAKT2 research rebuild — verified progress report','',
           '**The complete research rebuild is not yet finished.** This report distinguishes executed work from queued work. Historical manuscript values are not used as new results.','',
           '## Completed implementation and verification','',
           '* Verified root `/workspace/LightSAKT2`; all development and new artifacts are inside it.',
           '* Preserved 1,613 inventoried original files with matching byte sizes, SHA256 manifests, and a readable archival copy. Original Drive files were never modified. Read-only local artifact copies have explicitly limited provenance.',
           '* Documented and protected the original architecture; exact original parameter names/shapes and counts are tested. No replacement architecture or new LightSAKT component was added.',
           '* Corrected future-dependent global KV, incorrect normalization, key/value padding, positional slicing, next-response alignment and prediction-changing softmax visualization.',
           '* Rebuilt student-disjoint 70/10/20 splits before chunking, train-only vocabularies, binary-label rules, simultaneous-tag collapse and response-event chronology. Raw, processed and split hashes are saved.',
           '* Corrected baseline masking, shape handling, temporal-memory updates, uninitialized GKT parameters, and DKT+ two-endpoint penalty masks. Historical simplified GKT PAM/MHA remain explicitly distinct from canonical published GKT.',
           '* Built Adam training, validation-AUC early stopping, best/latest checkpoints, atomic writes, gradient/NaN checks, full RNG recovery and code-fingerprint resume protection.',
           '* Implemented 108 LightSAKT grid configurations and equal 36-configuration tuning budgets for every baseline variant; frozen validation selections drive matched repeated seeds.',
           '* Implemented four distinct component ablations. User-approved no_regularization aliases no_dropout and reuses its evidence; default reuses final LightSAKT results.',
           '* Implemented programmatic tables, paired Wilcoxon/Holm analysis, controlled efficiency benchmarks, true kernel heatmaps, and heldout skill-probability case studies. Implementation does not mean those dependent analyses have executed.','',
           '## Executed tests','']
    test=Path('reports/tests_final.log')
    if test.exists():lines+=['```',test.read_text().strip(),'```','']
    lines+=['Tests cover causality, the original leakage regression, dense float64 kernel/gradient equivalence, padding, all baseline forwards/backwards, batch independence, CPU/CUDA agreement, CUDA repeatability, data coverage/chronology, validation-only selection, heldout-label update independence, no evaluation updates, exact interruption/resume, checkpoints, undefined AUC, tuning budgets and regularization alias semantics.','',
            '## Genuine completed primary training results','',
            '|Dataset|Model|Seed|Epochs trained|Selected epoch|Test AUC|Accuracy|F1|MSE|BCE|','|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted(primary,key=lambda r:(r['dataset'],r['model'])):
        m=r['test'];lines.append('|'+ '|'.join([r['dataset'],r['model'],str(r['seed']),str(r['epochs_trained']),str(r['best_epoch'])]+[f"{m[k]:.6f}" for k in ['auc','accuracy','f1','mse','bce']])+'|')
    lines+=['',f"Completed smoke runs: {len(smoke)}. Smoke validation is not a final research comparison.",'',
            '## Experiment schedule status','',json.dumps(counts,indent=2),'',
            f"Total requested schedule slots: {len(manifest['jobs'])}. Counts include 21 seed-42 checkpoint reuse slots and 15 regularization aliases, which are not new independent trainings. The five-seed primary result is the mean, never the best seed. All remaining pending slots have no experimental result yet.",'',
            '## Findings and limitations','',
            'The original future-response leakage and non-normalized attention are reproduced by executed tests. The corrected default retains the original contribution but historical scores are no longer valid evidence. Completed primary results are new fixed-configuration single-seed measurements; a superiority claim requires the unfinished matched final comparisons. No fresh statistical significance or measured efficiency conclusion is established until the corresponding artifact files exist. O(L) sequence-length complexity alone does not establish lower latency or lower measured memory.',
            '', 'Raw data came from available local original/reference materials because the shared folder lacked required raw CSV/TSV files. Exact hashes and publisher references are documented, but publisher byte identity and an explicit redistribution license are not established. Algebra uses the labelled training subset, not complete official master coverage. GKT is the historical simplified implementation; canonical published GKT reproducibility remains a limitation. All 27 manuscript pages were rendered and visually inspected, including Figures 1–11; inconsistencies are documented in PAPER_CORRECTIONS.md. No editable manuscript source was present.',
            '', 'The no_regularization row is scientifically identical to no_dropout by user instruction and cannot be treated as another independent ablation. Five paired seeds yield weak statistical power, with minimum exact two-sided Wilcoxon p=.0625 for five nonzero pairs before correction; seed variation on one fixed split does not estimate new-student sampling uncertainty.',
            '', '## Reproduce and continue','', '```',
            'cd /workspace/LightSAKT2','python -m pytest -q','python -m scripts.schedule',
            'python -m scripts.summarize','python -m scripts.analyze','python -m scripts.verify_artifacts','```','',
            'Do not start another scheduler while `experiments/scheduler.lock` is held. Inspect `reports/scheduler.log`, `reports/run_logs/`, and per-run status/config JSON files. For failed runs, inspect the failure then use `python -m scripts.schedule --retry-failed`. Each run status includes its resume command. Configurations and completed outputs are immutable; no test-set tuning is allowed.',
            '', 'The detached scheduler process can survive a terminal disconnect; it cannot automatically restart after a machine shutdown. The checkpoint state is designed for manual restart from durable storage. Do not assume future automatic completion after environment loss.',
            '', '## Storage and synchronization','',
            'Private repository: https://github.com/Tanishgupta28/LightSAKT2, branch `research-rebuild`. No force pushes. Local Git bundles and the original/raw archive are under `backups/`; their references are under `reports/`. Git synchronization details are in GITHUB_SYNC_PENDING.md. Large raw files and checkpoints remain on the durable /workspace network mount, excluded from ordinary Git. These local archives are not independent offsite artifact copies.',
            '', 'Originals: `original_snapshot/`; raw datasets: `data/raw/`; fresh preprocessing: `data/processed/`; student manifests: `data/splits/`; fresh weights: `checkpoints/<run>/`; raw metrics/configs: `results/raw/<run>/`; predictions: `results/predictions/`; generated tables/figures: `results/tables/`, `results/figures/`; paper evidence: `paper/`, PAPER_CORRECTIONS.md.']
    Path('FINAL_RESEARCH_REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
