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
    completion=('**All scheduled experiments and final output checks have passed.** The dataset provenance and historical simplified GKT limitations below still apply.' if progress['research_verified_complete'] else
                '**The complete research rebuild is not yet finished.** This report distinguishes executed work from queued work. Historical manuscript values are not used as new results.')
    lines=['# LightSAKT2 research rebuild — verified progress report','',
           completion,'',
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
    latest=Path('reports/latest_test_run.json')
    if latest.exists():test=Path(json.load(open(latest))['log'])
    analyses=Path('experiments/analyses.json')
    if analyses.exists():
        task=json.load(open(analyses))['tasks'].get('full_tests',{})
        if task.get('status')=='completed':test=Path(task['log'])
    if test.exists():lines+=['```',test.read_text().strip(),'```','']
    cpu_test=Path('reports/latest_cpu_test_run.json')
    if cpu_test.exists():
        record=json.load(open(cpu_test))
        lines+=['Additional CPU verification during active GPU training:','', '```',Path(record['log']).read_text().strip(),'```','',
                'CUDA tests were skipped in this additional invocation to avoid GPU contention; the preceding full CPU/CUDA correctness gate remains the GPU evidence. Model/training source fingerprints are unchanged.','']
    lines+=['Tests cover causality, the original leakage regression, dense float64 kernel/gradient equivalence, padding, all baseline forwards/backwards, batch independence, CPU/CUDA agreement, CUDA repeatability, data coverage/chronology, validation-only selection, heldout-label update independence, no evaluation updates, exact interruption/resume, checkpoints, undefined AUC, tuning budgets and regularization alias semantics.','',
            '## Genuine completed primary training results','',
            '|Dataset|Model|Seed|Epochs trained|Selected epoch|Test AUC|Accuracy|F1|MSE|BCE|','|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted(primary,key=lambda r:(r['dataset'],r['model'])):
        m=r['test'];lines.append('|'+ '|'.join([r['dataset'],r['model'],str(r['seed']),str(r['epochs_trained']),str(r['best_epoch'])]+[f"{m[k]:.6f}" for k in ['auc','accuracy','f1','mse','bce']])+'|')
    lines+=['',f"Completed smoke runs: {len(smoke)}. Smoke validation is not a final research comparison.",'',
            '## Experiment schedule status','',json.dumps(counts,indent=2),'',
            f"Total requested schedule slots: {len(manifest['jobs'])}. Counts include 21 seed-42 checkpoint reuse slots, 15 regularization aliases, and {progress['additional_primary_grid_reuses']} verified primary/grid reuse slots, which are not new independent trainings. Distinct planned trainings: {progress['distinct_training_runs_planned']}. Every required grid row is retained. The five-seed primary result is the mean, never the best seed. All remaining pending slots have no experimental result yet.",'',
            '## Findings and limitations','',
            'The original future-response leakage and non-normalized attention are reproduced by executed tests. The corrected default retains the original contribution but historical scores are no longer valid evidence. Completed primary results are new fixed-configuration single-seed measurements; a superiority claim requires the unfinished matched final comparisons. No fresh statistical significance or measured efficiency conclusion is established until the corresponding artifact files exist. O(L) sequence-length complexity alone does not establish lower latency or lower measured memory.',
            '', 'Raw data came from available local original/reference materials because the shared folder lacked required raw CSV/TSV files. Exact hashes and publisher references are documented, but publisher byte identity and an explicit redistribution license are not established. Algebra uses the labelled training subset, not complete official master coverage. GKT is the historical simplified implementation; canonical published GKT reproducibility remains a limitation. All 27 manuscript pages were rendered and visually inspected, including Figures 1–11; inconsistencies are documented in PAPER_CORRECTIONS.md. No editable manuscript source was present.',
            '', 'The no_regularization row is scientifically identical to no_dropout by user instruction and cannot be treated as another independent ablation. Five paired seeds yield weak statistical power, with minimum exact two-sided Wilcoxon p=.0625 for five nonzero pairs before correction; seed variation on one fixed split does not estimate new-student sampling uncertainty.',
            '', 'Measured runtimes, phase/model/dataset extrapolations, uncertainty scenarios and conditional completion dates are in reports/ETA_REPORT.md and reports/ETA.json. The schedule audit and before/after effective manifests are in reports/SCHEDULE_AUDIT.md. Live health samples and recovery checks are in reports/EXECUTION_HEALTH.md. GPU utilization is unavailable on the current MIG partition.',
            '', '## Reproduce and continue','', '```',
            'cd /workspace/LightSAKT2','python -m pytest -q','python -m scripts.schedule',
            'python -m scripts.summarize','python -m scripts.analyze','python -m scripts.verify_artifacts','```','',
            'Do not start another scheduler while `experiments/scheduler.lock` is held. Inspect `reports/scheduler.log`, `reports/run_logs/`, and per-run status/config JSON files. For failed runs, inspect the failure then use `python -m scripts.schedule --retry-failed`. Each run status includes its resume command. Configurations and completed outputs are immutable; no test-set tuning is allowed.',
            '', 'The detached scheduler process can survive a terminal disconnect; it cannot automatically restart after a machine shutdown. The checkpoint state is designed for manual restart from durable storage. Do not assume future automatic completion after environment loss.',
            '', '## Storage and synchronization','',
            'Private repository: https://github.com/Tanishgupta28/LightSAKT2, branch `research-rebuild`. No force pushes. Local Git bundles and the original/raw archive are under `backups/`; their references are under `reports/`. Git synchronization details are in GITHUB_SYNC_PENDING.md. Large raw files and checkpoints remain on the /workspace NFS mount, excluded from ordinary Git. Retention after environment deprovisioning is unverified; local archives are not independent offsite artifact copies.',
            '', 'Originals: `original_snapshot/`; raw datasets: `data/raw/`; fresh preprocessing: `data/processed/`; student manifests: `data/splits/`; fresh weights: `checkpoints/<run>/`; raw metrics/configs: `results/raw/<run>/`; predictions: `results/predictions/`; generated tables/figures: `results/tables/`, `results/figures/`; paper evidence: `paper/`, PAPER_CORRECTIONS.md.']
    Path('FINAL_RESEARCH_REPORT.md').write_text('\n'.join(lines)+'\n')
    summary_path=Path('results/tables/LightSAKT_final_summary.csv')
    if summary_path.exists():
        summary=pd.read_csv(summary_path)
        with Path('FINAL_RESEARCH_REPORT.md').open('a') as output:
            output.write('\n## Verified selected five-seed LightSAKT results\n\n')
            output.write('|Dataset|Seeds|Test AUC|Accuracy|F1|MSE|BCE|\n|---|---|---|---|---|---|---|\n')
            for row in summary.to_dict('records'):
                cells=[row['dataset'],str(row['completed_seeds'])]+[f'{row[m+"_mean"]:.6f} ± {row[m+"_std"]:.6f}' for m in ['auc','accuracy','f1','mse','bce']]
                output.write('|'+ '|'.join(cells)+'|\n')
            output.write('\nMean ± sample SD across prespecified seeds on one frozen student split. Selected hyperparameters, strict checkpoint/prediction checks, completed-only baseline evidence and model-specific ETA decomposition are in reports/LIGHTSAKT_FINAL_RESULTS.md. Single-seed primary baselines are explicitly descriptive; pending selected baseline seed sets are excluded from final comparisons. Reproduce with the CPU-only scripts.light_final_report command recorded there.\n')
            proof_path=Path('reports/selected_final_inference_verification.json')
            if proof_path.exists():
                proof=json.load(open(proof_path));runs=[r for r in proof['runs'] if r['verified'] and r.get('group','final')=='final']
                output.write(f'\nIndependent CPU inference reproduced validation metrics and every saved test probability for {len(runs)} selected final checkpoints, including {sum(r["model"]=="LightSAKT" for r in runs)} LightSAKT seeds. Exact identities, target labels, per-probability tolerances, hashes and timings are in {proof_path}. All individual five-metric results, selected epochs, seeds, run IDs and checkpoint/Git references are in reports/LIGHTSAKT_FINAL_SEED_RESULTS.md. Complete matched baseline comparisons and pending models are in reports/BASELINE_COMPARISON.md. The forensic explanation of the earlier 391-hour forecast is in reports/ETA_HISTORY_AUDIT.md. This does not mark the unfinished research schedule complete.\n')
    offsite=Path('reports/offsite_backup.json')
    efficiency=Path('paper/EFFICIENCY_RESULTS.md')
    if efficiency.exists():
        with Path('FINAL_RESEARCH_REPORT.md').open('a') as output:
            output.write('\n## Executed efficiency evidence\n\nThe exclusive 120-case benchmark has completed; its measured latency, memory, throughput, parameter counts and scope-specific limits are reported in paper/EFFICIENCY_RESULTS.md. The current measurements contradict an unconditional LightSAKT speed claim. Training continues on the preserved complete schedule; the efficiency benchmark is reused, and full epoch-time documentation is refreshed at final analysis.\n')
            verified=Path('reports/efficiency_verification.json')
            if verified.exists():
                record=json.load(open(verified))
                output.write(f"\nIndependent CPU rederivation verified {record['timed_repetitions_verified']} original timing repetitions across {record['cases']} cases. LightSAKT was slower in {record['higher_light_latency']}/{record['matched_comparisons']} matched comparisons; peak allocated memory was lower in {record['lower_light_peak_allocated_memory']}/{record['matched_comparisons']}. Primary parameter counts favor LightSAKT only on Algebra2005, with different documented widths. The generated main manuscript now separates theoretical O(L) attention work, measured latency, allocated/reserved memory and parameter counts. Evidence hashes and numeric checks are in reports/efficiency_verification.json; no training or benchmark mathematics was changed.\n")
    optimization=Path('reports/optimization/SCHEDULE_COMPARISON.json')
    if optimization.exists():
        record=json.load(open(optimization))
        with Path('FINAL_RESEARCH_REPORT.md').open('a') as output:
            output.write('\n## Execution optimization and unchanged scientific scope\n\n')
            output.write(f"Latest schedule audit: {record['fresh_remaining']} fresh trainings remain plus {record['reuse_remaining']} reuse/evaluation entries. All 108 LightSAKT candidates are retained, with one seed per grid candidate and five matched seeds only for selected final configurations and ablations. The author chose the complete schedule with execution improvements only; all 162 GKT configurations discussed in the archived reduction proposal remain scheduled. Evidence: experiments/schedule_decision.json. Full inventory and proposal: reports/optimization/SCHEDULE_AUDIT.md and PROPOSED_DIFFERENCES.csv.\n\n")
            output.write('Exact immutable sequence caches preserve every tensor, identity, batch order and RNG sequence; CPU training/resume equivalence is tested. Inter-run GPU probes and CPU/CUDA gates verify independent worker updates before two-worker GKT execution. Full required scope, FP32, batch sizes, Adam, stopping criteria, LightSAKT architecture, splits and untouched final test selections remain unchanged. Prioritization runs LightSAKT grid/finals/ablations before slow remaining baseline tuning. See reports/optimization/EXECUTION_OPTIMIZATION.md and reports/profiling/.\n')
    if offsite.exists():
        record=json.load(open(offsite))
        with Path('FINAL_RESEARCH_REPORT.md').open('a') as output:
            output.write(f"\nIndependent private GitHub draft recovery artifacts: {len(record['assets'])} assets have been downloaded and SHA256 verified. All available completed archives synchronized: {record.get('all_available_completed_archives_synced',False)}. See reports/OFFSITE_RECOVERY.md for recovery commands and reports/offsite_backup.json for immutable asset IDs/checksums. Running checkpoints are saved on NFS until their completed-run archive is uploaded. Cloud artifacts preserve evidence after workspace loss; they do not keep GPU training alive after shutdown.\n")

if __name__=='__main__':main()
