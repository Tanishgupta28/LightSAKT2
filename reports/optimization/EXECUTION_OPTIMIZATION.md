# Execution optimization: measured evidence and protected scope

The current full 957-entry research ledger is retained. The complete and accelerated proposal files are reviewable snapshots; the 162-candidate GKT reduction is not active without author approval. The 912 figure is the total distinct planned trainings, not the remaining count. All 108 unique LightSAKT candidates remain; each grid candidate uses seed 42. Five seeds apply only to frozen selected final configurations and controlled ablations. Default ablations reuse the final model; no_regularization aliases no_dropout by author instruction. Nine exact primary/grid duplicates also reuse genuine checkpoints. No other duplicate candidate or repeated grid seed was found.

## Measured forecasts

|Schedule|Remaining isolated serial hours|Occupied MIG hours with admitted concurrency|Scope|
|---|---|---|---|
|Complete|501.3|391.3|All distinct requested and expanded tuning runs|
|Accelerated proposal|172.1|136.3|Only supplementary GKT tuning 36→9 per variant/dataset; requires approval|

These are conditional training-only forecasts, using measured full epochs for all 21 model/dataset strata and primary/wide GKT step measurements on all three datasets. They exclude process startup, final test inference, exclusive efficiency benchmarks and final artifact verification. Stopping epoch is the largest unknown: many future GKT configurations have only a single primary stopping-count reference. Full low/high scenarios are {'low': 87.26070784241692, 'central': 391.27396164515596, 'high': 1465.3843023882318}; these are an earliest-stopping versus maximum-epoch/runtime envelope, not a confidence interval. Parallel throughput uses the smallest observed MHA gain. The high scenario assumes no sustained concurrency gain. Epochs include validation CPU work; isolated worker hours, summed concurrent process hours, and occupied GPU wall hours are distinct quantities.

## Profiling and accepted implementation changes

27 actual train-split diagnostic configurations cover all seven model labels on all three datasets plus the widest PAM/MHA variants. These untrained probes measure implementation cost, not predictive research metrics. Algebra PAM executed 49,679 CUDA launches, 200 recurrent backward calls and 1,832 stream synchronizations in a profiled step. Data collation and transfers were small compared with GKT recurrent computation and host dispatch. Profiling overhead is excluded from timed steps and full-epoch evidence remains the ETA anchor. MIG utilization percentage is unavailable; no utilization percentage is inferred from board power.

Exact sequence arrays were built using three independent CPU preprocessing workers, stored as immutable SHA256-verified memory maps, and checked tensor-by-tensor and identity-by-identity on all nine dataset splits. Loading avoids repeated CSV grouping without changing preprocessing or vocabulary. Bit-exact CPU training, shuffle, RNG and interrupted-resume tests passed. Real construction/load timings are in REAL_DATA_CACHE_VERIFICATION.json. Training uses unchanged DataLoader batch sizes, order and workers=0; collation measured roughly 0.5–1.2 milliseconds and transfers roughly 0.1–0.3 milliseconds. More DataLoader workers have no demonstrated benefit for these cached rows.

Two independently seeded GKT-MHA processes produced exactly matching parameters and gradients on all datasets and the widest Algebra configuration, with observed throughput gains of 1.51–1.61×. The controller and shared/exclusive GPU admission lock permit two MHA workers; PAM and all other models stay exclusive. Wide Algebra PAM measured 0.90× throughput with two workers, so PAM concurrency was rejected. There is at most one active non-MHA model; waiting workers allocate no CUDA tensors. Shared locks do not change model operations, gradients or random-number state. Concurrency is recorded in environment manifests. Final controlled efficiency benchmarks run exclusively after training; concurrent epoch timings must carry their execution metadata.

The original healthy GPU run finished and saved its checkpoint before inter-run diagnostics and the fresh CPU/CUDA gate. The original scheduler PID remains alive; an execution coordinator prioritizes LightSAKT grids, selected LightSAKT seeds, and ablations, then fast baseline results and slower tuning. No scope reduction, completed-run reset or test-dependent priority was used. Materialized configurations are durably saved in experiments/runtime_jobs/ and per-run result/status files without racing the original scheduler ledger. Restarts reload these configurations and freeze the same validation selection.

## Changes investigated but not enabled

BF16 probes on identical real train batches and dropout RNG exceeded strict 1e-6 probability and gradient tolerances on all datasets (maximum probability differences 0.00267–0.00311). AMP/TF32 remain off. Larger batches alter Adam update sequences; padding trimming and length bucketing can change dropout RNG and batches, so these were not applied. torch.compile is not enabled: the full recurrent graph and compilation overhead have not passed numerical/RNG/resume equivalence or demonstrated a net runtime gain. No compilation speedup is included in forecasts. Aggregating finite checks could reduce synchronization, but no model/training-source rewrite is adopted without gradient/resume proof; correctness checks remain intact.

## Scientific impact and pending decision

Executed changes affect storage, orchestration and implementation overhead. LightSAKT embeddings, learned positions, ELU+1 learned multi-head projections, causal prefix aggregation, reduced FFN, residuals, LayerNorm, dropout and shifted sigmoid response objective are unchanged. The source-tree training fingerprint remains 0a04dc026130226cf58cd1fad8e9f9a696d91088097b2190cdf9e1bcd1fa48f6. Student-disjoint split hashes, seeds, Adam settings, epochs/patience and validation-only frozen selection are preserved. Historical simplified GKT limitations remain disclosed.

The proposal removes only 162 supplementary GKT tuning candidates: PAM learning rates [.0005,.001,.005] × widths [16,30,64]; MHA the same rates × widths [16,32,64], heads=2. Completed/running evidence is always kept. The completed MHA primary width 30/head 2 would be an additional validation candidate. Every other baseline retains 36 candidates, all LightSAKT grid rows and all matched seeds/ablations remain. Table 7 and significance text would disclose smaller GKT tuning budgets. This changes the established comparison protocol and requires author approval. The full run-ID difference is PROPOSED_DIFFERENCES.csv.

## Progressive artifacts and recovery

CPU-only progressive work already generated primary performance figures, parameter counts, three real causal attention maps and ASSIST2009 skill-probability case plots with PNG/SVG/PDF and source arrays. Their labels include primary/final phase; primary figures are not a final repeated-seed comparison. Final frozen seed-42 checkpoints trigger new correctly attributed plots. Paired significance awaits all matched final seeds; it cannot be invented from engineering probes.

Completed checkpoints/results are content-verified and independently downloaded/checksum-verified from private draft GitHub release assets. An additional CPU-only active recovery worker snapshots consistent best/latest checkpoints, optimizer, RNG, histories and config after 15 minutes of recorded run time, then hourly per long-running run. Snapshots upload to separate daily private draft containers and are downloaded for SHA verification. Its end-to-end workflow was verified on a completed Algebra long-run checkpoint. Running latest states are saved on NFS every epoch. Git bundles are locally verified and pushed source history is independently recoverable; cloud storage preserves evidence, not active GPU processes.

The environment is a Kubernetes workload with a 15-core CPU quota, roughly 90 GiB RAM, and H100 MIG 3g.40gb. No OOM events or shutdown deadline were observed, but lifetime/deprovision retention is unverified. Detached workers survive terminal disconnect while the container stays up. No automatic restart after environment shutdown is configured. Restore artifacts then run the following commands inside the existing project; never start a second scheduler while its lock is held.

```bash
cd /workspace/LightSAKT2
python -m scripts.progress
python -m scripts.schedule --retry-failed
# In separate detached CPU-only workers after checking their locks/PIDs:
CUDA_VISIBLE_DEVICES= python -m scripts.active_recovery
CUDA_VISIBLE_DEVICES= python -m scripts.progressive --watch
python -m scripts.preserve --watch
python -m scripts.offsite --watch
python -m scripts.eta --watch
```

CPU/CUDA gate: reports/test_runs/20261014T004136-execution-gate.log. Current CPU-only regression results: reports/latest_cpu_test_run.json. Checkpoints: reports/completed_checkpoint_loads.json and active_checkpoint_verification.json. Detailed timing CSVs: reports/profiling/MEASURED_STEP_TIMINGS.csv, CONCURRENCY_EQUIVALENCE.csv, ETA_MEASURED_TIMINGS.csv and ETA_PHASE_MODEL_DATASET.csv.
