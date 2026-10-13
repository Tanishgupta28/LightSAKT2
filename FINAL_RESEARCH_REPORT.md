# LightSAKT2 research rebuild — verified progress report

**The complete research rebuild is not yet finished.** This report distinguishes executed work from queued work. Historical manuscript values are not used as new results.

## Completed implementation and verification

* Verified root `/workspace/LightSAKT2`; all development and new artifacts are inside it.
* Preserved 1,613 inventoried original files with matching byte sizes, SHA256 manifests, and a readable archival copy. Original Drive files were never modified. Read-only local artifact copies have explicitly limited provenance.
* Documented and protected the original architecture; exact original parameter names/shapes and counts are tested. No replacement architecture or new LightSAKT component was added.
* Corrected future-dependent global KV, incorrect normalization, key/value padding, positional slicing, next-response alignment and prediction-changing softmax visualization.
* Rebuilt student-disjoint 70/10/20 splits before chunking, train-only vocabularies, binary-label rules, simultaneous-tag collapse and response-event chronology. Raw, processed and split hashes are saved.
* Corrected baseline masking, shape handling, temporal-memory updates, uninitialized GKT parameters, and DKT+ two-endpoint penalty masks. Historical simplified GKT PAM/MHA remain explicitly distinct from canonical published GKT.
* Built Adam training, validation-AUC early stopping, best/latest checkpoints, atomic writes, gradient/NaN checks, full RNG recovery and code-fingerprint resume protection.
* Implemented 108 LightSAKT grid configurations and equal 36-configuration tuning budgets for every baseline variant; frozen validation selections drive matched repeated seeds.
* Implemented four distinct component ablations. User-approved no_regularization aliases no_dropout and reuses its evidence; default reuses final LightSAKT results.
* Implemented programmatic tables, paired Wilcoxon/Holm analysis, controlled efficiency benchmarks, true kernel heatmaps, and heldout skill-probability case studies. Implementation does not mean those dependent analyses have executed.

## Executed tests

```
Running 67 items in this shard
...................................................................                                                                                  [100%]
67 passed in 13.83s
```

Additional CPU verification during active GPU training:

```
Running 87 items in this shard
.....................................sssssssssssssss...................................                                                              [100%]
72 passed, 15 skipped in 12.99s
```

CUDA tests were skipped in this additional invocation to avoid GPU contention; the preceding full CPU/CUDA correctness gate remains the GPU evidence. Model/training source fingerprints are unchanged.

Tests cover causality, the original leakage regression, dense float64 kernel/gradient equivalence, padding, all baseline forwards/backwards, batch independence, CPU/CUDA agreement, CUDA repeatability, data coverage/chronology, validation-only selection, heldout-label update independence, no evaluation updates, exact interruption/resume, checkpoints, undefined AUC, tuning budgets and regularization alias semantics.

## Genuine completed primary training results

|Dataset|Model|Seed|Epochs trained|Selected epoch|Test AUC|Accuracy|F1|MSE|BCE|
|---|---|---|---|---|---|---|---|---|---|
|ASSIST2009|DKT|42|14|9|0.755294|0.736889|0.822058|0.189177|0.563864|
|ASSIST2009|DKT+|42|21|16|0.753788|0.734248|0.816238|0.193745|0.574353|
|ASSIST2009|DKVMN|42|49|44|0.753237|0.735931|0.820487|0.178819|0.535500|
|ASSIST2009|GKT|42|13|8|0.757626|0.734993|0.820675|0.178737|0.535156|
|ASSIST2009|GKT_MHA|42|21|16|0.762630|0.739190|0.818543|0.176886|0.530880|
|ASSIST2009|LightSAKT|42|11|6|0.749537|0.731947|0.814641|0.180871|0.540315|
|ASSIST2009|SAKT|42|21|16|0.750656|0.733438|0.818968|0.180896|0.540641|
|ASSIST2015|DKT|42|15|10|0.726490|0.749211|0.845112|0.181661|0.546080|
|ASSIST2015|DKT+|42|18|13|0.720112|0.745265|0.842337|0.187538|0.559814|
|ASSIST2015|DKVMN|42|36|31|0.726157|0.750421|0.845956|0.171135|0.515765|
|ASSIST2015|GKT|42|14|9|0.729880|0.750481|0.845187|0.170689|0.514919|
|ASSIST2015|GKT_MHA|42|23|18|0.732967|0.752894|0.847879|0.169458|0.511468|
|ASSIST2015|LightSAKT|42|12|7|0.723090|0.749128|0.846615|0.171970|0.518306|
|ASSIST2015|SAKT|42|15|10|0.724443|0.749143|0.847193|0.172060|0.518539|
|Algebra2005|DKT|42|27|22|0.822137|0.800368|0.872351|0.149487|0.466205|
|Algebra2005|DKT+|42|33|28|0.823264|0.800227|0.872682|0.153568|0.478078|
|Algebra2005|DKVMN|42|50|50|0.809766|0.795944|0.869996|0.143876|0.444578|
|Algebra2005|GKT|42|16|11|0.821627|0.802637|0.874831|0.140674|0.436897|
|Algebra2005|SAKT|42|31|26|0.812828|0.796911|0.870883|0.143335|0.443372|

Completed smoke runs: 1. Smoke validation is not a final research comparison.

## Experiment schedule status

{
  "completed": 19,
  "running": 1,
  "pending": 937
}

Total requested schedule slots: 957. Counts include 21 seed-42 checkpoint reuse slots, 15 regularization aliases, and 9 verified primary/grid reuse slots, which are not new independent trainings. Distinct planned trainings: 912. Every required grid row is retained. The five-seed primary result is the mean, never the best seed. All remaining pending slots have no experimental result yet.

## Findings and limitations

The original future-response leakage and non-normalized attention are reproduced by executed tests. The corrected default retains the original contribution but historical scores are no longer valid evidence. Completed primary results are new fixed-configuration single-seed measurements; a superiority claim requires the unfinished matched final comparisons. No fresh statistical significance or measured efficiency conclusion is established until the corresponding artifact files exist. O(L) sequence-length complexity alone does not establish lower latency or lower measured memory.

Raw data came from available local original/reference materials because the shared folder lacked required raw CSV/TSV files. Exact hashes and publisher references are documented, but publisher byte identity and an explicit redistribution license are not established. Algebra uses the labelled training subset, not complete official master coverage. GKT is the historical simplified implementation; canonical published GKT reproducibility remains a limitation. All 27 manuscript pages were rendered and visually inspected, including Figures 1–11; inconsistencies are documented in PAPER_CORRECTIONS.md. No editable manuscript source was present.

The no_regularization row is scientifically identical to no_dropout by user instruction and cannot be treated as another independent ablation. Five paired seeds yield weak statistical power, with minimum exact two-sided Wilcoxon p=.0625 for five nonzero pairs before correction; seed variation on one fixed split does not estimate new-student sampling uncertainty.

Measured runtimes, phase/model/dataset extrapolations, uncertainty scenarios and conditional completion dates are in reports/ETA_REPORT.md and reports/ETA.json. The schedule audit and before/after effective manifests are in reports/SCHEDULE_AUDIT.md. Live health samples and recovery checks are in reports/EXECUTION_HEALTH.md. GPU utilization is unavailable on the current MIG partition.

## Reproduce and continue

```
cd /workspace/LightSAKT2
python -m pytest -q
python -m scripts.schedule
python -m scripts.summarize
python -m scripts.analyze
python -m scripts.verify_artifacts
```

Do not start another scheduler while `experiments/scheduler.lock` is held. Inspect `reports/scheduler.log`, `reports/run_logs/`, and per-run status/config JSON files. For failed runs, inspect the failure then use `python -m scripts.schedule --retry-failed`. Each run status includes its resume command. Configurations and completed outputs are immutable; no test-set tuning is allowed.

The detached scheduler process can survive a terminal disconnect; it cannot automatically restart after a machine shutdown. The checkpoint state is designed for manual restart from durable storage. Do not assume future automatic completion after environment loss.

## Storage and synchronization

Private repository: https://github.com/Tanishgupta28/LightSAKT2, branch `research-rebuild`. No force pushes. Local Git bundles and the original/raw archive are under `backups/`; their references are under `reports/`. Git synchronization details are in GITHUB_SYNC_PENDING.md. Large raw files and checkpoints remain on the durable /workspace network mount, excluded from ordinary Git. These local archives are not independent offsite artifact copies.

Originals: `original_snapshot/`; raw datasets: `data/raw/`; fresh preprocessing: `data/processed/`; student manifests: `data/splits/`; fresh weights: `checkpoints/<run>/`; raw metrics/configs: `results/raw/<run>/`; predictions: `results/predictions/`; generated tables/figures: `results/tables/`, `results/figures/`; paper evidence: `paper/`, PAPER_CORRECTIONS.md.

Independent private GitHub draft recovery artifacts: 20 assets have been downloaded and SHA256 verified. All available completed archives synchronized: True. See reports/OFFSITE_RECOVERY.md for recovery commands and reports/offsite_backup.json for immutable asset IDs/checksums. Running checkpoints are saved on NFS until their completed-run archive is uploaded. Cloud artifacts preserve evidence after workspace loss; they do not keep GPU training alive after shutdown.
