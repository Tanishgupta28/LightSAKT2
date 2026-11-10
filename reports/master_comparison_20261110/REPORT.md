# LightSAKT master research comparison — 10 November 2026

**75 existing verified seed records reused; 15 of 27 requested model/dataset groups complete; 12 groups PENDING. No new GPU training was launched.**

This is a complete inventory CSV with explicit missing results, not a claim that all nine models have finished on all datasets. The 45 previously published SAKT/LightSAKT/difficulty measurements are preserved. Thirty previously completed DKT/DKT+/DKVMN seeds on ASSIST2009 and Algebra2005 are added through their existing independent CPU replay evidence, without retraining or new checkpoint prediction inference.

## Master results

| Dataset | Model | Seeds | Validation AUC | Test AUC | Test accuracy | Test F1 | Status |
|---|---|---:|---|---|---|---|---|
| ASSIST2009 | DKT | 5/5 | 0.757492 ± 0.000684 | 0.757165 ± 0.000527 | 0.736135 ± 0.001022 | 0.818332 ± 0.000993 | VERIFIED_5_SEEDS |
| ASSIST2009 | DKT+ | 5/5 | 0.754946 ± 0.002044 | 0.751202 ± 0.000876 | 0.733004 ± 0.001809 | 0.816546 ± 0.002508 | VERIFIED_5_SEEDS |
| ASSIST2009 | DKVMN | 5/5 | 0.753992 ± 0.001149 | 0.755016 ± 0.001395 | 0.736225 ± 0.001179 | 0.819802 ± 0.001518 | VERIFIED_5_SEEDS |
| ASSIST2009 | GKT | 0/5 | PENDING | PENDING | PENDING | PENDING | PENDING |
| ASSIST2009 | GKT-MHA (implemented variant) | 0/5 | PENDING | PENDING | PENDING | PENDING | PENDING |
| ASSIST2009 | SAKT | 5/5 | 0.748284 ± 0.000767 | 0.750322 ± 0.000648 | 0.732718 ± 0.000669 | 0.817401 ± 0.000943 | VERIFIED_5_SEEDS |
| ASSIST2009 | LightSAKT | 5/5 | 0.749454 ± 0.001570 | 0.751056 ± 0.000766 | 0.732229 ± 0.000801 | 0.816533 ± 0.001517 | VERIFIED_5_SEEDS |
| ASSIST2009 | LightSAKT + Question Difficulty | 5/5 | 0.780248 ± 0.001153 | 0.779359 ± 0.000778 | 0.744460 ± 0.000439 | 0.822429 ± 0.000649 | VERIFIED_5_SEEDS |
| ASSIST2009 | SAKT + Question Difficulty | 0/5 | PENDING | PENDING | PENDING | PENDING | PENDING |
| Algebra2005 | DKT | 5/5 | 0.820171 ± 0.000798 | 0.822181 ± 0.000683 | 0.800524 ± 0.001175 | 0.872422 ± 0.000891 | VERIFIED_5_SEEDS |
| Algebra2005 | DKT+ | 5/5 | 0.822312 ± 0.000462 | 0.823456 ± 0.000649 | 0.801094 ± 0.000253 | 0.872982 ± 0.000347 | VERIFIED_5_SEEDS |
| Algebra2005 | DKVMN | 5/5 | 0.814134 ± 0.001502 | 0.817725 ± 0.001099 | 0.798950 ± 0.000818 | 0.872047 ± 0.000572 | VERIFIED_5_SEEDS |
| Algebra2005 | GKT | 0/5 | PENDING | PENDING | PENDING | PENDING | PENDING |
| Algebra2005 | GKT-MHA (implemented variant) | 0/5 | PENDING | PENDING | PENDING | PENDING | PENDING |
| Algebra2005 | SAKT | 5/5 | 0.809999 ± 0.000725 | 0.813373 ± 0.000367 | 0.796382 ± 0.000504 | 0.871012 ± 0.000372 | VERIFIED_5_SEEDS |
| Algebra2005 | LightSAKT | 5/5 | 0.807548 ± 0.000766 | 0.810590 ± 0.000422 | 0.795574 ± 0.000639 | 0.870356 ± 0.000350 | VERIFIED_5_SEEDS |
| Algebra2005 | LightSAKT + Question Difficulty | 5/5 | 0.835215 ± 0.000498 | 0.836871 ± 0.000217 | 0.807301 ± 0.000642 | 0.876874 ± 0.000771 | VERIFIED_5_SEEDS |
| Algebra2005 | SAKT + Question Difficulty | 0/5 | PENDING | PENDING | PENDING | PENDING | PENDING |
| EdNetKT1_5k_first | DKT | 0/5 | PENDING | PENDING | PENDING | PENDING | PENDING |
| EdNetKT1_5k_first | DKT+ | 0/5 | PENDING | PENDING | PENDING | PENDING | PENDING |
| EdNetKT1_5k_first | DKVMN | 0/5 | PENDING | PENDING | PENDING | PENDING | PENDING |
| EdNetKT1_5k_first | GKT | 0/5 | PENDING | PENDING | PENDING | PENDING | PENDING |
| EdNetKT1_5k_first | GKT-MHA (implemented variant) | 0/5 | PENDING | PENDING | PENDING | PENDING | PENDING |
| EdNetKT1_5k_first | SAKT | 5/5 | 0.665022 ± 0.001326 | 0.636359 ± 0.001981 | 0.662697 ± 0.001453 | 0.777653 ± 0.002664 | VERIFIED_5_SEEDS |
| EdNetKT1_5k_first | LightSAKT | 5/5 | 0.670306 ± 0.001088 | 0.642417 ± 0.000980 | 0.664176 ± 0.000810 | 0.778199 ± 0.001393 | VERIFIED_5_SEEDS |
| EdNetKT1_5k_first | LightSAKT + Question Difficulty | 5/5 | 0.729147 ± 0.000894 | 0.706876 ± 0.000691 | 0.693068 ± 0.000701 | 0.790115 ± 0.001670 | VERIFIED_5_SEEDS |
| EdNetKT1_5k_first | SAKT + Question Difficulty | 0/5 | PENDING | PENDING | PENDING | PENDING | PENDING |

Means and sample SD use exactly seeds42,123,2026,3407,7777. Validation AUC is the native validation-selected score; test AUC/accuracy/F1 use existing verified CPU replay. Classification threshold0.5 and globally pooled masked targets are unchanged. Numeric aggregate cells are empty for PENDING groups; their display fields say PENDING. No single-seed value is presented as a five-seed mean.

## Verification and comparable populations

All75 best checkpoints were SHA256-checked, loaded strictly on CPU, checked for finite weights, correct parameter counts, scientific configurations, processed/split hashes and maximum-validation-AUC epochs. Some original seed42 baselines reuse equivalent selected-tuning checkpoints with documented run-name aliases. This does not create additional independent seeds. The existing independent CPU inference cache contains repeated audit entries; it is de-duplicated by run/seed and matched against exact configuration/checkpoint/prediction/metrics hashes, then the latest matching verification timestamp is used, not the best test score.

All120 available saved prediction arrays (90 previously published validation/test arrays plus30 additional baseline test arrays) match the expected original per-student n−1 targets exactly. Independent rank-sum AUC/manual confusion counts/BCE match their saved-array reference metrics within1e−12. Old baseline caches already contain independent CPU validation/test inference; these values are reused. Native GPU test values and CPU replay values can differ in final floating-point digits, are checked within2e−6 and remain separately traceable. Existing historical reports are not rewritten. The previously published45 canonical measurements remain identical in this export.

All models share per-dataset data bytes, fixed student-disjoint70/10/20 partitions (split seed42), training-only vocabulary/UNK and target masks. ASSIST2009 retains274590interactions/4151students, length100. Algebra2005 retains607025interactions/574students, length200; its holdout is a local split of the original training subset, not the official challenge test. Item identity for the correction is authentic problem_id on ASSIST2009 and the lossless (Problem Name,Step Name) tutor-step pair on Algebra.

EdNet is EXACTLY EdNetKT1_5k_first:5000authentic users uniformly selected before labels/length/scores with seed20261104;563469interactions,11893authentic question IDs;student splits3500/500/1000;target counts403742/42875/111852. The first presented question per solving session is selected before answer validation; other current-session answers never enter history. Correctness comes from recorded answers and official answer keys; tags are full sorted composite categories, without interaction explosion. Length100 and the frozen1452-skill input includingUNK remain unchanged. Dataset/split hashes are in every master row. No full-KT1 leaderboard claim.

## Model-specific settings and budgets

All completed models use Adam,betas(.9,.999),epsilon1e−8,weight decay0,float32,noAMP/TF32,max50epochs,patience5,strict validation-AUC checkpoint selection. Actual epochs/runtime, learning rates, dimensions, regularization coefficients and config references are in model_configurations.csv and master training_budget JSON cells. Historical DKT/DKT+/DKVMN/SAKT/LightSAKT configurations came from their existing validation-only searches; new EdNet baseline templates have not undergone an equivalent search. There is no claim of equal hyperparameter search across datasets.

Batch64 applies to the completed five-seed groups. Existing GKT implementations use the documented batch8 override; do not silently change it. DKT/DKT+ retain native output-logit dropout0.5. GKT MLP dropout is0.5 and its MHA module default attention dropout0. All defaults are preserved rather than retuned.

Equal epoch ceilings do not imply equal optimizer-update counts: batch8 GKT makes more updates than batch64 models. Runtime fields are provenance gathered under different device/pacing policies; they are not a matched efficiency benchmark and must not be ranked as such.

“GKT” here is the repository's simplified PAM implementation. **GKT-MHA (implemented variant)** is the repository's MHA subclass, not an established canonical reproduction of the published GKT architecture. The source expressly omits erase-add gating and MLP batch normalization. Both labels must retain this qualification. Core original LightSAKT/SAKT names refer to the established corrected causal implementations; flawed historical snapshots are excluded.

## Missing experiments and reuse opportunities

| Dataset | Model | Missing verified seed results | Selected training checkpoints reusable | Preparation |
|---|---|---:|---:|---|
| ASSIST2009 | GKT | 5 | 1 | Reuse highest-validation checkpoint from complete original 36-point grid for seed42; freeze same settings for other four seeds |
| ASSIST2009 | GKT-MHA (implemented variant) | 5 | 1 | Reuse highest-validation checkpoint from complete original 36-point grid for seed42; freeze same settings for other four seeds |
| ASSIST2009 | SAKT + Question Difficulty | 5 | 0 | Exact native SAKT selected/frozen configuration plus identical existing one-scalar question feature |
| Algebra2005 | GKT | 5 | 0 | Algebra2005 GKT tuning grid not complete; primary seed42 cannot substitute for a selected repeated-seed protocol |
| Algebra2005 | GKT-MHA (implemented variant) | 5 | 0 | Algebra2005 GKT tuning grid not complete; primary seed42 cannot substitute for a selected repeated-seed protocol |
| Algebra2005 | SAKT + Question Difficulty | 5 | 0 | Exact native SAKT selected/frozen configuration plus identical existing one-scalar question feature |
| EdNetKT1_5k_first | DKT | 5 | 0 | Prespecified transfer of existing ASSIST2009 selected model settings, without EdNet test-guided tuning |
| EdNetKT1_5k_first | DKT+ | 5 | 0 | Prespecified transfer of existing ASSIST2009 selected model settings, without EdNet test-guided tuning |
| EdNetKT1_5k_first | DKVMN | 5 | 0 | Prespecified transfer of existing ASSIST2009 selected model settings, without EdNet test-guided tuning |
| EdNetKT1_5k_first | GKT | 5 | 0 | Untuned original primary template; graph memory/runtime profiling required on 1452 skills |
| EdNetKT1_5k_first | GKT-MHA (implemented variant) | 5 | 0 | Untuned original primary template; graph memory/runtime profiling required on 1452 skills |
| EdNetKT1_5k_first | SAKT + Question Difficulty | 5 | 0 | Exact native SAKT selected/frozen configuration plus identical existing one-scalar question feature |

There are60missing verified seed-result slots. Two ASSIST2009 GKT training checkpoints are already reusable: each is selected by maximum validation AUC from its complete original36-candidate seed42 grid, without test access. Their best validation AUCs are0.764002738 (PAM) and0.767020187 (MHA); these are training-reuse/validation evidence, **not five-seed test results**. Four additional training seeds and final CPU checkpoint evaluation are needed per group. Do not retrain their selected seed42 runs.

Thus at least58additional selected-configuration training runs remain if those two checkpoints are reused. This excludes extra configuration-selection trials: Algebra2005 GKT tuning/selection is incomplete, so ten final-run slots await validation-only configuration selection. Its completed seed42 primary runs use different fixed settings and cannot silently substitute for selected-final runs. Four historical single-seed GKT primary results are preserved separately in excluded_single_seed_primary_evidence.csv, never in the master aggregate cells. Seventy-two completed ASSIST2009 GKT tuning trials are not72independent seeds.

The EdNet gaps are DKT,DKT+,DKVMN,GKT andGKT-MHA:25seed runs. The SAKT+Difficulty control is absent on allthree datasets:15runs. Every control uses the exact native SAKT configuration and same item-feature sidecar as the corresponding completed LightSAKT+Difficulty protocol. Existing SAKTQuestion class is unchanged; the generic frozen-sidecar adapter has CPU tests for allthree datasets.

Fifty reviewed prepared configuration files cover EdNet's25slots, all15controls and the ten ASSIST2009 GKT final slots (including two checkpoint-reuse entries). Algebra's ten GKT slots are explicitly awaiting configuration selection. Preparations use isolated ignored workspaces with checksum-bound symlinks to original data/splits, not copies or new preprocessing. EdNet DKT/DKT+/DKVMN templates transfer the existing ASSIST2009-selected settings without using EdNet test outcomes. EdNet GKT templates retain original primary settings and require memory/runtime feasibility profiling on1452skills. Prepared configs are NOT an execution authorization; every training config disables test evaluation.

## GPU execution blocker

The old exact MIG device is visible and the read-only telemetry check has technical free-memory/host headroom. That does not establish current ownership or a new allocation/budget. The previous EdNet scope finished15/15 invocations; unused wall-time ceiling does not authorize more runs. No new current allocation confirmation or total execution budget has been verified for this expanded campaign. The user was asked for those missing resource details while CPU work continued. GPU_READINESS.json records the observation and distinction. No unrelated process was signalled, and no HiTSKT files/jobs were accessed.

Retain4GiB allocator/900s windows and previous shared safety floors, pacing, step bounds and admission locks if a new scope is granted. GKT on1452skills may require resource profiling and multiple explicitly budgeted exact-resume windows; completion within one900s window is not assumed. Do not bypass holds or launch an old scheduler. No new training or actual-data model prediction inference occurred in this task; CPU tests use tiny synthetic batches to verify integration/causality/padding/backpropagation.

## Findings and methodological limitations

Original LightSAKT does not lead every available baseline: DKT has higher mean test AUC on ASSIST2009 and DKT+ on Algebra2005. Adding the old missing-family measurements therefore changes the breadth of the comparison, not the historical model scores. The item-augmented LightSAKT has higher mean test AUC than the currently complete comparators, but this is an input-information comparison, not attention-architecture superiority. No conclusion can be drawn against PENDING models.

Matched-seed negative/inconsistent classification findings against newly included comparators:

No negative AUC/accuracy/F1 matched-seed differences against the available complete comparators were found.

Full paired AUC/accuracy/F1 differences and positive/negative seed counts are in paired_test_differences.csv and paired_test_difference_summary.csv. AUC gains and F1 gains are distinct; a positive mean does not imply every seed improved on every metric. No post-hoc significance or general superiority claim is made.

The difficulty readout still adds one scalar times logit(smoothed item success)−logit(smoothed skill success), with item smoothing10, Beta(1,1) skill prior, whole-own-student exclusion in both training aggregates and train-only held-out fitting; unseen items yield0. It supplies item information unavailable to skill-only baselines. No held-out-label/self-label leakage was identified under the documented OFFLINE student-disjoint protocol. Full training-corpus fitting contains later observations of other training students; EdNet's temporal audit flags99.6012%validation and99.5887%test targets. This does not establish chronological deployment performance.

Five initializations use one split per dataset; SD is optimization variability rather than independent cohort uncertainty. EdNet is largely known-item evaluation; long histories are interaction-weighted (largest validation student19.11%of targets). Its first/session subset and composite-tag skills differ from fullEdNet. Historical validation reuse and unequal hyperparameter-search exposure remain. The feature-matched SAKT+Difficulty control, independent-cohort/split replication and chronological evaluation remain necessary. Do not tune against released test outcomes or alter native models to obtain higher results. ASSIST2015 is excluded from this replacement-dataset master and retained historically.

## Files and reproduction

- master_comparison.csv/JSON:27dataset/model rows; separate numeric mean/std for each metric; generic mean/standard_deviation aliases refer ONLY to primary_metric=test_auc.
- individual_seed_results.csv:135prespecified slots,75verified and60PENDING; no missing measurements imputed.
- missing_experiments.csv and experiments/master_comparison_20261110/plan.json:60slots, reuse decisions,50prepared configs and configuration/resource blockers.
- model_configurations.csv, configurations/, experiment_provenance.json: exact settings, runtime/epoch budgets, checkpoints, SHA256 and original verification sources.
- selected_reusable_gkt_checkpoints.json and excluded_single_seed_primary_evidence.csv: explicit training-only reuse versus excluded single-seed evidence.
- verification_summary.json, export_verification.json, cpu_tests.xml and GPU_READINESS.json: audit scope and resource limits.

From the research root, CPU-only regeneration is `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m scripts.build_master_comparison_20261110`, followed by `python -B -m scripts.report_master_comparison_20261110` and `python -B scripts/verify_master_comparison_20261110.py`. The builder loads existing checkpoints and recomputes metrics from saved arrays; it never calls a trainer or executes model prediction forwards. Preserve all source artifacts. New raw data, predictions and weights are not committed; checkpoint paths are local provenance locators. Publication should use only the research branch, fresh ancestry checks, dry-run then normal push; never main or force.
