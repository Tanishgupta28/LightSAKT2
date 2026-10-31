# LightSAKT + Question Difficulty: verified professor report

31 October 2026. **40 measured model-seed results; five ASSIST2015 difficulty
slots unavailable under the current protocol. No new GPU experiment executed.**

## Findings and suitability to share

The existing ASSIST2009 seed-42 validation change from 0.750803537 to 0.781189972
is supported by the preserved checkpoints and replay evidence. It is a validation
result, not a test score. Across five seeds on each completed dataset, the
modified model improves both validation and held-out test AUC in every matched
seed versus both SAKT and LightSAKT. Mean test accuracy and F1 also improve.

**These results are sufficient to share as a verified research progress report.**
They do not establish publication readiness, complete three-dataset improvement,
attention-kernel superiority, or performance in chronological deployment. The
missing dataset experiment and feature-matched control remain material gaps.

## Complete comparison status

| Dataset | Model | Validation AUC | Test AUC | Test accuracy | Test F1 | Seeds |
|---|---|---|---|---|---|---|
| ASSIST2009 | SAKT | 0.748284 ± 0.000767 | 0.750322 ± 0.000648 | 0.732718 ± 0.000669 | 0.817401 ± 0.000943 | 5/5 |
| ASSIST2009 | LightSAKT | 0.749454 ± 0.001570 | 0.751056 ± 0.000766 | 0.732229 ± 0.000801 | 0.816533 ± 0.001517 | 5/5 |
| ASSIST2009 | LightSAKT+Difficulty | 0.780248 ± 0.001153 | 0.779359 ± 0.000778 | 0.744460 ± 0.000439 | 0.822429 ± 0.000649 | 5/5 |
| ASSIST2015 | SAKT | 0.728970 ± 0.000435 | 0.725180 ± 0.000258 | 0.749418 ± 0.000386 | 0.846664 ± 0.000609 | 5/5 |
| ASSIST2015 | LightSAKT | 0.729922 ± 0.000819 | 0.726088 ± 0.000315 | 0.749526 ± 0.000419 | 0.846282 ± 0.000392 | 5/5 |
| ASSIST2015 | LightSAKT+Difficulty | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | 0/5 |
| Algebra2005 | SAKT | 0.809999 ± 0.000725 | 0.813373 ± 0.000367 | 0.796382 ± 0.000504 | 0.871012 ± 0.000372 | 5/5 |
| Algebra2005 | LightSAKT | 0.807548 ± 0.000766 | 0.810590 ± 0.000422 | 0.795574 ± 0.000639 | 0.870356 ± 0.000350 | 5/5 |
| Algebra2005 | LightSAKT+Difficulty | 0.835215 ± 0.000498 | 0.836871 ± 0.000217 | 0.807301 ± 0.000642 | 0.876874 ± 0.000771 | 5/5 |

All displayed uncertainties are sample SD across training seeds
42, 123, 2026, 3407 and 7777. Accuracy/F1 here are TEST metrics. Validation and
test AUC are separate columns. Exact per-seed validation/test accuracy, F1,
precision, recall and BCE are in individual_seed_results.csv/JSON; summary files
contain mean/SD for every metric. No missing measurement is manufactured.

## Paired improvements: modified LightSAKT minus named comparator

| Dataset | Comparator | Validation AUC gain | Test AUC gain | Positive validation / test seeds |
|---|---|---|---|---|
| ASSIST2009 | LightSAKT | +0.030794 ± 0.000537 | +0.028303 ± 0.000484 | 5/5 / 5/5 |
| ASSIST2009 | SAKT | +0.031964 ± 0.001220 | +0.029036 ± 0.000567 | 5/5 / 5/5 |
| ASSIST2015 | LightSAKT | UNAVAILABLE | UNAVAILABLE | — |
| ASSIST2015 | SAKT | UNAVAILABLE | UNAVAILABLE | — |
| Algebra2005 | LightSAKT | +0.027667 ± 0.000377 | +0.026281 ± 0.000458 | 5/5 / 5/5 |
| Algebra2005 | SAKT | +0.025217 ± 0.000731 | +0.023497 ± 0.000424 | 5/5 / 5/5 |

The SD above describes paired per-seed differences, not subtraction of group SDs.
Positive means higher AUC; these are absolute AUC units. Full definitions and the
native-validation versus CPU-test provenance are in METRIC_DEFINITIONS.md.

## Architecture and information supplied

The baseline LightSAKT architecture has one causal normalized ELU+1 linear
attention block, learned skill/response and position embeddings, residual layer
normalization, a d -> d/2 -> d ReLU feed-forward network and sigmoid readout.
SAKT retains its native causal softmax attention and d -> d -> d feed-forward
network. “Original” denotes the established corrected causal implementations
with original trainable architecture/parameter shapes. The preserved historical
original_snapshot had attention/masking/test-selection defects; its unverified
historical scores are excluded. This distinction must accompany dissemination.

Modified LightSAKT changes only its readout:

`p = sigmoid(original_logit + alpha * feature)`, with zero-initialized learned alpha.

The smoothed skill success rate is `(successes+1)/(count+2)`; the question rate is
`(question_successes + 10*skill_rate)/(question_count+10)`; the feature is their
logit difference. Positive values represent relatively easier questions. The
encoder, depth, attention mechanism and per-dataset LightSAKT settings are unchanged.
The one extra neural parameter is accompanied by fitted item statistics: the
training corpus contains 16,511 question aggregates on ASSIST2009 and 119,581
target-step aggregates on Algebra2005. These are extra information/estimation
capacity and are not counted as neural weights.

## Leakage, causal history and unavailable data

For a training row, the complete student's responses are subtracted from both
skill and question statistics, including its current and future responses.
Validation/test feature fitting uses training students' labels only. Query
metadata is transformed without current outcome labels. Unseen questions receive
zero correction through the smoothed skill-rate fallback. Fresh real-data
held-out-label and whole-own-student-label perturbations passed on both datasets,
including a responsive other-student positive control. All original feature
sidecars were reproduced; no feature, split or label was edited.

The encoder remains causal. Earlier observed held-out responses form history
for later predictions, as in the baseline, but future held-out responses and the
current target are excluded. All expected per-student n-1 targets are present
in order without padding, duplication or discarded difficult examples.

Statistics fit the COMPLETE training corpus. The earlier audit found later-order
training observations for 95.045% of ASSIST2009 validation attempts. Offline
student-disjoint fitting is the stated protocol; forward-time availability has
not been demonstrated. Algebra predictions condition on the known target
problem/step identity, consistent with the documented benchmark fields.

For ASSIST2015, the publisher's original file is byte-identical to the local
four-column release and raw backup. No independent question ID or authentic
log-to-question crosswalk was found. Sequence IDs are problem-set/skill-builder
proxies and log IDs identify events; neither is an item-ID substitute. All five
modified slots are unavailable for this protocol. See ASSIST2015_LIMITATION.md
for evidence, primary links, exact search bounds and requirements to unblock.

## Dataset and evaluation consistency

All comparisons use the existing fixed student-disjoint 70/10/20 partitions,
split seed42, training-only vocabulary and identical per-dataset preprocessing.
ASSIST2009/ASSIST2015 sequences have length100; Algebra length200. Chunking
preserves all n-1 targets with a one-interaction boundary overlap. The full
KC(Default) composite is one Algebra skill; its raw training subset is locally
student-split. Its test column is not the official challenge test. Dataset
counts and fingerprints are in unchanged evidence_20261030/dataset_metadata.json.

| Dataset | Model | Dimension | Heads | Dropout | Learning rate | Neural parameters |
|---|---|---:|---:|---:|---:|---:|
| ASSIST2009 | SAKT | 64 | 4 | 0.2 | 0.0005 | 57985 |
| ASSIST2009 | LightSAKT | 64 | 2 | 0.1 | 0.001 | 53857 |
| ASSIST2009 | LightSAKT+Difficulty | 64 | 2 | 0.1 | 0.001 | 53858 |
| ASSIST2015 | SAKT | 128 | 4 | 0.2 | 0.0005 | 151297 |
| ASSIST2015 | LightSAKT | 128 | 4 | 0.1 | 0.0005 | 134849 |
| Algebra2005 | SAKT | 128 | 2 | 0.2 | 0.0005 | 274305 |
| Algebra2005 | LightSAKT | 128 | 4 | 0.2 | 0.001 | 257857 |
| Algebra2005 | LightSAKT+Difficulty | 128 | 4 | 0.2 | 0.001 | 257858 |

All runs use batch64, float32, no AMP/TF32, Adam betas(0.9,0.999), epsilon1e-8,
weight decay0, maximum50epochs, patience5 and strict validation-AUC checkpoint
selection. SAKT/LightSAKT retained their separately selected settings from the
same36-point validation grid. Difficulty models inherit their native LightSAKT
settings, without retuning. Each modified configuration was compared with its
matched baseline configuration and all40 saved checkpoint configurations.

Baseline final configs enable test evaluation AFTER training/checkpoint selection;
modified training disables it and uses a separate CPU evaluation. The table
consistently uses the previously saved CPU test replay, batch8, globally pooled
masked targets and fixed0.5 threshold. Validation values are the native selected
epoch scores, reproduced within AUC/BCE2e-6 and accuracy/F1 tolerance1e-4 by CPU
replay. The flag difference changes evaluation scheduling, not optimization or
checkpoint selection. No test-based tuning or new inference was performed here.

## Verification and provenance

Fresh checks read all77 files in the previous report directory, verify all40
checkpoint hashes, finite weights, configurations, shapes, parameter counts and
validation-best epochs, and recalculate metrics from all80 saved validation/test
prediction arrays. Independent rank-sum AUC and manual confusion counts agree.
Ten difficulty optimizer/RNG/shuffle-state audits and338 protected-file hashes
pass. Every aggregate mean/sample SD and all paired differences match per-seed
evidence. Original reports, checkpoints, raw files and scientific sources remain
unchanged; verification_summary.json and original_report_fingerprints.json record
the boundaries. This is a fresh audit of the prior checkpoint replay, not a new
independent training campaign or an additional inference measurement.

Results are anchored at c286bcbc; the preceding handoff is49982fa7. Three local
incremental Git bundles verify, with their prerequisite baseline commits
preserved. The bundle chain is not an independent full-history repository.
No remote push was attempted or claimed; the recorded CLI authentication blocker
remains. HiTSKT and unrelated workloads were not modified.

## Individual seed results

| Dataset | Model | Seed | Validation AUC | Test AUC | Test accuracy | Test F1 |
|---|---|---:|---:|---:|---:|---:|
| ASSIST2009 | SAKT | 42 | 0.749442606 | 0.749198555 | 0.732607677 | 0.816955628 |
| ASSIST2009 | SAKT | 123 | 0.748615381 | 0.750691833 | 0.732415967 | 0.817147016 |
| ASSIST2009 | SAKT | 2026 | 0.747664391 | 0.750761905 | 0.733885741 | 0.818918410 |
| ASSIST2009 | SAKT | 3407 | 0.748107389 | 0.750347336 | 0.732479870 | 0.816418412 |
| ASSIST2009 | SAKT | 7777 | 0.747588929 | 0.750612312 | 0.732202957 | 0.817564430 |
| ASSIST2009 | LightSAKT | 42 | 0.750803537 | 0.750508348 | 0.730860989 | 0.814765947 |
| ASSIST2009 | LightSAKT | 123 | 0.749794733 | 0.751549123 | 0.732309462 | 0.817761278 |
| ASSIST2009 | LightSAKT | 2026 | 0.749631395 | 0.752093588 | 0.732586376 | 0.817087741 |
| ASSIST2009 | LightSAKT | 3407 | 0.746766658 | 0.750216352 | 0.732948494 | 0.817985162 |
| ASSIST2009 | LightSAKT | 7777 | 0.750272203 | 0.750912266 | 0.732437268 | 0.815064560 |
| ASSIST2009 | LightSAKT+Difficulty | 42 | 0.781189972 | 0.778555249 | 0.744046351 | 0.821588716 |
| ASSIST2009 | LightSAKT+Difficulty | 123 | 0.780341551 | 0.779128643 | 0.744003749 | 0.823155478 |
| ASSIST2009 | LightSAKT+Difficulty | 2026 | 0.780865054 | 0.780599489 | 0.744451071 | 0.822951255 |
| ASSIST2009 | LightSAKT+Difficulty | 3407 | 0.778264523 | 0.778969884 | 0.744962297 | 0.822007820 |
| ASSIST2009 | LightSAKT+Difficulty | 7777 | 0.780578811 | 0.779540275 | 0.744834491 | 0.822441266 |
| ASSIST2015 | SAKT | 42 | 0.729264682 | 0.725195505 | 0.749135624 | 0.846851311 |
| ASSIST2015 | SAKT | 123 | 0.728275257 | 0.725120267 | 0.749789543 | 0.847558993 |
| ASSIST2015 | SAKT | 2026 | 0.729054699 | 0.725520410 | 0.749864706 | 0.846464376 |
| ASSIST2015 | SAKT | 3407 | 0.728871781 | 0.724804604 | 0.749015363 | 0.846553990 |
| ASSIST2015 | SAKT | 7777 | 0.729382553 | 0.725258151 | 0.749285951 | 0.845893702 |
| ASSIST2015 | LightSAKT | 42 | 0.730065294 | 0.725976675 | 0.749669282 | 0.845896512 |
| ASSIST2015 | LightSAKT | 123 | 0.730525180 | 0.725897266 | 0.750045098 | 0.846839378 |
| ASSIST2015 | LightSAKT | 2026 | 0.730859149 | 0.726010353 | 0.749165690 | 0.846253075 |
| ASSIST2015 | LightSAKT | 3407 | 0.729101602 | 0.726645479 | 0.749721897 | 0.846480834 |
| ASSIST2015 | LightSAKT | 7777 | 0.729059200 | 0.725912004 | 0.749030396 | 0.845941606 |
| ASSIST2015 | LightSAKT+Difficulty | 42 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| ASSIST2015 | LightSAKT+Difficulty | 123 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| ASSIST2015 | LightSAKT+Difficulty | 2026 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| ASSIST2015 | LightSAKT+Difficulty | 3407 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| ASSIST2015 | LightSAKT+Difficulty | 7777 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| Algebra2005 | SAKT | 42 | 0.810922351 | 0.813542905 | 0.796708999 | 0.871056815 |
| Algebra2005 | SAKT | 123 | 0.810267779 | 0.813586642 | 0.795583269 | 0.870396289 |
| Algebra2005 | SAKT | 2026 | 0.809402593 | 0.813619139 | 0.796357208 | 0.871388977 |
| Algebra2005 | SAKT | 3407 | 0.809131769 | 0.812738091 | 0.796357208 | 0.871185781 |
| Algebra2005 | SAKT | 7777 | 0.810268412 | 0.813380470 | 0.796902484 | 0.871029896 |
| Algebra2005 | LightSAKT | 42 | 0.808126510 | 0.811009501 | 0.796242876 | 0.870200011 |
| Algebra2005 | LightSAKT | 123 | 0.807899180 | 0.810613347 | 0.795275452 | 0.870011950 |
| Algebra2005 | LightSAKT | 2026 | 0.808187028 | 0.810165996 | 0.794668613 | 0.870108989 |
| Algebra2005 | LightSAKT | 3407 | 0.806409391 | 0.811000490 | 0.796102160 | 0.870790838 |
| Algebra2005 | LightSAKT | 7777 | 0.807120089 | 0.810158174 | 0.795583269 | 0.870668885 |
| Algebra2005 | LightSAKT+Difficulty | 42 | 0.835287427 | 0.836898773 | 0.807975093 | 0.877310887 |
| Algebra2005 | LightSAKT+Difficulty | 123 | 0.835734921 | 0.837020733 | 0.806814184 | 0.876790702 |
| Algebra2005 | LightSAKT+Difficulty | 2026 | 0.835612277 | 0.836540567 | 0.806444804 | 0.877822930 |
| Algebra2005 | LightSAKT+Difficulty | 3407 | 0.834533667 | 0.836798930 | 0.807649687 | 0.876685141 |
| Algebra2005 | LightSAKT+Difficulty | 7777 | 0.834909079 | 0.837094660 | 0.807623303 | 0.875759676 |

## Limitations and next required comparison

Five initializations on one split per dataset do not establish population-level
uncertainty or cross-cohort replication. Earlier validation reuse introduces
selection exposure. These are local dataset variants and globally pooled
interaction metrics, so direct comparison with unrelated published protocols
would be inappropriate. This analysis is descriptive, with no post-hoc
significance or state-of-the-art claim. The released test split must not guide
further tuning.

The key missing control is **SAKT + the identical question-success correction**:
same item information, smoothing, whole-student exclusion, training-only held-out
fitting, scalar readout, partitions, five seeds, loss/masking/budget and validation
selection, while preserving SAKT's own native encoder and previously selected
settings. Compare its gain over SAKT with the difficulty gain over LightSAKT,
and compare both feature-matched models. That isolates input information from
encoder-family differences more clearly than the current richer-input comparison.
It is already prepared but unexecuted and requires a separately defined resource
budget. Do not repeat the completed LightSAKT seeds in the older control plan.

## Portable reporting package

The portable archive is `/workspace/LightSAKT2/backups/professor_release_20261031/LightSAKT_question_difficulty_verified_results_20261031.zip`. It includes this report and summary, all
CSV/JSON tables, copied original report evidence, scientific source snapshots,
40 frozen configs, provenance, checkpoint SHA256 references and a standalone
standard-library verifier. Manifest paths are relative and archive readback,
extraction, integrity and aggregation checks are performed. External local paths
are provenance locators, not portability dependencies for reading/verifying tables.

Raw datasets, per-student prediction/target arrays, model weights, credentials and
transient download URLs are excluded. Checkpoint references and data fingerprints
allow exact matching to the preserved local artifacts, but the reporting archive
is not a self-contained training or model-inference bundle. See README.md and
the external archive_verification.json for its exact integrity scope/checksum.
