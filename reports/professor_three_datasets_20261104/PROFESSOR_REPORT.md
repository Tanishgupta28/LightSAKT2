# LightSAKT + Question Difficulty: three-dataset research results

4 November 2026. 45/45 core model-seed measurements verified; 15/15 replacement experiments completed.

## Objective and measured findings

Evaluate whether the existing question-success correction improves LightSAKT on three student-disjoint datasets while retaining its causal linear attention. Original ASSIST2009 and Algebra2005 measurements are preserved and reused. ASSIST2015 is replaced with an explicitly reduced EdNet-KT1 protocol approved in principle by the professor. No HiTSKT files or jobs were accessed or changed.

| Dataset | Model | Validation AUC | Test AUC | Test accuracy | Test F1 | Seeds |
|---|---|---|---|---|---|---|
| ASSIST2009 | SAKT | 0.748284 ± 0.000767 | 0.750322 ± 0.000648 | 0.732718 ± 0.000669 | 0.817401 ± 0.000943 | 5/5 |
| ASSIST2009 | LightSAKT | 0.749454 ± 0.001570 | 0.751056 ± 0.000766 | 0.732229 ± 0.000801 | 0.816533 ± 0.001517 | 5/5 |
| ASSIST2009 | LightSAKT+Difficulty | 0.780248 ± 0.001153 | 0.779359 ± 0.000778 | 0.744460 ± 0.000439 | 0.822429 ± 0.000649 | 5/5 |
| Algebra2005 | SAKT | 0.809999 ± 0.000725 | 0.813373 ± 0.000367 | 0.796382 ± 0.000504 | 0.871012 ± 0.000372 | 5/5 |
| Algebra2005 | LightSAKT | 0.807548 ± 0.000766 | 0.810590 ± 0.000422 | 0.795574 ± 0.000639 | 0.870356 ± 0.000350 | 5/5 |
| Algebra2005 | LightSAKT+Difficulty | 0.835215 ± 0.000498 | 0.836871 ± 0.000217 | 0.807301 ± 0.000642 | 0.876874 ± 0.000771 | 5/5 |
| EdNet-KT1 (5k students; first/session) | SAKT | 0.665022 ± 0.001326 | 0.636359 ± 0.001981 | 0.662697 ± 0.001453 | 0.777653 ± 0.002664 | 5/5 |
| EdNet-KT1 (5k students; first/session) | LightSAKT | 0.670306 ± 0.001088 | 0.642417 ± 0.000980 | 0.664176 ± 0.000810 | 0.778199 ± 0.001393 | 5/5 |
| EdNet-KT1 (5k students; first/session) | LightSAKT+Difficulty | 0.729147 ± 0.000894 | 0.706876 ± 0.000691 | 0.693068 ± 0.000701 | 0.790115 ± 0.001670 | 5/5 |

Entries are mean ± **sample SD (ddof=1)** over seeds 42, 123, 2026, 3407 and 7777. Accuracy and F1 in this table are held-out TEST metrics; validation AUC is a separate selection metric. All metrics, including precision, recall, BCE, individual seeds and exact values, are in the paired CSV/JSON files. Incomplete measurements are PENDING; aggregates of partial groups must not be presented as five-seed results.

### Absolute AUC changes: modified LightSAKT minus comparator

| Dataset | Comparator | Validation AUC gain | Test AUC gain | Positive validation / test seeds |
|---|---|---|---|---|
| ASSIST2009 | SAKT | 0.031964 ± 0.001220 | 0.029036 ± 0.000567 | 5/5 / 5/5 |
| ASSIST2009 | LightSAKT | 0.030794 ± 0.000537 | 0.028303 ± 0.000484 | 5/5 / 5/5 |
| Algebra2005 | SAKT | 0.025217 ± 0.000731 | 0.023497 ± 0.000424 | 5/5 / 5/5 |
| Algebra2005 | LightSAKT | 0.027667 ± 0.000377 | 0.026281 ± 0.000458 | 5/5 / 5/5 |
| EdNet-KT1 (5k students; first/session) | SAKT | 0.064125 ± 0.000945 | 0.070516 ± 0.002336 | 5/5 / 5/5 |
| EdNet-KT1 (5k students; first/session) | LightSAKT | 0.058841 ± 0.000808 | 0.064458 ± 0.000648 | 5/5 / 5/5 |

ASSIST2009: modified minus SAKT mean test AUC = +0.029036; positive in 5/5 matched seeds.
ASSIST2009: modified minus LightSAKT mean test AUC = +0.028303; positive in 5/5 matched seeds.
Algebra2005: modified minus SAKT mean test AUC = +0.023497; positive in 5/5 matched seeds.
Algebra2005: modified minus LightSAKT mean test AUC = +0.026281; positive in 5/5 matched seeds.
EdNet-KT1 (5k students; first/session): modified minus SAKT mean test AUC = +0.070516; positive in 5/5 matched seeds.
EdNet-KT1 (5k students; first/session): modified minus LightSAKT mean test AUC = +0.064458; positive in 5/5 matched seeds.

The earlier ASSIST2009 seed-42 0.750803537 → 0.781189972 claim concerns validation AUC. It is reproduced by existing checkpoints and CPU replay, not confused with either five-seed mean or test AUC. Differences above are absolute AUC units; paired-difference SD is calculated from matched seeds rather than subtracting group SDs. No publication-readiness, statistical significance or attention-superiority claim follows from these descriptive comparisons.

## Verified architecture and modification

Established original LightSAKT has one normalized causal ELU+1 linear attention block, skill/response embeddings, learned positions, residual layer normalization and a d→d/2→d ReLU FFN. SAKT retains its native causal softmax and d→d→d FFN. “Original” means the previously corrected causal implementation; the preserved historical original_snapshot contained masking/attention/selection defects and its unverified results are excluded. No native source or architecture changed for this replacement.

Modified LightSAKT adds one zero-initialized learned scalar to the readout:
`p = sigmoid(original_logit + alpha * [logit(p_item) - logit(p_skill)])`.
`p_skill=(skill_successes+1)/(skill_count+2)`;
`p_item=(item_successes+10*p_skill)/(item_count+10)`.
Training rates exclude the ENTIRE current student's rows from both aggregates; validation and test use training students only. Unseen items yield exactly zero correction. The raw official correct answer and current user answer are used to construct labels only, never predictors. Alpha=0 preserves the baseline initialization and forward behavior. Attention, depth and native LightSAKT settings remain unchanged.

The fitted item rates are additional information and estimation capacity, even though the neural parameter count grows by only one. They condition on a known item ID that the skill-only baselines do not receive. Improvements do not establish linear-attention superiority.

## Dataset selection and ASSIST2015 replacement

EdNet-KT1 was selected over Junyi because genuine repeated `question_id` values, student sequences, official answer keys and tags were actually obtained from the [publisher's EdNet repository](https://github.com/riiid/ednet). The [EdNet paper](https://arxiv.org/abs/1912.03072) describes the source research dataset. Full archive byte sizes/hashes and actual schema checks appear in DATASET_AND_SPLITS.md. Source metadata has 13,169 unique questions. No synthetic item identifiers or skill IDs were substituted.

The original Junyi study distinguishes exercises containing multiple problems ([Chang, Hsu and Chen, 2015](https://ken77921.github.io/assets/pdf/proj8-education-student%20modeling-paper.pdf)). Individual repeated-question identity was not established from the examined exercise/problem-type metadata. Exercise-level difficulty would be a different labelled protocol. The older DataShop source was inaccessible (HTTP403), and no Junyi payload was needed once authentic EdNet was available. This is not a claim that every Junyi release lacks item IDs.

ASSIST2015's preserved official four-column release has user_id, log_id, sequence_id and correct. Its prior authoritative-source audit found no independent item crosswalk: event IDs and exercise/skill-builder sequences cannot be relabelled as questions. Its five difficulty slots remain historically unavailable, not completed. The old baseline/results/reports are preserved; see ../professor_release_20261031/ASSIST2015_LIMITATION.md. It is excluded from the final three-dataset comparison.

## Dataset protocol and statistics

EdNet is explicitly **5,000 uniformly selected authentic students, first question per solving session**, not the full KT1 benchmark. Selection uses sorted complete ZIP user filenames and NumPy default_rng seed20261104, before reading labels, lengths or model scores. The 5,000 selected ZIP members passed complete-file CRC checks. The full 784,309-user archive was downloaded but the unselected population was not exhaustively decompressed/CRC-tested. No publisher SHA was available; hashes are locally computed fingerprints of the publisher-linked payload.

| Count | EdNet subset |
|---|---:|
| Selected/retained students | 5,000 / 5,000 |
| Students with ≥2 retained observations | 4,816 |
| Raw selected interactions | 721,760 |
| Exact duplicate rows removed | 34 |
| Other questions in same solving session excluded | 158,161 |
| Invalid first-question answers excluded | 96 |
| Missing question metadata | 0 |
| Retained interactions | 563,469 |
| Authentic observed question IDs | 11,893 |
| Train / validation / test students | 3,500 / 500 / 1,000 |
| Train / validation / test interactions | 407,242 / 43,375 / 112,852 |
| Train / validation / test targets | 403,742 / 42,875 / 111,852 |
| Training tagset vocabulary + UNK | 1,451 + 1 |

Student IDs are genuine filename IDs. Chronology uses timestamp then original row order. Exact duplicates are removed; repeated questions in different sessions remain. First-in-session selection occurs before inspecting answer validity and invalid first answers are not replaced. Timestamp ranges must not overlap/tie between sessions; all selected students passed. This prevents other current-session responses from entering the history, but sacrifices bundle coverage and may bias toward first-question positions. It is a pre-registered operational definition, not the official full-question benchmark or proof of wall-clock deployment causality.

Answer correctness is equality between the recorded answer and official key; only a–d are valid (part2 a–c). Full sorted unique question tags form composite skills without row explosion. Vocabulary is fit only on training students; validation/test unseen tags map to UNK0 (1/4 rows respectively). Student split seed42 reuses the existing 70/10/20 function. Each student belongs to exactly one split. The 184 single-observation students remain in the population but have zero next-response targets.

ASSIST2009 retains 274,590 interactions and 2,905/415/831 students; Algebra2005 retains 607,025 interactions and 401/57/116 students. Algebra uses the local student split of the original training subset, not the challenge's official test; target item means lossless (Problem Name, Step Name). Full original counts, fingerprints and identifier semantics are preserved in dataset_metadata.json.

## Training, selection and evaluation

All replacement models use d64, 2 heads, dropout0.1, learning rate0.001, batch64, length100, Adam betas(0.9,0.999), epsilon1e-8, decay0, float32, no AMP/TF32, max50 epochs and patience5. The common compact setting was frozen before EdNet model scores; no new grid search occurred. Historical datasets retain their previously model-specific selected configurations (their original 36-point validation grid). Equal tuning search across all three datasets is not claimed. Per-run configs list exact dimensions, heads, dropout and neural parameters. Algebra sequence length remains200; ASSIST2009 remains100.

| Dataset | Model | Dimension | Heads | Dropout | Learning rate | Neural parameters |
|---|---|---:|---:|---:|---:|---:|
| ASSIST2009 | SAKT | 64 | 4 | 0.2 | 0.0005 | 57985 |
| ASSIST2009 | LightSAKT | 64 | 2 | 0.1 | 0.001 | 53857 |
| ASSIST2009 | LightSAKT+Difficulty | 64 | 2 | 0.1 | 0.001 | 53858 |
| Algebra2005 | SAKT | 128 | 2 | 0.2 | 0.0005 | 274305 |
| Algebra2005 | LightSAKT | 128 | 4 | 0.2 | 0.001 | 257857 |
| Algebra2005 | LightSAKT+Difficulty | 128 | 4 | 0.2 | 0.001 | 257858 |
| EdNet-KT1 (5k students; first/session) | SAKT | 64 | 2 | 0.1 | 0.001 | 310465 |
| EdNet-KT1 (5k students; first/session) | LightSAKT | 64 | 2 | 0.1 | 0.001 | 306337 |
| EdNet-KT1 (5k students; first/session) | LightSAKT+Difficulty | 64 | 2 | 0.1 | 0.001 | 306338 |

Native one-overlap chunks preserve every student's n−1 next-response targets, reset context at chunk boundaries and mask padding explicitly. All three models share processed data, student population, split IDs, masks, targets, seed list and budgets. Selection uses STRICT improvement in validation AUC only; ties do not replace the earliest best. Every EdNet config disables test evaluation during training. After completion the frozen selected checkpoint is restored on CPU (batch8) for validation and test, without changing configurations based on test results. Historical baseline flags allowed post-training test scoring; the reported TEST values consistently use their saved CPU replay. Validation values are native selected scores, independently reproduced within AUC/BCE2e-6 and classification1e-4.

Metrics pool all valid interactions globally, not equally weighted students or batch means. Accuracy/F1/precision/recall use threshold0.5 fixed before inference. BCE clips probabilities to [1e−7,1−1e−7]. Independent rank-sum AUC, manual confusion counts and BCE must agree within 1e−12; prediction target arrays must equal concatenated per-student labels[1:] exactly. Dataset/split hashes and the parameter/selection provenance are bound to each checkpoint.

## Leakage, restoration and resource checks

Real EdNet audits verify training-only vocabulary, authentic item/tag alignment, student disjointness, invariance of a student's features to flipping ALL its own labels, invariance to flipping held-out labels and a responsive other-training-student control. All frozen features were independently recomputed. Unseen-item row fractions are 0.000669 validation and 0.001648 test. Six CPU tests verify session preprocessing, invalid-answer policy, padding, causality, finite gradients and zero-alpha equivalence. The optional existing SAKT+Difficulty class also passes the generic EdNet adapter CPU check.

Every new completed latest checkpoint is checked for finite model/Adam state, exact configuration, contiguous epochs, best-validation selection, valid Python/NumPy/Torch/CUDA RNG and a loader RNG matching the exact seeded epoch shuffle stream. Adam step counts match completed batches. Best checkpoints are restored strictly, parameters counted and all saved prediction arrays hashed. Old checkpoint/prediction arrays and independent metrics are reused and rechecked without repeating their training/inference. Checkpoint hashes/configurations, audit outputs and provenance accompany this report; preservation checks cover 490 historical source/report artifacts plus holds and frozen new data/science.

The previously authorized shared MIG allocation was verified immediately before launch and guarded throughout; it was not claimed exclusive. The existing 4GiB allocator cap, 900s per invocation, free GPU/host memory floors, RSS bound, 0.1s batch pacing and 30s step bound remain. Core scope is 15 serial windows, at most13,500s, without automatic retries. Checkpoints are saved every completed epoch. No unrelated process is signalled or changed. Actual timings/peaks are in resource_summary.json; wall time includes deliberate pacing and is not exclusive GPU time or kernel profiling.

Actual core: 15/15 runs, 56.95 admitted window minutes, 0.251484GiB maximum Torch allocation, 298.61s longest native training invocation. Resource-floor/RSS breaches: 0.

As an explicit reproducibility check beyond the fifteen unique new checkpoint replays, seed42 was restored again separately for all three model families. All six validation/test prediction arrays reproduced bit for bit on CPU, without overwriting the saved arrays or executing additional GPU inference. See fresh_prediction_reproduction.json.

## Scientific limitations and next comparison

This is **offline student-disjoint** fitting. Item/skill rates use the complete permitted training corpus, including later events of other training students. The explicit timestamp audit finds later training observations in the same item/skill aggregates for 99.6012% of validation targets and 99.5887% of test targets (same-item alone: 96.8% / 95.3930%). These statistics would be unavailable at those historical times in a strict forward-time study. This is a material temporal limitation, not held-out-label contamination under the stated offline protocol; no online-deployment improvement is established. Earlier held-out outcomes may be used as observed causal history for later predictions, identically across models; current/future held-out outcomes do not enter predictors. Within-training self-label leakage is prevented by whole-student exclusion, but a chronological/cross-cohort study remains necessary for deployment claims.

Five initializations share one fixed split per dataset. Their SD captures optimization variation, not independent population/split uncertainty. EdNet's longest validation student contributes 8,194/42,875 targets (19.11%); the longest test student contributes 8,020/111,852 (7.17%). These globally pooled interaction metrics can be strongly influenced by long histories, reinforcing the need for independent cohorts/splits and student-level uncertainty checks. ASSIST2009 validation guided prior design; historical selection exposure remains. EdNet's random subset, composite tags, first/session sampling and untuned compact configuration limit broader generalization. The splits contain unseen students, largely known items: only 0.0669% of validation and 0.1648% of test rows have unseen question IDs. This is permitted item overlap, not evidence of cold-start item performance. The fitted rates are population-dependent success estimates, potentially influenced by student ability and item selection; they are not validated intrinsic difficulty or causal-effect parameters. Released tests must not guide further tuning. These are local controlled protocols and must not be compared directly to unrelated published leaderboard values. Negative or inconsistent results are retained.

The essential feature-matched control is **SAKT + the identical correction**, with the same item statistics, smoothing, own-student exclusion, native SAKT encoder, data, masks, optimizer, seeds, budget and validation selection. Its EdNet five configs and tested CPU adapter are prepared separately; no control GPU runs were charged to the core budget. A distinct resource budget is needed before executing the control campaign. Compare within-encoder gains and both feature-matched models before attributing benefits to attention architecture.

## Reproducibility and artifacts

See REPRODUCIBILITY.md for authoritative download links, exact hashes, reconstruction commands, fixed configs, checkpoint references and archive verification. Machine-readable tables include means/sample SD for validation/test AUC, accuracy, F1, precision, recall and BCE, individual seed metrics, paired deltas and neural parameters. Graphs under graphs/ use actual verified values only. Existing checkpoints/raw data are preserved locally; the portable reporting archive excludes raw data, student-level arrays and model binaries and contains explicit checkpoint references instead. It is not a self-contained model-inference bundle.

## Individual-seed results

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
| EdNet-KT1 (5k students; first/session) | SAKT | 42 | 0.665573541 | 0.639857916 | 0.663877266 | 0.778740334 |
| EdNet-KT1 (5k students; first/session) | SAKT | 123 | 0.664315887 | 0.635434537 | 0.663609055 | 0.779616939 |
| EdNet-KT1 (5k students; first/session) | SAKT | 2026 | 0.663147633 | 0.635445602 | 0.660426278 | 0.773755063 |
| EdNet-KT1 (5k students; first/session) | SAKT | 3407 | 0.666612587 | 0.635084398 | 0.662062368 | 0.776100130 |
| EdNet-KT1 (5k students; first/session) | SAKT | 7777 | 0.665460554 | 0.635974661 | 0.663510711 | 0.780051076 |
| EdNet-KT1 (5k students; first/session) | LightSAKT | 42 | 0.669818334 | 0.642402337 | 0.664342166 | 0.780061159 |
| EdNet-KT1 (5k students; first/session) | LightSAKT | 123 | 0.671166150 | 0.641733238 | 0.663680578 | 0.778459618 |
| EdNet-KT1 (5k students; first/session) | LightSAKT | 2026 | 0.669889885 | 0.641390304 | 0.663287201 | 0.776252941 |
| EdNet-KT1 (5k students; first/session) | LightSAKT | 3407 | 0.671666214 | 0.642636231 | 0.664145478 | 0.777652824 |
| EdNet-KT1 (5k students; first/session) | LightSAKT | 7777 | 0.668988675 | 0.643924468 | 0.665423953 | 0.778568681 |
| EdNet-KT1 (5k students; first/session) | LightSAKT+Difficulty | 42 | 0.728430952 | 0.706335369 | 0.692379215 | 0.790959903 |
| EdNet-KT1 (5k students; first/session) | LightSAKT+Difficulty | 123 | 0.729207732 | 0.707111775 | 0.692495440 | 0.788534961 |
| EdNet-KT1 (5k students; first/session) | LightSAKT+Difficulty | 2026 | 0.728356029 | 0.705994719 | 0.693085506 | 0.789293102 |
| EdNet-KT1 (5k students; first/session) | LightSAKT+Difficulty | 3407 | 0.730579894 | 0.707264974 | 0.694131531 | 0.792634437 |
| EdNet-KT1 (5k students; first/session) | LightSAKT+Difficulty | 7777 | 0.729160771 | 0.707671874 | 0.693246433 | 0.789152517 |
