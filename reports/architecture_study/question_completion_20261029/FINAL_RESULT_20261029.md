# Completed question-difficulty pilot — 29 October 2026

**The improvement survived leakage checks, exact continuation and independent
CPU validation replay for the frozen OFFLINE STUDENT-DISJOINT benchmark.**
Final validation AUC0.781189971878,delta+0.030386434766
(+3.038643percentage points) over original0.7508035371114405.
One seed and reused validation;not independent training replication or proof of
forward-time online performance or attention-kernel superiority.

| Architecture/input | Validation AUC | F1 | Accuracy | BCE | Best epoch | Total epochs |
|---|---:|---:|---:|---:|---:|---:|
| Original one-layer linear | 0.750803537111 | 0.814805489110 | 0.732116841492 | 0.542363721265 | 24 | 29 |
| Two-layer linear | 0.750790932280 | 0.816262117682 | 0.732845279720 | 0.542330543346 | 24 | 29 |
| Causal softmax | 0.745639378878 | 0.812546863284 | 0.726835664336 | 0.546582257528 | 16 | 21 |
| Recency softmax | 0.748237788473 | 0.813433394001 | 0.727454836830 | 0.544185487825 | 16 | 21 |
| Gated linear | 0.750645233679 | 0.815458840084 | 0.732517482517 | 0.542621742998 | 24 | 29 |
| Question difficulty | 0.781189971878 | 0.824088667446 | 0.747960372960 | 0.517380792618 | 24 | 29 |

## Leakage audit and prediction-time boundary

All numerical frozen features exactly match an independent rebuild. Flipping
all real validation labels changes no feature value. Flipping the longest
training student's entire response history changes none of that student's
features. Positive controls show other training students' labels do influence
fitted difficulty,as intended. Full-builder synthetic tests show validation/test
labels and raw correctness do not influence numerical features;test students
are absent from feature files. Whole-file provenance hashes change if input
bytes change,which is not a predictor-value dependency. No real test evaluation,
test feature construction or actual test-label perturbation was performed.

Difficulty is a fitted training-corpus statistic,not a within-student running
estimate. For training rows,ALL own-student outcomes are subtracted from both
question and skill rates,including the current target and future own responses.
For validation rows,all rate estimates use training students only. Training and
validation students are disjoint. Question ID is known metadata,joined one-to-one
by existing student/order keys;raw correctness is not used in that join. Unknown
questions fall back to the skill prior. Original causal input indexing remains
q/r through t predicting response at t+1;observed validation history may be used
just as in original KT evaluation,but future validation outcomes cannot enter
the attention or fitted difficulty feature.

**Temporal limitation:**95.045% of validation attempts have a later-order training
observation for the same problem. order_id is an ordering proxy,not an audited
wall-clock timestamp. Thus features use globally fitted full-training statistics
that need not have existed before each historical event. This is permissible
under the existing retrospective student-disjoint benchmark,where model fitting
also uses the complete training partition. It is NOT a causal forward-time
deployment claim. A chronological evaluation would require a separately frozen
time-aware training/statistics design;we did not silently alter the protocol.

Validation was reused for diagnostic feature choice and early stopping. This
selection exposure remains even though no validation label enters fitted rates.
Fixed-seed replication and feature-matched controls are needed. Input is richer
than the original skill-only model;training-derived per-problem statistics add
information/capacity despite only one additional neural scale. Same split and
evaluation procedure do not mean identical input information.

## Exact continuation verification

Initial execution b24e154a stopped after18saved epochs;resumed from latest.pt
under continuation commitd958cff4. Same run ID/config,not a new initialization.
The first18history entries remain exactly unchanged. Continued epochs19..29,
selected24,ended with bad_epochs5 under original patience5 (max50).

All22 Adam states advanced from step1134 to1827 (63batches/epoch,11new epochs).
The restored loader's full permutation stream produces the final saved loader
RNG exactly;epoch19order SHA256
2b0b05474cbf7d1a5c59d61f724f97fd0baf87689e6c7f52fe78b8cd1453ba3f.
CPU global RNG matches the expected validation-loader draws;Python/NumPy states
are preserved. CUDA state was present and restored by unchanged engine code;
no extra CUDA audit updates were performed. Source/model/adapter/launcher/observer
identities match the original execution. Hyperparameters,data,feature file,
split,precision and early-stop/checkpoint rules were not changed.

Independent CPU selected-checkpoint replay AUC difference
1.19363933182e-08;F1/accuracy/precision/recall match
exactly. Full finite model/Adam/RNG/loader state and config/data/feature/grant
bindings verified. All240 protected old files unchanged,including the original
interruption archive and28October reports. Five leakage/pipeline audit CPU tests
and two feature-matched SAKT preparation tests pass. No original scientific
source was edited. Verification performs zero optimizer updates.

## Resource use and artifacts

29epoch cumulative saved training time327.944s;
continuation training113.740s;
new invocation118.733s,below900s.
Peak allocated254.856MiB,reserved
368.000MiB,below4GiB allocator cap.
Minimum continuation host headroom9.335GiB;
GPU free27.996GiB. Shared peer
workload untouched;general holds,pacing and all resource floors retained. New
grant closed_completed. Prior interrupted window remains recorded,not erased.
Cumulative saved training time excludes discarded unsaved work/queue gaps;
shared-device timings are not clean model-throughput comparisons.

Best/latest checkpoint hashes and selected alpha=0.929290950
are in completion_verification.json. Run ID:ASSIST2009-LightSAKT-pilot-one_layer_question_difficulty-42-v1.
Checkpoint directory:pilot-workspace/checkpoints/<run ID>/.
Old epoch18 binary archive remains intact;new completed binary/feature/raw-record
archive is indexed by recovery_archive_20261029.json. Feature sidecar is required
for replay and remains local/ignored,not in memory or Git. Both old and new
executed grants are retained. CSV one_layer_question_difficulty_epochs.csv
contains every epoch's validation AUC/F1/accuracy/BCE/precision/recall,online
training metrics,runtime and GPU/RSS peaks. PNG/PDF curves are standalone exports.

CPU replay:CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m scripts.verify_question_completion_20261029
State continuity:python -m scripts.verify_question_state_continuity

## Replication packet — prepared,not launched

experiments/question_replication_20261029/plan.json freezes five seeds
42,123,2026,3407,7777 for LightSAKTQuestion and a feature-matched SAKT control.
Reuse this completed LightSAKT seed42;only four new LightSAKT runs plus five SAKT
controls remain. Both receive exactly the same frozen feature values and one
zero-initialized scale. SAKT control preserves native SAKT forward/attention via
a scoped logit hook;CPU init/dropout/gradient/exact-resume tests pass. Each model
keeps its previously selected settings;feature-matched is not hyperparameter-
identical (SAKT lr0.0005,4heads,dropout0.2;LightSAKT lr0.001,2heads,dropout0.1).
This control can distinguish general information gain from model-family effects.

Proposed SEPARATE budget ceiling:9sequential new invocations,each<=900seconds
and4GiB allocator,total<=8100GPU-seconds (135minutes);same host/GPU safeguards,
no automatic extra windows/retries. This is a conservative ceiling,not a runtime
estimate or approval. Requires explicit campaign budget and a budget-enforcing
seed-general launcher;current pilot launcher intentionally allows seed42 only.
No real-data SAKT control or additional LightSAKT seed was launched.

Report paired validation deltas,mean and sample SD across all five;never best
seed. Five seeds measure initialization variance,not new student-cohort uncertainty.
Keep final test locked until architecture/input selection is frozen and separately
authorized. Algebra2005 needs verified problem/step mapping;current ASSIST2015
four-column release lacks separate problem IDs,no silent input/schema substitution.

Professor update:The audited question-difficulty correction completed29epochs
and reached validation AUC0.781190 versus0.750804 (+0.030386),with exact state
continuity and CPU replay. It is a real one-seed offline validation gain using
richer known-item information. Five-seed and feature-matched SAKT configurations
are prepared;campaign execution and online generalization are not yet established.
