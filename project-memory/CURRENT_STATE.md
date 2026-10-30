# Verified current state — 30 October 2026, professor campaign complete

Worktree `/workspace/LightSAKT2-architecture`, branch
`research/architecture-variants-20261025`. Frozen execution commit a37b17ed.
All nine new runs completed; owned GPU controller exited0. No training remains
active under this campaign and its scope is consumed. Do not launch more runs.
Reused30baseline seeds plus completed ASSIST2009 difficulty seed42; executed four
remaining ASSIST2009 difficulty seeds and five Algebra2005 difficulty seeds.
All use seeds42,123,2026,3407,7777 and unchanged per-dataset LightSAKT settings.

| Dataset | Model | Mean validation AUC | Mean test AUC |
|---|---|---:|---:|
| ASSIST2009 | SAKT | 0.748283739 | 0.750322388 |
| ASSIST2009 | LightSAKT | 0.749453705 | 0.751055935 |
| ASSIST2009 | LightSAKT+Difficulty | 0.780247982 | 0.779358708 |
| ASSIST2015 | SAKT | 0.728969794 | 0.725179788 |
| ASSIST2015 | LightSAKT | 0.729922085 | 0.726088355 |
| ASSIST2015 | LightSAKT+Difficulty | PENDING | PENDING |
| Algebra2005 | SAKT | 0.809998581 | 0.813373449 |
| Algebra2005 | LightSAKT | 0.807548440 | 0.810589502 |
| Algebra2005 | LightSAKT+Difficulty | 0.835215474 | 0.836870733 |

All non-PENDING rows have five seeds; exact sample SD, F1, accuracy, precision,
recall, BCE, epochs, runtime, neural parameters and provenance are in
`/workspace/LightSAKT2-architecture/reports/professor_results_20261030/`.
Primary report `PROFESSOR_REPORT.md`, concise `PROFESSOR_SUMMARY.md`, actual
`individual_seed_results.csv`, `summary_results.csv`, `absolute_auc_improvements.csv`.

Both completed difficulty datasets improve validation AND test AUC in all five
matched seeds versus BOTH original LightSAKT and SAKT. Mean test gains versus
LightSAKT: ASSIST2009 +0.028302772; Algebra2005 +0.026281231. Test F1/accuracy
means also improve. This supports the item-information modification on these
fixed offline student-disjoint partitions, not attention-kernel superiority,
forward-time deployment or all-three-dataset generality.

ASSIST2015 difficulty: five slots PENDING because the available four-column
source has no independent question ID. Do not substitute skill IDs or another
release silently. Algebra target identity (Problem Name,Step Name) is aligned
row-for-row and confirmed by the official SIGKDD format; its local test is a
student partition of the original training subset, not official challenge test.

Verification:40checkpoints independently CPU-replayed on validation/test;
80prediction-array checks reproduce every student's n-1targets and independent
rank-sum AUC/confusion metrics. Ten difficulty best/latest checkpoint states,
finite Adam tensors/counters and exact seeded shuffle/RNG states PASS. All338
protected original files unchanged. Real held-out-label and whole-own-student
label perturbations PASS on both feature datasets. Seven pre-GPU CPU tests PASS.

Question scalar formula/smoothing10 and causal linear encoder unchanged.
Training estimates exclude whole own student; held-out estimates use training
labels only. ASSIST2009 original numerical features reused exactly. Full training
corpus timing, richer item inputs, reused validation and one fixed split per
dataset are disclosed limitations. Seed42 headline0.750803537->0.781189972 is
reproduced, best24/completed29 after prior exact epoch18 continuation; no restart.
Historical original_snapshot defects are disclosed; corrected causal baselines
retain original architecture/parameter shapes. Original artifacts are preserved.

Resource summary:9new admitted windows3850.931233s (64.18min), native training
3832.883621s; each<900s. Maximum allocated818MiB, workerRSS1.584GiB;
minimum GPUfree18.818GiB/hostheadroom32.568GiB. No sampled guard breach or peer
interference. FourGiB cap and original holds remain. Test scoring was explicitly
authorized only after validation selection; all training has evaluate_test=false.

Verified result commit `c286bcbcee46a5317dbb3071c1f6c3a66335d04a` on intended research branch.
Local recovery archives/checksums: checkpoint_backups.json under report folder,
pointing to `/workspace/LightSAKT2/backups/professor_results_20261030/`.
Thirteen readback-verified archives retain10difficulty best/latest checkpoints,
raw run records, both datasets' frozen features and all80prediction arrays.
Incremental Git bundle `/workspace/LightSAKT2/backups/professor_results_20261030/research_results-c286bcbc.bundle` verifies and requires preserved a4414cf2.
The prior verified bundle in backups/question_completion_20261029 retains that
predecessor and requires preserved baseline a6bdc6c0. No bundle/history overwritten.
Git results are local only; existing authentication blocker retained, no external
push or force-push. A following memory-only handoff may advance HEAD; scientific
results are anchored at c286bcbc. See Git log for latest handoff commit.
