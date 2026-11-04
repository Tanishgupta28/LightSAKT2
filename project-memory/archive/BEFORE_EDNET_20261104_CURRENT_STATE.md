# Verified current state — 31 October 2026, portable professor package

Research worktree `/workspace/LightSAKT2-architecture`, branch
`research/architecture-variants-20261025`. Measurements anchored at c286bcbc;
prior handoff49982fa7, training plan a37b17ed. The 30October nine-run GPU campaign
is complete and consumed. Current task ran ZERO GPU experiments and no fresh
model inference. No active campaign worker was found. Never duplicate those runs.

Fresh verification:40best checkpoint hashes/configurations/finite states/shapes/
parameter counts and validation-best epochs PASS. All80saved validation/test
prediction arrays were checked against exact student n-1 targets and independent
rank-AUC/confusion/BCE calculations. Ten difficulty optimizer/RNG/data-order
checks and338protected original-file hashes PASS. All77files in the original
professor_results_20261030 report directory remain byte-unchanged. New evidence:
`/workspace/LightSAKT2-architecture/reports/professor_release_20261031/verification_summary.json`.

Five seeds42,123,2026,3407,7777;fixed student-disjoint split seed42.
- ASSIST2009 modified mean validation AUC0.780247982;test0.779358708.
  Original LightSAKT test0.751055935;paired test gain+0.028302772.
- Algebra2005 modified mean validation AUC0.835215474;test0.836870733.
  Original LightSAKT test0.810589502;paired test gain+0.026281231.
- Each of those five seeds improves validation AND test AUC versus BOTH LightSAKT
  and SAKT. Mean test F1/accuracy also improve. Full sample SD and all seed/metric
  values are in the new report/CSV tables; no broader superiority claim follows.

ASSIST2015 difficulty is UNAVAILABLE_CURRENT_PROTOCOL (5slots), superseding the
historical PENDING queue label. The publisher-linked original CSV was retrieved
through Google Drive and is byte-identical to local raw AND raw archive:
SHA256 75e1f46131d5897a388c82411bc3032279899cb1f1e66a9043354d41234407a0;18,334,729bytes;708,631rows;
only user_id/log_id/sequence_id/correct. Local headers/archives, official field
semantics and maintained pyKT source yielded no authentic log-to-question map.
sequence_id denotes a skill-builder/problem set (existing input skill proxy),
log_id denotes an interaction event;neither is an independent repeated question.
The release alone does not determine item assignments. Provider-side mapping
might exist;do not claim universal impossibility. An authoritative log_id->problem_id
crosswalk with retained-row/student coverage is required to unblock. No invented
ID, replacement dataset, new feature or training run was introduced.

Evaluation clarification:baseline FINAL configs have evaluate_test=true for
post-training scoring;modified TRAINING configs false and separate CPU scoring.
All reported TEST values consistently use prior CPU replay(batch8);validation
values are native scores at the validation-selected epoch,reproduced by replay.
This supersedes the overly broad earlier shorthand 'all training false';the
selection rule remains validation-only. Test threshold0.5;globally pooled valid
targets;mean/sample SD across seeds(ddof1);paired deltas=modified minus comparator.

Leakage safeguards pass for offline fitting:whole own student excluded for train
features,training labels only for held-out estimates,causal history,no current or
future held-out labels in predictors. Full training-corpus statistics,richer item
inputs,prior validation reuse and one student split limit interpretation. Algebra
uses (Problem Name,Step Name) metadata and a local split of its raw training
subset,not official challenge test. Corrected causal baselines preserve original
trainable architectures;flawed historical original_snapshot scores are excluded.
Feature-matched SAKT augmentation remains prepared but unexecuted;separate budget.

Portable archive: `/workspace/LightSAKT2/backups/professor_release_20261031/LightSAKT_question_difficulty_verified_results_20261031.zip`
SHA256 `abf2feec4c31669006f6731ef7689742f7e549063d699e620e74cf83c4ffe8f0`;844874bytes.
CRC,safe extraction,184payload SHA checks,45slot/40measurement aggregations and
standalone standard-library verifier from /tmp PASS. Archive excludes weights,
raw datasets,prediction/target arrays,credentials and temporary download URLs.
It includes40configs/checkpoint SHA references and source snapshots,but is NOT a
self-contained inference/training bundle. New report and concise summary:
`/workspace/LightSAKT2-architecture/reports/professor_release_20261031/PROFESSOR_REPORT.md` and PROFESSOR_SUMMARY.md.

Three existing incremental Git bundles freshly verified;base prerequisites are
preserved. No remote push attempted/claimed;recorded CLI auth blocker remains.
Existing raw/checkpoint archives and main dirty scientific state are preserved.
These results are sufficient to share as verified research progress,not complete
three-dataset evidence or publication readiness. See NEXT_CHECKPOINT.md.

Verified portable-release commit: `2103a6e3e2ff79fa09af3b01406b02f404b58733`. Local incremental
Git bundle `/workspace/LightSAKT2/backups/professor_release_20261031/research_release-2103a6e3.bundle` verifies;requires preserved49982fa7.
Scientific measurements remain anchored at c286bcbc. A following memory-only
handoff may advance Git HEAD;no remote push was attempted.
