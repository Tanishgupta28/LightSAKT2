# Current durable directives — 30 October 2026

Campaign scope is now CONSUMED:9/9new runs completed. No additional GPU run is
authorized by this completed campaign. Missing ASSIST2015 metadata and optional
feature-matched SAKT controls are next separately scoped actions, not automatic
continuations. Test is released for reporting only; no further tuning against it.

The current user requests professor-ready SAKT, LightSAKT and LightSAKT with
question difficulty comparisons on ASSIST2009, ASSIST2015 and Algebra2005.
Prefer the existing five training seeds:42,123,2026,3407,7777. Reuse verified
results and checkpoints; independently replay validation and final test metrics.
Report all actual means/sample SD and individual seeds; missing values PENDING.

Current explicit final-test request supersedes the prior test-locked scope ONLY
for frozen validation-selected checkpoints. No tuning or architecture/feature
changes after test release. Preserve original architectures and established
student-disjoint splits, preprocessing, loss/masks, stopping and selection.
Disclose corrected causal baselines versus the flawed historical snapshot,
richer item inputs, offline full-training-corpus fitting and validation reuse.
Exclude whole own student from training target encoding; use training labels
only for held-out estimates. Do not claim forward-time deployment validity.

Current Phase3 scope authorizes nine necessary new bounded serial runs, frozen
in a37b17ed:four remaining ASSIST2009 difficulty seeds and five Algebra2005
seeds. Reuse completed difficulty42 and30existing baseline seeds. Existing shared
GPU permission applies only with exact-device/resource admission. Each run has
4GiB allocator cap/900s maximum; total<=8100GPU-seconds, concurrency1. Admission
free>=10GiB, runtimefree>=6GiB, workerRSS<=4GiB, hostheadroom>=2GiB, pacing0.1s,
stepstop30s; pre-existing unsupported-MIG-utilization fallback applies. Stop the
whole campaign on unsafe interruption; no automatic retry/extra window. Never
signal or modify peer/HiTSKT work. Preserve hold markers and their original bytes.

ASSIST2015 difficulty is blocked by missing independent question identity. Do not
substitute another feature or dataset. Algebra identity is the verified lossless
(Problem Name,Step Name) tutor-step composite; same numerical formula/smoothing10.
No feature-matched SAKT augmentation campaign or new architecture is launched in
this scope. The older optional control plan is a different, still-unlaunched plan.

Preserve existing checkpoints, raw results, original dirty state and Git history.
Commit validated changes only on intended research branch if safe. Keep verified
local bundles and artifact archives. No push to main, force push or credentials
exposure. Report unavailable Git authentication without discarding local work.
Update/synchronize canonical memory after material milestones and validate it.
Earlier user directives remain preserved in archive and decision log.
