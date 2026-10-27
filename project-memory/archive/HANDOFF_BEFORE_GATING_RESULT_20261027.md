# Historical handoffs before the completed gating result

Archived27October2026 to keep current memory compact. Earlier assertions of
current state,permissions or next actions below are historical,including the
transient in-progress gating note. The latest root files and actual artifacts
supersede them. Relative links were adjusted to the memory root; scientific
claims were not revised. This is history within the existing memory system.


## Prior README.md

Original text SHA256 `e0258505c7170cf1e843357aaa28bc7f99aaed834ddb1222b33733b8f8044d1a`.

## Latest verified outcome — recency completed, 2026-10-27

All four initial ASSIST2009 seed42 arms completed. Recency AUC0.7482377884727862,
F1 0.8134333940013463; original one-layer linear0.7508035371114405 remains best.
Recency improved plain softmax by0.0025984095949505 but missed baseline by
0.0025657486386543. No tested variant warrants an improvement claim or promotion.
Next hypothesis: normalized decayed ELU+1 attention, not yet implemented/trained.
Grant closed; no retry/additional GPU scope. CPU replay and116 protected hashes PASS.
See [CURRENT_STATE.md](../CURRENT_STATE.md), [NEXT_CHECKPOINT.md](../NEXT_CHECKPOINT.md),
and `/workspace/LightSAKT2-architecture/reports/architecture_study/recency_pilot/RESULT_20261027.md`.
Existing missing Git-transport authentication remains; local history preserved.
Earlier current/pending-recency descriptions below are historical.

## Latest verified outcome — 2026-10-27 causal-softmax completion

Softmax completed21epochs, best16: validation AUC0.7456393788778357,
F1 0.8125468632841789. Original one-layer0.7508035371114405 remains best;
softmax delta -0.005164158233604832 is a valid one-seed negative result.
CPU validation replay and historical artifact preservation PASS. No test use.
Grant closed; no active training or automatic retry. Next hypothesis is recency
bias, not another softmax run. Read [CURRENT_STATE.md](../CURRENT_STATE.md) and
[NEXT_CHECKPOINT.md](../NEXT_CHECKPOINT.md). Full evidence:
`/workspace/LightSAKT2-architecture/reports/architecture_study/softmax_pilot/RESULT_20261027.md`.
Current session authorizes commit and safe non-force push to the research branch.
Earlier recovery/hold summaries below are historical where superseded.

## Current handoff — 2026-10-27 (supersedes older operational summary below)

Research branch `research/architecture-variants-20261025` is recovered in
`/workspace/LightSAKT2-architecture`; main remains `research-rebuild`.
Read [CURRENT_STATE.md](../CURRENT_STATE.md), [USER_DIRECTIVES.md](../USER_DIRECTIVES.md),
[DECISION_LOG.md](../DECISION_LOG.md) and [NEXT_CHECKPOINT.md](../NEXT_CHECKPOINT.md).
Canonical memory remains `/workspace/LightSAKT2/project-memory`; branch copies
are synchronized versions. Separate older LightSAKT_Project memory is unrelated.
Installed skill: `/root/.codex/skills/durable-project-memory/SKILL.md`.
Current evidence: `/workspace/LightSAKT2-architecture/reports/architecture_study/RECOVERY_20261027.md`.
Depth results recovered; softmax has no surviving valid outcome. Safe CPU
preparation is complete; no GPU run or hold release. Older status below is history.

# LightSAKT2 durable project memory

Authority: `/workspace/LightSAKT2`, branch `research-rebuild`, existing
repository `Tanishgupta28/LightSAKT2` (public in the current GitHub API). Artifacts and live process evidence take
precedence over stale progress prose. This is separate from the older
`/workspace/LightSAKT_Project/project-memory` (a different study); do not modify
or publish that sibling as part of this work.

Read [USER_DIRECTIVES.md](../USER_DIRECTIVES.md), [CURRENT_STATE.md](../CURRENT_STATE.md),
[DECISION_LOG.md](../DECISION_LOG.md), then [NEXT_CHECKPOINT.md](../NEXT_CHECKPOINT.md).
The current conclusion is **user hold active; no live LightSAKT2 training;
813 completed /142 pending /2 interrupted**. Completed artifacts and epoch-2
recovery state verified. At the latest09:24UTC recheck no visible GPU handles remain; reuse is still blocked
by missing existing-reservation owner/transfer/window/no-conflict evidence. No
automatic resume or watcher. Earlier busy snapshots are historical.

Primary evidence: `reports/LIGHTSAKT2_HITSKT_PRIORITY_HANDOFF_20261020.md` and
`reports/priority_handoff_20261020/`. Preserve the substantial pre-existing
uncommitted scientific state; never reset it to the base commit.


## Prior USER_DIRECTIVES.md

Original text SHA256 `7e9567f29ab301362fe8393976f8838cb7daf390ea50a8b8c7fea658df5e2f84`.

## Current authorization — gated linear pilot, 2026-10-27

User supersedes the prior decayed-linear next step with a separate minimal gating
variant retaining one-block causal ELU+1 attention. Authorizes implementation,
correctness validation and exactly one controlled ASSIST2009 seed42 GPU run under
existing shared permissions,4GiB allocator/900second cap,unchanged scientific
controls and validation-only selection. No test use or further architecture
change/run after a negative result. Preserve originals, prior results,local
commits and backups; update/validate canonical memory. Gate design is frozen in
/workspace/LightSAKT2-architecture/reports/architecture_study/gating_pilot/DESIGN_20261027.md.
Earlier no-additional-GPU/decay priority statements below are historical.

## Execution scope consumed — 2026-10-27

The authorized single recency run completed and its grant is closed_completed.
No additional GPU run is authorized; original controls and general holds remain.
Previous authorization text is retained below as historical provenance.

## Latest authorization — single recency run, 2026-10-27

User authorizes one fresh ASSIST2009 seed42 recency-biased softmax training run,
4GiB allocator/900second invocation limit, unchanged scientific controls and
shared safeguards. No baseline/depth/softmax rerun, tuning, test evaluation,
automatic retry or additional experiment. Verify checkpoints and record outcome.
Preserve local commits; report existing missing Git authentication without
exposing secrets or rewriting history. Supersedes older recency-not-authorized
text below for this one invocation only. General holds remain active.

## Latest bounded execution authorization — 2026-10-27

User explicitly authorizes one ASSIST2009 seed42 causal-softmax GPU run,4GiB cap,
15-minute limit, existing shared admission safeguards, unchanged training controls,
validation-only evaluation. Also authorizes local commit and safe non-force push
to research/architecture-variants-20261025. No recency or multi-seed execution.
General holds remain active with only an exact expiring single-arm exception.
This supersedes the preparation-only budget boundary below for this run only.

## Current research authorization — 2026-10-27

User requests LightSAKT2 recovery and architecture research continuation;
safe preparation is authorized, replacing the older instruction to stop all
project work. User grants full tool access and asks not to repeat permission
questions. Maintain all original code/results/checkpoints, use existing memory,
verify memory against artifacts, and avoid duplicate experiments. No test
inference while selecting models. Do not automatically launch expensive GPU
experiments: check workloads and applicable execution scope; report any required
new budget for approval. General hold remains active; no automatic recovery or
peer termination. Earlier HiTSKT-only next steps below are historical.

# User directives

## Latest authorization — verified shutdown and existing-reservation reuse

20 October2026: user again authorizes stopping only verified owned LightSAKT2
training and disabling its restart/submissions, preserving all checkpoints/results/
configs/logs. They do not want LightSAKT2 continued now. This supersedes the prior
allocation-first no-stop instruction for verified LightSAKT2 only. No unrelated
termination or reservation override. One bounded status/allocation check, no
polling/redundant audits. HiTSKT may reuse an existing allocation if its actual
permissions allow transfer under the same account/project/reservation without
conflicts; do not unnecessarily request a new allocation. Otherwise stop before
CUDA and explain missing permission. Staged CUDA then <=5 ASSIST train-only
updates authorized conditionally; no full training/test evaluation/DKT changes.

## Earlier clarification — allocation first, 20 October2026

User has no exclusive MIG assignment evidence. Preserve all checkpoints/results;
do not stop LightSAKT2 solely for GPU release before HiTSKT allocation is confirmed.
Project ownership is not permission to displace other reservations. No CUDA or
HiTSKT training without verified authorization. Do not restart anything or alter
the existing hold. Current clarification requests administrator guidance only.

## Earlier handoff authorization — historical order

20 October 2026 UTC — explicit user authorization supersedes earlier protection
against stopping only the user's verified LightSAKT2 jobs. Inspect ownership,
preserve all checkpoints/results/configs/logs, disable future automatic launches,
gracefully stop verified training if active. Never kill unrelated jobs/services,
or force-kill during artifact writing. Ask if ownership or safe preservation
cannot be established. All actual workers were already interrupted before this
turn, so no signals were necessary or sent.

Keep `experiments/USER_TRAINING_HOLD.json` active. No future LightSAKT2 resumption
or automatic recovery without a later explicit user decision. Preserve pending
configurations and both interrupted statuses; do not mark unfinished work complete.

User conditionally authorizes HiTSKT FAVOR+ staged CUDA checks then <=5 ASSIST2017
train-only updates after genuine exclusive allocation and correctness PASS.
Stopping LightSAKT2/free memory does not prove allocation. No unrelated workload
termination, allocation change, full training, test evaluation, DKT/EdNet
resumption/reset or historical ELU+1 modification is authorized.

Commit/push verified changes to the appropriate existing research branches;
no merging, force-push or history rewriting. Leave unrelated dirty work intact.


## Prior CURRENT_STATE.md

Original text SHA256 `56285318ed437fb6154a7ba5fd8ce3c3d83a730a5956f0fcd68c6bf3dd787bbe`.

## Gated-linear pilot in progress — 2026-10-27

Current user requested minimal gating instead of the earlier proposed decay arm.
Implemented LightSAKTGated: g=2sigmoid(w*E(next_skill)+b),65new parameters,
zero weights/bias give exact identity. Original attention,one block,residual,
norms,FFN/head/settings unchanged.89 CPU checks PASS before execution.
Single authorized seed42 ASSIST2009 run launched from7026c660,4GiB/900s/shared
safeguards. Outputs:
/workspace/LightSAKT2-architecture/pilot-workspace/results/raw/ASSIST2009-LightSAKT-pilot-one_layer_gated_linear-42-v1/.
Current grant:
/workspace/LightSAKT2-architecture/experiments/architecture_study/pilot/gpu_authorization_gating_20261027.json.
Check actual terminal state before any recovery; no automatic retry or second run.
Earlier no-further-GPU/decay-priority text is historical where superseded.

## Verified recency completion — 2026-10-27

Branch `research/architecture-variants-20261025`, persistent research worktree
`/workspace/LightSAKT2-architecture`; recency execution commit `27e5a486`.
Fresh authorized recency-softmax run completed21epochs, best16:
validation AUC0.7482377884727862, F1 0.8134333940013463,
accuracy0.7274548368298368, BCE0.544185487825003.
Controls unchanged. Original linear0.7508035371114405; depth0.7507909322800659;
softmax0.7456393788778357. Recency-baseline delta -0.0025657486386543304;
recency-softmax delta +0.0025984095949505015. Valid negative versus baseline,
preliminary one-seed benefit over softmax; no overall improvement/replication gate.

82 CPU tests PASS before launch (causality/padding/dense-reference decay gradients,
finite optimizer/save-load, scope and exact reporting/resume). Independent CPU
validation replay PASS (AUC difference8.9523e-9); full finite model/Adam/RNG/loader/
early-stop/history state present. All116 pre-existing source/checkpoint/raw-report/
manifest/hold file hashes unchanged. No test loader/prediction access.
Decay per head learned0.0381679125,0.0389851369 from0.03125.
Runtime167.035566s; peak allocated140561408bytes (134.050MiB),reserved160MiB.
Within4GiB/900s shared safeguards; peer compute process retained untouched.
Grant closed_completed, no automatic retries or extra experiment.

Best/latest checkpoints:
`/workspace/LightSAKT2-architecture/pilot-workspace/checkpoints/ASSIST2009-LightSAKT-pilot-one_layer_recency_softmax-42-v1/`.
Best SHA256 `988b4302c2bc0f3c3855f7c365f9bd154f2b66fefea83b3cb384e3774796355b`.
Persistent read-back-verified archive:
`/workspace/LightSAKT2/backups/architecture_recency_20261027/pilot_checkpoints_and_records.tar.gz`.
Evidence `/workspace/LightSAKT2-architecture/reports/architecture_study/recency_pilot/` includes raw history/config/metrics/status/environment,
epoch CSVs, comparison, replay, resource and archive-hash records.
Missing Git transport authentication unchanged; retain local commits and report
blocker rather than exposing secrets or reconstructing history remotely.
Prior active/no-recency-scope sections below are historical and superseded.

## Publication checkpoint — 2026-10-27

Verified result commit `f15b28b9a30117a9cfdcea677f398ba2b0967713` is local. Normal non-force push failed:
shell Git has no HTTPS credentials; gh has no logged-in account and no SSH key
is present. Connected GitHub app authentication works, but its exposed commit API
cannot push the exact existing local commit graph. Do not recreate/rewrite local
research history to mask this limitation. Remote research branch was absent at
last ls-remote check. No remote update succeeded.

Local Git bundle `/workspace/LightSAKT2/backups/architecture_softmax_20261027/research_commits.bundle` preserves research commits since a6bdc6c0
(requires that base); bundle verify PASS. Model checkpoints have their separate
verified archive. Remaining publication action after authenticated Git transport
is available: `git push --set-upstream origin HEAD:refs/heads/research/architecture-variants-20261025`
from /workspace/LightSAKT2-architecture, then verify remote HEAD. No force push.
No further GPU scope; recency remains a recommendation only.

## Completed and verified softmax result — 2026-10-27

Research worktree `/workspace/LightSAKT2-architecture`, branch
`research/architecture-variants-20261025`; executed from `b63ec478`.
One authorized fresh causal-softmax run completed21epochs; selected epoch16.
Validation AUC0.7456393788778357, F1 0.8125468632841789,
accuracy0.7268356643356644, BCE0.5465822575284047.
Original one-layer0.7508035371114405; two-layer0.7507909322800659.
Softmax delta -0.005164158233604832; no improvement or promotion signal.
Conclusion is a valid single-seed negative, not generalized inferiority.

Training161.204seconds; peak allocated140479488bytes (133.972MiB),
reserved167772160bytes (160MiB), within4GiB/900second caps. CPU replay AUC
absolute difference1.49205e-8; all six validation metrics pass. Best/latest
checkpoints contain verified model/Adam/RNG/loader/history state. Prior source,
depth results/checkpoints, manifest and holds remain hash-identical. Test untouched.
Grant closed_completed; no retry, recency training or unrelated workload changes.

Run ID `ASSIST2009-LightSAKT-pilot-one_layer_softmax-42-v1`; checkpoint directory
`/workspace/LightSAKT2-architecture/pilot-workspace/checkpoints/ASSIST2009-LightSAKT-pilot-one_layer_softmax-42-v1/`.
Best SHA256 `4d74a39f00e4e36a04c074649e37018cc5856557b9b3ff5b87b6fe14b9ac7816`.
Persistent archive:
`/workspace/LightSAKT2/backups/architecture_softmax_20261027/pilot_checkpoints_and_records.tar.gz`;
member and archive hashes in `/workspace/LightSAKT2-architecture/reports/architecture_study/softmax_pilot/recovery_archive_20261027.json`.
Numerical records, full comparison and verification in `/workspace/LightSAKT2-architecture/reports/architecture_study/softmax_pilot/`.
Commit/push authorized for verified scoped results; main dirty work is retained.
All earlier active/pending-softmax descriptions below are historical.

## Verified current research state — 2026-10-27

Recovered prior research HEAD `6beced52` on
`research/architecture-variants-20261025`; persistent worktree
`/workspace/LightSAKT2-architecture`. Main stays `research-rebuild` at `a6bdc6c0`
with pre-existing dirty science preserved. Missing /tmp worktree registration
retained, not pruned. Commit alone did not preserve weights: verified persistent
archive SHA256 `1899e200690f6897bfe9af27a003a1f91f324b3f48e63c8b09996c8487aca8d6`
at `/workspace/LightSAKT2/backups/architecture_depth_20261025/pilot_checkpoints_and_records.tar.gz`;
all22 member hashes verified,14 immutable run files restored into persistent
worktree `pilot-workspace/` (four best/latest checkpoints plus raw records).

Verified ASSIST2009 seed42 validation AUC: one-layer0.7508035371114405,
two-layer0.7507909322800659; both29 epochs, selected24. No depth gain.
Execution commit `f46e7d81`, completion commit `7b650cb9`.
Causal softmax was prepared/authorized at `6beced52`, but no surviving metrics,
checkpoint, history or terminal marker establish execution. Lost uncommitted
/tmp work cannot be excluded. Recency likewise has no verified result.

Dataset/split/protocol/selected-baseline hashes pass; frozen seed42 config
unchanged.80 targeted CPU tests pass. Hash-bound relocation support was added
only to the summary reader; original models/trainer untouched this recovery.
Main versus branch trainer already differs by observer instrumentation, checked
by existing update-equivalence/exact-resume tests. Initial pytest temp setup
failure and corrected assertion are documented, not experimental failures.

One read-only process/GPU snapshot found no LightSAKT2 trainer/controller,
37MiB/40320MiB MIG usage and no visible compute process; unrelated jobs untouched.
No new GPU execution or test inference. Historical general holds and GKT
interruptions retained. No current unexpired softmax grant exists.
Detailed evidence and restored checkpoint paths:
`/workspace/LightSAKT2-architecture/reports/architecture_study/recovery_20261027/{readiness,comparison}.json`;
report `/workspace/LightSAKT2-architecture/reports/architecture_study/RECOVERY_20261027.md`.
Skill installed at `/root/.codex/skills/durable-project-memory`, upstream
`0a06f1bb8231ace20e4405281a263012fedb5e51`. Earlier current-state text below is historical.

# Current state

Latest09:24UTC,20 October2026: renewed user authorization permits stopping
verified owned LightSAKT2 now and disabling restarts. Single fresh check found
no live trainer/controller/recovery or GPU handles; persistent hold intact,
813 completed/142 pending/2 interrupted, zero signals needed/sent. All7915 old
files present/same size;8 key hashes and local recovery archive unchanged. No
repeated full audit or tests. Sole MIG nearly empty with no visible GPU owners,
but existing allocation owner/account/transfer/window/conflicts are unverified
(no scheduler evidence, Kubernetes403). A new allocation is not required if the
existing one authorizes HiTSKT. Stop before CUDA; do not resume LightSAKT2,
bypass the hold, poll automatically or change artifacts. HiTSKT evidence/report:
`/workspace/capstone/reports/favor_plus_v1/EXISTING_ALLOCATION_REUSE_STATUS_20261020.md`.
Earlier conclusions/instructions below are historical where superseded.

Latest user clarification: preserve all work; do not stop LightSAKT2 solely to
free the GPU before HiTSKT allocation is confirmed. No CUDA/training without
verified authorization, no hold release/restart. No new process/GPU observation
was made in this documentation-only clarification. Prior handoff below remains
historical evidence; allocation-first sequencing supersedes earlier ordering.

20 October 2026 UTC. Base `34eeefb7d6b99ad079eab4da340877e46e48a8ec`,
`research-rebuild`. No live training or known local controller/recovery worker;
old PIDs absent/zombie and stale heartbeat are not activity. **813 completed,
142 pending,2 interrupted**. No signals sent. Persistent user hold blocks known
launch paths;9 targeted CPU tests pass, no CUDA or training run.

Preservation PASS:7915 scientific files/1,391,693,255 bytes identical before/after;
814 recorded selected-checkpoint references (776 unique paths), all declared
prediction hashes/configs match.814 completed backup paths exist; no new offsite
download verification. Epoch2 GKT interruption is CPU-validated and separately
archived locally with optimizer/RNG/loader state. Waiting GKT has no saved epoch.
No incomplete work marked complete, no scientific code/config/result changed.

Evidence: `reports/LIGHTSAKT2_HITSKT_PRIORITY_HANDOFF_20261020.md`,
`reports/priority_handoff_20261020/{before_hold,after_hold}/snapshot.json`,
`preservation_comparison.json`, `interrupted_checkpoint_and_archive.json`,
`hold_cpu_tests_final.log` in that evidence directory.

HiTSKT CUDA/pilot remains BLOCKED: sole H100 MIG also has unrelated GPU handle
owners; no authoritative exclusive HiTSKT allocation. Do not stop them. Current
user authorization does not manufacture ownership. No automatic observer started.

Substantial older dirty manifests/status/logs/reports are retained on disk and
not included in the scoped control commit. Checkpoint binaries/new recovery
archive remain ignored/local; the Git commit alone cannot restore epoch2.


## Prior NEXT_CHECKPOINT.md

Original text SHA256 `12868cadf64f0391a69775e01b59f44b0d87dd0edcdfa27594ff9317a9178ceb`.

## Next research action after all four initial arms — 2026-10-27

Do not repeat baseline/depth/softmax/recency seed42 runs; all are complete.
Original one-layer ELU+1 remains best, AUC0.7508035371114405.
Recency-softmax0.7482377884727862 is below baseline, so no five-seed promotion
of that arm based on this screening. Exact result: `/workspace/LightSAKT2-architecture/reports/architecture_study/recency_pilot/RESULT_20261027.md`.

Next hypothesis: retain normalized ELU+1 and add learned per-head exponential
decay to both KV and K prefix states. Recency recovered0.00259841 over softmax,
while ELU+1 still leads; combining those factors is untested, not proven useful.
AKT/RetNet primary-source motivation, equations, numerical hazards and verification
requirements are recorded in the result report. No implementation or training
of this new arm exists yet. Preserve baseline source, settings and data.

Next bounded development action: implement a separate normalized decayed-linear
attention arm; verify dense-vs-recurrent/scan forward and gradients, lambda1
baseline limit, causality/padding/finite gradients/determinism. Initialize decay
at1/32 per head without a new tuning sweep; decay numerator and denominator
identically. Recheck original-state mapping/initialization. Only then propose a
separately authorized seed42 ASSIST2009 comparison with4GiB/900s shared safeguards.
No further GPU authorization in current task. Promote only credible improvements
to five seeds42/123/2026/3407/7777 over ASSIST2009/ASSIST2015/Algebra2005, reporting
paired validation deltas and mean/sample SD. Test stays locked during selection.

Publication remains blocked by existing missing Git transport authentication.
Keep local commits/bundles; no force push or rewrite. Earlier next-recency,
recency-not-authorized and softmax-pending text below is historical.

## Publication checkpoint — 2026-10-27

Verified result commit `f15b28b9a30117a9cfdcea677f398ba2b0967713` is local. Normal non-force push failed:
shell Git has no HTTPS credentials; gh has no logged-in account and no SSH key
is present. Connected GitHub app authentication works, but its exposed commit API
cannot push the exact existing local commit graph. Do not recreate/rewrite local
research history to mask this limitation. Remote research branch was absent at
last ls-remote check. No remote update succeeded.

Local Git bundle `/workspace/LightSAKT2/backups/architecture_softmax_20261027/research_commits.bundle` preserves research commits since a6bdc6c0
(requires that base); bundle verify PASS. Model checkpoints have their separate
verified archive. Remaining publication action after authenticated Git transport
is available: `git push --set-upstream origin HEAD:refs/heads/research/architecture-variants-20261025`
from /workspace/LightSAKT2-architecture, then verify remote HEAD. No force push.
No further GPU scope; recency remains a recommendation only.

## Exact next checkpoint after softmax — 2026-10-27

Do not rerun baseline, depth or causal softmax: all three have valid completed
seed42 ASSIST2009 results. Original linear remains best at0.7508035371114405;
softmax0.7456393788778357 failed the +0.001 screening gate.

Recommend the already prepared one-layer recency-biased softmax experiment on
ASSIST2009 seed42 under identical controls, comparing both original and plain
softmax. Its learned per-head nonnegative distance penalty tests recency bias.
No recency execution scope was granted in this task. Obtain separate bounded
scope before any GPU run; keep general holds, shared admission and4GiB/900s caps.
No automatic launch or retry. Exact config:
`/workspace/LightSAKT2-architecture/experiments/architecture_study/pilot/configs/one_layer_recency_softmax.json`.
Current runner supports recency only in two-arm attention stage; before future
execution add a single-recency stage so plain softmax is not launched again.
Do not use --stage attention unmodified. This preparation is future work.

After a promising screening result, propose five seeds42/123/2026/3407/7777 on
ASSIST2009/ASSIST2015/Algebra2005 and report paired validation deltas and mean±SD.
No test selection. Completed softmax evidence: `/workspace/LightSAKT2-architecture/reports/architecture_study/softmax_pilot/RESULT_20261027.md`.
End current task by validating memory, scoped commit and safe non-force push;
verify remote branch HEAD rather than assuming publication succeeded.
Earlier softmax preparation or in-progress instructions below are historical.

## Exact next research checkpoint — 2026-10-27

Safe preparation is complete. Next uncompleted experiment is one-layer causal
softmax on ASSIST2009 seed42 against the restored same-seed baseline. No baseline
or depth retraining needed. Use the unchanged config in
`/workspace/LightSAKT2-architecture/experiments/architecture_study/pilot/configs/one_layer_softmax.json`.

Before GPU execution, obtain a current bounded execution budget/window (one run,
<=900seconds,4GiB allocator cap) and recheck shared admission (>=10GiB free at
entry,>=6GiB during execution), exact MIG UUID, unchanged hold/config hashes.
Do not reuse expired grants or remove the general hold. Command and full policy
are in `/workspace/LightSAKT2-architecture/reports/architecture_study/RECOVERY_20261027.md`.
Use `--stage softmax`; `attention` also launches recency and exceeds this scope.

After running: verify terminal status/config/data/checkpoint hashes, replay only
validation, export all six epoch metrics, compare against0.7508035371114405;
promotion screening needs >=0.001 AUC gain. Preserve negative/interrupted outcomes,
close grant, archive new checkpoints outside /tmp, update and validate memory.
If softmax fails, investigate prepared recency bias under separately bounded
scope. Promising changes: five seeds42/123/2026/3407/7777 over ASSIST2009,
ASSIST2015 and Algebra2005 with paired validation deltas. Keep test locked.
Earlier HiTSKT next steps below are historical, superseded for this request.

# Next checkpoint

Latest09:24UTC,20 October2026: renewed user authorization permits stopping
verified owned LightSAKT2 now and disabling restarts. Single fresh check found
no live trainer/controller/recovery or GPU handles; persistent hold intact,
813 completed/142 pending/2 interrupted, zero signals needed/sent. All7915 old
files present/same size;8 key hashes and local recovery archive unchanged. No
repeated full audit or tests. Sole MIG nearly empty with no visible GPU owners,
but existing allocation owner/account/transfer/window/conflicts are unverified
(no scheduler evidence, Kubernetes403). A new allocation is not required if the
existing one authorizes HiTSKT. Stop before CUDA; do not resume LightSAKT2,
bypass the hold, poll automatically or change artifacts. HiTSKT evidence/report:
`/workspace/capstone/reports/favor_plus_v1/EXISTING_ALLOCATION_REUSE_STATUS_20261020.md`.
Earlier conclusions/instructions below are historical where superseded.

Latest user clarification: preserve all work; do not stop LightSAKT2 solely to
free the GPU before HiTSKT allocation is confirmed. No CUDA/training without
verified authorization, no hold release/restart. No new process/GPU observation
was made in this documentation-only clarification. Prior handoff below remains
historical evidence; allocation-first sequencing supersedes earlier ordering.

Preservation/hold complete; stop operational work. Leave
`experiments/USER_TRAINING_HOLD.json` present, all scientific artifacts and older
dirty state intact. Publish scoped controls/evidence/memory to `research-rebuild`
only; do not merge or stage old unrelated modifications.

Next bounded task is external allocation clarification for HiTSKT, handled in
`/workspace/capstone/project-memory/`. Do not poll automatically, signal unrelated
GPU owners, initialize CUDA or resume LightSAKT2.

Any future user-authorized LightSAKT2 recovery must first recheck ownership and
checkpoint/data/config/source hashes. Epoch2 GKT (`...e01b589e46cc`) is validated;
waiting GKT (`...e1c69a92d6b5`) has no epoch. Consult
`reports/priority_handoff_20261020/interrupted_checkpoint_and_archive.json` and
the still-local original checkpoints. Only then make an explicitly reviewed hold
release decision. Do not use stale PID files or reset research state to Git HEAD.
