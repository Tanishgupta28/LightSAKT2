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
