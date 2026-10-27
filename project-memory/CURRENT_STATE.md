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
