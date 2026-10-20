# LightSAKT2 preservation and HiTSKT priority handoff

20 October 2026 UTC. Base `34eeefb7d6b99ad079eab4da340877e46e48a8ec`,
branch `research-rebuild`. The user explicitly authorized stopping only their
verified LightSAKT2 workload, preserving all research and disabling automatic
relaunches. No training configuration or scientific model code was changed.

## Outcome

**LightSAKT2 was already stopped before this inspection. No signals were sent.**
The old 19 October heartbeat was stale. The existing trainer log ends with
`KeyboardInterrupt` during backward; the other worker stopped waiting for GPU
admission. The earlier interruption's initiating actor is not established.
Saved PIDs are absent or zombies, not running training. Interactive shells and
coding sessions were left untouched.

| Check | Status / evidence |
| --- | --- |
| Current schedule | **813 completed, 142 pending, 2 interrupted, 0 running** out of 957, from per-run status overlaid on the manifest |
| Completed research preservation | **PASS**: 7,915 files / 1,391,693,255 bytes identical before and after the hold |
| Stored selected checkpoints | **PASS**: 814 result references, 776 unique selected paths, all recorded hashes match; includes one extra smoke result outside the 957 slots |
| Stored prediction hashes / completed configs | **PASS** for every declared prediction and result config in the capture |
| Existing completed backup archives | **814 referenced local archives present**; no new completed-archive download/hash verification claimed |
| Interrupted epoch-boundary recovery | **PASS**: strict CPU model load, finite model/Adam, RNG/loader states, matching config/data/split hashes and epoch history |
| Known local automatic relaunch paths | **Held** by persistent marker; final targeted tests **9 PASS** |
| External startup/reservations | **NOT VERIFIED**: no scheduler assignment exposed; Kubernetes read denied (403) |
| GPU reassignment / HiTSKT CUDA / HiTSKT pilot | **BLOCKED / NOT RUN / NOT RUN**: unrelated GPU workload and no verified exclusive allocation |

The inventory verifies byte preservation and recorded hash consistency. It is
not an independent numerical replay of all completed results or a fresh
offsite-backup audit.

## Interrupted work and recoverable state

`ASSIST2015-GKT-baseline_tuning-42-e01b589e46cc` has two completed epochs.
Best and latest are epoch 2, validation AUC `0.7327774052048939`, bad-epoch
counter 0. Epoch 1/2 took about 614.5/615.8 seconds respectively. No current
training or completion ETA is appropriate while the schedule is on user hold.
Any partial work after epoch 2 is not represented by the saved checkpoint.

| Checkpoint | SHA256 |
| --- | --- |
| `checkpoints/ASSIST2015-GKT-baseline_tuning-42-e01b589e46cc/best.pt` | `a36930ca8a0a5d9e0325405de50dc83c82584408080bf2f0f56cd321fb455608` |
| `checkpoints/ASSIST2015-GKT-baseline_tuning-42-e01b589e46cc/latest.pt` | `1a508b505f07f48491e489d06cb837e660cb6dd5378c7ba33ab8ff61a8a0e551` |

`ASSIST2015-GKT-baseline_tuning-42-e1c69a92d6b5` was waiting for the GPU
admission lock and has no saved epoch. Its configuration and interruption log
are preserved. Neither run was resumed or marked complete.

An additional **local**, content-verified recovery archive is at
`backups/priority_handoff_20261020/interrupted_gkt_epoch2.tar.gz`, 3,869,675
bytes, 1,155 members, SHA256
`6e3b03248307f609059a4bec7ea0ac203cbbb35054360a871b3b5920bf0a45c0`.
It includes both checkpoints, interrupted logs/status/history/configurations,
all configs, scientific source and manifest. It remains ignored in Git; this
commit does not upload model/data binaries or establish offsite durability for
this new archive. Existing checkpoints and completed backups remain in place.

## Persistent hold and verification

`experiments/USER_TRAINING_HOLD.json` records the user decision separately from
immutable experimental configurations. Marker presence fails closed even if its
JSON is malformed. Guards cover `scripts.run`, `schedule`, `coordinator`,
`recover --start/--watch`, GPU admission, post-run launch hooks and `finalize`.
Recovery returns without probing CUDA or launching children. Queued admission
rechecks the hold after acquiring its lock. No watcher was started.

Nine targeted CPU cases verify the actual entrypoints under CUDA and child-launch
traps, plus recovery/admission/hooks and malformed-marker behavior. Final:
`reports/priority_handoff_20261020/hold_cpu_tests_final.log` / `.xml`,
**9 passed in 11.38s**. An initial test harness trapped Python's harmless
dependency-import `uname` subprocess, producing 3 passes/6 failures before any
entrypoint ran; those logs/XML are retained. Corrected import/trap ordering
passed, then private test markers removed dependency on the real operational
hold and the final nine cases passed again. No training or CUDA ran in these
tests. The full scientific test suite was not rerun for control-only changes.

No local live training/controller/recovery process was found. The only project
module in the capture is the one-shot inspector itself. The marker blocks the
known project launch paths; inspection cannot certify invisible external
schedulers or prevent someone deliberately bypassing project entrypoints.
Do not remove it or resume anything without a later explicit user decision.

## GPU blocker

Only parent `GPU-5a568e3c-e96c-169f-b063-f4478340ff51` and exposed
`MIG-d2fcfd79-720e-5218-82c9-0a132be68920` (H100 3g.40gb) were visible.
At initial capture, unrelated PID 2643837, outside both research roots, held
the same NVIDIA capability/device handles as the old LightSAKT2 workers.
It was **not** signalled. Later HiTSKT resource evidence observes a different
unrelated worker group; process identity must always be rechecked.

The HiTSKT snapshot at 06:55:23 UTC records MIG capacity 42,278,584,320 bytes,
9,922,347,008 used and 32,356,237,312 free. Those are transient NVML readings,
not permission or exclusivity. Parent/process queries are restricted; no
Slurm/PBS/LSF assignment is exposed, and Kubernetes RBAC denies the attempted
allocation read. Thus no CUDA initialization or five-step HiTSKT pilot occurred.
Ask the allocation owner/administrator for an exclusive UUID/owner/job/time-window
assignment and clearance of unrelated reservations; do not kill that workload.

## Evidence and continuation

- `reports/priority_handoff_20261020/before_hold/`: initial snapshot, compressed
  SHA256 inventory and copied progress/manifest/controller/recovery metadata.
- `reports/priority_handoff_20261020/after_hold/`: matching final capture.
- `reports/priority_handoff_20261020/preservation_comparison.json`: exact equality.
- `reports/priority_handoff_20261020/interrupted_checkpoint_and_archive.json`:
  CPU recovery verification, archive identity and scope.
- `reports/priority_handoff_20261020/completed_backup_presence.json`:
  existing local archive presence, explicitly not fresh offsite verification.
- `project-memory/`: current hold and exact resume restrictions.
- HiTSKT `/workspace/capstone/reports/favor_plus_v1/priority_handoff_20261020/resource_and_process_evidence.json`:
  sanitized read-only allocation/resource observation, no CUDA context.

Pre-existing dirty manifests, research reports and interrupted outputs were
preserved and excluded from this scoped commit. They must not be reset to Git
HEAD or mistaken for changes introduced by this handoff. Original HiTSKT ELU+1
artifacts and unfinished DKT/EdNet were not modified.
