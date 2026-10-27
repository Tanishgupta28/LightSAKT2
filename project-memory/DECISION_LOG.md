# Decision log

## 2026-10-20 — Preserve stopped workload and hold automatic restart

New user priority permits stopping verified LightSAKT2 only. Direct OS state and
logs supersede old active-heartbeat interpretation: both workers had already
interrupted. No further signal warranted. Full before/after digest inventory and
strict epoch2 CPU recovery check pass. Added marker and operational launch gates,
preserving all scientific source/config/results and pending schedule.

9 final CPU hold tests pass; initial import-trap failures retained as harness
failures, not hidden. No GPU probing by these tests. A separate local archive
preserves the interrupted epoch boundary; no claim of fresh offsite durability.

Unrelated GPU job observed on same device capabilities, and no allocation
authority available. Do not infer exclusivity from stopped LightSAKT2 or free
VRAM. HiTSKT CUDA/five-step pilot remains unrun pending allocation evidence.
External scheduling visibility is limited; known local recovery/entrypoints
refuse the hold. No watcher, automatic resume or new training authorized.


## 2026-10-20 — Allocation-first user clarification

User has no administrator/scheduler exclusivity evidence. Both projects are
owned by user, but this does not override other reservations. Preserve all
LightSAKT2 checkpoints/results; no stopping solely for GPU release before HiTSKT
assignment. No CUDA/training without verified authorization. Supersedes earlier
shutdown-first order, without requesting any restart or hold release. Current
work limited to memory and administrator-request documentation; no fresh process/
GPU probes or operational changes. Prior stopped status/zero signals preserved
as historical evidence. Next task is obtaining authoritative assignment details.


## 2026-10-20,09:24UTC — Renewed shutdown authorization, existing-allocation check

User again permits verified LightSAKT2 shutdown/prevention of restart and actual
existing-reservation reuse for bounded HiTSKT stages; no unnecessary new booking.
One read-only check finds no live LightSAKT2 trainer/controller/GPU handles; hold
intact,813 completed/142 pending/2 interrupted. Zero signals needed or sent.
7915 files present/same size,8 key hashes and recovery archive match; no repeated
full audit/tests. MIG now nearly empty, zero visible GPU owners, superseding older
busy snapshots only at observation time. Still no existing owner/account/project/
transfer/window/no-conflict reservation evidence, Kubernetes403/no scheduler.
Stop before CUDA. Specific blocker is existing allocation permissions, not new
user pilot authorization or mandatory separate hardware. No CUDA/optimizer/test
metrics, model/data/DKT edits, watcher or allocation change. Fresh report/evidence
under reports/favor_plus_v1/EXISTING_ALLOCATION_REUSE_STATUS_20261020.md and
existing_allocation_reuse_20261020_v1/inspection.json in HiTSKT. Preserve all
original artifacts and scoped branch publication; later full training needs approval.


## 2026-10-27 — Architecture recovery, memory installation and bounded preparation

Recovered research commit6beced52 from missing /tmp worktree into persistent
/workspace/LightSAKT2-architecture without resetting main or pruning history.
Verified depth archive and all22 recorded member hashes; restored14 immutable
files. Baseline0.7508035371114405 versus depth0.7507909322800659 is a valid
single-seed null result,29 epochs each/selected24. Softmax preparation survives;
no valid completed run survives. Unknown lost execution is not claimed absent.
Hash-bound summary checkpoint relocation implemented and80 CPU checks pass.
Original models/trainer, archived metrics and holds preserved. Initial pytest
setup errors and overly broad trainer-equality assertion were corrected and
recorded in the recovery report. No GPU training or test evaluation.

Installed reviewed durable-project-memory upstream0a06f1bb8231ace20e4405281a263012fedb5e51
through standard installer. Resume/Update/Audit use canonical existing memory.
New user research scope supersedes old documentation-only HiTSKT priority,
while retaining explicit GPU budget boundaries. Next: current budget/window for
one900s/4GiB seed42 softmax pilot, then validation-only replay/comparison.
Full evidence: /workspace/LightSAKT2-architecture/reports/architecture_study/RECOVERY_20261027.md.
No push or external message is authorized by this checkpoint.


## 2026-10-27 — Authorized softmax completed; valid negative result

User explicitly authorized exactly one ASSIST2009 seed42 causal-softmax run,
4GiB/900s cap and shared safeguards, plus safe commit/push. No valid prior softmax
artifacts existed; existing controls and recovered depth checkpoints passed
preflight. Fresh run from b63ec478 completed21epochs, selected16:
AUC0.7456393788778357/F1 0.8125468632841789/accuracy0.7268356643356644/
BCE0.5465822575284047, versus original AUC0.7508035371114405.
Delta -0.005164158233604832: retain original; no softmax promotion.
Runtime161.204s, allocated133.972MiB/reserved160MiB. CPU validation replay,
finite full resume state and original-file preservation checks PASS.
No test evaluation; unchanged general holds; grant closed; no retry or peer signals.
Archived best/latest and raw records persist outside /tmp. Report and hashes:
/workspace/LightSAKT2-architecture/reports/architecture_study/softmax_pilot/RESULT_20261027.md and recovery_archive_20261027.json.
Next prepared hypothesis: recency bias, separately authorized and single-arm
runner needed to avoid repeating completed softmax. No recency run in this task.


## 2026-10-27 — Verified result committed; exact-history push blocked

Result commit f15b28b9a30117a9cfdcea677f398ba2b0967713. Non-force push failed because shell Git has no
HTTPS credential; gh logged out/no SSH key. Connected GitHub app profile works,
but exposed commit/tree APIs cannot transfer exact local commits. No remote
mutation or reconstructed history. Preserved a verified local Git bundle at
/workspace/LightSAKT2/backups/architecture_softmax_20261027/research_commits.bundle; checkpoint archive remains separate. Publish existing branch with
ordinary push when authenticated Git transport is available; no training retry.


## 2026-10-27 — Single recency pilot completed; original still best

User authorized exactly one bounded recency-softmax ASSIST2009 seed42 run.
Added single-recency launch stage without scientific config/model/trainer edits;
82 CPU tests PASS. Three prior arms verified and no prior recency artifacts.
Execution from27e5a486 completed21epochs,best16: AUC0.7482377884727862,
F1 0.8134333940013463,accuracy0.7274548368298368,BCE0.544185487825003.
Delta-baseline -0.0025657486386543304,delta-softmax +0.0025984095949505015.
Valid one-seed negative against original; no overall improvement or automatic
five-seed promotion. CPU replay/full resume/116 historical file hashes PASS.
Learned head decays0.0381679125/0.0389851369 versus initial0.03125.
Runtime167.035566s,allocated134.050MiB,reserved160MiB. No test use or peer signals;
shared headroom/cap/pacing respected; exact grant closed_completed, no retry.
Archived checkpoints/raw records and epoch metrics: /workspace/LightSAKT2-architecture/reports/architecture_study/recency_pilot/RESULT_20261027.md.
Next hypothesis after source/literature review: normalized recency decay inside
ELU+1 prefix attention. AKT/RetNet motivate decay but do not establish this
transplant's performance. No new arm implementation/training performed.
Missing shell Git authentication remains; preserve all local commits.
