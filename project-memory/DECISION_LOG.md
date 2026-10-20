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
