# Bounded LightSAKT depth execution authorization

The current user explicitly states: “I authorize lifting the project-specific
training hold for this bounded experiment, subject to the existing shared-GPU
permissions and resource safeguards.” The requested execution is fresh one-layer
versus two-layer LightSAKT on ASSIST2009, one common seed, 4 GiB allocation cap,
15-minute limit, validation only, with resumable checkpoints. The user forbids
interrupting other GPU jobs or overwriting original results.

Existing intentional shared-H100 permission is documented in
`/workspace/capstone/reports/favor_plus_v1/shared_gpu_v1/USER_SHARED_AUTHORIZATION.md`
and remains affirmed in `/workspace/capstone/project-memory/USER_DIRECTIVES.md`.
It does not require exclusive ownership or complete host PID mapping. The current
user supplies the new LightSAKT execution scope; no HiTSKT optimizer allowance or
expired exclusive allocation record is reused as LightSAKT training permission.

The exposed existing research device is
`MIG-d2fcfd79-720e-5218-82c9-0a132be68920`. Read-only NVML inspection found
42,238,803,968 free bytes and no reported compute processes at 07:16 UTC.
MIG utilization reports unsupported (NVML status 3), not zero load; the existing
headroom/cap/pacing fallback applies. Namespace PID attribution remains unresolved.

The general hold markers stay present and byte-identical so automatic schedulers,
recovery workers and unrelated experiments stay blocked. A narrowly scoped
exception releases only the two exact depth-stage configuration hashes and the
two current hold-marker hashes. It expires with the local bounded execution
window. Any marker/config change, expiry or missing grant blocks the pilot.
This is the user-authorized bounded hold release, not a global restart.

The execution record's time window is an operator ceiling for this invocation,
not a claimed administrator lease or scheduler reservation. Existing shared
permission plus the present user instruction are its permission evidence. We
retain a 4 GiB allocator cap, 6 GiB runtime free floor, 10 GiB admission floor,
4 GiB host RSS limit, 2 GiB host headroom, 0.1-second inter-batch pacing and
30-second step/time-between-batches stop. The original batch size, model/data,
optimizer, learning rate, seed and 50-epoch/patience-5 budget stay unchanged.

No other process, device allocation, MPS setting or scheduler is changed. No
attention-variant execution or five-seed campaign is authorized by this bounded
depth launch. No actual GPU work began before the scoped-release tests passed.
