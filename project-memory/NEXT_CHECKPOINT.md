# Exact next checkpoint — completed comparison, metadata gap

No active campaign job. Scope for nine new runs is consumed. Inspect actual
`/workspace/LightSAKT2-architecture/reports/professor_results_20261030/campaign_status.json`
(status completed,9completed/9launched) and branch Git HEAD before any recovery.
Never restart the completed controller or any of its runs. Forty model-seed
checkpoints are verified; five ASSIST2015 difficulty slots are metadata-blocked.

Deliver/review the professor report and tables in
`/workspace/LightSAKT2-architecture/reports/professor_results_20261030/`.
Use existing checkpoint/source/config/feature hashes, archive manifests and
CPU audits as ground truth; no fresh baseline training is needed.
Report generation: python -m scripts.professor_report_20261030.
Read-only audit: CUDA_VISIBLE_DEVICES='' python -m scripts.audit_professor_campaign_20261030 --leakage.
Prediction arrays and checkpoint/features/raw records have local archive backups.
If artifacts are absent, restore verified archives before replay rather than
silently trusting cache JSON or discarding historical records.

ASSIST2015 requires an authoritative mapping to independent question/exercise
identity, aligned to existing log/student records. Validate mapping and absence
of outcome-dependent metadata before defining a frozen experiment. Preserve
current splits and preprocessing; do not substitute skill as question or report
an invented equivalent result. Alternative dataset releases require an explicitly
new, disclosed protocol if exact partitions cannot be maintained.

Next scientifically justified control: already implemented feature-matched SAKT
augmentation, five ASSIST2009 seeds with unchanged native SAKT settings. It is
NOT authorized under consumed campaign scope. Define a separate bounded budget
(e.g. five serial900s windows /4GiB,75GPU-minute ceiling with existing guards)
before execution. The old question_replication_20261029 plan's LightSAKT slots
are now fulfilled; DO NOT repeat them. Verified equivalence mapping is in
older_replication_plan_equivalence.json in the report folder. Only its five SAKT
augmentation controls remain unexecuted.

Do not tune against the released test split. A chronological/deployment study or
new student-cohort partitions would require a separately frozen protocol and
resource scope. Current evidence is retrospective, richer-input, fixed-split
replication; no all-three-dataset or attention superiority claim. Keep all prior
checkpoints, original dirty state, holds, peers and Git history untouched.
