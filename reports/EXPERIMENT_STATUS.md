# Experiment status

Updated UTC: 2026-10-14T14:03:03.470201+00:00

Completed: **661**. Remaining: **296**. Total schedule slots: 957.

```json
{
  "completed": 661,
  "pending": 295,
  "running": 1
}
```

Scheduler process: running (PID 78472).

Preservation process: running (PID 212401).

Eta process: running (PID 508657).

Offsite process: running (PID 49211).

Coordinator process: running (PID 191761).

Progressive process: running (PID 185610).

Active_recovery process: running (PID 203682).

Current run: `ASSIST2015-DKVMN-baseline_tuning-42-e8423ad3f7f0`; last completed epoch: 0.

The 45 reuse/alias slots are not independent new training experiments. Default ablation reuses final LightSAKT evidence. The separate smoke run is excluded from schedule counts.

Distinct planned trainings: 912. The live manifest plus experiments/reuse_plan.json form the effective schedule; 9 exact primary/grid duplicates reuse verified evidence without removing grid rows. See reports/SCHEDULE_AUDIT.md.

Measured-reference serial equivalent: 488.7 training hours; scenario envelope 104.7–1,435.2 hours. Current execution forecast: 378.6 occupied MIG wall hours; scenario envelope 85.6–1,435.2 hours, using independently verified two-worker GKT-MHA throughput only. Estimate updated 2026-10-14T14:02:58.900436+00:00; startup, test inference and final verification add time. This is a planning scenario, not a confidence interval or guarantee. See reports/ETA_REPORT.md.

Process existence alone is not a health guarantee. Last independently sampled CPU/checkpoint progress and checkpoint integrity are recorded in reports/EXECUTION_HEALTH.md. A detached scheduler survives a terminal disconnect, but no automatic restart after environment shutdown is configured.

Private draft recovery assets independently download/checksum verified: 663. All available completed archives synchronized: True. See reports/OFFSITE_RECOVERY.md and reports/offsite_backup.json.

The 108 LightSAKT grid configurations, equal baseline tuning budgets, and five matched final seeds use the frozen student splits. See experiments/manifest.json and per-run status files.

Verified existing efficiency evidence: 120 cases and 1200 raw timed repetitions. LightSAKT was slower in 60/60 matched latency comparisons and used lower peak allocated memory in 7/60. Original measurements are preserved; theoretical scaling, latency, memory and parameters are discussed separately in paper/CORRECTED_RESULTS.md and paper/EFFICIENCY_RESULTS.md.

Research completion requires every scheduled slot and all dependent analyses, plots, checkpoints, and traceability checks to pass. Raw-data provenance limitations and the historical simplified GKT label remain disclosed.
