# Experiment status

Updated UTC: 2026-10-18T23:35:46.614529+00:00

Completed: **804**. Remaining: **153**. Total schedule slots: 957.

```json
{
  "completed": 804,
  "pending": 152,
  "running": 1
}
```

Scheduler process: running (PID 78472).

Preservation process: running (PID 212401).

Eta process: running (PID 986001).

Offsite process: running (PID 1722517).

Coordinator process: running (PID 191761).

Progressive process: running (PID 185610).

Active_recovery process: running (PID 818712).

Recovery_supervisor process: running (PID 818850).

Current run: `ASSIST2015-GKT-baseline_tuning-42-af210e4205b8`; last completed epoch: 0.

The 45 reuse/alias slots are not independent new training experiments. Default ablation reuses final LightSAKT evidence. The separate smoke run is excluded from schedule counts.

Distinct planned trainings: 912. The live manifest plus experiments/reuse_plan.json form the effective schedule; 9 exact primary/grid duplicates reuse verified evidence without removing grid rows. See reports/SCHEDULE_AUDIT.md.

Measured-reference serial equivalent: 367.2 training hours; scenario envelope 71.5–969.7 hours. Current execution forecast: 269.9 occupied MIG wall hours; scenario envelope 55.4–969.7 hours, using independently verified two-worker GKT-MHA throughput only. Estimate updated 2026-10-18T23:34:55.525269+00:00; startup, test inference and final verification add time. This is a planning scenario, not a confidence interval or guarantee. See reports/ETA_REPORT.md.

Process existence alone is not a health guarantee. Last independently sampled CPU/checkpoint progress and checkpoint integrity are recorded in reports/EXECUTION_HEALTH.md. A detached scheduler survives a terminal disconnect, but no automatic restart after environment shutdown is configured.

Private draft recovery assets independently download/checksum verified: 790. All available completed archives synchronized: True. See reports/OFFSITE_RECOVERY.md and reports/offsite_backup.json.

The 108 LightSAKT grid configurations, equal baseline tuning budgets, and five matched final seeds use the frozen student splits. See experiments/manifest.json and per-run status files.

Verified existing efficiency evidence: 120 cases and 1200 raw timed repetitions. LightSAKT was slower in 60/60 matched latency comparisons and used lower peak allocated memory in 7/60. Original measurements are preserved; theoretical scaling, latency, memory and parameters are discussed separately in paper/CORRECTED_RESULTS.md and paper/EFFICIENCY_RESULTS.md.

Research completion requires every scheduled slot and all dependent analyses, plots, checkpoints, and traceability checks to pass. Raw-data provenance limitations and the historical simplified GKT label remain disclosed.
