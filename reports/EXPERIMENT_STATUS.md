# Experiment status

Updated UTC: 2026-10-13T15:09:15.734900+00:00

Completed: **12**. Remaining: **945**. Total schedule slots: 957.

```json
{
  "completed": 12,
  "running": 1,
  "pending": 944
}
```

Scheduler process: running (PID 32646).

Preservation process: running (PID 42683).

Current run: `ASSIST2015-LightSAKT-primary-42-301d9bff60e9`; last completed epoch: 3.

The 45 reuse/alias slots are not independent new training experiments. Default ablation reuses final LightSAKT evidence. The separate smoke run is excluded from schedule counts.

Distinct planned trainings: 912. The live manifest plus experiments/reuse_plan.json form the effective schedule; 9 exact primary/grid duplicates reuse verified evidence without removing grid rows. See reports/SCHEDULE_AUDIT.md.

Measured-reference training ETA: 664.5 GPU reserved hours; scenario envelope 110.0–5,583.0 hours. Estimate updated 2026-10-13T15:08:22.356410+00:00; unmeasured final GPU analysis adds time. See reports/ETA_REPORT.md.

Process existence alone is not a health guarantee. Last independently sampled CPU/checkpoint progress and checkpoint integrity are recorded in reports/EXECUTION_HEALTH.md. A detached scheduler survives a terminal disconnect, but no automatic restart after environment shutdown is configured.

The 108 LightSAKT grid configurations, equal baseline tuning budgets, and five matched final seeds use the frozen student splits. See experiments/manifest.json and per-run status files.

Research completion requires every scheduled slot and all dependent analyses, plots, checkpoints, and traceability checks to pass. Raw-data provenance limitations and the historical simplified GKT label remain disclosed.
