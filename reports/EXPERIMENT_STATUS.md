# Experiment status

Updated UTC: 2026-10-13T23:29:53.567384+00:00

Completed: **94**. Remaining: **863**. Total schedule slots: 957.

```json
{
  "completed": 94,
  "running": 1,
  "pending": 862
}
```

Scheduler process: running (PID 78472).

Preservation process: running (PID 51530).

Eta process: running (PID 51531).

Offsite process: running (PID 49211).

Current run: `ASSIST2009-DKVMN-baseline_tuning-42-5e4b7351be03`; last completed epoch: 0.

The 45 reuse/alias slots are not independent new training experiments. Default ablation reuses final LightSAKT evidence. The separate smoke run is excluded from schedule counts.

Distinct planned trainings: 912. The live manifest plus experiments/reuse_plan.json form the effective schedule; 9 exact primary/grid duplicates reuse verified evidence without removing grid rows. See reports/SCHEDULE_AUDIT.md.

Measured-reference training ETA: 748.2 GPU reserved hours; scenario envelope 107.1–4,108.1 hours. Estimate updated 2026-10-13T23:29:40.367449+00:00; unmeasured final GPU analysis adds time. See reports/ETA_REPORT.md.

Process existence alone is not a health guarantee. Last independently sampled CPU/checkpoint progress and checkpoint integrity are recorded in reports/EXECUTION_HEALTH.md. A detached scheduler survives a terminal disconnect, but no automatic restart after environment shutdown is configured.

Private draft recovery assets independently download/checksum verified: 96. All available completed archives synchronized: True. See reports/OFFSITE_RECOVERY.md and reports/offsite_backup.json.

The 108 LightSAKT grid configurations, equal baseline tuning budgets, and five matched final seeds use the frozen student splits. See experiments/manifest.json and per-run status files.

Research completion requires every scheduled slot and all dependent analyses, plots, checkpoints, and traceability checks to pass. Raw-data provenance limitations and the historical simplified GKT label remain disclosed.
