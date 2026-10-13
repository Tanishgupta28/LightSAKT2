# Experiment status

Updated UTC: 2026-10-13T23:24:31.369734+00:00

Completed: **92**. Remaining: **865**. Total schedule slots: 957.

```json
{
  "completed": 92,
  "running": 1,
  "pending": 864
}
```

Scheduler process: running (PID 78472).

Preservation process: running (PID 51530).

Eta process: running (PID 51531).

Offsite process: running (PID 49211).

Current run: `ASSIST2009-DKTplus-baseline_tuning-42-0bc7596d9685`; last completed epoch: 3.

The 45 reuse/alias slots are not independent new training experiments. Default ablation reuses final LightSAKT evidence. The separate smoke run is excluded from schedule counts.

Distinct planned trainings: 912. The live manifest plus experiments/reuse_plan.json form the effective schedule; 9 exact primary/grid duplicates reuse verified evidence without removing grid rows. See reports/SCHEDULE_AUDIT.md.

Measured-reference training ETA: 746.5 GPU reserved hours; scenario envelope 107.1–4,101.4 hours. Estimate updated 2026-10-13T23:23:33.530116+00:00; unmeasured final GPU analysis adds time. See reports/ETA_REPORT.md.

Process existence alone is not a health guarantee. Last independently sampled CPU/checkpoint progress and checkpoint integrity are recorded in reports/EXECUTION_HEALTH.md. A detached scheduler survives a terminal disconnect, but no automatic restart after environment shutdown is configured.

Private draft recovery assets independently download/checksum verified: 82. All available completed archives synchronized: True. See reports/OFFSITE_RECOVERY.md and reports/offsite_backup.json.

The 108 LightSAKT grid configurations, equal baseline tuning budgets, and five matched final seeds use the frozen student splits. See experiments/manifest.json and per-run status files.

Research completion requires every scheduled slot and all dependent analyses, plots, checkpoints, and traceability checks to pass. Raw-data provenance limitations and the historical simplified GKT label remain disclosed.
