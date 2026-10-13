# Experiment status

Updated UTC: 2026-10-13T12:30:59.563810+00:00

Completed: **9**. Remaining: **948**. Total schedule slots: 957.

```json
{
  "completed": 9,
  "running": 1,
  "pending": 947
}
```

Scheduler process: running (PID 25426).

Preservation process: running (PID 24074).

Current run: `ASSIST2015-DKVMN-primary-42-c91ed69c5400`; last completed epoch: 0.

The 36 reuse/alias slots are not independent new training experiments. Default ablation reuses final LightSAKT evidence. The separate smoke run is excluded from schedule counts.

The 108 LightSAKT grid configurations, equal baseline tuning budgets, and five matched final seeds use the frozen student splits. See experiments/manifest.json and per-run status files.

Research completion requires every scheduled slot and all dependent analyses, plots, checkpoints, and traceability checks to pass. Raw-data provenance limitations and the historical simplified GKT label remain disclosed.
