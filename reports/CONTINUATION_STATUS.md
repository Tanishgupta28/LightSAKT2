# Continued baseline execution verification

Observed UTC: 2026-10-15T00:54:01.359450+00:00

Completed 720/957; remaining 237. The approved complete schedule, original architecture, locked splits and selection rules remain unchanged. Healthy GPU workers, scheduler and coordinator were not signalled or restarted.

Actual recent logs forecast 365.45 occupied MIG hours (15.23 days of continuous availability), versus 475.49 isolated-equivalent serial hours. Planning envelope: 82.84–1400.04 occupied hours; this is not a confidence interval. Future GKT stopping epochs and sustained MHA concurrency are unknown; startup, test evaluation, final analysis and outages add wall time.

## Verified five-seed LightSAKT evidence

|Dataset|Learning rate|Width|Heads|Dropout|AUC|Accuracy|F1|MSE|BCE|
|---|---|---|---|---|---|---|---|---|---|
|ASSIST2009|0.001|64|2|0.1|0.751056 ± 0.000766|0.732229 ± 0.000801|0.816533 ± 0.001517|0.180233 ± 0.000319|0.538615 ± 0.000886|
|ASSIST2015|0.0005|128|4|0.1|0.726088 ± 0.000315|0.749526 ± 0.000419|0.846282 ± 0.000392|0.171233 ± 0.000106|0.516098 ± 0.000340|
|Algebra2005|0.001|128|4|0.2|0.810590 ± 0.000422|0.795574 ± 0.000639|0.870356 ± 0.000350|0.144069 ± 0.000364|0.445272 ± 0.001018|

The arithmetic means and sample standard deviations use seeds 42, 123, 2026, 3407 and 7777, split seed 42. All 15 LightSAKT and all 60 completed baseline final-seed checkpoint validation/test outputs are independently reproduced. Hyperparameter selection uses validation AUC only. Sources: LIGHTSAKT_FINAL_SEED_RESULTS.md, BASELINE_COMPARISON.md and selected_final_inference_verification.json. GKT-PAM/MHA matched final groups remain pending; manuscript values are never substituted.

The complete 108-row LightSAKT grid and 75 ablation slots, including 15 equivalent aliases, have verified tables and 72 CSV/PNG/SVG/PDF plot artifacts. The original efficiency benchmark is reused unchanged: LightSAKT is slower in all 60 matched latency comparisons; theoretical linear attention work, latency, memory and parameters remain separately reported.

Checkpoint/backup verification: 720 scheduled result references, 721 unique archives with both local and independently downloaded bytes SHA256 verified. Active backup evidence: `[{"run_id": "ASSIST2009-GKT-baseline_tuning-42-30db8df014cb", "last_completed_epoch": 14, "latest_verified_offsite_epoch": 13, "offsite_verified_utc": "2026-10-15T00:52:36.673486+00:00", "asset_id": 642196935, "snapshot_bytes_rechecked": true}]`. The latest recorded upload failure has a newer successful download-verified snapshot: True.

The CPU suite log `reports/test_runs/20261015T005034-continued-baseline-cpu.log` records 110 passes and 15 CUDA deferrals; no GPU correctness gate was rerun over active training. CPU recovery supervisor remains active and respects live locks. Platform replacement/shutdown auto-restart is not configured; the tested manual recovery commands are in RECOVERY_PROCEDURE.md.

All available tables, matched statistics and figures were regenerated from completed evidence. paper/CORRECTED_RESULTS.md, reports/EXPERIMENT_STATUS.md, reports/ETA_REPORT.md and FINAL_RESEARCH_REPORT.md remain progressive reports, not a declaration of research completion. GitHub push and bundle verification for each milestone are recorded in github_sync.json and git_backup.json.

Actual completed GKT full-run counts and epoch averages by dataset/variant are saved in continuation_verification.json from ETA_MEASURED_TIMINGS.csv. New ASSIST2009 PAM timings update the stopping-count estimate; other GKT strata still have sparse full-run evidence, so their primary/wide execution measurements remain forecast anchors.
