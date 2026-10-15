# Continued research execution verification

Observed UTC: 2026-10-15T03:46:47.226981+00:00

Exact schedule counts: 725 completed, 1 running, 231 pending, 0 failed and 0 interrupted; 957 total entries. Remaining: 224 fresh trainings and 8 reuse/evaluation entries.

Active runs: [{"run_id": "ASSIST2009-GKT-baseline_tuning-42-5abd556796ed", "last_completed_epoch": 3, "best_checkpoint": "checkpoints/ASSIST2009-GKT-baseline_tuning-42-5abd556796ed/best.pt", "resume_command": "python -m scripts.run --config results/raw/ASSIST2009-GKT-baseline_tuning-42-5abd556796ed/config.json"}]. Sampled CPU/epoch progress is healthy: True. Scheduler and coordinator PIDs remain unchanged. No healthy training process was signalled, restarted or modified.

Measured-reference ETA: 363.55 occupied MIG hours (15.15 continuously available days), compared with 473.58 serial-equivalent training hours. Planning envelope: 81.64–1386.09 occupied hours; this is not a confidence interval. Conditional training finish: 2026-10-30T07:18:48.128396+00:00. Startup, test inference, final verification and outages add time. GKT full-run counts: {"ASSIST2009/GKT": 14, "ASSIST2009/GKT_MHA": 1, "ASSIST2015/GKT": 1, "ASSIST2015/GKT_MHA": 1, "Algebra2005/GKT": 1, "Algebra2005/GKT_MHA": 1}.

## Verified final LightSAKT results

|Dataset|LR|Embedding width|Heads|Dropout|Test AUC|Accuracy|F1|MSE|BCE|
|---|---|---|---|---|---|---|---|---|---|
|ASSIST2009|0.001|64|2|0.1|0.751056 ± 0.000766|0.732229 ± 0.000801|0.816533 ± 0.001517|0.180233 ± 0.000319|0.538615 ± 0.000886|
|ASSIST2015|0.0005|128|4|0.1|0.726088 ± 0.000315|0.749526 ± 0.000419|0.846282 ± 0.000392|0.171233 ± 0.000106|0.516098 ± 0.000340|
|Algebra2005|0.001|128|4|0.2|0.810590 ± 0.000422|0.795574 ± 0.000639|0.870356 ± 0.000350|0.144069 ± 0.000364|0.445272 ± 0.001018|

These are arithmetic means ± sample SD (ddof=1), using seeds 42, 123, 2026, 3407 and 7777 on split seed 42. Selection is validation-only among the complete 36 candidates per dataset. All 15 LightSAKT and all 60 completed baseline final-seed checkpoint outputs have independent validation/test inference proofs. GKT-PAM and GKT-MHA final-seed comparisons remain pending; primary single-seed results are separately labelled. Source: LIGHTSAKT_FINAL_SEED_RESULTS.md, BASELINE_COMPARISON.md, results/tables/LightSAKT_final_individual_seeds.csv and selected_final_inference_verification.json.

## All fifteen checkpoint references

|Dataset|Seed|Selected validation epoch|Checkpoint|
|---|---|---|---|
|ASSIST2009|123|29|checkpoints/ASSIST2009-LightSAKT-final-123/best.pt|
|ASSIST2009|2026|26|checkpoints/ASSIST2009-LightSAKT-final-2026/best.pt|
|ASSIST2009|3407|21|checkpoints/ASSIST2009-LightSAKT-final-3407/best.pt|
|ASSIST2009|42|24|checkpoints/ASSIST2009-LightSAKT-grid-42-f08e9494d270/best.pt|
|ASSIST2009|7777|20|checkpoints/ASSIST2009-LightSAKT-final-7777/best.pt|
|ASSIST2015|123|11|checkpoints/ASSIST2015-LightSAKT-final-123/best.pt|
|ASSIST2015|2026|12|checkpoints/ASSIST2015-LightSAKT-final-2026/best.pt|
|ASSIST2015|3407|13|checkpoints/ASSIST2015-LightSAKT-final-3407/best.pt|
|ASSIST2015|42|18|checkpoints/ASSIST2015-LightSAKT-grid-42-64640d03b0d3/best.pt|
|ASSIST2015|7777|13|checkpoints/ASSIST2015-LightSAKT-final-7777/best.pt|
|Algebra2005|123|26|checkpoints/Algebra2005-LightSAKT-final-123/best.pt|
|Algebra2005|2026|33|checkpoints/Algebra2005-LightSAKT-final-2026/best.pt|
|Algebra2005|3407|26|checkpoints/Algebra2005-LightSAKT-final-3407/best.pt|
|Algebra2005|42|26|checkpoints/Algebra2005-LightSAKT-grid-42-06884c8943b2/best.pt|
|Algebra2005|7777|23|checkpoints/Algebra2005-LightSAKT-final-7777/best.pt|

Backup verification: 725 completed scheduled results have 726 unique archives whose local and independently downloaded bytes were rehashed. 728 private offsite assets are download-verified. Current active snapshot references: [{"run_id": "ASSIST2009-GKT-baseline_tuning-42-5abd556796ed", "epoch": 1, "asset_id": 642561810, "release_tag": "research-active-recovery-20261015", "verified_utc": "2026-10-15T03:41:34.741060+00:00", "sha256": "296e63063828f57b1ae978d17041aedc4988ae3fb86ab6ac1d811674239cce71"}].

All 108 LightSAKT grid rows, all 75 ablation slots and 72 associated CSV/PNG/SVG/PDF artifacts remain verified. Completed efficiency evidence is retained without rerunning: LightSAKT is slower in all 60 matched latency comparisons. Training source fingerprint, architecture, gradients, student-disjoint split hashes and scientific scope remain unchanged.

Engineering/reporting correction: removed obsolete ETA prose claiming only one completed GKT run per stratum; the observer now records exact counts. Numeric ETA computation was unchanged. Only the CPU ETA observer was refreshed, as recorded in eta_observer_refresh.json. All seven existing ETA/history/final-summary observer tests passed; the unchanged core retains the 110-pass CPU suite and previous exclusive CUDA gate evidence.

Worker recovery, backups, progressive figures and reporting remain detached and active. Provider/environment replacement cannot start itself with current permissions; use the tested commands in RECOVERY_PROCEDURE.md. The research project is incomplete. Current status, ETA, corrected results and final research reports are regenerated from saved artifacts; no historical manuscript metrics or pending runs supply final evidence.
