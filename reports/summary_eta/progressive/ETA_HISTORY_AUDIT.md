## Historical ETA audit and runtime categories

Archived corrected-rebuild ETA snapshots were read from Git objects; these are execution estimates, not historical manuscript results. The nearest saved snapshot to each requested completion count is labelled explicitly.

|Requested milestone|Archived completed entries|Updated UTC|Serial equivalent hours|Occupied MIG hours|Git commit|
|---|---|---|---|---|---|
|183|183|2026-10-14T01:52:25.176860+00:00|501.140|391.106|3b53f28278b8237fc2dbd55e711e6e407335c1c1|
|254|255|2026-10-14T02:30:33.484506+00:00|500.585|390.551|1be758a7379d01102f07b54e3434c524f49ed7e4|
|551|551|2026-10-14T05:03:39.739302+00:00|499.323|389.289|ee6996362af625ed7dba9769e34a77b9cb590a5f|
|current|774|2026-10-16T19:44:51.206868+00:00|436.396|334.013|live saved artifacts|

The earlier approximately 391-hour number was a continuously refreshed, model/dataset-weighted forecast of remaining work. It was not a global average multiplied by total pending entries. Its small changes were obscured by rounding and by the fact that short prioritized jobs and evidence-reuse entries contributed little remaining time. The expensive GKT configurations remained untrained; new fast-model timings updated their own strata without removing those costs.

Current GKT contribution: 334.013/334.013 occupied MIG hours (100.00% of the central forecast). Six completed primary GKT dataset/variant runs and real primary/wide engineering profiles still anchor this dominant term. Most future GKT stopping epochs are unknown. The forecast is refreshed rather than stale, but the slow configurations have limited full-run timing evidence.

|Remaining category|Entries|Serial equivalent hours|Occupied MIG hours|
|---|---|---|---|
|baseline_tuning|153|385.621|294.950|
|repeated_seeds|30|50.775|39.063|

Isolated serial training work is 436.396 hours. With the admitted independent GKT-MHA concurrency, occupied-device time is 334.013 MIG hours. On the one available MIG partition, continuous-availability wall-clock training time is therefore 13.92 days, with conditional finish 2026-10-30T17:45:38.092381+00:00. These are separate quantities; actual hardware-active utilization is unavailable.

Planning envelope: 71.8–1234.3 occupied hours. This is an early-stop/maximum-epoch scenario envelope, not a confidence interval or guaranteed bound. The upper scenario does not assume sustained concurrency benefit. Unselected baseline configurations, GKT stopping counts, contention and environment availability dominate uncertainty.

Startup, selected test evaluation and independent final artifact/checkpoint verification add wall time beyond this training-only estimate. Completed LightSAKT grids, selected seeds, ablations, controlled efficiency measurements and trained case figures are retained; they are not scheduled for fresh training again. Remaining numerical graphs and paired statistics are generated progressively on CPU.

Per-category/model/dataset source: ETA_CATEGORY_OCCUPIED_HOURS.csv. Complete archived trajectory: ETA_ARCHIVED_HISTORY.csv. Current forecasts: ETA.json, ETA_REMAINING_RUNS.csv and ETA_MEASURED_TIMINGS.csv. Live scheduler logs: reports/scheduler.log and reports/optimization/coordinator.log; sampled processes/checkpoints: reports/execution_health.json.
