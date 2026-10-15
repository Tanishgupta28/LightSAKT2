# Measured runtime estimate

Updated UTC: 2026-10-15T22:35:33.440364+00:00

Remaining training: **470.3 GPU reserved hours**; scenario range **94.3–1,305.3 hours**. Serial wall time: 19.6 days; continuous-availability training finish 2026-11-04T12:52:41.456013+00:00.

Two-worker execution forecast: **356.4 MIG occupied wall hours** (14.9 continuous days), with measured GKT throughput factor 1.512. Scenario envelope 75.6–1,305.3 hours; the high scenario assumes no sustained concurrency benefit. Serial isolated-equivalent hours differ from MIG occupied wall hours. Two independent GKT-MHA workers only; PAM and other models remain serial. Use the smallest measured MHA throughput factor across tested datasets/widths. Wide PAM measured 0.90x and was rejected for parallel execution. Short-step throughput is extrapolated, not guaranteed. High scenario assumes no sustained benefit. Startup/test/analysis excluded. Aggregate worker elapsed-hours may increase under contention.

This is a provisional measured-reference extrapolation, not a validated confidence interval. Hardware-active GPU utilization is unavailable. Epochs include training and validation, and reserved hours include their CPU work. Startup, final test inference and artifact verification add time beyond this training-only date; completed efficiency benchmarking is reused.

Overall mean completed-run duration: 4.99 minutes, over 713 actual independent completed runs. This global average is reported descriptively and is not multiplied by the remaining count.

Saved cumulative training_seconds include training and validation through checkpoint restoration, excluding subprocess initialization and post-selection test inference. Epoch_seconds directly measure complete train/validation epochs; resumed histories are included once.

## Observed model/dataset averages

|Dataset|Model|Completed runs|Measured epochs|Mean epoch seconds|Mean completed run minutes|
|---|---|---|---|---|---|
|ASSIST2009|DKT|41|778|0.284|0.093|
|ASSIST2009|DKT+|41|1198|0.348|0.175|
|ASSIST2009|DKVMN|40|1570|6.179|4.052|
|ASSIST2009|GKT|36|478|169.363|37.580|
|ASSIST2009|GKT_MHA|4|184|216.022|104.623|
|ASSIST2009|SAKT|41|1061|0.421|0.187|
|ASSIST2009|LightSAKT|60|1726|0.864|0.420|
|ASSIST2015|DKT|41|645|1.167|0.310|
|ASSIST2015|DKT+|41|962|1.416|0.559|
|ASSIST2015|DKVMN|40|1500|22.332|13.976|
|ASSIST2015|GKT|1|14|605.806|141.706|
|ASSIST2015|GKT_MHA|1|23|633.193|243.188|
|ASSIST2015|SAKT|41|885|1.475|0.536|
|ASSIST2015|LightSAKT|60|1331|3.796|1.411|
|Algebra2005|DKT|41|1324|0.280|0.158|
|Algebra2005|DKT+|41|1503|0.354|0.225|
|Algebra2005|DKVMN|40|1852|6.949|5.376|
|Algebra2005|GKT|1|16|200.815|53.731|
|Algebra2005|GKT_MHA|1|48|204.052|163.480|
|Algebra2005|SAKT|41|1541|0.320|0.210|
|Algebra2005|LightSAKT|60|2201|0.977|0.607|

## Remaining phases

|Phase|Low hours|Central hours|High hours|
|---|---|---|---|
|baseline_tuning|84.20|419.84|1,151.76|
|repeated_seeds|10.09|50.45|153.58|

## Longest forecast runs

|Run|Reference|Central hours|Scenario hours|
|---|---|---|---|
|ASSIST2015-GKT_MHA-baseline_tuning-42-1aa2f6f065fe|ASSIST2015-GKT_MHA-primary-42-6a6d36ebb3d6;measured primary/wide engineering anchors|4.71|0.98–12.28|
|ASSIST2015-GKT_MHA-baseline_tuning-42-b53e443cf30e|ASSIST2015-GKT_MHA-primary-42-6a6d36ebb3d6;measured primary/wide engineering anchors|4.71|0.98–12.28|
|ASSIST2015-GKT_MHA-baseline_tuning-42-ebf0bd893198|ASSIST2015-GKT_MHA-primary-42-6a6d36ebb3d6;measured primary/wide engineering anchors|4.71|0.98–12.28|
|ASSIST2015-GKT_MHA-baseline_tuning-42-55ef256295ef|ASSIST2015-GKT_MHA-primary-42-6a6d36ebb3d6;measured primary/wide engineering anchors|4.71|0.98–12.28|
|ASSIST2015-GKT_MHA-baseline_tuning-42-277eb8941533|ASSIST2015-GKT_MHA-primary-42-6a6d36ebb3d6;measured primary/wide engineering anchors|4.71|0.98–12.28|
|ASSIST2015-GKT_MHA-baseline_tuning-42-bf0659f5cb99|ASSIST2015-GKT_MHA-primary-42-6a6d36ebb3d6;measured primary/wide engineering anchors|4.71|0.98–12.28|

## Dependent analysis duration

* benchmark: Exclusive 120-case benchmark status completed; actual execution seconds 25.469461232976755; results/tables/efficiency.csv. Completed benchmarking is reused by final analyses; CPU epoch-runtime documentation is refreshed later.
* statistics: 18 paired comparisons, five seeds; engineering microbenchmark timing is reported below; real paired research results still require final training.
* visualizations: Three final attention maps, two ASSIST2009 case figures; measured plot-render pilot below; final-case CPU inference remains unmeasured.
* verification: About 1.04 serial inference GPU hours from measured primary eval steps for 912 distinct projected states; future selected-config widths, AUC computation and model/cache setup add uncertainty and overhead. All unique checkpoints remain verified; identical aliases reuse inference.

Measured CPU pilot timing (actual completed primary plots; not final research completion): {"measured_utc": "2026-10-13T14:57:20.174838+00:00", "actual_primary_analysis_seconds": 4.55341371500981, "png_figures_existing_after": 11, "includes_png_svg_pdf_and_csv": true, "scope": "Incomplete primary-only genuine data; timing includes startup and summary verification. Final five-seed graphics workload differs.", "png_figures_rendered": 6, "mean_render_seconds_per_figure_including_all_exports": 0.7589022858349684, "eighteen_five_pair_signed_rank_compute_seconds": 0.002166819991543889, "statistical_timing_scope": "Algorithm workload microbenchmark only, synthetic differences; no research p-values or completed paired experiments produced.", "five_final_attention_case_figure_render_only_estimate_seconds": 3.7945114291748423, "render_estimate_limit": "Rendering reference only; future CPU inference, dataset reads, tables and GPU benchmarks are additional and unmeasured."}

Measured complete train/validation epochs and dataset/model-specific stopping counts. GKT epoch costs are calibrated with primary and wide configurations measured on each dataset, using weighted train/validation step ratios and interpolation by width; a minimum 20% runtime scenario spread is applied. Smaller-than-primary widths conservatively retain primary cost. Other models use closest measured configurations and batch-timestep/arithmetic proxies. Six-epoch earliest-stop and 50-epoch maximum scenarios bracket the observed dataset/model mean stopping count. Unknown selections use mean candidate cost, not selection probabilities. FP32 mathematics, batch sizes and stopping rules are unchanged.

All 21 model/dataset primary strata including Algebra2005 have measured complete epochs. GKT primary/wide steps were measured on all three datasets; newly completed full runs update dataset/model stopping counts. Exact independent completed-run counts are tabulated below and stored in measured_gkt_full_runs. Most future configuration stopping epochs remain unknown; timing evidence is still sparse in several GKT strata. Short concurrency probes do not guarantee whole-run throughput. Post-selection test inference, subprocess initialization, final artifact verification and remaining analyses add time; completed exclusive benchmarks are reused.

The preserved equal 36-configuration baseline tuning budget includes 216 GKT variant configurations across all datasets; GKT is the main measured bottleneck. No dimensions, batches, split assignments, seeds, learning rates or early-stopping rules were changed to shorten this forecast.

## Historical ETA audit and runtime categories

Archived corrected-rebuild ETA snapshots were read from Git objects; these are execution estimates, not historical manuscript results. The nearest saved snapshot to each requested completion count is labelled explicitly.

|Requested milestone|Archived completed entries|Updated UTC|Serial equivalent hours|Occupied MIG hours|Git commit|
|---|---|---|---|---|---|
|183|183|2026-10-14T01:52:25.176860+00:00|501.140|391.106|3b53f28278b8237fc2dbd55e711e6e407335c1c1|
|254|255|2026-10-14T02:30:33.484506+00:00|500.585|390.551|1be758a7379d01102f07b54e3434c524f49ed7e4|
|551|551|2026-10-14T05:03:39.739302+00:00|499.323|389.289|ee6996362af625ed7dba9769e34a77b9cb590a5f|
|current|750|2026-10-15T22:35:33.440364+00:00|470.286|356.423|live saved artifacts|

The earlier approximately 391-hour number was a continuously refreshed, model/dataset-weighted forecast of remaining work. It was not a global average multiplied by total pending entries. Its small changes were obscured by rounding and by the fact that short prioritized jobs and evidence-reuse entries contributed little remaining time. The expensive GKT configurations remained untrained; new fast-model timings updated their own strata without removing those costs.

Current GKT contribution: 356.423/356.423 occupied MIG hours (100.00% of the central forecast). Six completed primary GKT dataset/variant runs and real primary/wide engineering profiles still anchor this dominant term. Most future GKT stopping epochs are unknown. The forecast is refreshed rather than stale, but the slow configurations have limited full-run timing evidence.

|Remaining category|Entries|Serial equivalent hours|Occupied MIG hours|
|---|---|---|---|
|baseline_tuning|177|419.837|317.575|
|repeated_seeds|30|50.448|38.847|

Isolated serial training work is 470.286 hours. With the admitted independent GKT-MHA concurrency, occupied-device time is 356.423 MIG hours. On the one available MIG partition, continuous-availability wall-clock training time is therefore 14.85 days, with conditional finish 2026-10-30T19:00:54.934547+00:00. These are separate quantities; actual hardware-active utilization is unavailable.

Planning envelope: 75.6–1305.3 occupied hours. This is an early-stop/maximum-epoch scenario envelope, not a confidence interval or guaranteed bound. The upper scenario does not assume sustained concurrency benefit. Unselected baseline configurations, GKT stopping counts, contention and environment availability dominate uncertainty.

Startup, selected test evaluation and independent final artifact/checkpoint verification add wall time beyond this training-only estimate. Completed LightSAKT grids, selected seeds, ablations, controlled efficiency measurements and trained case figures are retained; they are not scheduled for fresh training again. Remaining numerical graphs and paired statistics are generated progressively on CPU.

Per-category/model/dataset source: ETA_CATEGORY_OCCUPIED_HOURS.csv. Complete archived trajectory: ETA_ARCHIVED_HISTORY.csv. Current forecasts: ETA.json, ETA_REMAINING_RUNS.csv and ETA_MEASURED_TIMINGS.csv. Live scheduler logs: reports/scheduler.log and reports/optimization/coordinator.log; sampled processes/checkpoints: reports/execution_health.json.
