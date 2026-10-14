# Measured runtime estimate

Updated UTC: 2026-10-14T02:50:08.241613+00:00

Remaining training: **500.3 GPU reserved hours**; scenario range **106.1–1,459.1 hours**. Serial wall time: 20.8 days; continuous-availability training finish 2026-11-03T23:08:05.155023+00:00.

Two-worker execution forecast: **390.3 MIG occupied wall hours** (16.3 continuous days), with measured GKT throughput factor 1.512. Scenario envelope 87.0–1,459.1 hours; the high scenario assumes no sustained concurrency benefit. Serial isolated-equivalent hours differ from MIG occupied wall hours. Two independent GKT-MHA workers only; PAM and other models remain serial. Use the smallest measured MHA throughput factor across tested datasets/widths. Wide PAM measured 0.90x and was rejected for parallel execution. Short-step throughput is extrapolated, not guaranteed. High scenario assumes no sustained benefit. Startup/test/analysis excluded. Aggregate worker elapsed-hours may increase under contention.

This is a provisional measured-reference extrapolation, not a validated confidence interval. Hardware-active GPU utilization is unavailable. Epochs include training and validation, and reserved hours include their CPU work. Unmeasured final benchmarking and artifact verification add time beyond this training-only date.

Overall mean completed-run duration: 3.55 minutes, over 259 actual independent completed runs. This global average is reported descriptively and is not multiplied by the remaining count.

Saved cumulative training_seconds include training and validation through checkpoint restoration, excluding subprocess initialization and post-selection test inference. Epoch_seconds directly measure complete train/validation epochs; resumed histories are included once.

## Observed model/dataset averages

|Dataset|Model|Completed runs|Measured epochs|Mean epoch seconds|Mean completed run minutes|
|---|---|---|---|---|---|
|ASSIST2009|DKT|37|706|0.273|0.090|
|ASSIST2009|DKT+|37|1061|0.342|0.168|
|ASSIST2009|DKVMN|15|733|6.135|5.007|
|ASSIST2009|GKT|1|13|168.209|36.544|
|ASSIST2009|GKT_MHA|1|21|175.622|61.598|
|ASSIST2009|SAKT|1|21|0.395|0.143|
|ASSIST2009|LightSAKT|60|1726|0.864|0.420|
|ASSIST2015|DKT|1|15|0.994|0.253|
|ASSIST2015|DKT+|1|18|1.271|0.386|
|ASSIST2015|DKVMN|1|36|23.941|14.382|
|ASSIST2015|GKT|1|14|605.806|141.706|
|ASSIST2015|GKT_MHA|1|23|633.193|243.188|
|ASSIST2015|SAKT|1|15|1.267|0.321|
|ASSIST2015|LightSAKT|55|1239|3.363|1.196|
|Algebra2005|DKT|1|27|0.285|0.138|
|Algebra2005|DKT+|1|33|0.322|0.185|
|Algebra2005|DKVMN|1|50|7.359|6.148|
|Algebra2005|GKT|1|16|200.815|53.731|
|Algebra2005|GKT_MHA|1|48|204.052|163.480|
|Algebra2005|SAKT|1|31|0.293|0.164|
|Algebra2005|LightSAKT|40|1533|0.785|0.511|

## Remaining phases

|Phase|Low hours|Central hours|High hours|
|---|---|---|---|
|ablations|0.04|0.30|1.00|
|baseline_tuning|95.88|449.44|1,298.95|
|repeated_seeds|10.22|50.56|159.15|

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

All 21 model/dataset primary strata including Algebra2005 have measured complete epochs. GKT primary/wide steps were measured on all three datasets; most future configuration stopping counts remain unknown, with only one completed GKT run per stratum. Short concurrency probes do not guarantee whole-run throughput. Post-selection test inference, subprocess initialization, final artifact verification and exclusive GPU benchmarks add time.

The preserved equal 36-configuration baseline tuning budget includes 216 GKT variant configurations across all datasets; GKT is the main measured bottleneck. No dimensions, batches, split assignments, seeds, learning rates or early-stopping rules were changed to shorten this forecast.
