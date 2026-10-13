# Measured runtime estimate

Updated UTC: 2026-10-13T23:45:58.432397+00:00

Remaining training: **749.3 GPU reserved hours**; scenario range **107.1–4,107.6 hours**. Serial wall time: 31.2 days; continuous-availability training finish 2026-11-14T05:03:31.075420+00:00.

This is a provisional measured-reference extrapolation, not a validated confidence interval. Hardware-active GPU utilization is unavailable. Epochs include training and validation, and reserved hours include their CPU work. Unmeasured final benchmarking and artifact verification add time beyond this training-only date.

Overall mean completed-run duration: 7.83 minutes, over 97 actual independent completed runs. This global average is reported descriptively and is not multiplied by the remaining count.

Saved cumulative training_seconds include training and validation through checkpoint restoration, excluding subprocess initialization and post-selection test inference. Epoch_seconds directly measure complete train/validation epochs; resumed histories are included once.

## Observed model/dataset averages

|Dataset|Model|Completed runs|Measured epochs|Mean epoch seconds|Mean completed run minutes|
|---|---|---|---|---|---|
|ASSIST2009|DKT|37|706|0.273|0.090|
|ASSIST2009|DKT+|37|1061|0.342|0.168|
|ASSIST2009|DKVMN|5|252|6.149|5.114|
|ASSIST2009|GKT|1|13|168.209|36.544|
|ASSIST2009|GKT_MHA|1|21|175.622|61.598|
|ASSIST2009|SAKT|1|21|0.395|0.143|
|ASSIST2009|LightSAKT|1|11|1.596|0.296|
|ASSIST2015|DKT|1|15|0.994|0.253|
|ASSIST2015|DKT+|1|18|1.271|0.386|
|ASSIST2015|DKVMN|1|36|23.941|14.382|
|ASSIST2015|GKT|1|14|605.806|141.706|
|ASSIST2015|GKT_MHA|1|23|633.193|243.188|
|ASSIST2015|SAKT|1|15|1.267|0.321|
|ASSIST2015|LightSAKT|1|12|5.794|1.164|
|Algebra2005|DKT|1|27|0.285|0.138|
|Algebra2005|DKT+|1|33|0.322|0.185|
|Algebra2005|DKVMN|1|50|7.359|6.148|
|Algebra2005|GKT|1|16|200.815|53.731|
|Algebra2005|GKT_MHA|1|48|204.052|163.480|
|Algebra2005|SAKT|1|31|0.293|0.164|
|Algebra2005|LightSAKT|1|22|0.674|0.253|

## Remaining phases

|Phase|Low hours|Central hours|High hours|
|---|---|---|---|
|LightSAKT_grid|0.20|0.71|4.24|
|ablations|0.02|0.37|2.83|
|baseline_tuning|102.30|672.80|3,273.25|
|repeated_seeds|4.59|75.41|827.30|

## Longest forecast runs

|Run|Reference|Central hours|Scenario hours|
|---|---|---|---|
|ASSIST2015-GKT_MHA-baseline_tuning-42-55ef256295ef|ASSIST2015-GKT_MHA-primary-42-6a6d36ebb3d6|11.52|1.06–40.03|
|ASSIST2015-GKT_MHA-baseline_tuning-42-40031d9f8cb4|ASSIST2015-GKT_MHA-primary-42-6a6d36ebb3d6|11.52|1.06–40.03|
|ASSIST2015-GKT_MHA-baseline_tuning-42-333c8f50dc55|ASSIST2015-GKT_MHA-primary-42-6a6d36ebb3d6|11.52|1.06–40.03|
|ASSIST2015-GKT_MHA-baseline_tuning-42-ebf0bd893198|ASSIST2015-GKT_MHA-primary-42-6a6d36ebb3d6|11.52|1.06–39.99|
|ASSIST2015-GKT_MHA-baseline_tuning-42-5d49bd1029f9|ASSIST2015-GKT_MHA-primary-42-6a6d36ebb3d6|11.52|1.06–39.99|
|ASSIST2015-GKT_MHA-baseline_tuning-42-6e76f8e10a56|ASSIST2015-GKT_MHA-primary-42-6a6d36ebb3d6|11.52|1.06–39.99|

## Dependent analysis duration

* benchmark: 120 cases, 3 warmups plus 10 repeats each. GPU duration not measured during active training; excluded from training ETA.
* statistics: 18 paired comparisons, five seeds; engineering microbenchmark timing is reported below; real paired research results still require final training.
* visualizations: Three final attention maps, two ASSIST2009 case figures; measured plot-render pilot below; final-case CPU inference remains unmeasured.
* verification: Fresh checkpoint inference for each unique source; redundant aliases will use cached inference. Duration not separately measured.

Measured CPU pilot timing (actual completed primary plots; not final research completion): {"measured_utc": "2026-10-13T14:57:20.174838+00:00", "actual_primary_analysis_seconds": 4.55341371500981, "png_figures_existing_after": 11, "includes_png_svg_pdf_and_csv": true, "scope": "Incomplete primary-only genuine data; timing includes startup and summary verification. Final five-seed graphics workload differs.", "png_figures_rendered": 6, "mean_render_seconds_per_figure_including_all_exports": 0.7589022858349684, "eighteen_five_pair_signed_rank_compute_seconds": 0.002166819991543889, "statistical_timing_scope": "Algorithm workload microbenchmark only, synthetic differences; no research p-values or completed paired experiments produced.", "five_final_attention_case_figure_render_only_estimate_seconds": 3.7945114291748423, "render_estimate_limit": "Rendering reference only; future CPU inference, dataset reads, tables and GPU benchmarks are additional and unmeasured."}

Per-model measured epochs; matching dataset references where available. Extrapolate by actual padded batch-timesteps and arithmetic proxies for nodes, widths and sequence lengths. Central is geometric mean of launch and arithmetic ratios. Six-epoch earliest-stop versus 50-epoch maximum scenarios; central epoch counts are completed-run model means. Unknown selected configs use the unweighted mean of candidate runtime estimates, not a claim about selection probabilities.

No Algebra timings yet; most model/dataset/config strata have zero or one completed observation. GKT hidden-width performance and future stopping epochs are unmeasured. Post-selection test inference, subprocess initialization, final checkpoint verification and exclusive GPU benchmarks remain additional work.

The preserved equal 36-configuration baseline tuning budget includes 216 GKT variant configurations across all datasets; GKT is the main measured bottleneck. No dimensions, batches, split assignments, seeds, learning rates or early-stopping rules were changed to shorten this forecast.
