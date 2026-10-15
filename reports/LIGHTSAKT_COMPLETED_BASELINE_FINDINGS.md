# LightSAKT against completed baseline evidence

Generated UTC: 2026-10-15T15:51:39.582076+00:00

This comparison includes only completed selected five-seed DKT, DKT+, DKVMN and SAKT results on the same locked splits. It excludes historical manuscript scores, primary single-seed runs, partial seed sets and both GKT variants. Best means highest completed five-seed mean test AUC among these four baselines. This is a descriptive leaderboard after evaluation; it does not select hyperparameters, checkpoints or future experiments.

|dataset|completed_baselines|best_completed_baseline|light_auc_mean|baseline_auc_mean|light_minus_best_auc|
|---|---|---|---|---|---|
|ASSIST2009|4|DKT|0.751056|0.757165|-0.006109|
|ASSIST2015|4|DKT|0.726088|0.728235|-0.002146|
|Algebra2005|4|DKT+|0.810590|0.823456|-0.012866|

ASSIST2009: LightSAKT mean test AUC is lower than the best completed baseline, DKT; difference -0.006109.
ASSIST2015: LightSAKT mean test AUC is lower than the best completed baseline, DKT; difference -0.002146.
Algebra2005: LightSAKT mean test AUC is lower than the best completed baseline, DKT+; difference -0.012866.

Each reported model has five prespecified seeds, and SD means sample SD (ddof=1). Full baseline AUC, accuracy, F1, MSE and BCE means/SDs are in BASELINE_COMPARISON.md. All 15 LightSAKT seed values, validation-selected epochs and checkpoint references are in LIGHTSAKT_FINAL_SEED_RESULTS.md. Small positive differences against individual models are descriptive, not established statistical superiority. Five nonzero pairs permit a minimum two-sided exact signed-rank p-value of 0.0625; multiplicity adjustment cannot lower it.

The original general superiority claim must be replaced by dataset- and metric-specific observations. Causal ELU+1 prefix attention retains theoretical O(L) sequence-length work at fixed head dimensions; that mathematical property does not imply faster measured execution, lower measured memory, better accuracy or better modelling of long histories.

The preserved controlled benchmark found higher LightSAKT latency in all 60 matched comparisons and lower peak allocated memory in 7/60. Parameter savings are configuration-dependent: only Algebra2005 has fewer LightSAKT parameters than SAKT in the documented primary comparisons, whose widths differ. These primary parameter counts are separate from validation-selected final configurations. Evidence: paper/EFFICIENCY_RESULTS.md and reports/efficiency_verification.json.

Completed ablations support a descriptive role for attention and dropout: removing either yields lower mean AUC on all three datasets under the controlled settings. They do not universally favor the default: single-head attention has higher mean AUC on ASSIST2015 and Algebra2005, and removing positions has higher mean AUC on Algebra2005. These are descriptive controlled observations, not significance claims. Kernel attention maps and held-out correctness trajectories support faithful model inspection; they do not directly establish student mastery or an educational intervention benefit.

Evidence: LightSAKT_final_summary.csv, LightSAKT_completed_baseline_comparisons.csv, LightSAKT_vs_best_completed_baseline.csv, reports/light_final_summary_verification.json, reports/selected_final_inference_verification.json and reports/LIGHTSAKT_GRID_ABLATION_RESULTS.md. The original architecture and approved experiment schedule remain unchanged.

A narrower error-loss result is supported descriptively: LightSAKT has lower mean MSE and BCE than DKT and DKT+ on all three datasets, while DKVMN has lower mean MSE and BCE than LightSAKT on all three. Lower error losses alone do not establish probability calibration.
