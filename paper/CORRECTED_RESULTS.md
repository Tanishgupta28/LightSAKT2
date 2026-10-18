# Corrected results text — generated from experiment artifacts

This document accompanies the preserved manuscript; it does not overwrite historical claims. Incomplete experiments remain explicitly pending.

## Corrected protocol

We split students into training (70%), validation (10%), and testing (20%) before sequence chunking. Split seed 42 was prespecified for the rebuilt protocol. Chronological response-event order was preserved, and vocabularies were fitted on training students. Histories were chunked at lengths 100, 100, and 200 for ASSIST2009, ASSIST2015, and Algebra2005, respectively, with one shared interaction at boundaries and no carried latent state. All valid next-response targets were retained; padding contributed to neither loss nor metrics.

LightSAKT retained interaction and target-question embeddings, learned positions, multi-head learned projections, the ELU+1 kernel, residual connections, layer normalization, dropout, reduced-width feed-forward layers, and sigmoid prediction. Causal key and key–value prefixes share the same boundary and exclude invalid keys. Historical global aggregation was corrected because it exposed future responses.

Adam used beta values (0.9, 0.999), epsilon 1e-8, and zero weight decay. Training lasted at most 50 epochs with patience five on validation AUC. Test data were excluded from hyperparameter and checkpoint selection. Each model variant received 36 validation-tuning configurations per dataset. The LightSAKT grid comprised learning rates {0.0005, 0.001, 0.005}, widths {32, 64, 128}, heads {2, 4}, and dropout {0.1, 0.2}. Final training seeds were {42, 123, 2026, 3407, 7777}; selected tuning checkpoints were reused for seed 42.

The GKT PAM and MHA rows refer to the corrected historical simplified implementation and do not establish canonical published GKT reproducibility. Local dataset hashes and processing are verified, while independent publisher byte identity and redistribution licensing remain unresolved; Algebra covers the available labelled training subset.

## Dataset statistics

|dataset|raw_rows|clean_interactions|students|unique_skill_identifiers|num_q|
|---|---|---|---|---|---|
|ASSIST2009|525534|274590|4151|136|137|
|ASSIST2015|708631|683801|19840|100|101|
|Algebra2005|809694|607025|574|436|388|

## LightSAKT validation selection

|dataset|learning_rate|embed_dim|num_heads|dropout|validation_auc|
|---|---|---|---|---|---|
|Algebra2005|0.001000|128|4|0.200000|0.808127|
|ASSIST2009|0.001000|64|2|0.100000|0.750804|
|ASSIST2015|0.000500|128|4|0.100000|0.730065|

These configurations maximize validation AUC among all 36 prespecified candidates per dataset, with a lexical run-ID tie-break. The selection records are frozen before final evaluation. Validation scores are separate from held-out test performance; no test metric determined these choices. Run IDs and frozen split hashes are in LightSAKT_selected_configurations.csv and experiments/selections/.

## Matched final comparisons

|dataset|model|completed_seeds|auc (mean ± sample SD)|accuracy (mean ± sample SD)|f1 (mean ± sample SD)|mse (mean ± sample SD)|bce (mean ± sample SD)|
|---|---|---|---|---|---|---|---|
|ASSIST2009|DKT|5|0.757165 ± 0.000527|0.736135 ± 0.001022|0.818332 ± 0.000993|0.189594 ± 0.000669|0.564732 ± 0.001604|
|ASSIST2009|DKT+|5|0.751202 ± 0.000876|0.733004 ± 0.001809|0.816546 ± 0.002508|0.194971 ± 0.000512|0.577168 ± 0.001301|
|ASSIST2009|DKVMN|5|0.755016 ± 0.001395|0.736225 ± 0.001179|0.819802 ± 0.001518|0.178742 ± 0.000262|0.535401 ± 0.000548|
|ASSIST2009|LightSAKT|5|0.751056 ± 0.000766|0.732229 ± 0.000801|0.816533 ± 0.001517|0.180233 ± 0.000319|0.538615 ± 0.000886|
|ASSIST2009|SAKT|5|0.750322 ± 0.000648|0.732718 ± 0.000669|0.817401 ± 0.000943|0.180373 ± 0.000252|0.539191 ± 0.000666|
|ASSIST2015|DKT|5|0.728235 ± 0.000327|0.749869 ± 0.000281|0.845662 ± 0.000528|0.182124 ± 0.000394|0.547349 ± 0.001032|
|ASSIST2015|DKT+|5|0.720802 ± 0.001036|0.745974 ± 0.000619|0.842588 ± 0.000766|0.187262 ± 0.000415|0.559082 ± 0.001053|
|ASSIST2015|DKVMN|5|0.727605 ± 0.000989|0.751034 ± 0.000445|0.847155 ± 0.000456|0.170776 ± 0.000228|0.515011 ± 0.000615|
|ASSIST2015|LightSAKT|5|0.726088 ± 0.000315|0.749526 ± 0.000419|0.846282 ± 0.000392|0.171233 ± 0.000106|0.516098 ± 0.000340|
|ASSIST2015|SAKT|5|0.725180 ± 0.000258|0.749418 ± 0.000386|0.846664 ± 0.000609|0.171635 ± 0.000129|0.517255 ± 0.000332|
|Algebra2005|DKT|5|0.822181 ± 0.000683|0.800524 ± 0.001175|0.872422 ± 0.000891|0.149683 ± 0.000862|0.466820 ± 0.002433|
|Algebra2005|DKT+|5|0.823456 ± 0.000649|0.801094 ± 0.000253|0.872982 ± 0.000347|0.152382 ± 0.000488|0.474735 ± 0.001334|
|Algebra2005|DKVMN|5|0.817725 ± 0.001099|0.798950 ± 0.000818|0.872047 ± 0.000572|0.141806 ± 0.000514|0.439274 ± 0.001405|
|Algebra2005|LightSAKT|5|0.810590 ± 0.000422|0.795574 ± 0.000639|0.870356 ± 0.000350|0.144069 ± 0.000364|0.445272 ± 0.001018|
|Algebra2005|SAKT|5|0.813373 ± 0.000367|0.796382 ± 0.000504|0.871012 ± 0.000372|0.143309 ± 0.000206|0.443098 ± 0.000598|

Means summarize every completed prespecified seed. Sample standard deviations and Student-t intervals describe conditional initialization variability on one fixed student split; they do not measure new-student sampling uncertainty. Partial seed counts are not complete five-seed results.

ASSIST2009: LightSAKT mean test AUC minus DKT was -0.006109 across the five prespecified seeds (Table 7).
ASSIST2009: LightSAKT mean test AUC minus DKT+ was -0.000146 across the five prespecified seeds (Table 7).
ASSIST2009: LightSAKT mean test AUC minus DKVMN was -0.003960 across the five prespecified seeds (Table 7).
ASSIST2009: LightSAKT mean test AUC minus SAKT was +0.000734 across the five prespecified seeds (Table 7).
ASSIST2015: LightSAKT mean test AUC minus DKT was -0.002146 across the five prespecified seeds (Table 7).
ASSIST2015: LightSAKT mean test AUC minus DKT+ was +0.005287 across the five prespecified seeds (Table 7).
ASSIST2015: LightSAKT mean test AUC minus DKVMN was -0.001517 across the five prespecified seeds (Table 7).
ASSIST2015: LightSAKT mean test AUC minus SAKT was +0.000909 across the five prespecified seeds (Table 7).
Algebra2005: LightSAKT mean test AUC minus DKT was -0.011592 across the five prespecified seeds (Table 7).
Algebra2005: LightSAKT mean test AUC minus DKT+ was -0.012866 across the five prespecified seeds (Table 7).
Algebra2005: LightSAKT mean test AUC minus DKVMN was -0.007135 across the five prespecified seeds (Table 7).
Algebra2005: LightSAKT mean test AUC minus SAKT was -0.002784 across the five prespecified seeds (Table 7).

## Comparison with the best completed baseline

Generated UTC: 2026-10-18T16:21:44.423892+00:00

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


Independent CPU inference reproduced validation metrics and saved test probabilities for 75 completed selected-final checkpoints. Verification checked exact frozen test identities/labels and FP32 cross-device probability tolerances (atol=1e-6, rtol=1e-6). Per-run hashes, source Git commits and metric checks are recorded in reports/selected_final_inference_verification.json; all 15 individual LightSAKT seed results and selected validation epochs are in reports/LIGHTSAKT_FINAL_SEED_RESULTS.md. Final verification of the remaining experimental schedule is still pending.

## Ablations and paired inference

Default LightSAKT reused the final-model evidence. Component variants removed dropout, positional contributions, multiple heads, or attention. Without attention, the attention contribution was zero while the original embedding residual and feed-forward pathway remained. By explicit author instruction, “without regularization” was equivalent to “without dropout,” because the original model supplied no other explicit regularizer and weight decay was zero. These two rows reuse the same evidence and are not independent observations.

Relative changes and sample standard deviations are generated in Table 8 and paired_ablations.csv. No outcome was required to favor the default architecture. The prespecified paired analysis uses matching completed model seeds and the same frozen student split, with exact sign enumeration, zero omission, midranks for ties, and Holm adjustment across 18 comparisons. Unfinished pairs supply no significance evidence. Five nonzero pairs permit a minimum two-sided exact p-value of 0.0625, so conventional significance cannot be established with that design even before correction.

## Efficiency and interpretation

**Measured runtime:** LightSAKT had higher mean latency in 60 of 60 matched comparisons and lower mean latency in 0. These genuine findings contradict an unconditional speed advantage. The saved 120 FP32 cases cover lengths 50, 100, 200, 500 and 1000, batch sizes 1 and 8, inference and forward/backward, and three separately labelled comparison scopes on the H100 MIG partition. Each case used three warm-ups and ten synchronized timed repetitions. We preserved the original architecture and these measurements.

**Measured memory:** LightSAKT had lower peak allocated memory in 7 of 60 matched comparisons. Allocated peaks, reserved peaks and baseline allocations are reported separately; these process measurements include model, inputs, workspace and warm-up state. They do not establish an unconditional memory advantage or the asymptotic storage behavior of a different attention backend.

|Comparison scope|Matched cases|LightSAKT slower|LightSAKT lower peak allocated memory|
|---|---|---|---|
|documented_complete|20|20|0|
|equal_attention|20|20|4|
|equal_complete|20|20|3|

**Parameter efficiency:** Counts for the documented primary configurations are:

|Dataset|LightSAKT parameters|SAKT parameters|LightSAKT/SAKT|
|---|---|---|---|
|ASSIST2009|148673|112201|1.325059|
|ASSIST2015|134849|101401|1.329859|
|Algebra2005|108449|197501|0.549106|

Primary model widths differ, so these counts cannot be attributed solely to the kernel or feed-forward reduction. Equal-width complete-model comparisons isolate parameter differences more closely. Benchmark positional capacity grows with tested length; parameter counts and serialized sizes are recorded for each case.

**Theoretical scaling and interpretation:** At fixed head dimensions, the causal ELU+1 kernel performs O(B H L d_k d_v) attention work. This vectorized training implementation also stores sequence-length key/value prefixes; its dimension-dependent constants matter. This theoretical O(L) work does not imply lower measured latency or lower memory throughout the tested range. The isolated dense SAKT reference requests attention weights, while the existing complete SAKT uses PyTorch MHA without requesting them. Backend and comparison scope remain explicit.

Benchmark weights were newly initialized, and forward/backward used a squared-output surrogate. These timings are distinct from dataset epoch times, which include training and validation. Ten timing repetitions measure engineering variability on one recorded device, not independent student or training-seed uncertainty. Scope-specific numerical results, complete raw repetitions and hardware metadata are preserved in results/tables/efficiency.csv, results/tables/efficiency_paired_comparisons.csv, results/tables/parameters.csv, experiments/efficiency/raw_timings.json and experiments/efficiency/environment.json. The full measured discussion is in paper/EFFICIENCY_RESULTS.md.

## Matched SAKT grid appendix

# LightSAKT versus SAKT: two distinct comparisons

## A. Identical hyperparameters — validation only

|dataset|configurations|LightSAKT_higher|SAKT_higher|exact_ties|mean_difference|median_difference|minimum_difference|maximum_difference|descriptive_sd|
|---|---|---|---|---|---|---|---|---|---|
|Algebra2005|36|6|30|0|-0.002002|-0.002387|-0.004669|0.002416|0.001774|
|ASSIST2009|36|31|5|0|0.002803|0.002287|-0.001221|0.011890|0.003232|
|ASSIST2015|36|29|7|0|0.002192|0.000610|-0.001760|0.014603|0.004215|

Difference is LightSAKT validation AUC minus SAKT validation AUC. The 36 rows are a deterministic hyperparameter sweep at one seed, not 36 independent random trials; no configuration-level significance test is performed. Mean/median/range/SD describe that specified grid. This comparison covers the complete architectures, including their attention and FFN differences, and does not isolate the attention kernel alone.

## B. Independently tuned configurations — held-out test

|dataset|model|learning_rate|embedding_dim|heads|dropout|validation_auc|
|---|---|---|---|---|---|---|
|Algebra2005|LightSAKT|0.001000|128|4|0.200000|0.808127|
|Algebra2005|SAKT|0.000500|128|2|0.200000|0.810922|
|ASSIST2009|LightSAKT|0.001000|64|2|0.100000|0.750804|
|ASSIST2009|SAKT|0.000500|64|4|0.200000|0.749443|
|ASSIST2015|LightSAKT|0.000500|128|4|0.100000|0.730065|
|ASSIST2015|SAKT|0.000500|128|4|0.200000|0.729265|

Both models independently selected their own highest-validation-AUC configuration from the same complete search space, then used seeds 42,123,2026,3407,7777 on the same frozen split. Seed 42 reuses its selected grid checkpoint. All five seeds contribute; SD is sample SD (ddof=1).

|dataset|metric|LightSAKT|SAKT|difference|
|---|---|---|---|---|
|Algebra2005|auc|0.810590 ± 0.000422|0.813373 ± 0.000367|-0.002784|
|Algebra2005|accuracy|0.795574 ± 0.000639|0.796382 ± 0.000504|-0.000807|
|Algebra2005|f1|0.870356 ± 0.000350|0.871012 ± 0.000372|-0.000655|
|Algebra2005|mse|0.144069 ± 0.000364|0.143309 ± 0.000206|0.000760|
|Algebra2005|bce|0.445272 ± 0.001018|0.443098 ± 0.000598|0.002173|
|ASSIST2009|auc|0.751056 ± 0.000766|0.750322 ± 0.000648|0.000734|
|ASSIST2009|accuracy|0.732229 ± 0.000801|0.732718 ± 0.000669|-0.000490|
|ASSIST2009|f1|0.816533 ± 0.001517|0.817401 ± 0.000943|-0.000868|
|ASSIST2009|mse|0.180233 ± 0.000319|0.180373 ± 0.000252|-0.000140|
|ASSIST2009|bce|0.538615 ± 0.000886|0.539191 ± 0.000666|-0.000576|
|ASSIST2015|auc|0.726088 ± 0.000315|0.725180 ± 0.000258|0.000909|
|ASSIST2015|accuracy|0.749526 ± 0.000419|0.749418 ± 0.000386|0.000108|
|ASSIST2015|f1|0.846282 ± 0.000392|0.846664 ± 0.000609|-0.000382|
|ASSIST2015|mse|0.171233 ± 0.000106|0.171635 ± 0.000129|-0.000402|
|ASSIST2015|bce|0.516098 ± 0.000340|0.517255 ± 0.000332|-0.001157|

Positive differences favor LightSAKT for AUC, accuracy and F1; negative differences favor LightSAKT for MSE and BCE. These are descriptive differences conditional on one student split. Five seeds cannot support a two-sided exact signed-rank p<0.05; grid rows are not used to inflate that sample size. Individual metrics, selected epochs and checkpoint references are in SAKT_matched_grid_individual_seeds.csv.

The existing efficiency evidence remains unchanged: LightSAKT was slower in all 60 matched latency comparisons on the recorded H100 MIG partition; lower peak allocated memory occurred in 7/60. Theoretical O(L) attention work, measured latency, memory and configuration-specific parameter counts remain separate. These additional outputs do not imply the remaining GKT research schedule is complete.

## Hyperparameter effects within the specified grid

SAKT_grid_hyperparameter_sensitivity.csv reports balanced marginal means over all other coordinates, with counts and observed minima/maxima. These are descriptive averages of the fixed grid, not independent replications or confidence intervals. They do not replace joint validation-based selection.
Algebra2005: highest SAKT marginal means occur at learning_rate=0.001 (marginal mean validation AUC 0.805189); embedding_dim=128 (marginal mean validation AUC 0.809021); heads=4 (marginal mean validation AUC 0.804285); dropout=0.2 (marginal mean validation AUC 0.803993). Interactions can make the jointly selected configuration differ.
ASSIST2009: highest SAKT marginal means occur at learning_rate=0.0005 (marginal mean validation AUC 0.747139); embedding_dim=64 (marginal mean validation AUC 0.747212); heads=4 (marginal mean validation AUC 0.745634); dropout=0.1 (marginal mean validation AUC 0.745454). Interactions can make the jointly selected configuration differ.
ASSIST2015: highest SAKT marginal means occur at learning_rate=0.0005 (marginal mean validation AUC 0.727923); embedding_dim=64 (marginal mean validation AUC 0.727062); heads=4 (marginal mean validation AUC 0.725498); dropout=0.2 (marginal mean validation AUC 0.725622). Interactions can make the jointly selected configuration differ.


The three complete 36-row SAKT tables and three matched comparison tables are in paper/sakt_grid/ as CSV, Markdown and LaTeX. Source checkpoints were reused only after protocol, code, split, integrity and independent inference verification; no existing training was interrupted or repeated.

Kernel heatmaps reconstruct normalized causal ELU+1 weights from selected trained checkpoints and genuine held-out sequences; requesting weights leaves predictions unchanged. Skill trajectories are predicted next-response correctness conditional on observed history, rather than directly observed mastery. Historical TransDKT/TransDKT+ figures are not relabelled as LightSAKT.

## Artifact sources

Tables: results/tables/. Numerical figure data: results/figures/*.csv, experiments/efficiency/raw_timings.json, and experiments/case_studies/*.npz. Each run records the checkpoint, split hash, configuration, seed, code version, predictions, and metrics. PAPER_CORRECTIONS.md documents page-specific discrepancies with suggested wording.
