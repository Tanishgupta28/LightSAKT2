# SAKT grid search: verified reuse and complete tables

All 108 requested configurations and 15 independently selected five-seed SAKT slots already existed in the approved schedule. Their checkpoints, configurations, histories, split/data hashes, code compatibility and validation-only selections were checked. No duplicate jobs were added, no completed training was repeated and no healthy worker was signalled. The main manifest is unchanged; experiments/sakt_matched_grid_manifest.json is a reference view of its existing jobs.

Paper Tables 4–6 specify the same 3 learning rates × 3 widths × 2 head counts × 2 dropout rates. Table order is preserved. Grid training seed and split seed are both 42. Context lengths are 100/100/200 for ASSIST2009/ASSIST2015/Algebra; batch size 64; Adam betas (0.9,0.999), epsilon 1e-8, weight decay 0, maximum 50 epochs and validation-AUC patience 5 match between models. Standard SAKT retains PyTorch multi-head attention; LightSAKT retains ELU+1 causal prefix attention and its reduced-width FFN.

## Complete independently selected validation configurations

|dataset|model|learning_rate|embedding_dim|heads|dropout|validation_auc|best_epoch|seed|run_id|checkpoint|checkpoint_sha256|
|---|---|---|---|---|---|---|---|---|---|---|---|
|Algebra2005|LightSAKT|0.001000|128|4|0.200000|0.808127|26|42|Algebra2005-LightSAKT-grid-42-06884c8943b2|checkpoints/Algebra2005-LightSAKT-grid-42-06884c8943b2/best.pt|14b64e7450e0610c300913e933a7dd2ab0d6d3881951e3475557a8e673ee1390|
|Algebra2005|SAKT|0.000500|128|2|0.200000|0.810922|46|42|Algebra2005-SAKT-baseline_tuning-42-6aa156e936e3|checkpoints/Algebra2005-SAKT-baseline_tuning-42-6aa156e936e3/best.pt|0420adc72834ffa11722acbd302aaa58824bdecf6819859316d3b864a0d5bb57|
|ASSIST2009|LightSAKT|0.001000|64|2|0.100000|0.750804|24|42|ASSIST2009-LightSAKT-grid-42-f08e9494d270|checkpoints/ASSIST2009-LightSAKT-grid-42-f08e9494d270/best.pt|d0af3a20f40cf3734883634c23053dbfbf187dee9e7a556c85fcb27b4841197d|
|ASSIST2009|SAKT|0.000500|64|4|0.200000|0.749443|36|42|ASSIST2009-SAKT-baseline_tuning-42-bc0a2a560ce1|checkpoints/ASSIST2009-SAKT-baseline_tuning-42-bc0a2a560ce1/best.pt|6fce0569bc2405231c11ae8ab6da3ab20d9b57d731e4d5f8448b172295ac7378|
|ASSIST2015|LightSAKT|0.000500|128|4|0.100000|0.730065|18|42|ASSIST2015-LightSAKT-grid-42-64640d03b0d3|checkpoints/ASSIST2015-LightSAKT-grid-42-64640d03b0d3/best.pt|c08f06c27d9bd31c155dcaff06bbe17b8464578df7e058930233ffbea68d7136|
|ASSIST2015|SAKT|0.000500|128|4|0.200000|0.729265|13|42|ASSIST2015-SAKT-baseline_tuning-42-4cc38d7c6af6|checkpoints/ASSIST2015-SAKT-baseline_tuning-42-4cc38d7c6af6/best.pt|d5a6980a25bb9dc689f4298a476757d5c3808dcecf650619d38b0f4dc3fe1c1c|

Selection maximizes full-precision validation AUC among all 36 candidates for each model/dataset. Exact ties select the lexically smallest run ID. Bold table cells include all exact maxima. Test results never select these configurations. Identical search spaces do not require identical selected configurations.

Tables A/B/C and their paired comparison tables are in results/tables/SAKT_Table_*.{csv,md,tex}, LightSAKT_vs_SAKT_grid_*.{csv,md,tex}, and paper/sakt_grid/. CSVs retain every source run ID, checkpoint and metrics-file hash. Original LightSAKT Tables 4–6 and the original paper are preserved.

## Measured execution budget

|dataset|grid_runs|measured_grid_training_seconds|mean_run_seconds|mean_epoch_seconds|new_grid_runs|new_final_seed_runs|additional_training_gpu_hours|
|---|---|---|---|---|---|---|---|
|Algebra2005|36|445.784614|12.382906|0.319636|0|0|0.000000|
|ASSIST2009|36|387.285844|10.757940|0.420540|0|0|0.000000|
|ASSIST2015|36|1201.952215|33.387562|1.473958|0|0|0.000000|

Genuinely new grid trainings: 0. Reused grid configurations: 108. Additional final-seed trainings: 0; all 15 final slots are verified, including three selected seed-42 checkpoint reuses. Additional training GPU hours: 0. Independent original-device validation verification is serialized through the existing GPU admission lock between completed runs. No active training is stopped; measured verification duration is recorded in reports/sakt_grid_verification_execution.json. Current overall forecast remains in reports/ETA_REPORT.md.

AUC proof: reports/sakt_grid_auc_cpu_validation.json. Checkpoint/protocol audit: experiments/sakt_matched_grid_manifest.json. Final test inference proof: reports/selected_final_inference_verification.json. The separate original-device full-metric proof is reports/sakt_grid_validation_inference.json when completed. One strict CPU validation-accuracy check differed from the saved GPU metric by the equivalent of one correct prediction; its failure remains preserved in sakt_grid_cpu_validation_failure.json. The original-device validation check uses the unchanged metric tolerances and original metrics; neither checkpoints nor scores are rewritten. One historical global LightSAKT fingerprint differs only through the audited Algebra preprocessing correction; the unchanged ASSIST2009 model/data equivalence is documented in reports/training_code_equivalence.json. SAKT code fingerprints match the current frozen source exactly. Explicit non-padding test masks and original valid-target predictions are preserved as additional artifacts under results/predictions/sakt_matched_grid/. Continuation commands and publication status are documented in reports/SAKT_GRID_CONTINUATION.md.

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
