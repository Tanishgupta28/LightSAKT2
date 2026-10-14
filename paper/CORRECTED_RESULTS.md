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
|ASSIST2015|DKVMN|2|0.727274 ± 0.001319|0.751026 ± 0.000494|0.846761 ± 0.000538|0.170801 ± 0.000307|0.515092 ± 0.000840|
|ASSIST2015|LightSAKT|5|0.726088 ± 0.000315|0.749526 ± 0.000419|0.846282 ± 0.000392|0.171233 ± 0.000106|0.516098 ± 0.000340|
|ASSIST2015|SAKT|5|0.725180 ± 0.000258|0.749418 ± 0.000386|0.846664 ± 0.000609|0.171635 ± 0.000129|0.517255 ± 0.000332|
|Algebra2005|DKT|5|0.822181 ± 0.000683|0.800524 ± 0.001175|0.872422 ± 0.000891|0.149683 ± 0.000862|0.466820 ± 0.002433|
|Algebra2005|DKT+|5|0.823456 ± 0.000649|0.801094 ± 0.000253|0.872982 ± 0.000347|0.152382 ± 0.000488|0.474735 ± 0.001334|
|Algebra2005|LightSAKT|5|0.810590 ± 0.000422|0.795574 ± 0.000639|0.870356 ± 0.000350|0.144069 ± 0.000364|0.445272 ± 0.001018|
|Algebra2005|SAKT|5|0.813373 ± 0.000367|0.796382 ± 0.000504|0.871012 ± 0.000372|0.143309 ± 0.000206|0.443098 ± 0.000598|

Means summarize every completed prespecified seed. Sample standard deviations and Student-t intervals describe conditional initialization variability on one fixed student split; they do not measure new-student sampling uncertainty. Partial seed counts are not complete five-seed results.

ASSIST2009: LightSAKT mean test AUC minus DKT was -0.006109 across the five prespecified seeds (Table 7).
ASSIST2009: LightSAKT mean test AUC minus DKT+ was -0.000146 across the five prespecified seeds (Table 7).
ASSIST2009: LightSAKT mean test AUC minus DKVMN was -0.003960 across the five prespecified seeds (Table 7).
ASSIST2009: LightSAKT mean test AUC minus SAKT was +0.000734 across the five prespecified seeds (Table 7).
ASSIST2015: LightSAKT mean test AUC minus DKT was -0.002146 across the five prespecified seeds (Table 7).
ASSIST2015: LightSAKT mean test AUC minus DKT+ was +0.005287 across the five prespecified seeds (Table 7).
ASSIST2015: LightSAKT mean test AUC minus SAKT was +0.000909 across the five prespecified seeds (Table 7).
Algebra2005: LightSAKT mean test AUC minus DKT was -0.011592 across the five prespecified seeds (Table 7).
Algebra2005: LightSAKT mean test AUC minus DKT+ was -0.012866 across the five prespecified seeds (Table 7).
Algebra2005: LightSAKT mean test AUC minus SAKT was -0.002784 across the five prespecified seeds (Table 7).

Independent CPU inference reproduced validation metrics and saved test probabilities for 60 completed selected-final checkpoints. Verification checked exact frozen test identities/labels and FP32 cross-device probability tolerances (atol=1e-6, rtol=1e-6). Per-run hashes, source Git commits and metric checks are recorded in reports/selected_final_inference_verification.json; all 15 individual LightSAKT seed results and selected validation epochs are in reports/LIGHTSAKT_FINAL_SEED_RESULTS.md. Final verification of the remaining experimental schedule is still pending.

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

Kernel heatmaps reconstruct normalized causal ELU+1 weights from selected trained checkpoints and genuine held-out sequences; requesting weights leaves predictions unchanged. Skill trajectories are predicted next-response correctness conditional on observed history, rather than directly observed mastery. Historical TransDKT/TransDKT+ figures are not relabelled as LightSAKT.

## Artifact sources

Tables: results/tables/. Numerical figure data: results/figures/*.csv, experiments/efficiency/raw_timings.json, and experiments/case_studies/*.npz. Each run records the checkpoint, split hash, configuration, seed, code version, predictions, and metrics. PAPER_CORRECTIONS.md documents page-specific discrepancies with suggested wording.
