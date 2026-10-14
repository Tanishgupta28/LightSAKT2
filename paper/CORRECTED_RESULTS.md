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

## Matched final comparisons

Final selected-configuration repeated-seed experiments are pending; primary single-seed results are recorded separately in FINAL_RESEARCH_REPORT.md.

## Ablations and paired inference

Default LightSAKT reused the final-model evidence. Component variants removed dropout, positional contributions, multiple heads, or attention. Without attention, the attention contribution was zero while the original embedding residual and feed-forward pathway remained. By explicit author instruction, “without regularization” was equivalent to “without dropout,” because the original model supplied no other explicit regularizer and weight decay was zero. These two rows reuse the same evidence and are not independent observations.

Relative changes and sample standard deviations are generated in Table 8 and paired_ablations.csv. No outcome was required to favor the default architecture. Paired signed-rank comparisons used matching model seeds and the same frozen student split, with exact sign enumeration, zero omission, midranks for ties, and Holm adjustment across the 18 prespecified comparisons. Five nonzero pairs permit a minimum two-sided exact p-value of 0.0625, so conventional significance cannot be established with that design even before correction.

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
