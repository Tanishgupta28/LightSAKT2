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

The ELU+1 prefix mechanism scales linearly with sequence length at fixed head dimensions; this theoretical fact does not establish measured speed. Controlled timings separately compare equal-width attention kernels, equal-width complete models, and complete models using documented primary ASSIST2009 dimensions and dropout. Benchmark positional capacity equals the tested length, and weights are freshly initialized. Measured forward/backward timings use a squared-output surrogate; recorded dataset epoch times include validation. All raw repeats, memory measurements, hardware, and parameter counts remain available.

Kernel heatmaps reconstruct normalized causal ELU+1 weights from selected trained checkpoints and genuine held-out sequences; requesting weights leaves predictions unchanged. Skill trajectories are predicted next-response correctness conditional on observed history, rather than directly observed mastery. Historical TransDKT/TransDKT+ figures are not relabelled as LightSAKT.

## Artifact sources

Tables: results/tables/. Numerical figure data: results/figures/*.csv, experiments/efficiency/raw_timings.json, and experiments/case_studies/*.npz. Each run records the checkpoint, split hash, configuration, seed, code version, predictions, and metrics. PAPER_CORRECTIONS.md documents page-specific discrepancies with suggested wording.
