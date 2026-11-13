# Verified paper-update assets

The original manuscript is unchanged. This package contains genuine measured
results and reproducible replacement assets, with incomplete experiments marked.
Audit performed 13 November 2026; the requested directory date is retained.

- [Research asset report](REPORT.md) and [professor summary](PROFESSOR_SUMMARY.md).
- [Full numerical comparison](tables/full_model_comparison/master_comparison_all_metrics.csv):
  27 groups, 15 complete five-seed groups, two partial groups, 77 verified seeds.
- [Individual seeds](tables/full_model_comparison/individual_seed_results.csv),
  [completion matrix](tables/full_model_comparison/current_completion_matrix.csv)
  and [58 remaining slots](tables/full_model_comparison/remaining_seed_slots.csv).
- [Tables 2–8 mapping](documentation/PAPER_TABLE_MAPPING.md) and
  [Figures 1–9 mapping](documentation/PAPER_GRAPH_MAPPING.md).
- [All graph programs](python_scripts/generate_all_paper_graphs.py),
  [reproduction instructions](REPRODUCE.md) and
  [graph verification](verification/graph_reproducibility/verification.json).
- [Manual replacement guide](documentation/MANUAL_REPLACEMENT_GUIDE.md),
  [separate suggested corrections](documentation/SUGGESTED_MANUSCRIPT_CORRECTIONS.md)
  and [remaining limitations](documentation/REMAINING_LIMITATIONS.md).

Every graph is available as PNG (320 DPI), vector PDF and SVG:

| Requested graph | Path stem under this directory |
|---|---|
| A / original Figure 4 | `graphs/ASSIST2009/training_performance_seed42` |
| B / original Figure 5 replacement | `graphs/EdNet_KT1_5k/training_performance_seed42` |
| C / original Figure 6 | `graphs/Algebra2005/training_performance_seed42` |
| D | `graphs/model_comparisons/test_auc` |
| E | `graphs/model_comparisons/test_accuracy` |
| F | `graphs/model_comparisons/test_f1` |
| G | `graphs/model_comparisons/test_mse` |
| H | `graphs/model_comparisons/test_bce` |
| I / original Figure 7 | `graphs/parameters/model_parameters` |
| J | `graphs/ablations/corrected_ablation_test_auc` |
| K | `graphs/model_comparisons/question_difficulty_comparison` |
| L | `graphs/model_comparisons/individual_seed_test_auc` |
| M, descriptive alternatives | `graphs/student_knowledge/{ASSIST2009,Algebra2005,EdNet_KT1_5k}_prediction_sequence` |

Graph M shows genuine next-response probabilities, not verified mastery. No latent
mastery replacement for original Figures 8–9 is certified. No missing curves or
measurements are imputed. GKT-MHA denotes the repository's implemented variant.

Zero new training runs: current allocation ownership and an expanded GPU budget
are unconfirmed. Two frozen ASSIST2009 GKT checkpoints were evaluated on CPU.
The feature-matched SAKT + Difficulty control remains untrained. Positive offline
item-information gains do not establish attention-architecture superiority.

Portable ZIP, outside Git and the original artifacts:
`/workspace/LightSAKT2/backups/paper_update_assets_20261112/LightSAKT_paper_update_assets_20261112.zip`.
Its external receipt records SHA256, size, CRC and payload verification. Raw
datasets, checkpoint binaries, full prediction arrays and the original PDF are
excluded; exact checkpoint/source hashes are retained.
