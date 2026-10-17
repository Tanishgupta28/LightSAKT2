# Updated paper tables for the supervisor

Seven original table images were inspected and recreated from verified corrected artifacts. The combined PDF is in paper order: Tables 2, 3, 4, 5, 6, 7, 8. No training or checkpoint was started, stopped or changed.

[Download the combined PDF](All_Updated_Tables.pdf)

Untouched extracted reference images are kept separately in `paper/table_image_inputs/`. The PNGs in this sharing folder are the updated versions.

|Table|Original image inside allTables.zip|PNG / individual PDF|Meaning and verified source|GKT handling|
|---|---|---|---|---|
|2|`allTables/Screenshot 2026-10-17 234727.png`|[PNG](Table_02_Dataset_Statistics.png) · [PDF](Table_02_Dataset_Statistics.pdf)|Table 2: Overview of datasets. reports/*_statistics.json; results/tables/Table_2.csv; raw headers and locked split checks|Not present|
|3|`allTables/Screenshot 2026-10-17 235129.png`|[PNG](Table_03_Training_Configurations.png) · [PDF](Table_03_Training_Configurations.pdf)|Table 3: Training network configurations for different datasets. configs/protocol.json; selected LightSAKT final configs; experiments/selections/|Not present|
|4|`allTables/Screenshot 2026-10-17 234747.png`|[PNG](Table_04_LightSAKT_Grid_Algebra2005.png) · [PDF](Table_04_LightSAKT_Grid_Algebra2005.pdf)|Table 4: Hyperparameter tuning results for Algebra 2005–2006. results/tables/Table_4.csv; completed grid metrics/checkpoints; reports/light_grid_inference_verification.json|Not present|
|5|`allTables/Screenshot 2026-10-17 234803.png`|[PNG](Table_05_LightSAKT_Grid_ASSIST2009.png) · [PDF](Table_05_LightSAKT_Grid_ASSIST2009.pdf)|Table 5: Hyperparameter tuning results for ASSIST2009. results/tables/Table_5.csv; completed grid metrics/checkpoints; reports/light_grid_inference_verification.json|Not present|
|6|`allTables/Screenshot 2026-10-17 234809.png`|[PNG](Table_06_LightSAKT_Grid_ASSIST2015.png) · [PDF](Table_06_LightSAKT_Grid_ASSIST2015.pdf)|Table 6: Hyperparameter tuning results for ASSIST2015. results/tables/Table_6.csv; completed grid metrics/checkpoints; reports/light_grid_inference_verification.json|Not present|
|7|`allTables/Screenshot 2026-10-17 234818.png`|[PNG](Table_07_Verified_Model_Comparison.png) · [PDF](Table_07_Verified_Model_Comparison.pdf)|Table 7: Performance comparison of knowledge-tracing models across datasets. results/tables/Table_7.csv; 75 final-seed metrics; reports/selected_final_inference_verification.json; reports/BASELINE_COMPARISON.md|Pending in all metric rows|
|8|`allTables/Screenshot 2026-10-17 234828.png`|[PNG](Table_08_LightSAKT_Ablation_Study.png) · [PDF](Table_08_LightSAKT_Ablation_Study.pdf)|Table 8: Ablation study. results/tables/Table_8.csv; 90 logical default/ablation seed records and independent inference proofs|Not present|

## Explicit limitations and corrections

- Table 8 retains the original “No layer normalization” row as **Not run**. It has no verified experiment and no inferred numbers. The approved “Without regularization” row is added and explicitly reuses No Dropout evidence.
- Table 7 marks all GKT final results **Pending**. Historical scores, primary runs, incomplete seeds and both unfinished variants are excluded.
- Table 7 shows five-seed held-out test means ± sample SD. Table 8 shows five-seed test AUC. Tables 4–6 show validation AUC, never test AUC. Best values use full precision; lower MSE/BCE is better.
- Table 2 labels students, cleaned records, observed composite/sequence skill identifiers and raw column counts explicitly. Algebra is the labelled training subset, with training-only vocabulary and an unknown-ID bucket.
- The original screenshot of Table 5 spans two manuscript pages; the updated table presents all 36 configurations together.
- Formatting changes do not establish statistical significance or alter the preserved efficiency findings. There are no efficiency/significance tables in this particular zip.

## Editable sources and verification

`source/` contains one CSV, Markdown, LaTeX and HTML per table, `tables.json` with rendered cells, full-precision values and row-level provenance, and the generator script. The LaTeX snippets require `booktabs` and `graphicx`. PDFs were generated directly as vector tables, not with LaTeX.

`source/project_snapshot/` freezes every consumed config, metric, summary and independent proof. `source/evidence_manifest.json` binds these copies to project paths and SHA256 checksums. `source/verified_runs.json` records run IDs, seeds, training Git commits, checkpoints and hashes. `source/verification.json` records source checks, image dimensions, searchable PDF cell checks and output checksums.

Rebuild from current verified project inputs (uses retained input snapshots when present):

```bash
cd /workspace/LightSAKT2
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.supervisor_tables
```

Render the frozen editable table specifications without accessing datasets or checkpoints:

```bash
CUDA_VISIBLE_DEVICES='' python paper/updated_tables_for_sir/source/generate_tables.py --render-only
```

All seven images and the combined PDF are ready to share **with the visible Pending/Not run disclosures**. The remaining GKT research schedule continues independently.
