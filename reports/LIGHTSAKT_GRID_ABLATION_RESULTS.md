# Verified LightSAKT grids and ablations

Generated UTC: 2026-10-14T18:24:54.008454+00:00

All 108 unique validation-grid rows and their checkpoint inference are verified. Selection remains validation-only. Tables 4–6 retain all 36 candidates per dataset; no test result selected a configuration.

All 75 logical ablation slots passed original-GPU validation/test inference. Every saved test probability reproduced bit-for-bit. There are 60 fresh component-removal trainings and 15 equivalent no-regularization aliases. The default reference uses the 15 selected final seeds; it is not retrained. Split seed 42 and the five prescribed training seeds are unchanged.

|dataset|variant|AUC|ACCURACY|F1|MSE|BCE|
|---|---|---|---|---|---|---|
|ASSIST2009|no_attention|0.708307 ± 0.001107|0.701103 ± 0.000972|0.796294 ± 0.003038|0.194852 ± 0.000103|0.573400 ± 0.000269|
|ASSIST2009|no_dropout|0.749340 ± 0.000716|0.731232 ± 0.000978|0.815848 ± 0.001307|0.180887 ± 0.000237|0.540356 ± 0.000537|
|ASSIST2009|no_position|0.750798 ± 0.000583|0.731969 ± 0.000404|0.815202 ± 0.001292|0.180684 ± 0.000286|0.539905 ± 0.000725|
|ASSIST2009|no_regularization|0.749340 ± 0.000716|0.731232 ± 0.000978|0.815848 ± 0.001307|0.180887 ± 0.000237|0.540356 ± 0.000537|
|ASSIST2009|single_head|0.750791 ± 0.000765|0.732855 ± 0.000672|0.817172 ± 0.000928|0.180159 ± 0.000180|0.538468 ± 0.000452|
|ASSIST2015|no_attention|0.699283 ± 0.000443|0.739957 ± 0.000307|0.843957 ± 0.000783|0.177296 ± 0.000150|0.532379 ± 0.000427|
|ASSIST2015|no_dropout|0.725149 ± 0.000593|0.749069 ± 0.000905|0.845043 ± 0.001508|0.171643 ± 0.000273|0.517042 ± 0.000511|
|ASSIST2015|no_position|0.725749 ± 0.000183|0.749277 ± 0.000491|0.845961 ± 0.000911|0.171370 ± 0.000115|0.516442 ± 0.000414|
|ASSIST2015|no_regularization|0.725149 ± 0.000593|0.749069 ± 0.000905|0.845043 ± 0.001508|0.171643 ± 0.000273|0.517042 ± 0.000511|
|ASSIST2015|single_head|0.726622 ± 0.000426|0.749967 ± 0.000487|0.845997 ± 0.000420|0.171153 ± 0.000190|0.515931 ± 0.000478|
|Algebra2005|no_attention|0.772550 ± 0.000515|0.776597 ± 0.000502|0.861749 ± 0.000576|0.157202 ± 0.000675|0.480300 ± 0.002178|
|Algebra2005|no_dropout|0.809160 ± 0.000459|0.794013 ± 0.000337|0.868754 ± 0.000951|0.144857 ± 0.000327|0.447452 ± 0.000957|
|Algebra2005|no_position|0.811410 ± 0.000603|0.795652 ± 0.000959|0.870663 ± 0.000623|0.143908 ± 0.000385|0.444912 ± 0.000949|
|Algebra2005|no_regularization|0.809160 ± 0.000459|0.794013 ± 0.000337|0.868754 ± 0.000951|0.144857 ± 0.000327|0.447452 ± 0.000957|
|Algebra2005|single_head|0.811755 ± 0.000448|0.795773 ± 0.000467|0.870187 ± 0.000817|0.143973 ± 0.000158|0.445168 ± 0.000603|
|ASSIST2009|default|0.751056 ± 0.000766|0.732229 ± 0.000801|0.816533 ± 0.001517|0.180233 ± 0.000319|0.538615 ± 0.000886|
|ASSIST2015|default|0.726088 ± 0.000315|0.749526 ± 0.000419|0.846282 ± 0.000392|0.171233 ± 0.000106|0.516098 ± 0.000340|
|Algebra2005|default|0.810590 ± 0.000422|0.795574 ± 0.000639|0.870356 ± 0.000350|0.144069 ± 0.000364|0.445272 ± 0.001018|

Mean ± sample SD. All individual values and checkpoint/Git references are in results/tables/Table_8_partial.csv and reports/light_outputs_verification.json. Table_8.csv contains the complete 18-row summary and relative changes of means. paired_ablations.csv instead records per-seed relative changes; their averaged ratios need not equal the ratio of means.

No regularization equals no dropout by author instruction. Their weights, probabilities and metrics are the same evidence; these rows are not independent samples. Their locked inference identities are resolved through checksum-bound physical source artifacts.

The ablations do not universally favor the default. Single-head attention has higher mean AUC on ASSIST2015 and Algebra2005; removing positions has higher mean AUC on Algebra2005. These five-seed descriptive effects do not establish conventional statistical significance.

The three grid heatmaps show the maximum validation AUC over heads/dropout for each learning-rate/width cell, with the full 36-row data retained in each grid table. All five ablation metrics have PNG/SVG/PDF plots and source CSVs for all three datasets. Source values, means, sample SDs and counts were checked against immutable metrics.

The CPU no-position validation-AUC discrepancy was retained as a failed strict CPU metric comparison. Original-GPU inference reproduced the saved metrics; the measured cross-device probability/rank-bound diagnosis is in reports/ablation_cpu_gpu_ranking_diagnostic.json. No tolerance was loosened, checkpoint rewritten or experiment retrained.

Source evidence: reports/light_grid_inference_verification.json; selected_final_inference_verification.json; light_outputs_verification.json; results/tables/Table_4.csv–Table_6.csv and Table_8.csv. Full project completion and remaining baseline analyses are still pending.
