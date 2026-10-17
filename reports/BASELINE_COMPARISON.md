# Verified baseline comparison

Generated UTC: 2026-10-17T03:05:58.860320+00:00

Only completed artifacts on the same locked data/splits and context lengths are used. Matched final comparisons require all five prespecified seeds. Incomplete rows are pending, with no historical-paper substitution.

|dataset|model|completed_seeds|required_seeds|status|
|---|---|---|---|---|
|ASSIST2009|DKT|5|5|completed and verified|
|ASSIST2009|DKT+|5|5|completed and verified|
|ASSIST2009|DKVMN|5|5|completed and verified|
|ASSIST2009|GKT|0|5|pending; partial evidence excluded from final comparison|
|ASSIST2009|GKT_MHA|0|5|pending; partial evidence excluded from final comparison|
|ASSIST2009|SAKT|5|5|completed and verified|
|ASSIST2015|DKT|5|5|completed and verified|
|ASSIST2015|DKT+|5|5|completed and verified|
|ASSIST2015|DKVMN|5|5|completed and verified|
|ASSIST2015|GKT|0|5|pending; partial evidence excluded from final comparison|
|ASSIST2015|GKT_MHA|0|5|pending; partial evidence excluded from final comparison|
|ASSIST2015|SAKT|5|5|completed and verified|
|Algebra2005|DKT|5|5|completed and verified|
|Algebra2005|DKT+|5|5|completed and verified|
|Algebra2005|DKVMN|5|5|completed and verified|
|Algebra2005|GKT|0|5|pending; partial evidence excluded from final comparison|
|Algebra2005|GKT_MHA|0|5|pending; partial evidence excluded from final comparison|
|Algebra2005|SAKT|5|5|completed and verified|

## Completed selected five-seed results

|dataset|baseline|AUC|ACCURACY|F1|MSE|BCE|LightSAKT minus baseline AUC|
|---|---|---|---|---|---|---|---|
|ASSIST2009|DKT|0.757165 ± 0.000527|0.736135 ± 0.001022|0.818332 ± 0.000993|0.189594 ± 0.000669|0.564732 ± 0.001604|-0.006109|
|ASSIST2009|DKT+|0.751202 ± 0.000876|0.733004 ± 0.001809|0.816546 ± 0.002508|0.194971 ± 0.000512|0.577168 ± 0.001301|-0.000146|
|ASSIST2009|DKVMN|0.755016 ± 0.001395|0.736225 ± 0.001179|0.819802 ± 0.001518|0.178742 ± 0.000262|0.535401 ± 0.000548|-0.003960|
|ASSIST2009|SAKT|0.750322 ± 0.000648|0.732718 ± 0.000669|0.817401 ± 0.000943|0.180373 ± 0.000252|0.539191 ± 0.000666|0.000734|
|ASSIST2015|DKT|0.728235 ± 0.000327|0.749869 ± 0.000281|0.845662 ± 0.000528|0.182124 ± 0.000394|0.547349 ± 0.001032|-0.002146|
|ASSIST2015|DKT+|0.720802 ± 0.001036|0.745974 ± 0.000619|0.842588 ± 0.000766|0.187262 ± 0.000415|0.559082 ± 0.001053|0.005287|
|ASSIST2015|DKVMN|0.727605 ± 0.000989|0.751034 ± 0.000445|0.847155 ± 0.000456|0.170776 ± 0.000228|0.515011 ± 0.000615|-0.001517|
|ASSIST2015|SAKT|0.725180 ± 0.000258|0.749418 ± 0.000386|0.846664 ± 0.000609|0.171635 ± 0.000129|0.517255 ± 0.000332|0.000909|
|Algebra2005|DKT|0.822181 ± 0.000683|0.800524 ± 0.001175|0.872422 ± 0.000891|0.149683 ± 0.000862|0.466820 ± 0.002433|-0.011592|
|Algebra2005|DKT+|0.823456 ± 0.000649|0.801094 ± 0.000253|0.872982 ± 0.000347|0.152382 ± 0.000488|0.474735 ± 0.001334|-0.012866|
|Algebra2005|DKVMN|0.817725 ± 0.001099|0.798950 ± 0.000818|0.872047 ± 0.000572|0.141806 ± 0.000514|0.439274 ± 0.001405|-0.007135|
|Algebra2005|SAKT|0.813373 ± 0.000367|0.796382 ± 0.000504|0.871012 ± 0.000372|0.143309 ± 0.000206|0.443098 ± 0.000598|-0.002784|

Differences compare five-seed means on the same split; they are descriptive effect estimates, not significance claims. Paired significance and Holm correction are generated from genuinely matched seeds.

## Fresh primary runs — separate descriptive evidence

Independent CPU validation/test inference verified for 60 completed baseline final-seed runs. Exact per-probability checks, tolerance values and checkpoint hashes are recorded in reports/selected_final_inference_verification.json.

Fresh completed seed-42 primary runs are fixed-configuration, single-seed measurements. They do not substitute for pending validation-selected five-seed comparisons. Original simplified GKT-PAM/MHA implementations remain separately labelled and do not establish canonical GKT reproduction.

|dataset|baseline|baseline_auc_mean|light_minus_baseline_auc_mean|
|---|---|---|---|
|ASSIST2009|DKT|0.755294|-0.004238|
|ASSIST2009|DKT+|0.753788|-0.002732|
|ASSIST2009|DKVMN|0.753237|-0.002181|
|ASSIST2009|GKT|0.757626|-0.006570|
|ASSIST2009|GKT_MHA|0.762630|-0.011575|
|ASSIST2009|SAKT|0.750656|0.000400|
|ASSIST2015|DKT|0.726490|-0.000401|
|ASSIST2015|DKT+|0.720112|0.005976|
|ASSIST2015|DKVMN|0.726157|-0.000069|
|ASSIST2015|GKT|0.729880|-0.003792|
|ASSIST2015|GKT_MHA|0.732967|-0.006879|
|ASSIST2015|SAKT|0.724443|0.001646|
|Algebra2005|DKT|0.822137|-0.011548|
|Algebra2005|DKT+|0.823264|-0.012675|
|Algebra2005|DKVMN|0.809766|0.000824|
|Algebra2005|GKT|0.821627|-0.011038|
|Algebra2005|GKT_MHA|0.821574|-0.010985|
|Algebra2005|SAKT|0.812828|-0.002238|

All five metrics and deltas: results/tables/LightSAKT_completed_baseline_comparisons.csv. Exact per-run Git commits, checkpoint/prediction hashes, locked test verification and source paths: reports/light_final_summary_verification.json. No statistical superiority or completed research-project claim is made from pending comparisons.
