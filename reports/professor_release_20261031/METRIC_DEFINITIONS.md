# Meaning of comparison columns

| Column | Definition |
|---|---|
| Validation AUC | Native recorded validation AUC at that seed's validation-selected checkpoint; the prior CPU replay reproduces it within the declared numerical tolerance. |
| Test AUC | AUC on the fixed held-out student test split, from CPU replay of the same selected checkpoint. It is not another validation score. |
| Test accuracy / F1 / precision / recall | Globally pooled valid next-response targets; fixed probability threshold 0.5. No threshold optimization. |
| BCE | Mean binary cross-entropy on valid targets, clipping probabilities to [1e-7,1-1e-7] for reporting. |
| Mean ± SD | Unweighted mean across five prescribed training seeds and SAMPLE standard deviation (ddof=1). No SD is estimated for unavailable results. |
| Validation AUC gain | For each matched seed: modified LightSAKT validation AUC MINUS the named comparator's validation AUC; then mean/sample SD of those paired differences. |
| Test AUC gain | The corresponding paired subtraction using held-out test AUC. Positive favors the modified model; units are absolute AUC, not relative percent. |
| Comparator | Either original causal LightSAKT or original causal SAKT, identified separately in every difference row. |
| Positive seed count | Number of paired differences strictly greater than zero; not a significance test or confidence interval. |
| UNAVAILABLE_CURRENT_PROTOCOL | No valid modified ASSIST2015 measurement, because independent question IDs cannot be obtained from the inspected release. Never a zero score or estimated baseline-equivalent score. |
| Checkpoint reference | External logical weight locator and SHA256. Portable configurations are included; weights and raw datasets are not in this reporting archive. |

Validation scores are recorded native scores; test columns consistently use the
2026-10-30 CPU replay (batch size 8). Both refer to exactly the same frozen
checkpoint. Current verification recalculates metrics from all 80 saved replay
arrays and checks all 40 checkpoint hashes/configurations; it performs no new
training or model inference. The original replay evidence is included unchanged.

Baseline final configurations have evaluate_test=true for test scoring AFTER
training/checkpoint selection. Difficulty training has evaluate_test=false and
separate post-selection CPU scoring. This flag does not alter training loss,
optimizer, early stopping or checkpoint selection. Selection is maximum validation
AUC for all models. The fixed 0.5 threshold applies to both splits.
