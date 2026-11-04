# Professor progress summary — 4 November 2026

45/45 core model-seed measurements verified; 15/15 replacement experiments completed.

ASSIST2015 was replaced with EdNet-KT1 because the preserved ASSIST2015 release lacks independent question identifiers. EdNet has authentic publisher question IDs and answer metadata. We fixed a random 5,000-student subset and retained the first question per solving session to avoid within-bundle response history. All three models use matched student splits, five seeds and validation-selected checkpoints.

| Dataset | Model | Validation AUC | Test AUC | Test accuracy | Test F1 | Seeds |
|---|---|---|---|---|---|---|
| ASSIST2009 | SAKT | 0.748284 ± 0.000767 | 0.750322 ± 0.000648 | 0.732718 ± 0.000669 | 0.817401 ± 0.000943 | 5/5 |
| ASSIST2009 | LightSAKT | 0.749454 ± 0.001570 | 0.751056 ± 0.000766 | 0.732229 ± 0.000801 | 0.816533 ± 0.001517 | 5/5 |
| ASSIST2009 | LightSAKT+Difficulty | 0.780248 ± 0.001153 | 0.779359 ± 0.000778 | 0.744460 ± 0.000439 | 0.822429 ± 0.000649 | 5/5 |
| Algebra2005 | SAKT | 0.809999 ± 0.000725 | 0.813373 ± 0.000367 | 0.796382 ± 0.000504 | 0.871012 ± 0.000372 | 5/5 |
| Algebra2005 | LightSAKT | 0.807548 ± 0.000766 | 0.810590 ± 0.000422 | 0.795574 ± 0.000639 | 0.870356 ± 0.000350 | 5/5 |
| Algebra2005 | LightSAKT+Difficulty | 0.835215 ± 0.000498 | 0.836871 ± 0.000217 | 0.807301 ± 0.000642 | 0.876874 ± 0.000771 | 5/5 |
| EdNet-KT1 (5k students; first/session) | SAKT | 0.665022 ± 0.001326 | 0.636359 ± 0.001981 | 0.662697 ± 0.001453 | 0.777653 ± 0.002664 | 5/5 |
| EdNet-KT1 (5k students; first/session) | LightSAKT | 0.670306 ± 0.001088 | 0.642417 ± 0.000980 | 0.664176 ± 0.000810 | 0.778199 ± 0.001393 | 5/5 |
| EdNet-KT1 (5k students; first/session) | LightSAKT+Difficulty | 0.729147 ± 0.000894 | 0.706876 ± 0.000691 | 0.693068 ± 0.000701 | 0.790115 ± 0.001670 | 5/5 |

ASSIST2009: modified minus SAKT mean test AUC = +0.029036; positive in 5/5 matched seeds.
ASSIST2009: modified minus LightSAKT mean test AUC = +0.028303; positive in 5/5 matched seeds.
Algebra2005: modified minus SAKT mean test AUC = +0.023497; positive in 5/5 matched seeds.
Algebra2005: modified minus LightSAKT mean test AUC = +0.026281; positive in 5/5 matched seeds.
EdNet-KT1 (5k students; first/session): modified minus SAKT mean test AUC = +0.070516; positive in 5/5 matched seeds.
EdNet-KT1 (5k students; first/session): modified minus LightSAKT mean test AUC = +0.064458; positive in 5/5 matched seeds.

The earlier ASSIST2009 0.7508→0.7812 statement is a seed-42 validation result. The table distinguishes five-seed validation and held-out test performance. All reported numbers come from preserved checkpoint/prediction evidence; means use sample SD.

The correction adds training-derived item information and one neural scalar while preserving causal linear attention. Whole-student exclusion removes training self-label contamination; held-out labels do not fit the feature. This remains an offline protocol using complete training-corpus statistics, not demonstrated chronological deployment. EdNet is the explicitly reduced first-question/session subset, not the full KT1 benchmark.

These measurements are appropriate to share as controlled research progress. They do not establish publication readiness or attention superiority. The next key experiment is the separately budgeted, feature-matched SAKT+Difficulty control, followed by independent split/cohort and chronological evaluation. No further tuning should use these released test outcomes.
