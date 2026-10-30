# Professor progress summary — 30 October 2026

The ASSIST2009 seed-42 LightSAKT validation improvement from **0.7508 to 0.7812**
reproduces from the saved checkpoints. The modified model retains causal linear
attention and adds a scalar correction based on training-derived question
statistics. Training features exclude the whole student's labels; validation/test
estimates use training labels only. Real-data leakage checks passed.

- ASSIST2009: 5/5 paired seeds; mean validation/test AUC gains +0.030794 / +0.028303; test AUC higher in 5/5 seeds.
- ASSIST2015: PENDING — separate question identifiers are unavailable.
- Algebra2005: 5/5 paired seeds; mean validation/test AUC gains +0.027667 / +0.026281; test AUC higher in 5/5 seeds.

| Dataset | Model | Validation AUC | Test AUC | Test F1 | Test accuracy | Seeds/status |
|---|---|---|---|---|---|---|
| ASSIST2009 | SAKT | 0.748284 ± 0.000767 | 0.750322 ± 0.000648 | 0.817401 ± 0.000943 | 0.732718 ± 0.000669 | 5/5 COMPLETE |
| ASSIST2009 | LightSAKT | 0.749454 ± 0.001570 | 0.751056 ± 0.000766 | 0.816533 ± 0.001517 | 0.732229 ± 0.000801 | 5/5 COMPLETE |
| ASSIST2009 | LightSAKT+Difficulty | 0.780248 ± 0.001153 | 0.779359 ± 0.000778 | 0.822429 ± 0.000649 | 0.744460 ± 0.000439 | 5/5 COMPLETE |
| ASSIST2015 | SAKT | 0.728970 ± 0.000435 | 0.725180 ± 0.000258 | 0.846664 ± 0.000609 | 0.749418 ± 0.000386 | 5/5 COMPLETE |
| ASSIST2015 | LightSAKT | 0.729922 ± 0.000819 | 0.726088 ± 0.000315 | 0.846282 ± 0.000392 | 0.749526 ± 0.000419 | 5/5 COMPLETE |
| ASSIST2015 | LightSAKT+Difficulty | PENDING | PENDING | PENDING | PENDING | 0/5 PENDING |
| Algebra2005 | SAKT | 0.809999 ± 0.000725 | 0.813373 ± 0.000367 | 0.871012 ± 0.000372 | 0.796382 ± 0.000504 | 5/5 COMPLETE |
| Algebra2005 | LightSAKT | 0.807548 ± 0.000766 | 0.810590 ± 0.000422 | 0.870356 ± 0.000350 | 0.795574 ± 0.000639 | 5/5 COMPLETE |
| Algebra2005 | LightSAKT+Difficulty | 0.835215 ± 0.000498 | 0.836871 ± 0.000217 | 0.876874 ± 0.000771 | 0.807301 ± 0.000642 | 5/5 COMPLETE |

Values are measured mean ± sample SD across the listed seeds; missing results are
PENDING. Campaign status: **completed**. Thirty baseline seeds were
already complete and independently replayed. The authorized missing-seed campaign
contains nine new difficulty runs, with original seed 42 reused. ASSIST2015
requires question metadata before the same modification can be tested fairly.
These results concern a retrospective student-disjoint benchmark with richer item
inputs, not forward-time deployment or attention-kernel superiority. The full
report includes individual seeds, precision/recall, configurations and provenance.
