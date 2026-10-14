# Schedule and scientific-scope audit

Observed completion: 106/957; remaining fresh trainings 806, reuse/evaluation slots 45. The 912 figure was the planned total of distinct trainings, not the remaining count.

|Category|Dataset|Completed|Fresh remaining|Reuse remaining|
|---|---|---|---|---|
|LightSAKT_grid|ASSIST2009|0|35|1|
|LightSAKT_grid|ASSIST2015|0|35|1|
|LightSAKT_grid|Algebra2005|0|35|1|
|ablations|ASSIST2009|0|20|5|
|ablations|ASSIST2015|0|20|5|
|ablations|Algebra2005|0|20|5|
|baseline_tuning|ASSIST2009|85|129|2|
|baseline_tuning|ASSIST2015|0|214|2|
|baseline_tuning|Algebra2005|0|214|2|
|primary|ASSIST2009|7|0|0|
|primary|ASSIST2015|7|0|0|
|primary|Algebra2005|7|0|0|
|repeated_selected|ASSIST2009|0|28|7|
|repeated_selected|ASSIST2015|0|28|7|
|repeated_selected|Algebra2005|0|28|7|

Every one of the 756 grid rows uses seed 42 only. Five matched seeds are scheduled only for validation-selected final configurations and controlled ablations. There is no grid-seed replication to remove. Splits remain frozen and identical across model seeds. No-regularization remains a no-dropout alias; default ablations reuse final LightSAKT evidence.

Primary comparisons, all 108 LightSAKT candidates, matched five-seed final comparisons and four distinct ablations per dataset remain essential to the requested paper. The equal 36-candidate baseline tuning budget was an expanded fairness/reproducibility design; it is not an exact search space stated in Tables 4–6. Those tables specify LightSAKT only. Both historical simplified GKT variants remain present. Removing supplementary tuning still changes the established comparison protocol and requires approval.

Complete remaining training forecast: 501.1 hours (106.3–1465.8 scenarios). Proposal: 171.9 hours (34.7–493.7). Proposed deferred fresh trainings: 162; remaining fresh trainings under proposal: 644. This is a training-only forecast, not a confidence interval. Selected-configuration uncertainty remains.

Proposed GKT-PAM grid: learning rates [.0005,.001,.005] × hidden widths [16,30,64]. Proposed GKT-MHA grid: the same learning rates × widths [16,32,64], heads fixed at 2; historical primary width 30/head 2 remains an additional validation candidate using completed evidence. Every other baseline keeps all 36 configurations. Previously completed/running entries are retained regardless of membership. All five final seeds and all ablations remain.

Paper impact: LightSAKT grid tables and ablation design remain intact. Table 7 and significance comparisons would describe GKT chosen using a smaller, disclosed tuning budget, not an exhaustive 36-configuration GKT search. Equal validation access is preserved; tuning-count equality is lost, while GKT still receives substantially more measured compute than LightSAKT. No test score or response label determines the retained configurations.

Full run-level before/after membership is in PROPOSED_DIFFERENCES.csv and the two schedule JSON files. The proposal is not active. No completed result, architecture, batch size, optimizer, split, precision, maximum epoch or early-stopping rule was changed.
