# Verified final LightSAKT results and ongoing comparisons

Generated UTC: 2026-10-15T15:51:39.582076+00:00

All numbers below are generated from fresh completed experiment files. Five selected seeds per dataset are complete; the overall research schedule remains incomplete. The three fixed-configuration LightSAKT primary runs are outside this selected-final summary; baseline primary runs are labelled separately.

## Selected configurations

|dataset|learning_rate|embed_dim|num_heads|dropout|seq_len|batch_size|
|---|---|---|---|---|---|---|
|ASSIST2009|0.001000|64|2|0.100000|100|64|
|ASSIST2015|0.000500|128|4|0.100000|100|64|
|Algebra2005|0.001000|128|4|0.200000|200|64|

Each dataset selection used all 36 prespecified validation candidates and a lexical run-ID tie-break. Adam betas (0.9, 0.999), epsilon 1e-8, weight decay zero, maximum 50 epochs and validation-AUC patience five are unchanged. Student split seed 42 remains fixed at 70/10/20 before chronological chunking.

## Five-seed test results

|dataset|AUC|ACCURACY|F1|MSE|BCE|
|---|---|---|---|---|---|
|ASSIST2009|0.751056 ± 0.000766|0.732229 ± 0.000801|0.816533 ± 0.001517|0.180233 ± 0.000319|0.538615 ± 0.000886|
|ASSIST2015|0.726088 ± 0.000315|0.749526 ± 0.000419|0.846282 ± 0.000392|0.171233 ± 0.000106|0.516098 ± 0.000340|
|Algebra2005|0.810590 ± 0.000422|0.795574 ± 0.000639|0.870356 ± 0.000350|0.144069 ± 0.000364|0.445272 ± 0.001018|

Mean ± sample standard deviation (ddof=1), with seeds 42, 123, 2026, 3407 and 7777. Seed 42 reuses the selected genuine tuning checkpoint. These quantities describe initialization variability on one fixed student split; they do not estimate uncertainty across new student samples. They never report the best seed.

Verification recalculated every saved test-prediction metric, checked checkpoint and prediction hashes, strictly loaded finite checkpoint states, verified the best validation epoch, confirmed unchanged training-source fingerprints and independently reproduced validation-only configuration selection. Independent selected-final CPU inference evidence is reported below; the completed LightSAKT grid has its own original-device inference proof, and remaining baseline checkpoints stay in the final verification schedule.

## Completed baseline evidence

Independent CPU validation and test checkpoint inference verified: 15/15 selected LightSAKT runs. Per-probability and metric tolerances, CPU timings and exact artifact hashes: reports/selected_final_inference_verification.json. Independent verification of all remaining grid/baseline checkpoints remains in the final research schedule.

|dataset|baseline|baseline_seed_count|baseline_auc_mean|baseline_auc_std|light_minus_baseline_auc_mean|
|---|---|---|---|---|---|
|ASSIST2009|DKT|5|0.757165|0.000527|-0.006109|
|ASSIST2009|DKT+|5|0.751202|0.000876|-0.000146|
|ASSIST2009|DKVMN|5|0.755016|0.001395|-0.003960|
|ASSIST2009|SAKT|5|0.750322|0.000648|0.000734|
|ASSIST2015|DKT|5|0.728235|0.000327|-0.002146|
|ASSIST2015|DKT+|5|0.720802|0.001036|0.005287|
|ASSIST2015|DKVMN|5|0.727605|0.000989|-0.001517|
|ASSIST2015|SAKT|5|0.725180|0.000258|0.000909|
|Algebra2005|DKT|5|0.822181|0.000683|-0.011592|
|Algebra2005|DKT+|5|0.823456|0.000649|-0.012866|
|Algebra2005|DKVMN|5|0.817725|0.001099|-0.007135|
|Algebra2005|SAKT|5|0.813373|0.000367|-0.002784|

The following fresh completed baseline primary runs are seed-42, fixed-configuration measurements on the same split. Differences against the five-seed LightSAKT mean are descriptive comparisons with unequal replication and tuning stages, not a matched final ranking or statistical significance result. Both GKT labels refer to the corrected original simplified PAM/MHA implementations, not canonical GKT reproduction.

|dataset|baseline|baseline_auc_mean|light_minus_baseline_auc_mean|light_seed42_minus_baseline_auc|
|---|---|---|---|---|
|ASSIST2009|DKT|0.755294|-0.004238|-0.004786|
|ASSIST2009|DKT+|0.753788|-0.002732|-0.003280|
|ASSIST2009|DKVMN|0.753237|-0.002181|-0.002729|
|ASSIST2009|GKT|0.757626|-0.006570|-0.007117|
|ASSIST2009|GKT_MHA|0.762630|-0.011575|-0.012122|
|ASSIST2009|SAKT|0.750656|0.000400|-0.000148|
|ASSIST2015|DKT|0.726490|-0.000401|-0.000513|
|ASSIST2015|DKT+|0.720112|0.005976|0.005864|
|ASSIST2015|DKVMN|0.726157|-0.000069|-0.000180|
|ASSIST2015|GKT|0.729880|-0.003792|-0.003903|
|ASSIST2015|GKT_MHA|0.732967|-0.006879|-0.006990|
|ASSIST2015|SAKT|0.724443|0.001646|0.001534|
|Algebra2005|DKT|0.822137|-0.011548|-0.011128|
|Algebra2005|DKT+|0.823264|-0.012675|-0.012255|
|Algebra2005|DKVMN|0.809766|0.000824|0.001244|
|Algebra2005|GKT|0.821627|-0.011038|-0.010618|
|Algebra2005|GKT_MHA|0.821574|-0.010985|-0.010565|
|Algebra2005|SAKT|0.812828|-0.002238|-0.001818|

All five baseline metrics, deltas and evidence scope are in LightSAKT_completed_baseline_comparisons.csv. Partial selected baseline seed sets are excluded until all five finish. Paired statistical analysis proceeds only from genuinely matched completed observations.

## Recalculated remaining training time

ETA measurements refreshed: 2026-10-15T15:51:30.782962+00:00. Serial isolated-equivalent remaining training: 461.052 hours. Current execution forecast: 351.018 occupied MIG wall hours (14.63 continuous days), applying the measured 1.512 two-worker throughput factor only to GKT-MHA.

|model|remaining_entries|fresh_training_runs|serial_equivalent_hours|occupied_mig_hours|
|---|---|---|---|---|
|GKT|90|85|136.229717|136.229717|
|GKT_MHA|123|120|324.822509|214.788672|

GKT-PAM and GKT-MHA account for 351.018 of 351.018 remaining occupied MIG hours (100.00%). Completion count is a poor proxy for compute progress. The priority order completed short LightSAKT grids, selected seeds, ablations and fast baseline tuning first. Reuse/alias entries add no training; the current category table identifies remaining GKT work.

The GKT forecast is calibrated by six completed primary model/dataset runs plus real primary/wide engineering profiles, rather than multiplying the global mean runtime by remaining entries. Recent fast-model completions refine those strata but add little evidence about the still-untrained slow GKT configurations. One GKT epoch has measured durations of minutes, whereas many DKT/SAKT/LightSAKT epochs take seconds or less; planned configuration widths and stopping counts also differ.

The occupied-MIG planning envelope is 77.0–1327.6 hours. It is an early-stop/maximum-epoch scenario envelope, not a confidence interval or guarantee. Most future GKT stopping counts and sustained concurrency performance are unknown. Startup, selected test inference and final artifact verification are additional; the completed controlled efficiency benchmark is reused.

## Continuing execution and preserved findings

Live schedule snapshot: 744/957 entries complete; 213 remain. Phase counts: {"5": {"completed": 21}, "6": {"completed": 573, "pending": 182, "running": 1}, "7": {"completed": 75, "pending": 30}, "8": {"completed": 75}}.

Scheduler and independent CPU reporting, Git preservation, checkpoint recovery and offsite workers remain active. No completed training was restarted and no experimental scope was removed. Detached execution survives client disconnect while the environment stays up; no automatic restart after environment shutdown is configured.

The original LightSAKT architecture, causal/padding controls and dataset splits remain unchanged. The preserved efficiency evidence shows LightSAKT slower in all 60 matched latency comparisons. Theoretical O(L) work, runtime, memory and parameter efficiency remain separate in paper/CORRECTED_RESULTS.md and paper/EFFICIENCY_RESULTS.md.

Local data and split hashes are verified. Independent publisher byte identity and explicit redistribution licensing remain unresolved, Algebra uses the available labelled training subset, and simplified GKT fidelity is disclosed.

## Reproduce this summary

```bash
cd /workspace/LightSAKT2
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.light_final_report --refresh-eta --eta-dir reports/summary_eta/latest
```

This command performs CPU-only evidence checks and regenerates reports/tables. It does not start, stop or reset GPU training. Source metric/checkpoint/prediction hashes and per-run verification are recorded in reports/light_final_summary_verification.json.
