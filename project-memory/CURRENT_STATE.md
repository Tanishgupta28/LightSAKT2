# Verified current state — 29 October 2026, completed audited pilot

Branch research/architecture-variants-20261025;worktree /workspace/LightSAKT2-architecture.
Question initial execution b24e154a,exact continuation d958cff4. Original sources,
main dirty state,old epoch18 archive and28October partial reports preserved.

| Variant | Validation AUC | F1 | Status |
|---|---:|---:|---|
| Original one-layer linear | 0.7508035371114405 | 0.8148054891099081 | completed29,best24 |
| Two-layer linear | 0.7507909322800659 | 0.8162621176824227 | completed29,best24 |
| Causal softmax | 0.7456393788778357 | 0.8125468632841789 | completed21,best16 |
| Recency softmax | 0.7482377884727862 | 0.8134333940013463 | completed21,best16 |
| Gated linear | 0.7506452336787726 | 0.8154588400844306 | completed29,best24 |
| Question-difficulty correction | 0.7811899718778572 | 0.8240886674462352 | completed29,best24 |

Question accuracy0.747960372960373,BCE0.5173807926180807,AUCdelta+0.0303864347664167.
Resumed epochs19..29;first18history rows identical,early stop bad_epochs5.
All22Adam counters1134->1827 (63batches/epoch);loader RNG matches exact11epoch
permutation stream andCPU/Python/NumPy RNG continuity passes. Source/config/
feature/observer unchanged. CPU replay AUCdifference1.19364e-8;240protected hashes
unchanged. Five audit/pipeline CPU tests and two SAKT preparation tests PASS.

Leakage scope:validation/test labels do NOT enter fitted difficulty rates;whole
own-student labels excluded for training rows. Real recomputation/perturbations
and synthetic full-builder tests pass. However FULL training-corpus statistics
are used;95.045% of validation rows have later-order training observations of
same problem(order_id proxy). Valid retrospective student-disjoint benchmark,
NOT forward-time causal deployment proof. Observed validation history is allowed
as in originalKT;future validation responses do not enter feature/attention.
Validation reuse/one seed/richer problem-ID input limit claims. No general
architecture superiority or matched-input competitor win established yet.

Saved totaltraining327.943915s;continuation113.740098s;peakallocated254.856MiB,
reserved368MiB. Shared guards passed;no peer interference. Grant closed_completed.
Run ASSIST2009-LightSAKT-pilot-one_layer_question_difficulty-42-v1.
Best/latest under pilot-workspace/checkpoints/<run>/;best SHA256
8be7fb4a227d6c58b7be0691aeccf10ebec29db58674675cff99a417753bffeb;
latest451c3d001fa35d0e21d6c32fab70bc5e5c5149e83fa8e3ef1e12ef1c614e825d.
Source SHA256cf083a73dc0dce9a875c296b4450c7b6f591bb2df6124566dc50078f3113adc0;
feature6231fc5b51317ab921be89647b5101f654dbf8fdb5d841027073df0f29e1e464.
Completed archive/checkpoints/features/raw records:
/workspace/LightSAKT2/backups/question_completion_20261029/completed_checkpoints_features_records.tar.gz.
Full report,verification,epochCSV,PNG/PDF,archive hashes in `/workspace/LightSAKT2-architecture/reports/architecture_study/question_completion_20261029/`.

Replication packet `/workspace/LightSAKT2-architecture/experiments/question_replication_20261029/plan.json`:reuse this seed42;4new LightSAKT seeds+5feature-matched
SAKT controls. Control implemented outside protected src,preserves nativeSAKT,
adds same one-scalar feature;CPU initialization/dropout/gradient/resume tests pass.
Each model retains own original selected hyperparameters;featurematched,not
hyperparameter-identical.9new runs not authorized/launched. Proposed ceiling
9sequential900s windows,4GiB each,total135GPU-minutes. Existing Git transport auth
blocker retained;local commits/bundles only. Test remains untouched.
