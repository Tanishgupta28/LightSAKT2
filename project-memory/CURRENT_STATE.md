# Verified state — 28 October 2026, interrupted promising pilot

Branch research/architecture-variants-20261025;execution commit b24e154a.
Worktree /workspace/LightSAKT2-architecture;originals/main dirty work preserved.

| Model | Validation AUC | F1 | Status |
|---|---:|---:|---|
| Original one-layer linear | 0.7508035371114405 | 0.8148054891099081 | completed29epochs,best24 |
| Two-layer linear | 0.7507909322800659 | 0.8162621176824227 | completed29,best24 |
| Causal softmax | 0.7456393788778357 | 0.8125468632841789 | completed21,best16 |
| Recency softmax | 0.7482377884727862 | 0.8134333940013463 | completed21,best16 |
| Gated linear | 0.7506452336787726 | 0.8154588400844306 | completed29,best24 |
| Question-difficulty correction | 0.7784249958819442 | 0.8239629026193759 | INTERRUPTED18,best18 |

New saved accuracy0.744208916083916,BCE0.5198850343652965;deltaAUC+0.027621458770503704.
One seed,reused validation and additional known problem-ID input;not general
superiority over same-input competitors. Original kernel/depth/settings unchanged.
One neural scale on a frozen problem-vs-skill difficulty feature;training rows
exclude ALL own-student labels,validation uses training only. Fixed smoothing10;
scale0initially,learned0.790180266. Per-problem training statistics add information
and representational capacity despite only one new neural parameter.

Diagnosis: train-eval AUC0.765195 vs validation0.750804;zero-attention0.688005;
36-grid settings already explored. Skill-only input misses question difficulty.
Question-feature diagnostic probe0.780203 versus weaker practice-count0.751670;
not independent confirmation. Error/frequency/history/curves/grid/checkpoint
review: `/workspace/LightSAKT2-architecture/reports/diagnostics_20261028/DIAGNOSIS_AND_SELECTION.md` and neighboring aggregates.

Resource interruption: Host memory headroom floor breached;last saved epoch18,
bad_epochs0,early stopping NOT reached. Full model/Adam/RNG/loader/history saved.
Any unsaved work beyond last complete epoch is not claimed preserved. No retry.
Grant closed_interrupted. Saved elapsed214.203818s;recorded completed-epoch peak
allocated254.627MiB/reserved368MiB. CPU validation replay difference2.08887e-8;
180 protected-file hashes unchanged.93 CPU tests PASS prelaunch;no test dataset.
Current normal metrics.json intentionally absent;separate partial_result.json
and partial_checkpoint_verification_20261028.json preserve honest status.

Run ASSIST2009-LightSAKT-pilot-one_layer_question_difficulty-42-v1;
best/latest under /workspace/LightSAKT2-architecture/pilot-workspace/checkpoints/<run>/.
Best SHA256 80a2ccc0b7f81d99f459bec3a1b8094f934d4d0df9b89123f78939643323eb03.
Latest SHA256 e134577a436c125274e00629b510739482b77b443b3db8d49d7c79092aca2c24.
Feature SHA256 6231fc5b51317ab921be89647b5101f654dbf8fdb5d841027073df0f29e1e464.
Source SHA256 cf083a73dc0dce9a875c296b4450c7b6f591bb2df6124566dc50078f3113adc0.
Read-back-verified archive includes REQUIRED feature sidecar and checkpoints:
/workspace/LightSAKT2/backups/question_pilot_20261028/interrupted_epoch18_checkpoints_features_records.tar.gz.
Full evidence `/workspace/LightSAKT2-architecture/reports/architecture_study/question_pilot_20261028/PARTIAL_RESULT_20261028.md`.
No push claimed;existing Git transport authentication blocker retained.
