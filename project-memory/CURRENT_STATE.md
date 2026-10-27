# Verified current state — 27 October 2026, gating completed

Branch `research/architecture-variants-20261025`;execution commit `7026c660`.
Worktree `/workspace/LightSAKT2-architecture`. Original main branch/dirty work kept.
All five ASSIST2009 seed42 arms completed (validation only):

| Architecture | AUC | F1 |
|---|---:|---:|
| Original one-layer linear | 0.7508035371114405 | 0.8148054891099081 |
| Two-layer linear | 0.7507909322800659 | 0.8162621176824227 |
| Causal softmax | 0.7456393788778357 | 0.8125468632841789 |
| Recency softmax | 0.7482377884727862 | 0.8134333940013463 |
| Gated one-layer linear | 0.7506452336787726 | 0.8154588400844306 |

Gating is a valid one-seed negative versus baseline:delta AUC -0.0001583034326679833.
F1 +0.0006533509745224597 does not replace the primary AUC criterion. Retain
original;no tested variant has demonstrated an AUC gain on this screening.
Gated run completed29epochs,selected24,accuracy0.7325174825174825,
BCE0.5426217429983549. Runtime412.977059s;peak allocated254.580MiB,reserved368MiB.
Gate65parameters,2sigmoid(w*E(next_skill)+b),identity initialization;linear
attention,one block,residuals,FFN/head unchanged. Selected vocabulary gates
0.4326034784..1.6664240360,mean1.0659102201;gate learned but AUC did not improve.

89 CPU tests PASS;independent CPU validation replay AUC difference8.9523e-9,
all six validation metrics pass;finite full model/Adam/RNG/loader/history state.
All146 protected prior source/checkpoint/raw-record/report/manifest/hold hashes
unchanged. No test loader/prediction access. Shared safeguards passed;peer GPU
workload untouched. Grant closed_completed,no retry/additional GPU scope.

Run `ASSIST2009-LightSAKT-pilot-one_layer_gated_linear-42-v1`.
Best/latest: `/workspace/LightSAKT2-architecture/pilot-workspace/checkpoints/ASSIST2009-LightSAKT-pilot-one_layer_gated_linear-42-v1/`.
Best SHA256 `847a042cc340f890da1616ad8718814892aa8d6ccfae932f56491782e34db7db`.
Persistent verified archive:
`/workspace/LightSAKT2/backups/architecture_gating_20261027/pilot_checkpoints_and_records.tar.gz`.
Detailed config/raw metrics/history/status/resource/replay/archive records:
`/workspace/LightSAKT2-architecture/reports/architecture_study/gating_pilot/RESULT_20261027.md` and neighboring JSON/CSV files.
Code source SHA256 `763839be531388d44be79bd278cb49bd129c264b6820637be8e0d75285875aa4`.
New gated model/factory entry explains changed combined source hash;baseline
model/attention/trainer unchanged. Existing Git transport authentication blocker
remains;local research commits/checkpoints preserved. No push claimed.
Historical GKT interruptions and prior archive paths remain in the archive/log;
no old pending baseline work was resumed.
