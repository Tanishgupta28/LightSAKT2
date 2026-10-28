# Historical handoff before the question-difficulty outcome — 28 October 2026

All current/next statements below are historical snapshots,superseded by root
files and actual artifacts. Relative index links were adjusted to the memory root.


## Prior README.md

Original text SHA256 `cb4fdafe0c9463412abf2ebcead59811724bba5dc138f5d4dc2ff8b9621c74c5`.

# LightSAKT2 durable project memory

Canonical memory: `/workspace/LightSAKT2/project-memory`.
Active research worktree: `/workspace/LightSAKT2-architecture`,branch
`research/architecture-variants-20261025`. Main remains `research-rebuild` with
pre-existing dirty scientific state preserved. Actual code,Git,checkpoints and
raw results outrank memory. Branch copies are synchronized versions of this
same memory,not an independent study. Older LightSAKT_Project memory is unrelated.

Read [USER_DIRECTIVES.md](../USER_DIRECTIVES.md),[CURRENT_STATE.md](../CURRENT_STATE.md),
[DECISION_LOG.md](../DECISION_LOG.md),then [NEXT_CHECKPOINT.md](../NEXT_CHECKPOINT.md).
The [historical handoffs](../archive/HANDOFF_BEFORE_GATING_RESULT_20261027.md)
preserve superseded recovery/progress/directive text,including interruptions.

Latest: gated linear completed29epochs,best24,AUC0.750645234 versus original
0.750803537. No AUC improvement. Five seed42 arms completed; retain original.
No active training/additional scope. CPU checkpoint replay and146 protected-file
hashes PASS;89 pre-run CPU tests PASS. Grant closed;test untouched.
Skill: `/root/.codex/skills/durable-project-memory/SKILL.md`.
Use Resume/Update/Audit and strict structural plus manual artifact validation.


## Prior USER_DIRECTIVES.md

Original text SHA256 `93f6a4093d549c597d6fcecb2c7df5c20cd4640265d3136d304880800d43b35f`.

## Current diagnostic and pilot scope — 28 October 2026

User authorizes train/validation failure analysis and one strongest defensible
ASSIST2009 seed42 pilot,conditional on CPU correctness and existing shared GPU
checks,4GiB/900s caps. Diagnostics selected a known-question difficulty correction
with train-student exclusion;practice-count correction remains a deferred
independent hypothesis. Preserve original data/splits/models/results;no test
inference or tuning on test. Richer known problem-ID input is explicitly disclosed.
Same optimizer/budget/selection. No arbitrary second pilot. This supersedes the
previous task's consumed-scope statement for this one new invocation only.

# Durable user directives — current through 27 October 2026

Objective: improve LightSAKT validation AUC with reproducible isolated architecture
experiments. Artifacts,checkpoints,Git and reports are truth;memory is an index.
Preserve professor-shared original LightSAKT/SAKT,prior results and dirty main work.
Do not rerun completed investigations without a specific new justification.

Latest user instruction authorized one minimal gating variant preserving causal
ELU+1 linear attention and one encoder block,plus exactly one controlled
ASSIST2009 seed42 run. This scope was consumed successfully;grant closed.
Same splits,vocabulary,seed,Adam,hyperparameters,epoch budget,early stopping and
validation-AUC checkpoint selection. No test evaluation or settings changes to
obtain better scores. The gate did not improve AUC;no further unapproved
architecture changes or GPU experiments. Future research needs a new decision.

Existing intentional shared-GPU permission is conditional on scoped current
execution and admission. Preserve4GiB allocator/900s invocation limit,10GiB
admission/6GiB runtime-free floors,4GiB RSS/2GiB host headroom,0.1s pacing,30s step
stop and recorded unsupported-MIG-utilization fallback. Never displace peers,
remove general holds or retry automatically. Free memory is not exclusivity.

User grants autonomous tools for authorized work and asks not to repeat permission
questions. Preserve local commits and verified backups. Existing Git transport
credentials are missing;report without exposing secrets,reset,force push or
reconstructing history. Do not message others. Future publication needs working
authenticated transport and applicable user scope.

Earlier instructions/reversals are retained in the historical handoff archive
and append-only decision log. Existing general hold markers remain byte-identical;
latest scope used only an expiring exact-config/hold-hash exception.


## Prior CURRENT_STATE.md

Original text SHA256 `ede20d067a6bc2460ddad67f709ab4cdf60f8db8caac1d314abb0fcf80dd5833`.

# Current investigation in progress — 28 October 2026

Prior five seed42 architectures remain completed;original is best among them.
Current user authorized evidence-led diagnostics and one conditional pilot.
Diagnostics selected missing known-question difficulty,not more capacity or
attention changes:train-fitted question-feature probe0.780203 versus0.750804.
Own-student labels excluded for training features;validation uses training only.
93 CPU tests PASS;one-scalar readout preserves baseline at initialization.
Execution b24e154a,question stage,4GiB/900s/shared safeguards. Richer known item
information means this is an input-representation comparison,not kernel superiority.
Check actual terminal state before acting;no second run or automatic retry.
Outputs:/workspace/LightSAKT2-architecture/pilot-workspace/results/raw/ASSIST2009-LightSAKT-pilot-one_layer_question_difficulty-42-v1/.
Feature sidecar is ignored/local,data/auxiliary in pilot-workspace;hashed in config.
Prior task's closed-scope next actions are superseded for this invocation only.

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


## Prior NEXT_CHECKPOINT.md

Original text SHA256 `f95d9690892c32825eebfba72ba8ed4b2190d488fbd7c5e3932df067c02cb3a4`.

# Exact next checkpoint — gating result recorded

Current task's authorized experiment is complete. Retain original one-layer
linear LightSAKT:AUC0.7508035371114405;gated0.7506452336787726 did not improve.
Do not launch another variant,initialization sweep or retry under the consumed
scope. Do not repeat the five completed seed42 architecture comparisons.

Next action is a user research decision,with the measured comparison at
`/workspace/LightSAKT2-architecture/reports/architecture_study/gating_pilot/RESULT_20261027.md`.
No new architecture is approved or required to finish this experiment. A later
credible positive AUC result would justify proposing five seeds42/123/2026/3407/
7777 on ASSIST2009/ASSIST2015/Algebra2005,using paired validation deltas and mean/
sample SD. This gating result does not meet that criterion. Keep test locked.

On resume,check branch/current terminal statuses and checkpoint/archive hashes,
read current directives and decision-log tail before any work. General holds stay
active;no automatic scheduling. Use exact execution commit7026c660 for source
reconstruction and the verified checkpoint archive above;Git alone has no weights.
Preserve local result commits and Git bundles. Publication remains blocked by
missing authenticated Git transport;no reset/force push/secret exposure. No
repeated authentication probing is needed unless credentials actually change.
