# Decision log

## 2026-10-20 — Preserve stopped workload and hold automatic restart

New user priority permits stopping verified LightSAKT2 only. Direct OS state and
logs supersede old active-heartbeat interpretation: both workers had already
interrupted. No further signal warranted. Full before/after digest inventory and
strict epoch2 CPU recovery check pass. Added marker and operational launch gates,
preserving all scientific source/config/results and pending schedule.

9 final CPU hold tests pass; initial import-trap failures retained as harness
failures, not hidden. No GPU probing by these tests. A separate local archive
preserves the interrupted epoch boundary; no claim of fresh offsite durability.

Unrelated GPU job observed on same device capabilities, and no allocation
authority available. Do not infer exclusivity from stopped LightSAKT2 or free
VRAM. HiTSKT CUDA/five-step pilot remains unrun pending allocation evidence.
External scheduling visibility is limited; known local recovery/entrypoints
refuse the hold. No watcher, automatic resume or new training authorized.


## 2026-10-20 — Allocation-first user clarification

User has no administrator/scheduler exclusivity evidence. Both projects are
owned by user, but this does not override other reservations. Preserve all
LightSAKT2 checkpoints/results; no stopping solely for GPU release before HiTSKT
assignment. No CUDA/training without verified authorization. Supersedes earlier
shutdown-first order, without requesting any restart or hold release. Current
work limited to memory and administrator-request documentation; no fresh process/
GPU probes or operational changes. Prior stopped status/zero signals preserved
as historical evidence. Next task is obtaining authoritative assignment details.


## 2026-10-20,09:24UTC — Renewed shutdown authorization, existing-allocation check

User again permits verified LightSAKT2 shutdown/prevention of restart and actual
existing-reservation reuse for bounded HiTSKT stages; no unnecessary new booking.
One read-only check finds no live LightSAKT2 trainer/controller/GPU handles; hold
intact,813 completed/142 pending/2 interrupted. Zero signals needed or sent.
7915 files present/same size,8 key hashes and recovery archive match; no repeated
full audit/tests. MIG now nearly empty, zero visible GPU owners, superseding older
busy snapshots only at observation time. Still no existing owner/account/project/
transfer/window/no-conflict reservation evidence, Kubernetes403/no scheduler.
Stop before CUDA. Specific blocker is existing allocation permissions, not new
user pilot authorization or mandatory separate hardware. No CUDA/optimizer/test
metrics, model/data/DKT edits, watcher or allocation change. Fresh report/evidence
under reports/favor_plus_v1/EXISTING_ALLOCATION_REUSE_STATUS_20261020.md and
existing_allocation_reuse_20261020_v1/inspection.json in HiTSKT. Preserve all
original artifacts and scoped branch publication; later full training needs approval.


## 2026-10-27 — Architecture recovery, memory installation and bounded preparation

Recovered research commit6beced52 from missing /tmp worktree into persistent
/workspace/LightSAKT2-architecture without resetting main or pruning history.
Verified depth archive and all22 recorded member hashes; restored14 immutable
files. Baseline0.7508035371114405 versus depth0.7507909322800659 is a valid
single-seed null result,29 epochs each/selected24. Softmax preparation survives;
no valid completed run survives. Unknown lost execution is not claimed absent.
Hash-bound summary checkpoint relocation implemented and80 CPU checks pass.
Original models/trainer, archived metrics and holds preserved. Initial pytest
setup errors and overly broad trainer-equality assertion were corrected and
recorded in the recovery report. No GPU training or test evaluation.

Installed reviewed durable-project-memory upstream0a06f1bb8231ace20e4405281a263012fedb5e51
through standard installer. Resume/Update/Audit use canonical existing memory.
New user research scope supersedes old documentation-only HiTSKT priority,
while retaining explicit GPU budget boundaries. Next: current budget/window for
one900s/4GiB seed42 softmax pilot, then validation-only replay/comparison.
Full evidence: /workspace/LightSAKT2-architecture/reports/architecture_study/RECOVERY_20261027.md.
No push or external message is authorized by this checkpoint.


## 2026-10-27 — Authorized softmax completed; valid negative result

User explicitly authorized exactly one ASSIST2009 seed42 causal-softmax run,
4GiB/900s cap and shared safeguards, plus safe commit/push. No valid prior softmax
artifacts existed; existing controls and recovered depth checkpoints passed
preflight. Fresh run from b63ec478 completed21epochs, selected16:
AUC0.7456393788778357/F1 0.8125468632841789/accuracy0.7268356643356644/
BCE0.5465822575284047, versus original AUC0.7508035371114405.
Delta -0.005164158233604832: retain original; no softmax promotion.
Runtime161.204s, allocated133.972MiB/reserved160MiB. CPU validation replay,
finite full resume state and original-file preservation checks PASS.
No test evaluation; unchanged general holds; grant closed; no retry or peer signals.
Archived best/latest and raw records persist outside /tmp. Report and hashes:
/workspace/LightSAKT2-architecture/reports/architecture_study/softmax_pilot/RESULT_20261027.md and recovery_archive_20261027.json.
Next prepared hypothesis: recency bias, separately authorized and single-arm
runner needed to avoid repeating completed softmax. No recency run in this task.


## 2026-10-27 — Verified result committed; exact-history push blocked

Result commit f15b28b9a30117a9cfdcea677f398ba2b0967713. Non-force push failed because shell Git has no
HTTPS credential; gh logged out/no SSH key. Connected GitHub app profile works,
but exposed commit/tree APIs cannot transfer exact local commits. No remote
mutation or reconstructed history. Preserved a verified local Git bundle at
/workspace/LightSAKT2/backups/architecture_softmax_20261027/research_commits.bundle; checkpoint archive remains separate. Publish existing branch with
ordinary push when authenticated Git transport is available; no training retry.


## 2026-10-27 — Single recency pilot completed; original still best

User authorized exactly one bounded recency-softmax ASSIST2009 seed42 run.
Added single-recency launch stage without scientific config/model/trainer edits;
82 CPU tests PASS. Three prior arms verified and no prior recency artifacts.
Execution from27e5a486 completed21epochs,best16: AUC0.7482377884727862,
F1 0.8134333940013463,accuracy0.7274548368298368,BCE0.544185487825003.
Delta-baseline -0.0025657486386543304,delta-softmax +0.0025984095949505015.
Valid one-seed negative against original; no overall improvement or automatic
five-seed promotion. CPU replay/full resume/116 historical file hashes PASS.
Learned head decays0.0381679125/0.0389851369 versus initial0.03125.
Runtime167.035566s,allocated134.050MiB,reserved160MiB. No test use or peer signals;
shared headroom/cap/pacing respected; exact grant closed_completed, no retry.
Archived checkpoints/raw records and epoch metrics: /workspace/LightSAKT2-architecture/reports/architecture_study/recency_pilot/RESULT_20261027.md.
Next hypothesis after source/literature review: normalized recency decay inside
ELU+1 prefix attention. AKT/RetNet motivate decay but do not establish this
transplant's performance. No new arm implementation/training performed.
Missing shell Git authentication remains; preserve all local commits.


## 2026-10-27 — Minimal gated linear completed; no AUC improvement

User selected gating instead of prior decayed-linear proposal. Added separate
LightSAKTGated with g=2sigmoid(w*E(qry)+b),65new parameters,zero-initialized
identity preserving original init/RNG/forward/gradients. Original attention,
one block,residual/norms/FFN/head and baseline source/settings unchanged.
89 CPU correctness/pipeline/scope/resume tests PASS. Execution7026c660,
exactly one ASSIST2009 seed42 run,29epochs,best24: AUC0.7506452336787726,
F1 0.8154588400844306,accuracy0.7325174825174825,BCE0.5426217429983549.
AUC delta -0.0001583034326679833 versus original0.7508035371114405;
valid one-seed negative,not generalized equivalence. Small F1 increase is not
AUC success. No five-seed promotion or further architecture changes after outcome.
Runtime412.977059s,allocated254.580MiB,reserved368MiB. CPU validation replay,
full finite resume state and146 protected-file hashes PASS;test untouched.
Gate parameters learned (vocabulary range0.4326..1.6664) without endpoint gain.
Shared safeguards maintained amid visible peer workload;no signals;grant closed.
Checkpoint/archive/raw records: /workspace/LightSAKT2-architecture/reports/architecture_study/gating_pilot/RESULT_20261027.md.
Consolidated current memory;preserved superseded overview/directives/state/next
text in archive/HANDOFF_BEFORE_GATING_RESULT_20261027.md and old decisions here.
Git auth blocker retained;local scientific commits backed up without rewriting.


## 2026-10-28 — Diagnostics select question representation; pilot interrupted

Current user authorized diagnostics and one strongest conditional pilot. Audited
five completed checkpoint pairs/curves and36-grid settings,CPU train/validation
inference,error/frequency/history and response interventions. Baseline uses
history (zero attention AUC0.688005) and modest gap(train0.765195,val0.750804);
composite-skill input hides problem differences. Question difficulty probe0.780203,
stable training-student subsets;weaker count probe0.751670 deferred. All exploratory
on reused validation,no test. Whole-chunk shuffle is noncausal corruption and is
excluded from valid alternate-prediction/selection claims. Full diagnostic report:
/workspace/LightSAKT2-architecture/reports/diagnostics_20261028/DIAGNOSIS_AND_SELECTION.md.

Implemented LightSAKTQuestion:one zero-init neural scale on frozen known-problem
relative difficulty;training target encoding removes whole own student from BOTH
rates,validation uses training labels only;fixed smoothing10,no sweep. Richer
input,not a same-input attention superiority claim.93 CPU tests PASS,including
leakage,causality,padding,gradients,adapter/reporting/exact-resume and test refusal.
Executed fromb24e154a with unchanged baseline split/Adam/settings/budget/selection.
Host headroom guard stopped it after18complete epochs,best18,bad_epochs0. Saved
AUC0.7784249958819442/F1 0.8239629026193759/accuracy0.744208916083916/
BCE0.5198850343652965;delta original+0.027621458770503704. Promising PARTIAL
endpoint,not completed early stopping. Saved elapsed214.203818s,recorded epoch
allocated254.627MiB/reserved368MiB. Grant closed_interrupted;no retry/resume.

Independent CPU partial replay AUC difference2.08887e-8,full finite checkpoint
state/source/data/feature/grant bindings and180 protected-file hashes PASS.
No metrics.json fabricated;separate partial records and raw interrupted status.
Archive contains best/latest AND required feature sidecar;member hashes verified.
Evidence /workspace/LightSAKT2-architecture/reports/architecture_study/question_pilot_20261028/PARTIAL_RESULT_20261028.md. Next: finish same run with new scope and
headroom before five seeds/matched-input controls. ASSIST2015 current release
lacks problem IDs;do not promise an exact transplant or silently change schema.
Canonical memory consolidated;prior handoff archived,local history/bundles kept.


## 2026-10-29 — Audited exact continuation completed; gain retained

New user authorized audit and same-run completion if valid,no hyperparameter
change. Feature rebuild matches exactly;real validation-label and whole-own-
training-student perturbations invariant;synthetic full-builder test/validation/
raw correctness perturbations invariant. No actual test evaluation/statistics.
95.045% validation rows have later-order training same-question observations:
full-training fitted statistics are valid for originaloffline student-disjoint
protocol,NOT forward-time deployment. This boundary is explicit,no silent fix.
Checkpoint18/optimizer/RNG/data order/source/adapter/launcher/observer verified.

Continuation d958cff4 used unchanged originalengine/config/model;restored latest18,
ran19..29,first18history entries unchanged,best24,bad_epochs5. FinalAUC
0.7811899718778572/F1 0.8240886674462352/accuracy0.747960372960373/
BCE0.5173807926180807;delta original+0.0303864347664167. CPU replay difference
1.19364e-8;240protected file hashes unchanged. Adam22states1134->1827;loader
permutation stream andCPU/Python/NumPy RNG continuity exact. Total saved training
327.943915s,continuation113.740098s,allocated254.856MiB/reserved368MiB. Shared
safety passed,no peer signals,grant closed_completed. Old18archive/reports kept;
new finalcheckpoint/features/rawarchive member hashes verified. Evidence:
/workspace/LightSAKT2-architecture/reports/architecture_study/question_completion_20261029/FINAL_RESULT_20261029.md. Five current audit tests+two control tests PASS.

Prepared10configs:reusecompleted LightSAKTseed42,4new LightSAKT seeds+5new
featurematched SAKT. Control preserves originalsource,one same feature/logit scale,
CPU init/dropout/gradient/exactresume passes. Model-specific selected settings
retained;matchedinput does not mean identicalhyperparameters. Proposed9serial
900s/4GiB windows=135GPU-min ceiling NOT APPROVED;no campaign launched. Need
budget and guardedseed-general execution. Paired5seed deltas/mean/sampleSD,
no general/online claim yet. Source/test/history/Git preserved,current memory
consolidated with prior state archived. Local commits/bundles only.

## 30 October 2026 — checkpoint replay and feature perturbations verified

All 30 existing SAKT/LightSAKT baseline seeds passed independent CPU validation/test
replay. Reused modified ASSIST2009 seed42 reproduces validation AUC 0.781189972
and test AUC 0.778555249. First new seed123 completed with validation AUC
0.780341551 and passed CPU held-out replay; remaining frozen campaign is active.
Evidence: reports/professor_results_20261030/replays/ in research worktree.
Real-data validation-label and entire-own-student-label flip invariance passed
for ASSIST2009 and Algebra2005, with responsive other-student positive control.
Saved optimizer/RNG and exact data-order stream verified for completed runs.
Evidence: real_data_leakage_audit.json and checkpoint_state_audit.json in that
report directory. No forward-time or equal-input attention superiority claim.
Current user explicitly requests final test scores and missing seed runs; this
supersedes older preparation-only/test-locked instructions for frozen campaign.
ASSIST2015 modified row stays PENDING because its source lacks item metadata.

## 30 October 2026 — ASSIST2009 five-seed difficulty replication complete

All five frozen seeds passed CPU validation/test replay and independent target/
rank-AUC/confusion-metric audit. Modified validation AUC 0.780247982153
(sampleSD 0.001153303456); test AUC 0.779358707815
(sampleSD 0.000778165242). Paired mean test AUC gain over LightSAKT
0.028302772406, positive5/5seeds. Both AUC and F1 improve on
this fixed student split; no broader deployment/kernel-superiority claim. All
completed checkpoints, optimizer/RNG/data order and backups verified. Evidence:
reports/professor_results_20261030/summary_results.json, paired_seed_auc_deltas.csv,
masking_and_metrics_audit.json, checkpoint_state_audit.json, checkpoint_backups.json.
Algebra2005 campaign active; ASSIST2015 missing question metadata stays PENDING.

## 30 October 2026 — first Algebra2005 difficulty seed verified

Seed42 completed43/selected38 with validation AUC0.8352874266647624; independent
CPU validation/test replay passed, test AUC0.836898773. Original matched LightSAKT
test AUC0.811009501. Other four seeds remain the active frozen campaign, not a
completed cross-seed claim. Checkpoint archive readback passed. Official SIGKDD
format confirms the frozen (Problem Name,Step Name) identity as supplied predictor
metadata; no feature/configuration change. Evidence: replays/Algebra2005-LightSAKT+Difficulty-42.json,
external_field_provenance.json and checkpoint_backups.json in professor report folder.

## 30 October 2026 — second Algebra2005 seed verified

Seed123 completed40/selected35: validation AUC 0.835734920590, test AUC 0.837020732773. CPU replay, checkpoint/optimizer/RNG/data-order, independent mask/metric and archive readback audits PASS. Evidence: professor_results_20261030/replays/Algebra2005-LightSAKT+Difficulty-123.json and current audit manifests. Three remaining frozen Algebra seeds are active/pending; no all-seed or all-three-dataset claim.

## 30 October 2026 — third Algebra2005 seed verified

Seed2026 completed39/selected34: validation AUC 0.835612277030, test AUC 0.836540566670. CPU replay, checkpoint/optimizer/RNG/data-order, independent mask/metric, real-data perturbation and archive readback audits PASS for all eight completed difficulty runs. Evidence: professor_results_20261030/replays/Algebra2005-LightSAKT+Difficulty-2026.json and audit manifests. Two prescribed Algebra seeds remain active/pending. No configuration changes.

## 30 October 2026 — fourth Algebra2005 seed verified

Seed3407 completed44/selected39: validation AUC 0.834533667248, test AUC 0.836798929931. CPU replay, checkpoint/optimizer/RNG/data-order, independent mask/metric and archive readback audits PASS for all nine completed difficulty runs. Resource summary confirms no sampled runtime-floor/RSS breach. Evidence: professor_results_20261030/replays/Algebra2005-LightSAKT+Difficulty-3407.json and audit manifests. Final prescribed seed7777 is active; no extra run or settings change.

## 30 October 2026 — final professor comparison verified

All9new runs completed, no retries/settings changes/unsafe stops. Final forty
model-seed checkpoints passed CPU validation/test replay;80prediction arrays
passed exact target/mask and independent metric checks. Ten difficulty checkpoint
optimizer/RNG/data-order audits PASS;338original fingerprints unchanged;7CPUtests
PASS. Complete five-seed difficulty results on ASSIST2009 and Algebra2005 have
positive validation/test AUC deltas in every matched seed versus both baselines.
Mean test gains versus LightSAKT:+0.028302772 and+0.026281231. Mean F1/accuracy
also improve. ASSIST2015 difficulty remains5PENDING slots because item identity
is absent, with no substitute or fabricated observation. All limitations are in
PROFESSOR_REPORT.md. Native training3832.883621s, admitted windows3850.931233s;
max allocated818MiB;no sampled floor/RSS breach. Scope consumed, controller exited0.
Evidence: summary_results.json, paired_seed_auc_deltas.csv, delivery_checks.json,
checkpoint_state_audit.json, masking_and_metrics_audit.json, resource_summary.json,
checkpoint_backups.json in reports/professor_results_20261030/. Final validation,
research-branch result commit and verified incremental Git bundle follow.

## 30 October 2026 — verified results committed and locally bundled

Result commit c286bcbcee46a5317dbb3071c1f6c3a66335d04a on research/architecture-variants-20261025.
Incremental bundle /workspace/LightSAKT2/backups/professor_results_20261030/research_results-c286bcbc.bundle verifies; prerequisite a4414cf2 and its earlier bundle
are preserved. Thirteen checkpoint/feature/rawrecord/prediction archives passed
readback. Both memory roots passed strict validation without errors/warnings;
manual claims agree with40replays/80mask-metric checks/10state audits/338protected
hashes. No push or credential access. Current next action is to resolve ASSIST2015
metadata or separately scope the five feature-matched SAKT controls; current
nine-run GPU grant is consumed. No active campaign job or automatic continuation.

## 31 October 2026 — verified portable professor release; ASSIST2015 bounded blocker

Fresh read-only checks pass:40best checkpoints/configs/shapes/selectedepochs,
80saved prediction arrays and independent metrics,10difficulty optimizer/RNG
states,338protected files,and every mean/sample SD/paired difference. All77old
report files are unchanged. Zero new training/inference,GPU use or HiTSKT edits.
Official publisher CSV fetched through Drive matches local raw AND preserved
raw archive exactly (SHA75e1f46131d5897a388c82411bc3032279899cb1f1e66a9043354d41234407a0).
Only user_id/log_id/sequence_id/correct;100skill-builder sets and708631unique
events. Official semantics and pyKT primary source corroborate no independent
item identity;no authentic mapping found. Mark5modified slots unavailable under
current protocol. Provider database mapping might exist;do not claim otherwise.
Configuration clarification:baseline final evaluate_test=true means post-training
scoring,modified trainingfalse;alltabletest values use saved CPU replay and
validation-only selection. This corrects earlier broad 'all training false'
wording without changing any configuration or metric. Paired gain columns are
modified minus named comparator,separate validation/test,absolute AUC units.
Portable ZIP /workspace/LightSAKT2/backups/professor_release_20261031/LightSAKT_question_difficulty_verified_results_20261031.zip (SHAabf2feec4c31669006f6731ef7689742f7e549063d699e620e74cf83c4ffe8f0)
passes CRC,safe extraction,184payload SHA checks and standalone verification.
40checkpoint references/configs included;no weights/rawstudentdata/credentials.
Three existing incremental bundles verify;no remote push. Suitable to send as
verified progress,not publication readiness;feature-matched SAKT remains missing.
Evidence:/workspace/LightSAKT2-architecture/reports/professor_release_20261031/verification_summary.json,assist2015_official_release_audit.json,
ASSIST2015_LIMITATION.md,archive_verification.json and main/concise reports.

## 31 October 2026 — release committed and safely backed up

Commit 2103a6e3e2ff79fa09af3b01406b02f404b58733 records the verified new reports/tables/provenance/checkpoint
references and portable archive SHA. Bundle /workspace/LightSAKT2/backups/professor_release_20261031/research_release-2103a6e3.bundle verifies against preserved
49982fa7. All three preceding bundles also verify;no full-history/self-contained
bundle claim. Portable ZIP SHAabf2feec4c31669006f6731ef7689742f7e549063d699e620e74cf83c4ffe8f0
and184payload checks PASS. Original77report files remain byte-identical. Canonical
and worktree memory both strict PASS;zero GPU/inference or remote publication.

## 4 November 2026 — replacement scope and real EdNet sources recovered

Current user approves ASSIST2015replacement and missing five-seed three-model
comparison,conditional on safe allocation. EdNet official links resolve to Drive;
connector256MiB limit prevents KT1fetch,so official publicusercontent endpoint
was streamed locally without credentials. Complete1201163816byte KT1ZIP SHA
0d13933f90201c5101c7fe8659e44474fa049e3fb93181a8ba6fb3e63267b535;
Contents ZIP173976bytes,metadata13169uniquequestionIDs/1792tagsets/answer keys.
Same approved shared MIG has no listed compute processes and finite~39GiBfree;
current admission passes. No CUDA/training launched yet. EdNet subset/session/
common-hyperparameter protocol frozen before any new model scores in
experiments/ednet_replacement_20261104/protocol.json. Full prior338protected
hashes and152historical report/config hashes preserved. HiTSKT remains untouched.

## 4 November 2026 — real EdNet data/feature audits and core configs frozen

Dataset EdNetKT1_5k_first retains563469interactions/5000students/11893authentic
questions;seededstudentchoicebeforeoutcomes,onefirstitempersession toavoidbundle
response leakage.3500/500/1000students,407242/43375/112852interactions. SixCPUtests
PASS;realwholeownstudent/validationlabel perturbation and feature reproduction
PASS. Commonnative d64/h2/dropout.1/lr.001/B64/L100/max50/patience5 fixed before
model scores;15configs andfivepreparedoptionalcontrols boundinnewplan. Native
models/splits/loss/metrics unchanged;previous datasets not rerun. Exactexisting
sharedMIG currently passesadmission;source/config/hold/data fingerprints retained.
Next safe action:commit andstartguarded15corewindowsunder currentuserscope.

## 2026-11-04 — EdNet execution milestone

Preparation commit5f1b830d and prerequisite-bound ednet_preparation.bundle verified.
Same approved shared-MIG admission passed before CUDA; launched core ONCE, PID1082738.
SAKT seed42 completed13epochs, best8 validation AUC0.6655735410712708;
next LightSAKT seed42 is running. All15configs frozen;independent CPU replay
started for completed checkpoints without test-guided changes. No HiTSKT access.

## 2026-11-04 — First complete matched EdNet seed

SAKT seed42: epochs13, best8, validationAUC0.665573541, CPUtestAUC0.639857916; exact checkpoint/config/Adam/RNG/loader/masks/metrics PASS.
LightSAKT seed42: epochs19, best14, validationAUC0.669818334, CPUtestAUC0.642402337; exact checkpoint/config/Adam/RNG/loader/masks/metrics PASS.
LightSAKT+Difficulty seed42: epochs19, best14, validationAUC0.728430952, CPUtestAUC0.706335369; exact checkpoint/config/Adam/RNG/loader/masks/metrics PASS.
Other four seeds remain running, allconfigsunchanged; no generalizationclaimyet.
CPU replay initial formatting assertion was corrected: frozen JSON SHA stays bound,
native config parsed equality allows trailing-newline difference. No training/data
change or repeated completed inference; successful replays cached withhashes.

## 2026-11-04 — Full raw-row and temporal audit

Independently aligned all563469retained rows to officialZIP/user source rows and
correct_answer metadata;source IDs/chronology/labels/first-session-selection
exact,without reusing clean_student. Offline temporal limitation quantified:
99.6012%validation and99.5887%testtargets have later-dated training observations
in their item/skill aggregates;item-only96.8%/95.3930%. These rates are not
available at historical prediction time in a forward-time deployment. Current
experiment explicitly preserves offline student-disjoint full-training-corpus
fitting;no held-out labels enter fitting. Do not claim online improvement.
See raw_row_alignment_audit.json and offline_temporal_availability_audit.json.

## 2026-11-04 — Two matched EdNet seeds completed

Six core runs complete; native validation and CPUtest metrics recorded in
reports/ednet_replacement_20261104/replays/. Second difficulty seed123
completed under unchanged protocol;all490protected originalfiles stillmatch.
Corecontinues withseed2026;no guardfailure,no HiTSKT access. Completion/replay
checkpoint/Adam/RNG/loader/mask audits and compressed backups current.

## 2026-11-04 — Four matched EdNet seeds verified

Twelve core runs complete and CPU-replayed/backed up. Difficulty seed3407
selected validation AUC0.730579894; see replay JSON for exact test/metrics.
All completed runs passed state/mask/metric checks. Seed7777 coregroup nowactive.
No model/config/data change or guardfailure; no HiTSKT access.

## 2026-11-04 — Core three-dataset completion, verified offline gains

All15 new EdNet runs complete:265epochs,total3416.978429s windows,peakTorch
270028800bytes,longestnative298.61s,zero guardfailures/retries. Controller exited.
All45 relevant records verified,90prediction arrays,15new optimizer/RNG/data-order
audits;extra six fresh CPU split predictions bitwise match. 15checkpoint archives
andfrozendata backup member-hash readback PASS;490old protectedfilesunchanged.
EdNettestmeans±sampleSD:SAKT0.636359±0.001981,Light0.642417±0.000980,
Difficulty0.706876±0.000691. Newdiff gain+0.064458427vsLight/+0.070516319vsSAKT;
AUC/accuracy/F1 improve5/5seeds vsboth on each ofthree datasets. Preserve offline/
known-item/subset/long-history/fixedsplit caveats. NoHiTSKT access. Nextdelivery:
commit/bundle/archive/pushcheck;nextscience separatelybudgeted featurematchedcontrol.
