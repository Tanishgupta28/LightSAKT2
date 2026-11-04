# Current research state — 4 November 2026, EdNet prepared for approved training

Worktree /workspace/LightSAKT2-architecture;branch research/architecture-variants-20261025.
Latest measured results remain c286bcbc,portable release2103a6e3,handoffcf866961.
Current user approves replacing ASSIST2015 with EdNet-KT1/Junyi and executing
missing five-seed SAKT/LightSAKT/LightSAKT+Difficulty comparison once data/resources
are verified. HiTSKT+FAVOR+ is strictly ON HOLD;no access/changes/jobs there.

Selected EdNet-KT1. Genuine publisher-linked fullZIP1201163816bytes SHA
0d13933f90201c5101c7fe8659e44474fa049e3fb93181a8ba6fb3e63267b535;
Contents173976bytes,13169uniquequestionIDs with official answer key andtags.
All5000selected source student files passed ZIP memberCRC. No invented IDs.
Junyi inspected as alternative:exercise/template granularity is not established
as individual generated-question identity;not selected. Do not relabel exercises.

Frozen protocol experiments/ednet_replacement_20261104/protocol.json and plan.json.
Population:uniform5000authentic student filenames,subsetseed20261104,beforelabels;
first presented question per solving session,chosen before answer validation,
to exclude other current-bundle responses from history. Explicit reduced EdNet
protocol,not fullKT1/officialleaderboard claim. Original student/item IDs remain.
721760rawselectedrows;34duplicatesremoved;158161other-sessionquestions excluded;
96invalidanswers excluded;563469retained,5000students,11893authenticitems.
3500/500/1000student splits(seed42),407242/43375/112852interactions and
403742/42875/111852validnextresponse targets.1452inputskillcodes includingUNK0;
fullsortedunique tagsets,training-only vocabulary. Allrepeateditems acrosssessions
retained. Test/current answers and official answer key are label-only,not inputs.

Real data audit PASS:whole-own-student label flip and validationlabel flip leave
features unchanged;independent held-out train-only transform matches frozen
features,positivecontrolresponds. SixCPUtests PASS includingfirst-per-session
selection,labelvalidity,causality,padding,gradients,zero-alpha equivalence forboth
LightSAKT andpreparedSAKTQuestion. Original one-scalar formula/smoothing10unchanged.
Global rate fitting remains OFFLINE full-training-corpus,not chronologicaldeployment.

Core configs15(five seeds42,123,2026,3407,7777 x3models) frozen before new model
scores. Common d64/2heads/dropout0.1/lr0.001/batch64/Adam betas0.9,0.999/eps1e-8/
decay0,max50epochs/patience5;L100 andoriginalmask/chunk/evaluation. No EdNet tuning
search. Existing datasets retain own prior selected settings;do not claim equal
hyperparameter search acrossdatasets. Allnewtrainingevaluate_test=false.

Same previously authorized shared MIG-d2fcfd79-720e-5218-82c9-0a132be68920 is
present,finite~39GiBfree,no listedcomputeprocess;sharedfallbackadmission PASS.
No exclusiveclaim or peeroperations. Currentuser authorizes15serial900s windows,
4GiBallocator,total<=13500s(225minceiling),original10GiBadmission/6GiBruntimefree/
4GiBRSS/2GiBhostheadroom/.1spacing/30sstep limits. Guardedlauncherexists and
source/config/data/hold hashes bound. Preserve general holdbytes. Noautomaticretry.
Five feature-matched SAKT control configs are prepared,NOT launched/corebudget.

New workspaces/raw/checkpoints local ignored under
/workspace/LightSAKT2-architecture/.tools/ednet_replacement_20261104/.
Current evidence reports/ednet_replacement_20261104/;real_data_feature_audit.json,
dataset_statistics.json,cpu_tests.xml,feature_specification.json andpreparation.log.
Before launch/continuation read actual campaign_status.json and training.log;
DO NOT duplicate if a run/controller exists. NewGPUtraining not yet launched at
this memory checkpoint. Exactnextaction:commit/freshadmission/startguarded core.

Old ASSIST2009/Algebra2005 five-seed measurements stay fixed;testmeans:
SAKT0.750322/0.813373,Light0.751056/0.810590,Difficulty0.779359/0.836871.
ASSIST2015 documented replaced/unavailable,historical reports retained. All338
oldprotected hashes and152report/config fingerprints unchanged. No original
models,data/checkpoints/reports or HiTSKT modified. LocalGitpush only ifauthworks;
otherwisepreserveverifiedcommits/bundles,no forcepush or claimedunverifiedpush.
