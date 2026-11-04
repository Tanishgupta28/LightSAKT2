# Current state — 4 November 2026: three-dataset core complete

Research worktree `/workspace/LightSAKT2-architecture`, branch
`research/architecture-variants-20261025`. Canonical memory remains
`/workspace/LightSAKT2/project-memory`; synchronized worktree view, one system.
Verified result/report commit `9c6a3e71fc16541145c1842397090c8cf948945c`.
Training/source preparation commit `5f1b830d3aff3cce146fb5cce215d1e93283cac7`.
Prior results c286bcbc, release2103a6e3, handoffcf866961 remain preserved.

## Completed research

All fifteen approved EdNet core runs completed normally, without retries or
resource breaches: five seeds42,123,2026,3407,7777 x SAKT/LightSAKT/Difficulty.
Owned controller PID1082738 exited normally; terminal session90383 closed.
No LightSAKT GPU run remains. Never duplicate the completed campaign.
Reused thirty matching ASSIST2009/Algebra2005 runs; no old training/inference
repeated. Historical ASSIST2015 baselines and unavailable difficulty slots
are retained, but ASSIST2015 is replaced in the final three datasets.

Final table test AUC (mean ± sampleSD, five matched seeds):

| Dataset | SAKT | LightSAKT | LightSAKT+Difficulty |
|---|---|---|---|
| ASSIST2009 | 0.750322 ± 0.000648 | 0.751056 ± 0.000766 | 0.779359 ± 0.000778 |
| Algebra2005 | 0.813373 ± 0.000367 | 0.810590 ± 0.000422 | 0.836871 ± 0.000217 |
| EdNet-KT1 5k first/session | 0.636359 ± 0.001981 | 0.642417 ± 0.000980 | 0.706876 ± 0.000691 |

Difficulty increases test AUC/accuracy/F1 over BOTH baselines in ALL five seeds
on EACH dataset. Absolute test AUC gains versus LightSAKT:
+0.028302772/+0.026281231/+0.064458427 respectively; versus SAKT:
+0.029036320/+0.023497283/+0.070516319. This is descriptive offline evidence,
not attention superiority, online deployment or publication readiness.
Earlier 0.750803537→0.781189972 is ASSIST2009 seed42 VALIDATION, not test/mean.

## EdNet protocol and verification

Authentic publisher-linked KT1/Contents downloaded, official question IDs/answer
keys/tags verified. Junyi individual generated-item identity was not established;
EdNet chosen with actual files available. 5000 user filenames sampled uniformly
from784309, seed20261104, before labels/length/performance. First question of each
solving session chosen before answer validity; no same-session response history.
Explicit reduced EdNet protocol, not fullKT1. Full tagsets are composite skills;
original student/item IDs preserved. 721760raw rows,34exact duplicates removed,
158161additional within-session questions and96invalid first answers excluded;
563469retained,5000students,11893observed authentic items (metadata13169).
Student splits3500/500/1000, seed42; targets403742/42875/111852.
Training-only1451tagsets+UNK; L100/native n−1 overlapping chunks/masks.
Common fixed d64,h2,dropout.1,lr.001,b64,Adam(.9,.999),eps1e-8,decay0,max50,
patience5,strict validation-AUC selection. No EdNet tuning search or source/model
change; historical datasets retain prior selected model-specific settings.

Original correction unchanged: scalar alpha times logit(item success) minus
logit(skill success), smoothing10, Beta(1,1) skill prior. Whole own student removed
from BOTH training aggregates; held-out features fit training labels only; unseen0.
Six CPU tests and real feature/label perturbations PASS; all563469raw source rows,
authentic IDs, labels, chronology and first-session selection independently match.
All15 new checkpoints CPU restored; Adam/RNG/exact loader-order checks pass.
All90 relevant saved validation/test prediction arrays pass exact target order and
independent AUC/confusion/BCE checks. Three seed42 families additionally restored
fresh onCPU: allsix split arrays bitwise reproduce, no predictions overwritten.
490 historical scientific/report hashes and frozen science/data/config/hold hashes
unchanged. All15 best/latest checkpoint recovery archives and frozen dataset
backup verified by member hashes; checkpoints/raw arrays remain ignored/local.

Resource: same authorized SHARED MIG-d2fcfd79-720e-5218-82c9-0a132be68920.
4GiB allocator/900s invocation/13500s total ceilings preserved with original
shared safeguards. Actual3416.978429s admitted wall time (56.95min),265epochs,
longest native training298.61s, peakTorch270028800bytes (0.251484GiB); zero breaches.
No exclusive allocation claim; no unrelated workload signalled/changed.
HiTSKT+FAVOR+ strictly ON HOLD: no files, checkpoints or jobs accessed/changed.

## Limitations and next scientific step

Offline full-training-corpus statistics include later OTHER training students'
observations for99.6012%validation/99.5887%test targets. No held-out/self-label
contamination under stated offline protocol; historical-time availability and
online benefit NOT established. Mostly known items (unseen val0.0669%,test0.1648%).
One fixed split; largest EdNet validation student contributes19.11%of targets.
SD measures initialization variability, not cohort/split uncertainty. Empirical
success rates also reflect ability/item selection, not proven intrinsic difficulty.

Essential next control: SAKT+identical difficulty, with native SAKT encoder and
matched information/protocol. Five EdNet configs and generic CPU adapter tested,
NOT GPU-executed or charged to corebudget. Define a separate control budget first.
Then independent cohort/split and chronological studies; no tuning on released tests.
No additional architecture or GPU experiment is currently part of unfinished core.

## Deliverables and preservation

Complete report/tables/graphs/provenance: `reports/professor_three_datasets_20261104/`.
New-run/source/feature/resource audits: `reports/ednet_replacement_20261104/`.
Ignored raw/work/checkpoint/predictions: `.tools/ednet_replacement_20261104/`.
Verified checkpoint/data archives: `/workspace/LightSAKT2/backups/ednet_replacement_20261104/`.
Seven historical/preparation/result local incremental bundles verified before
packaging, with prerequisites preserved. Result/report commit9c6a3e71; final
handoff may advance HEAD without changing measurements.
Portable reporting archive VERIFIED: `/workspace/LightSAKT2/backups/professor_three_datasets_20261104/LightSAKT_question_difficulty_three_datasets_20261104.zip`.
SHA256 `a4741fef489150bf50eaacd8c76f371cc51255fd31475ad9b08fa3f206999424`; 1292024bytes,374payloadfiles,45measuredslots,0pending.
ZIP CRC, extracted manifest/means/sampleSD/paired-delta/config-reference checks PASS.
Standalone standard-library CLI verifier also PASS. Reporting package includes
source/configs/split-population manifests and checkpoint refs, not raw responses,
student-level predictions or weights. These are separately backed up locally.

Non-force research push attempted on result commit9c6a3e71: FAILED
AUTHENTICATION_BLOCKED(exit128). No remote push success claimed. Local commits,
verified result bundle and all existing checkpoints/results preserved. Final
handoff bundle/receipt are under `/workspace/LightSAKT2/backups/ednet_replacement_20261104/`
(`ednet_delivery_handoff.bundle`, `delivery_handoff_verification.json`).
Canonical/worktree memory synchronized; strict validation and manual scientific
claims audit completed. Core delivery complete; no GPU run remains.
