# Current state — 10 November 2026: expanded comparison verified, GPU gaps pending

Worktree `/workspace/LightSAKT2-architecture`; branch
`research/architecture-variants-20261025`. Canonical memory at
`/workspace/LightSAKT2/project-memory`, synchronized tracked views. Starting
published HEAD24a7db63; prior core results9c6a3e71/source5f1b830d preserved.
Previous full handoff retained in archive/BEFORE_MASTER_20261110_CURRENT_STATE.md.

## New completed milestone

reports/master_comparison_20261110/master_comparison.csv covers all27 requested
model/dataset combinations. Fifteen groups have five matched verified seeds:
DKT,DKT+,DKVMN,SAKT,LightSAKT,LightSAKT+Question Difficulty on ASSIST2009 and
Algebra2005; SAKT/LightSAKT/LightSAKT+Difficulty on EXACT EdNetKT1_5k_first.
75existing seed records reused, zero retraining/new checkpoint prediction passes.
All75 checkpoint hashes/scientific configs/finite strict CPU loads/parameters/
max-validation epochs pass.120saved prediction arrays match exact n−1 targets
and independent metrics. Existing CPU replay cache reused, repeated audit entries
de-duplicated by actual seed/run/config/hash, not treated as independent seeds.

All27 rows have separate validationAUC/testAUC/testaccuracy/testF1 numeric mean/std
and formatted fields, seed counts/status/source/checkpoint/config/budget refs.
Generic mean/standard_deviation alias TEST AUC only. Pending numeric cells EMPTY,
not single-seed means. individual_seed_results.csv covers135slots:75verified,
60PENDING. Existing45 core measurements unchanged; old original baseline tiny
GPU/CPU numerical differences documented; canonical CPU test values reused.

## Missing experiments

Twelve PENDING groups:
- ASSIST2009 andAlgebra2005: GKT(PAM), GKT-MHA(implemented variant), SAKT+Difficulty.
- EdNetKT1_5k_first: DKT,DKT+,DKVMN,GKT,GKT-MHA,SAKT+Difficulty.
Both GKT classes omit paper erase-add/batchnorm features; not canonical GKT claims.
Four old seed42 primary GKT results remain a separate descriptive appendix,
excluded from master aggregates. ASSIST2009 has two complete36-candidate GKT
grids; selected highest-validation seed42 checkpoints reusable, no test scored.
PAM bestval0.764002738;MHA bestval0.767020187. Do NOT repeat these trainings.
Thus60missing verified final slots require at least58additional selected-config
trainings ifthese two reused; extra AlgebraGKT configuration-selection trials
notincluded. AlgebraGKT tuning not complete/no selected config; primary differs.

experiments/master_comparison_20261110/plan.json has60entries,50prepared configs:
25EdNet baseline slots,15controls,10ASSIST2009GKT slots(including2reuse). Ten
AlgebraGKT slots explicitlyawait configurationselection. Prepared configs are
not executable authority; no launcher/automatic retries supplied. Isolated ignored
workspaces symlink exact original data/splits/stats. Controls use exact matched
native SAKT settings plus unchanged scalar feature/frozen sidecar via generic
existing adapter. EdNet baseline templates transfer previous selected/primary
settings, no test-guided tuning. Models/source/data remain unchanged.

## Resources and verification

Same historical MIG visible; NVML technical memory/headroom check passes, no
listed compute processes at observation. Current allocation ownership and NEW
expanded-campaign window budget are NOT verified. Old15invocations consumed.
User asked asynchronously for current allocation and budget; no answer assumed.
No GPU training launched. Existing4GiB/900s shared safeguards remain candidate
bounds, not fresh permission. EdNetGKT1452skills needs feasibility profiling;
no assumption it fits or completes one window. No peer signals or HiTSKT access.

12CPUtests PASS: aggregation duplicate/split-drift/incomplete/invalid-result
rejection plus native forward/backward/causality/padding and all3control adapters.
Standalone CSV/JSON verifier PASS:27groups,135slots,75configs,sampleSD/PENDING cells.
504frozen/protected hashes and original1.29MBreport ZIP match; historical results,
checkpoints, raw data and sources preserved. No extra architecture modification.

## Deliverables and limitations

REPORT.md, master_comparison.csv/JSON, individual_seed_results.csv,
missing_experiments.csv, model_configurations.csv, checkpoint/provenance evidence,
paired metric differences, excluded primary appendix and export/resource audits
are under reports/master_comparison_20261110/. Builder/report/verifier scripts
new; native source untouched. Ready for verified commit and safe research-branch
push; remote ancestry recheck compatible with starting24a7db63.

OriginalLightSAKT below DKT onASSIST2009 andDKT+onAlgebra; input-augmented gains do
not establish attention superiority. Feature-matched SAKT+Difficulty remains
untrained all3. Offline full-training-corpus rates, futureOTHERtrain observations,
known-item/subset/task-granularity limits, fixedsplit/initialization SD, validation
reuse and unequal HPO exposure remain. Timings have different pacing/device
policies; do not rank as matched efficiency. No tuning on released test data.
HiTSKT+FAVOR+ strictly ON HOLD, untouched. No completed expensive run repeated.
