# Current state — 13 November 2026: verified paper-update assets

Research worktree /workspace/LightSAKT2-architecture; research branch
research/architecture-variants-20261025. Starting published HEAD9f64bfbb.
Canonical memory remains /workspace/LightSAKT2/project-memory, synchronized views.
Prior state archived in archive/BEFORE_PAPER_ASSETS_20261113_CURRENT_STATE.md.

New deliverables: reports/paper_update_assets_20261112/. Exact original PDF exists,
25pages/title independently confirmed with PyMuPDF/pypdf; allpages visual/text
reviewed. SHA256bcfc09657ccad41f36a9624dcbdd0d77979cb9232e32c3aed3074b23691d76cd.
PDF remains untracked/user-provided/unchanged, not in Git/ZIP. Old27page unused.

All75prior seeds reused. Two already-validation-selected ASSIST2009GKTseed42
checkpoints newly evaluated onCPU, no training: PAM testAUC0.7617558917744343,
MHAimplemented variant0.7659203269028207. Replay times106.43s/45.12s. Actual
probabilities local ignored.tools/paper_update_assets_20261112/replays/; source
checkpoints unchanged. Full proof in verification/checkpoint_verification/
new_GKT_CPU_replays.json under newassets. Preserve these; do not duplicate.

Current77verified seed results,15complete groups/27,2partial groups,58missing
slots.48pendingtraining and10missingAlgebraGKTselection. All9models NOT complete.
No numeric aggregate forpartial/incomplete groups. Seeds42/123/2026/3407/7777.
GKTgrids areconfiguration candidates, not independent seeds. Atleast58selected
trainingoutcomes plus necessaryselection remain. Controls untrained all3datasets.

New audit:77best checkpoints strictly restored/finite/config/hash/max-validation
selection,40corrected componentablation checkpoints,72actualvalidation-onlyLight
gridrecords,164saved arrays with exactn-1labelorder and independent metrics PASS.
MSE/BCE/precision/recall added. Thirty oldnativeGPUarrays differ fromprior CPU
replay by<=1.85e-8 incheckedmetrics; bothorigins/deltas explicit,oldreports unchanged.
Freshpaper metrics consistently recomputedfloat64 fromreferenced savedtestarrays.

15actual graphsets/45PNG/PDF/SVG;320DPI,sourcehash/caption sidecars;byte-identical
regeneration PASS.12CPUtests PASS. Tables2-8CSV/LaTeX/PDF;EdNetTable6protocolonly
because noEdNetgrid. Correctedablations old2datasets only;normremoval unavailable.
Seed42curves stop at actualepochs;DKT+objective penalties explicit. Realanonymized
studentprediction examples are NOTmastery/latent projections. OriginalFigs8/9
notcertified. Mapping/correction/manualinstructions/report separate frompaper.

Current actual-data featureperturbation and sidecarreconstruction PASSall3 under
OFFLINEprotocol. LaterOTHERtrainobservations remain: nochronologicalavailability
claim. FeaturematchedSAKT+QD/cohort/chronologicalgaps remain. Nineexactn5Wilcoxon
comparisons Holm9 allnon-significant;oldW78/p.00048 not reproduced. Search/settings/
GKTimplementation/timing limits documented. No architectural superiority claim.

VisiblehistoricalMIGfree~42.24GB/no listedcomputeprocesses at observation;current
allocationownership and NEWtotalcampaignbudget unverified. Old15scope consumed.
NoGPUcontext/newtraining/peer signals/HiTSKTaccess. Preserve4GiB/900s and shared
safeguards ifnewgrant; profileEdNetGKT underauthorized window. No automaticretry.

PortableZIP external /workspace/LightSAKT2/backups/paper_update_assets_20261112/
LightSAKT_paper_update_assets_20261112.zip; externalreceipt hashash/CRC/payload audit.
Publication result/remote receipts and verified localbundle in samebackupdirectory.
Commit/publication status must be read from actualGit/receipt, not inferred here.
