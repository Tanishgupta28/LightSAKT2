# Verified paper-update assets

Audit date: 13 November 2026. The directory retains the requested `20261112` identifier. This is a research asset report, not a regenerated manuscript.

The exact PDF `research_paper/original_research_paper.pdf` has 25 pages and the expected title, **An Efficient Attention-Driven Framework for Adaptive Education-based Continuous Learning**. PyMuPDF and pypdf independently confirmed the page count. All pages were rendered and reviewed; experimental pages were enlarged. SHA256: `bcfc09657ccad41f36a9624dcbdd0d77979cb9232e32c3aed3074b23691d76cd`. The older manuscript was not used. The original PDF is unchanged.

**New training runs: 0.** Current GPU allocation ownership and a new total campaign budget are unconfirmed. The idle-looking MIG device and consumed 15-run budget do not authorize further training. Two already-selected ASSIST2009 GKT checkpoints were evaluated on CPU. We reused all 75 prior seed results, audited 40 existing component ablations and verified 72 real LightSAKT grid candidates. There are now **77 verified seed results, 15/27 complete five-seed groups, two partial groups and 58 missing seed slots**.

## Full genuine comparison

Mean ± sample SD across exactly seeds 42, 123, 2026, 3407 and 7777. PENDING means no five-seed aggregate, including the two groups with one verified GKT seed. Validation AUC is the native selected epoch's score. Test measurements are independently recomputed from the referenced saved probabilities. Original reports and measurements were preserved.

| Dataset                       | Model           | Validation AUC      | Test AUC            | Accuracy            | F1                  | MSE                 | BCE                 | Precision           | Recall              | Seeds/status         |
|:------------------------------|:----------------|:--------------------|:--------------------|:--------------------|:--------------------|:--------------------|:--------------------|:--------------------|:--------------------|:---------------------|
| ASSIST2009                    | DKT             | 0.757492 ± 0.000684 | 0.757165 ± 0.000527 | 0.736135 ± 0.001022 | 0.818332 ± 0.000993 | 0.189594 ± 0.000669 | 0.564732 ± 0.001604 | 0.759306 ± 0.001825 | 0.887326 ± 0.004050 | 5/5 VERIFIED_5_SEEDS |
| ASSIST2009                    | DKT+            | 0.754946 ± 0.002044 | 0.751202 ± 0.000876 | 0.733004 ± 0.001809 | 0.816546 ± 0.002508 | 0.194971 ± 0.000512 | 0.577168 ± 0.001301 | 0.756331 ± 0.002510 | 0.887250 ± 0.009219 | 5/5 VERIFIED_5_SEEDS |
| ASSIST2009                    | DKVMN           | 0.753992 ± 0.001149 | 0.755016 ± 0.001395 | 0.736225 ± 0.001179 | 0.819802 ± 0.001518 | 0.178742 ± 0.000262 | 0.535401 ± 0.000548 | 0.755689 ± 0.004667 | 0.895913 ± 0.009739 | 5/5 VERIFIED_5_SEEDS |
| ASSIST2009                    | GKT             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | 1/5 PENDING          |
| ASSIST2009                    | GKT-MHA (impl.) | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | 1/5 PENDING          |
| ASSIST2009                    | SAKT            | 0.748284 ± 0.000767 | 0.750322 ± 0.000648 | 0.732718 ± 0.000669 | 0.817401 ± 0.000943 | 0.180373 ± 0.000252 | 0.539191 ± 0.000666 | 0.753460 ± 0.001458 | 0.893216 ± 0.004049 | 5/5 VERIFIED_5_SEEDS |
| ASSIST2009                    | LightSAKT       | 0.749454 ± 0.001570 | 0.751056 ± 0.000766 | 0.732229 ± 0.000801 | 0.816533 ± 0.001517 | 0.180233 ± 0.000319 | 0.538615 ± 0.000886 | 0.754522 ± 0.002874 | 0.889705 ± 0.007347 | 5/5 VERIFIED_5_SEEDS |
| ASSIST2009                    | LightSAKT + QD  | 0.780248 ± 0.001153 | 0.779359 ± 0.000778 | 0.744460 ± 0.000439 | 0.822429 ± 0.000649 | 0.172033 ± 0.000335 | 0.517652 ± 0.000968 | 0.769227 ± 0.002328 | 0.883561 ± 0.004450 | 5/5 VERIFIED_5_SEEDS |
| ASSIST2009                    | SAKT + QD       | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | 0/5 PENDING          |
| Algebra2005                   | DKT             | 0.820171 ± 0.000798 | 0.822181 ± 0.000683 | 0.800524 ± 0.001175 | 0.872422 ± 0.000891 | 0.149683 ± 0.000862 | 0.466820 ± 0.002433 | 0.819655 ± 0.002363 | 0.932469 ± 0.004070 | 5/5 VERIFIED_5_SEEDS |
| Algebra2005                   | DKT+            | 0.822312 ± 0.000462 | 0.823456 ± 0.000649 | 0.801094 ± 0.000253 | 0.872982 ± 0.000347 | 0.152382 ± 0.000488 | 0.474735 ± 0.001334 | 0.819068 ± 0.000928 | 0.934496 ± 0.001951 | 5/5 VERIFIED_5_SEEDS |
| Algebra2005                   | DKVMN           | 0.814134 ± 0.001502 | 0.817725 ± 0.001099 | 0.798950 ± 0.000818 | 0.872047 ± 0.000572 | 0.141806 ± 0.000514 | 0.439274 ± 0.001405 | 0.815787 ± 0.003242 | 0.936678 ± 0.005079 | 5/5 VERIFIED_5_SEEDS |
| Algebra2005                   | GKT             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | 0/5 PENDING          |
| Algebra2005                   | GKT-MHA (impl.) | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | 0/5 PENDING          |
| Algebra2005                   | SAKT            | 0.809999 ± 0.000725 | 0.813373 ± 0.000367 | 0.796382 ± 0.000504 | 0.871012 ± 0.000372 | 0.143309 ± 0.000206 | 0.443098 ± 0.000598 | 0.811539 ± 0.001389 | 0.939898 ± 0.002329 | 5/5 VERIFIED_5_SEEDS |
| Algebra2005                   | LightSAKT       | 0.807548 ± 0.000766 | 0.810590 ± 0.000422 | 0.795574 ± 0.000639 | 0.870356 ± 0.000350 | 0.144069 ± 0.000364 | 0.445272 ± 0.001018 | 0.811711 ± 0.001952 | 0.938147 ± 0.002899 | 5/5 VERIFIED_5_SEEDS |
| Algebra2005                   | LightSAKT + QD  | 0.835215 ± 0.000498 | 0.836871 ± 0.000217 | 0.807301 ± 0.000642 | 0.876874 ± 0.000771 | 0.135253 ± 0.000471 | 0.421769 ± 0.001170 | 0.823183 ± 0.005323 | 0.938154 ± 0.008587 | 5/5 VERIFIED_5_SEEDS |
| Algebra2005                   | SAKT + QD       | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | 0/5 PENDING          |
| EdNet-KT1 (5k, first/session) | DKT             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | 0/5 PENDING          |
| EdNet-KT1 (5k, first/session) | DKT+            | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | 0/5 PENDING          |
| EdNet-KT1 (5k, first/session) | DKVMN           | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | 0/5 PENDING          |
| EdNet-KT1 (5k, first/session) | GKT             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | 0/5 PENDING          |
| EdNet-KT1 (5k, first/session) | GKT-MHA (impl.) | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | 0/5 PENDING          |
| EdNet-KT1 (5k, first/session) | SAKT            | 0.665022 ± 0.001326 | 0.636359 ± 0.001981 | 0.662697 ± 0.001453 | 0.777653 ± 0.002664 | 0.215359 ± 0.000589 | 0.620486 ± 0.001433 | 0.680672 ± 0.001781 | 0.906940 ± 0.010241 | 5/5 VERIFIED_5_SEEDS |
| EdNet-KT1 (5k, first/session) | LightSAKT       | 0.670306 ± 0.001088 | 0.642417 ± 0.000980 | 0.664176 ± 0.000810 | 0.778199 ± 0.001393 | 0.214097 ± 0.000192 | 0.617404 ± 0.000401 | 0.682150 ± 0.001706 | 0.905766 ± 0.006434 | 5/5 VERIFIED_5_SEEDS |
| EdNet-KT1 (5k, first/session) | LightSAKT + QD  | 0.729147 ± 0.000894 | 0.706876 ± 0.000691 | 0.693068 ± 0.000701 | 0.790115 ± 0.001670 | 0.199781 ± 0.000209 | 0.584691 ± 0.000520 | 0.711542 ± 0.002475 | 0.888255 ± 0.007922 | 5/5 VERIFIED_5_SEEDS |
| EdNet-KT1 (5k, first/session) | SAKT + QD       | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | PENDING             | 0/5 PENDING          |

## Absolute held-out test AUC gains

| dataset           | baseline   |   absolute_test_AUC_gain |   sample_SD_paired_gain |   positive_seeds |   negative_seeds |   n |
|:------------------|:-----------|-------------------------:|------------------------:|-----------------:|-----------------:|----:|
| ASSIST2009        | SAKT       |                0.0290363 |             0.000567236 |                5 |                0 |   5 |
| ASSIST2009        | LightSAKT  |                0.0283028 |             0.000484456 |                5 |                0 |   5 |
| Algebra2005       | SAKT       |                0.0234973 |             0.000424336 |                5 |                0 |   5 |
| Algebra2005       | LightSAKT  |                0.0262812 |             0.00045845  |                5 |                0 |   5 |
| EdNetKT1_5k_first | SAKT       |                0.0705163 |             0.00233589  |                5 |                0 |   5 |
| EdNetKT1_5k_first | LightSAKT  |                0.0644584 |             0.000647787 |                5 |                0 |   5 |

LightSAKT + Difficulty improves test AUC, accuracy and F1 against both SAKT and original LightSAKT in all five matched seeds of each dataset. Original LightSAKT does not outperform every available baseline: DKT is stronger on ASSIST2009; DKT+ and SAKT are stronger on Algebra2005. Adding item information does not isolate attention-architecture quality. SAKT + Difficulty remains untrained on all three datasets.

The new GKT results are single-seed evidence only. Their checkpoint-selected seed-42 test AUCs are 0.761755892 (GKT/PAM) and 0.765920327 (GKT-MHA implemented variant). Training selection was frozen before test access. Neither result appears in a five-seed aggregate. See the individual-seed CSV and CPU replay provenance.

## Pending work and resource blocker

| dataset           | model                         | state                            |   Missing seeds |
|:------------------|:------------------------------|:---------------------------------|----------------:|
| ASSIST2009        | GKT                           | PENDING_TRAINING                 |               4 |
| ASSIST2009        | GKT-MHA (implemented variant) | PENDING_TRAINING                 |               4 |
| ASSIST2009        | SAKT + Question Difficulty    | PENDING_TRAINING                 |               5 |
| Algebra2005       | GKT                           | MISSING_HYPERPARAMETER_SELECTION |               5 |
| Algebra2005       | GKT-MHA (implemented variant) | MISSING_HYPERPARAMETER_SELECTION |               5 |
| Algebra2005       | SAKT + Question Difficulty    | PENDING_TRAINING                 |               5 |
| EdNetKT1_5k_first | DKT                           | PENDING_TRAINING                 |               5 |
| EdNetKT1_5k_first | DKT+                          | PENDING_TRAINING                 |               5 |
| EdNetKT1_5k_first | DKVMN                         | PENDING_TRAINING                 |               5 |
| EdNetKT1_5k_first | GKT                           | PENDING_TRAINING                 |               5 |
| EdNetKT1_5k_first | GKT-MHA (implemented variant) | PENDING_TRAINING                 |               5 |
| EdNetKT1_5k_first | SAKT + Question Difficulty    | PENDING_TRAINING                 |               5 |

At least 58 selected-configuration training outcomes remain, plus Algebra2005 GKT configuration selection. EdNet GKT feasibility also requires profiling under an authorized window. The resource check reports memory headroom and no visible compute processes, but cannot establish current allocation ownership. The old shared limits remain safeguards, not a new grant: 4 GiB allocator and 900-second windows, with the recorded memory, host-RAM, pacing and step limits. No GPU training, unrelated process signals or HiTSKT access occurred.

## Verification

All 77 result checkpoints were hashed and strictly restored on CPU, with finite weights, matching scientific configurations, dataset/split hashes and first maximum-validation-AUC checkpoint selection. Original complete epoch histories were checked. The two new GKT evaluations restored frozen checkpoints without training. All 164 available validation/test and ablation prediction arrays were checked against exact per-student n−1 labels in order; probabilities and metrics were independently recalculated. Forty ablation checkpoints and 72 validation-only grid records were also audited.

AUC uses average tied ranks and was crosschecked with sklearn. Classification uses P(correct) ≥ 0.5; precision/recall/F1 are for the positive correct-response class. MSE is mean squared probability error (Brier score). BCE uses natural logs and clips probabilities to [1e-7, 1−1e-7]. Metrics pool all valid target interactions; padding is excluded. Summary variability is sample SD, ddof=1. All means and SDs are independently recomputed in the export verifier.

Thirty older baseline files contain original GPU probabilities, whereas the previous master table used independently verified CPU replays. Their maximum checked AUC/accuracy/F1/BCE difference is 1.85e-8. Both origins and exact differences are preserved in `verification/metrics_verification/prior_CPU_vs_saved_array_differences.csv`. This package consistently reports float64 arithmetic on the actual referenced saved test files. Historical files were not rewritten.

Repeated real-data perturbation tests and reconstruction of frozen sidecars found no held-out-label or whole-own-student self-label leakage under the documented offline protocol. Train targets exclude every response of the same training student from both item and skill statistics. Held-out estimates fit training labels only, with item smoothing 10, skill Beta(1,1) and zero correction for unseen items. Future observations of OTHER training students can still contribute to fitted statistics; chronological deployment is not established.

## Architecture and dataset scope

Original LightSAKT has one causal ELU+1 linear-attention encoder, interaction/query embeddings, learned interaction-position embeddings, residual normalization and a d→d/2→d FFN. The difficulty model leaves that attention and encoder structure unchanged. One learned scalar adds logit(smoothed item success) minus logit(smoothed skill success) to its output logit. External fitted statistics add information and estimator capacity beyond that scalar.

ASSIST2009 uses authentic problem IDs; Algebra2005 uses a lossless problem/step composite and a local student split of the original training subset. EdNet uses authentic question IDs and the exact deterministic 5,000-student first-presented-question-per-session subset, not full KT1. Its 563,469 retained interactions use unchanged student splits, windowing and masks. Student populations, skill vocabularies and item granularity differ across datasets. Dataset and split documentation is in `source_data/dataset_provenance/`. Full model-specific configurations and actual epoch budgets are included.

## Tables, graphs and statistical limits

Every numerical graph was generated by executed Python using genuine CSV/JSON records, pandas, NumPy and Matplotlib. Fifteen graph sets have PNG at 320 DPI plus vector PDF/SVG. Graph sidecars record input hashes, output hashes, program and caption. Figures 4–6 use actual seed-42 histories, with no interpolation, smoothing or extension beyond early stopping. DKT+ objective penalties are labelled. Comparison bars require all five seeds. Real anonymized student examples show next-response probabilities, not mastery.

Tables 2–8 have CSV, LaTeX and PDF exports. Table 7 also has the original metric-row/model-column arrangement. Table 6 gives the actual EdNet selection protocol because no completed EdNet LightSAKT grid exists. Table 8 uses existing corrected component removals and marks missing EdNet/norm removals unavailable.

Nine seed-matched exact two-sided Wilcoxon AUC tests use n=5 and Holm correction; none rejects at 0.05. Five no-zero pairs have minimum attainable two-sided p=0.0625. The old W=78/p=0.00048 claim is not reproduced. Seeds share one dataset split, and some seed-42 models were reused from validation-only selection, so inference is conditional on the selection/split rather than independent cohort evidence. Grid candidates and epochs are not repetitions.

## Delivery and limitations

`documentation/PAPER_GRAPH_MAPPING.md` and `PAPER_TABLE_MAPPING.md` give original page/asset numbers, exact new paths, programs, inputs and replacement conditions. `SUGGESTED_MANUSCRIPT_CORRECTIONS.md` is separate from the unchanged paper. `REPRODUCE.md` explains portable regeneration; `verification/graph_reproducibility/verification.json` records byte-identical graph regeneration and source preservation. Checkpoint binaries, raw datasets and full prediction arrays are kept separately with hashes, outside Git and the ZIP.

The feature-matched SAKT control, independent cohorts and chronological evaluation remain needed. GKT variants differ from canonical GKT. Search exposure, batch sizes, hardware and pacing differ; timings are not matched efficiency comparisons. There is no publication-readiness or architectural-superiority claim. See `documentation/REMAINING_LIMITATIONS.md`. HiTSKT remains on hold and untouched.
