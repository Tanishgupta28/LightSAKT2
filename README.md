# LightSAKT2 corrected research rebuild

The original LightSAKT trainable architecture is preserved. Causal ELU+1 prefix attention replaces the incorrect global aggregation, with identical parameter names and shapes. Student-disjoint 70/10/20 splits, padding masks, next-response alignment and validation-only checkpoint selection are implemented and tested.

This is an ongoing experimental rebuild. Historical paper results are unverified. Completed, running, pending and blocked experiments are recorded in [reports/EXPERIMENT_STATUS.md](reports/EXPERIMENT_STATUS.md) and `experiments/manifest.json`. Do not interpret pending table cells as results.

```
python -m pytest -q
python -m scripts.schedule
python -m scripts.summarize
python -m scripts.verify_artifacts
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [PAPER_CORRECTIONS.md](PAPER_CORRECTIONS.md), [DATA_PROVENANCE.md](DATA_PROVENANCE.md), [reports/INITIAL_CODE_AUDIT.md](reports/INITIAL_CODE_AUDIT.md), and [FINAL_RESEARCH_REPORT.md](FINAL_RESEARCH_REPORT.md). Original source is in `original_snapshot/`, original checksums in `reports/original_checksums.json`, fresh weights in `checkpoints/<run>/`, and genuine numerical records in `results/raw/<run>/metrics.json`.

The separate [architecture study](reports/architecture_study/RESEARCH_REPORT.md) confirms one original encoder block and adds configurable stacked blocks, causal softmax, and recency-biased softmax under the `LightSAKTResearch` model name. Its validation-only configurations are prepared but unsubmitted; the original `LightSAKT` and existing training hold are preserved.

The [completed ASSIST2009 seed-42 pilot](reports/architecture_study/pilot/PILOT_STATUS.md) trained both depths from fresh initialization: validation AUC 0.750803537 (one layer) versus 0.750790932 (two layers), with 29 epochs each. Epoch metrics, verified checkpoints and recovery evidence are preserved. The bounded grant is closed; general automatic training remains held. Causal softmax is the next recommended experiment.

GKT results are explicitly historical simplified PAM/MHA variants, not claims of canonical published GKT reproduction. The no_regularization ablation is equivalent to no_dropout by user instruction; it reuses the same evidence.
