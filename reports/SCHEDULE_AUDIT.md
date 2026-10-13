# Schedule audit and safe reuse

Logical entries remain 957; no required paper row or distinct configuration is removed. Distinct planned trainings decrease from 921 to 912.

Found 9 additional exact primary/grid training duplicates. Every destination retains its immutable config, row and status; it references verified source training rather than rerunning it. The source primary may already have test predictions, but grid result files expose validation only and selection remains validation-only.

The 21 final seed-42 reuse slots, 15 no_regularization aliases, and reused default ablation remain unchanged. No-regularization is equivalent to no-dropout by author instruction. Repeated seeds 42,123,2026,3407,7777 are distinct trainings; one frozen student split is intentional. All 108 LightSAKT configurations and 36 per baseline variant per dataset remain present. Both historical simplified GKT PAM/MHA variants are retained.

The scheduler owns experiments/manifest.json in memory. Safe activation uses experiments/reuse_plan.json, read by future scripts.run children, avoiding a lost-update race or training interruption. Before/after effective snapshots and every merge are recorded in schedule_audit.json and SCHEDULE_MERGES.csv. No completed results, active configs, data splits, source architecture, or hyperparameters were modified.

Primary coverage, dependency order, all tuning candidates, matched seeds and ablation aliases were verified. No required training entry is missing. Dependent tables, figures, benchmarks, statistics, manuscript text and final verification are automatically invoked by scripts.finalize after training completes and tracked independently.

Repeated evaluations of identical weights/data/settings are cached during final verification, while every logical row still has its own checksum and metric checks. Frozen selected checkpoints reuse existing identical primary test predictions when available. Logical reuse results and new test predictions receive supplemental archives, without copying the weights repeatedly.

One ASSIST2009 LightSAKT source predates an Algebra-only preprocessing edit; source-tree hashes and the exact diff prove that ASSIST data/model/training behavior is unchanged. An explicit dataset-scoped source-equivalence allowlist is documented in TRAINING_CODE_EQUIVALENCE.md. No Algebra reuse or unknown code transition is authorized by that record.
