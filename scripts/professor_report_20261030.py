"""Professor-ready reports from verified measurements only; no estimation."""
import json
from pathlib import Path
from statistics import mean, stdev
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scripts.architecture_study import read_json, digest
from scripts.architecture_pilot import PROJECT, SOURCE, write_json

OUT = PROJECT / "reports/professor_results_20261030"
DATASETS = ["ASSIST2009", "ASSIST2015", "Algebra2005"]
MODELS = ["SAKT", "LightSAKT", "LightSAKT+Difficulty"]
SEEDS = [42, 123, 2026, 3407, 7777]


def formatted(value, sd, n):
    if value is None:return "PENDING"
    if n == 1:return f"{value:.6f} (n=1; PENDING replication)"
    return f"{value:.6f} ± {sd:.6f}" + ("; PENDING" if n < 5 else "")


def main():
    plan = read_json(PROJECT / "experiments/professor_campaign_20261030/plan.json")
    records = [read_json(p) for p in sorted((OUT / "replays").glob("*.json"))]
    lookup = {(r["dataset"], r["model"], r["seed"]): r for r in records}
    individual = []; summary = []; provenance = []
    for dataset in DATASETS:
        for model in MODELS:
            rows = []
            for seed in SEEDS:
                r = lookup.get((dataset, model, seed))
                item = {"dataset": dataset, "model": model, "seed": seed, "status": "VERIFIED" if r else "PENDING"}
                if r:
                    for split in ["validation", "test"]:
                        for metric in ["auc", "accuracy", "f1", "precision", "recall", "bce"]:
                            item[f"{split}_{metric}"] = r[split].get(metric, r["cpu_replay"][split].get(metric))
                    item.update(parameters=r["parameters"], best_epoch=r["best_epoch"], epochs=r["epochs_trained"],
                                training_seconds=r["training_seconds"], peak_allocated_bytes=r["peak_allocated_bytes"],
                                checkpoint=r["checkpoint"], checkpoint_sha256=r["checkpoint_sha256"], run_id=r["run_id"])
                    recorded = read_json(r["recorded_metrics_path"])
                    item["peak_reserved_bytes"] = recorded.get("peak_reserved_bytes")
                    rows.append(item)
                    source = {k: r[k] for k in ["dataset", "model", "seed", "run_id", "checkpoint", "checkpoint_sha256",
                        "config", "config_file", "config_sha256", "recorded_metrics_path", "recorded_metrics_sha256", "processed_sha256",
                        "split_sha256", "selection", "selected_git_commit", "recorded_training_code_sha256", "recorded_comparison_absolute_differences"]}
                    environment = Path(r["config_file"]).parent / "environment.json"
                    if environment.exists():
                        source["training_environment"] = {"path": str(environment), "sha256": digest(environment)}
                    if model == "LightSAKT+Difficulty":
                        source["feature_specification"] = plan["feature_specs"][dataset]
                        source["frozen_campaign_plan_sha256"] = digest(PROJECT / "experiments/professor_campaign_20261030/plan.json")
                        source["prediction_artifacts"] = r["prediction_artifacts"]
                    provenance.append(source)
                else:
                    item["reason"] = ("Question ID unavailable; no substitute authorized" if dataset == "ASSIST2015" and model.endswith("Difficulty")
                                      else "Training or independent checkpoint replay not yet completed")
                individual.append(item)
            aggregate = {"dataset": dataset, "model": model, "verified_seeds": len(rows), "expected_seeds": 5,
                         "status": "COMPLETE" if len(rows) == 5 else "PENDING"}
            for split in ["validation", "test"]:
                for metric in ["auc", "accuracy", "f1", "precision", "recall", "bce"]:
                    key = f"{split}_{metric}"
                    values = [r[key] for r in rows if r.get(key) is not None]
                    aggregate[key + "_mean"] = mean(values) if values else None
                    aggregate[key + "_sd"] = stdev(values) if len(values) > 1 else None
            aggregate["parameters"] = rows[0]["parameters"] if rows else None
            summary.append(aggregate)
    pd.DataFrame(individual).to_csv(OUT / "individual_seed_results.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "summary_results.csv", index=False)
    write_json(OUT / "individual_seed_results.json", individual)
    write_json(OUT / "summary_results.json", summary)
    write_json(OUT / "experiment_provenance.json", provenance)
    deltas = []; per_seed_deltas = []
    for dataset in DATASETS:
        for comparator in ["LightSAKT", "SAKT"]:
            paired = []
            for seed in SEEDS:
                modified = lookup.get((dataset, "LightSAKT+Difficulty", seed)); baseline = lookup.get((dataset, comparator, seed))
                if modified and baseline:
                    row = {"dataset": dataset, "comparator": comparator, "seed": seed,
                           "validation_auc_delta": modified["validation"]["auc"] - baseline["validation"]["auc"],
                           "test_auc_delta": modified["test"]["auc"] - baseline["test"]["auc"]}
                    paired.append(row); per_seed_deltas.append(row)
            row = {"dataset": dataset, "comparator": comparator, "paired_seeds": len(paired),
                   "status": "COMPLETE" if len(paired) == 5 else "PENDING"}
            for split in ["validation", "test"]:
                v = [r[split + "_auc_delta"] for r in paired]
                row[split + "_auc_delta_mean"] = mean(v) if v else None
                row[split + "_auc_delta_sd"] = stdev(v) if len(v) > 1 else None
                row[split + "_positive_seed_count"] = sum(x > 0 for x in v)
                row[split + "_minimum_delta"] = min(v) if v else None
            deltas.append(row)
    pd.DataFrame(deltas).to_csv(OUT / "absolute_auc_improvements.csv", index=False)
    pd.DataFrame(per_seed_deltas).to_csv(OUT / "paired_seed_auc_deltas.csv", index=False)
    write_json(OUT / "absolute_auc_improvements.json", deltas)
    table = ["| Dataset | Model | Validation AUC | Test AUC | Test F1 | Test accuracy | Seeds/status |",
             "|---|---|---|---|---|---|---|"]
    for r in summary:
        n = r["verified_seeds"]
        cells = [formatted(r[k + "_mean"], r[k + "_sd"], n) for k in ["validation_auc", "test_auc", "test_f1", "test_accuracy"]]
        table.append(f"| {r['dataset']} | {r['model']} | " + " | ".join(cells) + f" | {n}/5 {r['status']} |")
    table_text = "\n".join(table)
    (OUT / "comparison_table.md").write_text(table_text + "\n")
    data_rows = []; config_rows = []
    for ds in DATASETS:
        stats = read_json(SOURCE / f"reports/{ds}_statistics.json")
        data_rows.append({"dataset": ds, **{k: stats[k] for k in ["raw_rows", "clean_interactions", "split_students", "split_interactions", "num_q", "identifier_semantics", "processed_sha256", "split_sha256", "raw_sha256"]}})
        for model in ["SAKT", "LightSAKT"]:
            r = lookup.get((ds, model, 42))
            if r:
                cfg = r["config"]
                config_rows.append({"dataset": ds, "model": model, "parameters": r["parameters"], "learning_rate": cfg["learning_rate"],
                    "model_config": cfg["model_config"], "seq_len": cfg["seq_len"], "batch_size": cfg["batch_size"], "max_epochs": cfg["max_epochs"], "patience": cfg["patience"]})
        modified = lookup.get((ds, "LightSAKT+Difficulty", 42))
        if modified:
            cfg = modified["config"]
            config_rows.append({"dataset": ds, "model": "LightSAKT+Difficulty", "parameters": modified["parameters"],
                "learning_rate": cfg["learning_rate"], "model_config": cfg["model_config"], "seq_len": cfg["seq_len"],
                "batch_size": cfg["batch_size"], "max_epochs": cfg["max_epochs"], "patience": cfg["patience"],
                "question_difficulty": cfg["question_difficulty"], "feature_specification": plan["feature_specs"][ds]})
    write_json(OUT / "dataset_metadata.json", data_rows)
    write_json(OUT / "training_configurations.json", config_rows)
    campaign = read_json(OUT / "campaign_status.json") if (OUT / "campaign_status.json").exists() else {"status": "NOT_LAUNCHED"}
    delta_table = ["| Dataset | Compared with | Paired seeds | Validation AUC Δ | Test AUC Δ | Positive test seeds |",
                   "|---|---|---:|---|---|---|"]
    for r in deltas:
        n = r["paired_seeds"]
        a = formatted(r["validation_auc_delta_mean"], r["validation_auc_delta_sd"], n)
        b = formatted(r["test_auc_delta_mean"], r["test_auc_delta_sd"], n)
        delta_table.append(f"| {r['dataset']} | {r['comparator']} | {n} | {a} | {b} | {r['test_positive_seed_count']}/{n} |")
    dataset_table = ["| Dataset | Interactions train / validation / test | Students train / validation / test | Skills (incl. UNK) | Sequence length |",
                     "|---|---|---|---:|---:|"]
    for r in data_rows:
        counts = " / ".join(str(r["split_interactions"][k]) for k in ["train", "validation", "test"])
        students = " / ".join(str(r["split_students"][k]) for k in ["train", "validation", "test"])
        dataset_table.append(f"| {r['dataset']} | {counts} | {students} | {r['num_q']} | {200 if r['dataset'] == 'Algebra2005' else 100} |")
    config_table = ["| Dataset | Model | Embedding | Heads | Dropout | Learning rate | Parameters |",
                    "|---|---|---:|---:|---:|---:|---:|"]
    for r in config_rows:
        c = r["model_config"]
        dimension = c.get("embed_dim", c.get("d"))
        heads = c.get("num_heads", c.get("num_attn_heads"))
        config_table.append(f"| {r['dataset']} | {r['model']} | {dimension} | {heads} | {c['dropout']} | {r['learning_rate']} | {r['parameters']} |")
    seed_table = ["| Dataset | Model | Seed | Validation AUC | Test AUC | Test F1 | Test accuracy | Best / completed epoch |",
                  "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in individual:
        if r["status"] == "VERIFIED":
            values = " | ".join(f"{r[k]:.9f}" for k in ["validation_auc", "test_auc", "test_f1", "test_accuracy"])
            seed_table.append(f"| {r['dataset']} | {r['model']} | {r['seed']} | {values} | {r['best_epoch']} / {r['epochs']} |")
        else:
            seed_table.append(f"| {r['dataset']} | {r['model']} | {r['seed']} | PENDING | PENDING | PENDING | PENDING | PENDING |")
    conclusions = []
    for ds in DATASETS:
        d = next(r for r in deltas if r["dataset"] == ds and r["comparator"] == "LightSAKT")
        n = d["paired_seeds"]
        if not n:
            conclusions.append(f"**{ds}: PENDING.** No verified modified-model seed is available. " +
                ("The available source has no independent question identifier." if ds == "ASSIST2015" else "See campaign status for remaining training or replay."))
        else:
            mod = next(r for r in summary if r["dataset"] == ds and r["model"] == "LightSAKT+Difficulty")
            base = next(r for r in summary if r["dataset"] == ds and r["model"] == "LightSAKT")
            conclusions.append(f"**{ds}: {n}/5 paired seeds verified.** Modified minus original LightSAKT: "
                f"validation AUC {d['validation_auc_delta_mean']:+.9f}, test AUC {d['test_auc_delta_mean']:+.9f}; "
                f"positive validation/test seed deltas {d['validation_positive_seed_count']}/{n} and {d['test_positive_seed_count']}/{n}. "
                f"Minimum individual validation/test deltas: {d['validation_minimum_delta']:+.9f} / {d['test_minimum_delta']:+.9f}. "
                + (f"Mean test F1 difference {mod['test_f1_mean'] - base['test_f1_mean']:+.9f}; "
                   f"accuracy difference {mod['test_accuracy_mean'] - base['test_accuracy_mean']:+.9f}. " if n == 5 else "Replication remains incomplete. ")
                + "These are results on one fixed student split, not evidence of general superiority across student populations.")
    findings = "\n\n".join(conclusions)
    supported = [ds for ds in DATASETS if all(
        row["paired_seeds"] == 5 and row["validation_positive_seed_count"] == 5 and row["test_positive_seed_count"] == 5
        for row in deltas if row["dataset"] == ds)]
    conclusion = ("On the frozen offline student-disjoint benchmark, " + ", ".join(supported) +
        " show higher validation and held-out test AUC in every one of the five matched seeds versus both original LightSAKT and SAKT. "
        "This is a reproducible measured result for the question-information modification under the stated protocol. "
        "It does not isolate an attention-architecture advantage or establish forward-time deployment validity. "
        "ASSIST2015 remains unverified, so improvement across all three datasets is not established."
        if supported else "No dataset currently has verified positive validation and test AUC deltas in all five paired seeds versus both baselines. See every measured seed and remaining gaps above.")
    resource_text = "Resource summary is PENDING."
    resource_path = OUT / "resource_summary.json"
    if resource_path.exists():
        resource = read_json(resource_path)
        resource_text = (f"The verified resource snapshot covers {len(resource['completed_new_runs'])} completed new runs "
            f"(campaign: {resource['campaign_status']}). Admitted GPU-window wall time: "
            f"{resource['gpu_window_wall_seconds']:.3f} seconds of the 8,100-second ceiling; "
            f"summed native training time: {resource['new_training_seconds_sum']:.3f} seconds. "
            f"Maximum allocated GPU memory: {resource['maximum_allocator_peak_bytes'] / 1024**2:.3f} MiB; "
            f"maximum worker RSS: {resource['maximum_worker_rss_bytes'] / 1024**3:.3f} GiB. "
            f"Minimum shared GPU free memory: {resource['minimum_shared_gpu_free_bytes'] / 1024**3:.3f} GiB; "
            f"minimum host headroom: {resource['minimum_host_headroom_bytes'] / 1024**3:.3f} GiB. "
            f"Sampled runtime-floor/RSS breaches: {len(resource['runtime_floor_or_rss_breach_sample_indices'])}. "
            "These are allocation-window wall times including pacing, not kernel-active profiling or exclusive GPU time.")
    short_findings = []
    for ds in DATASETS:
        row = next(r for r in deltas if r["dataset"] == ds and r["comparator"] == "LightSAKT")
        n = row["paired_seeds"]
        if n:
            short_findings.append(f"- {ds}: {n}/5 paired seeds; mean validation/test AUC gains "
                f"{row['validation_auc_delta_mean']:+.6f} / {row['test_auc_delta_mean']:+.6f}; "
                f"test AUC higher in {row['test_positive_seed_count']}/{n} seeds.")
        else:
            short_findings.append(f"- {ds}: PENDING" + (" — separate question identifiers are unavailable." if ds == "ASSIST2015" else "."))
    report = f'''# LightSAKT with question difficulty: verified comparison

Report date: 30 October 2026. Campaign status: **{campaign['status']}**.
Verified model-seed checkpoints: **{len(records)} of 45** intended slots
(three datasets, three models, five seeds). Missing measurements are PENDING.

## Research objective and measured findings

Determine whether a training-derived question-difficulty correction improves
LightSAKT relative to the established LightSAKT and SAKT baselines. The reported
ASSIST2009 seed-42 validation AUCs, 0.750803537 and 0.781189972, reproduce from
saved checkpoints within CPU/GPU rounding tolerance. They are validation scores;
held-out test results are reported separately below.

{findings}

## Main comparison

{table_text}

Entries are mean ± **sample standard deviation** (ddof = 1) across the listed
seeds. F1 and accuracy in this table are **test** metrics. A single observation
has no standard deviation; incomplete seed sets remain PENDING. Precision,
recall, BCE and all validation metrics are included in the accompanying
`individual_seed_results.csv` and `summary_results.csv` (also available as JSON).
Threshold metrics use 0.5, without validation- or test-based threshold tuning.

![Held-out test AUC by dataset and model](test_comparison.png)

Points show observed means with sample-SD error bars. Missing results have no
plotted estimate. Standalone PDF versions accompany the figures.

## Absolute AUC improvements

Deltas use identical paired seeds. A partial modified seed set is compared only
with those same baseline seeds, rather than a five-seed baseline mean. Positive
values favor the modified model. These are absolute AUC units, not percentages.

{chr(10).join(delta_table)}

Per-seed deltas and minimum observed effects are preserved in
`paired_seed_auc_deltas.csv` and `absolute_auc_improvements.csv`.

## Original architectures and modification

“Original” in these comparisons means the established **corrected causal
baseline** with the author's original trainable architecture and parameter
shapes. The preserved historical `original_snapshot` had future-attention,
masking and test-selection defects identified in earlier research. Its unverified
historical scores are not included in these results.

LightSAKT has one encoder block. Interaction embedding M(q + K*r) plus learned
position embedding P supplies historical keys and values; E(next_skill) supplies
queries. Q/K/V/output projections have biases. Multihead normalized linear
attention uses ELU(x) + 1, includes exactly valid positions j <= t in both its
numerator and denominator, and adds epsilon 1e-8. Its first residual is
LayerNorm(Dropout(attention) + M + P + E). The feed-forward network is
*d → floor(d/2) → d*, with ReLU and dropout, followed by a second residual
LayerNorm, a linear prediction head and sigmoid. SAKT uses its native one-block
causal multihead softmax attention and *d → d → d* feed-forward network.

Modified LightSAKT preserves that encoder and adds one zero-initialized learned
scalar alpha to the prediction logit:

`prediction = sigmoid(original_logit + alpha * difficulty)`

The feature contains fitted per-question statistics. Thus one additional neural
parameter does **not** imply just one additional statistical degree of freedom.
Counts of fitted question/skill aggregates and unseen-question rates are retained
in `real_data_leakage_audit.json`; these aggregates are additional model inputs
and are not included in the neural parameter count.
No depth, attention, gating or other architecture changes were introduced here.

## Difficulty feature and leakage checks

With smoothing fixed at 10 before the original pilot:

- Skill rate = (skill successes + 1) / (skill observations + 2).
- Question rate = (question successes + 10 * skill rate) / (question observations + 10).
- Difficulty = logit(question rate) − logit(skill rate).

For a **training** example, all responses from that entire student are excluded
from both estimates, including its current and future responses. This prevents
self-label encoding even when a question is rare. Validation and test estimates
use training students' labels only. The held-out transformation accepts question
and skill metadata without the query outcome. Unseen questions fall back to the
skill rate, giving a zero correction; rates are smoothed and finite.

CPU tests cover causality, padding, gradients, zero-initialization equivalence,
feature/target alignment, numerical stability and integration with the original
loss/reporting hook. Real-data perturbation audits on ASSIST2009 and Algebra2005
verify validation-label invariance and entire-own-student-label invariance, with
an other-student perturbation as a responsive positive control. Frozen validation
and test sidecars are independently reproduced from training labels only.
See `cpu_tests.xml`, `real_data_leakage_audit.json` and `checkpoint_state_audit.json`.

ASSIST2009 uses raw `problem_id`. Its original training/validation feature file
was numerically reproduced and reused byte-for-byte. Algebra2005 uses the explicit
known target tutor-step identity `(Problem Name, Step Name)`. Every retained row
was aligned against original source-row index, student, transaction time and
skill. No outcome, hint, response duration or post-response timing field defines
item identity. No interaction was dropped or reordered for this modification.
The [official KDD Cup data format](https://kdd.org/kdd-cup/view/kdd-cup-2010-student-performance-evaluation/Data)
defines a step by this same problem-name/step-name pair and retains those fields
in test records while withholding correctness, hints and response timing fields.
This supports their use as known benchmark metadata. Predictions are conditional
on the target step identity, rather than forecasting which step a student chooses.

**ASSIST2015 is blocked:** the available raw file contains only `user_id`,
`log_id`, `sequence_id`, `correct`; `sequence_id` already serves as skill identity.
There is no independent question identifier. Reusing skills as questions would
change the feature definition. No substitute feature or alternative release was
silently introduced, and no modified ASSIST2015 scores are reported.

**Temporal scope:** estimates fit the complete training corpus under the fixed
retrospective student-disjoint protocol. An earlier audit found later-order
training observations for 95.045% of ASSIST2009 validation attempts. This is
permitted offline fitting, but it does not demonstrate forward-time deployment
performance. A chronological evaluation would require a separately frozen
protocol. During held-out KT evaluation, earlier responses from the same held-out
student form causal history as in the original baseline; future held-out responses
do not enter the prediction or fitted difficulty feature.

## Datasets and partitions

{chr(10).join(dataset_table)}

All models share the original student-disjoint 70/10/20 partition with split seed
42. Vocabulary is fitted on training students only, with UNK 0. All n−1
next-response targets per student are retained using length-L chunks with one
boundary interaction of overlap; context resets at chunk boundaries. Dataset
variants, input semantics and raw/processed/split SHA256 fingerprints are recorded
in `dataset_metadata.json`. These local releases should not be conflated with
other published variants. Existing cleaning excludes nonbinary ASSIST2015 rows
and Algebra rows without KC; preprocessing was not changed by this experiment.
ASSIST2009 simultaneous skill tags are collapsed into sorted composite skill-name
strings. ASSIST2015 uses the skill-builder `sequence_id`. Algebra2005 uses the
full `KC(Default)` composite string as one skill and starts from the original
training subset, which is then partitioned by student here. Its reported test
split is this local held-out student partition, not the official challenge test.

## Matched training and selection protocol

{chr(10).join(config_table)}

All runs use masked BCE, globally pooled metrics, float32, no AMP or TF32,
Adam with betas (0.9, 0.999), epsilon 1e-8 and zero weight decay, batch size 64,
at most 50 epochs, and patience 5. Strict improvement in **validation AUC** selects
the best checkpoint; test results never select checkpoints or settings.
Training seeds are 42, 123, 2026, 3407 and 7777.

Original SAKT and LightSAKT independently selected their settings using the same
36-point validation grid. Thus selected heads, dropout and learning rate can
differ between models. Modified LightSAKT inherits the original LightSAKT settings
for each dataset without a new search. Full configurations and provenance are in
`training_configurations.json` and `experiment_provenance.json`.

The modified model receives richer item information than the skill-only baselines.
Any improvement supports this feature and representation change; it does not
isolate attention-kernel superiority. A previously prepared feature-matched SAKT
augmentation control has not been executed in this three-model campaign.
Validation was reused during earlier architecture diagnosis and selection; this
exposure is disclosed. No frozen configuration changed after test release.

## Execution, checkpoint verification and reproducibility

Thirty existing baseline seeds and the completed ASSIST2009 difficulty seed 42
were reused. Baselines were not retrained. This request explicitly authorizes
held-out scoring of frozen validation-selected checkpoints, superseding the prior
test-locked development scope. Nine missing runs (four ASSIST2009 and five
Algebra2005) were frozen in commit `a37b17ed` before any new modified-model test
scoring. No feature-matched SAKT training or new architecture was launched.

The original difficulty pilot stopped after epoch 18. The earlier audited exact
continuation resumed epochs 19–29 from its saved optimizer/RNG/data-order state,
selected epoch 24 and completed early stopping. That completion, including the
preserved pre-resume checkpoint archive, is documented in
`reports/architecture_study/question_completion_20261029/`. Its best checkpoint
was reused here; the 0.7812 headline is not an incomplete-run estimate.

Every reported checkpoint is checked against its SHA256, configuration, dataset
and split fingerprints, parameter count and validation-selected epoch, then
independently replayed on CPU validation and test. Earlier GPU scores must agree
within AUC/BCE 2e-6 and accuracy/F1 1e-4; differences and raw scores are retained in
`replays/*.json`. CPU inference uses batch size 8 and performs no optimizer update.
Precision and recall are measured directly. Local ignored prediction arrays have
recorded hashes. An independent check compares targets with every original
student's labels[1:] in exact order, computes AUC using tied-rank Mann–Whitney
statistics, and recomputes classification scores from manual confusion counts
and BCE from probabilities. See `masking_and_metrics_audit.json` for the final
verification status and numerical differences. Completed difficulty checkpoints also retain finite optimizer,
Python/NumPy/Torch/CUDA RNG and data-loader state. The saved shuffle state is
checked against the exact seeded epoch stream; Adam counters match batch counts.
Epoch-wise validation and resource metrics are in `difficulty_epoch_results.csv`.

![Difficulty-model validation curves and allocated memory](difficulty_training_curves.png)

GPU work uses only the previously permitted shared MIG allocation. Each run has
a 4 GiB allocator cap and a 900-second execution window, with admission free memory
10 GiB, runtime free memory 6 GiB, host headroom 2 GiB, worker RSS ceiling 4 GiB,
0.1-second batch pacing and a 30-second maximum step time. The nine-window ceiling
is 8,100 GPU seconds. An unsafe stop halts the campaign without automatic retry.
Owned worker identity and resource samples are logged; unrelated GPU owners are
not assumed to belong to this project and no peer workload is modified.

Training times include their recorded execution environment. New guarded/paced
runs and historical unpaced baseline timings are **not** controlled speed
comparisons. Existing checkpoints, source, raw records, splits and hold markers
are fingerprinted and preserved. Execution status and resources are in
`campaign_status.json`, `campaign_training.log` and `campaign_resource_samples.jsonl`.

{resource_text}

## Individual seed results

{chr(10).join(seed_table)}

Exact precision, recall, BCE, memory, runtime, checkpoint locations and SHA256
values accompany each verified row in the machine-readable tables. PENDING
values are missing observations, not zero scores or estimates.

## Limitations and next actions

Five seeds assess initialization variability conditional on one student split;
they do not measure variation across new student cohorts. Consistency across all
three datasets cannot be established while ASSIST2015 lacks question metadata.
The current table is a verified status report, not a claim that all requested
modified-model experiments are complete. Report both positive and negative seeds.

Resolve ASSIST2015 item metadata from an authoritative source before defining its
modified experiment; preserve the current split or explicitly disclose if an
alternative release prevents equivalence. A feature-matched SAKT control would
separate the value of item information from the LightSAKT encoder. Any further
architecture changes, tuning or chronological studies require a new frozen
protocol; this test set must not become a tuning target. No general improvement
claim follows from a best seed or a validation-only gain.

## Evidence-backed conclusion

{conclusion}
'''
    (OUT / "PROFESSOR_REPORT.md").write_text(report)
    concise = f'''# Professor progress summary — 30 October 2026

The ASSIST2009 seed-42 LightSAKT validation improvement from **0.7508 to 0.7812**
reproduces from the saved checkpoints. The modified model retains causal linear
attention and adds a scalar correction based on training-derived question
statistics. Training features exclude the whole student's labels; validation/test
estimates use training labels only. Real-data leakage checks passed.

{chr(10).join(short_findings)}

{table_text}

Values are measured mean ± sample SD across the listed seeds; missing results are
PENDING. Campaign status: **{campaign['status']}**. Thirty baseline seeds were
already complete and independently replayed. The authorized missing-seed campaign
contains nine new difficulty runs, with original seed 42 reused. ASSIST2015
requires question metadata before the same modification can be tested fairly.
These results concern a retrospective student-disjoint benchmark with richer item
inputs, not forward-time deployment or attention-kernel superiority. The full
report includes individual seeds, precision/recall, configurations and provenance.
'''
    (OUT / "PROFESSOR_SUMMARY.md").write_text(concise)
    # A visual with only observed estimates. No placeholder heights for missing data.
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    x = np.arange(3); width = .24
    for j, model in enumerate(MODELS):
        group = [next(r for r in summary if r["dataset"] == ds and r["model"] == model) for ds in DATASETS]
        positions = []; values = []; errors = []
        for i, row in enumerate(group):
            if row["test_auc_mean"] is not None:
                positions.append(x[i] + (j - 1) * width); values.append(row["test_auc_mean"]); errors.append(row["test_auc_sd"] or 0)
            else:
                ax.text(x[i] + (j - 1) * width, .69, "PENDING", rotation=90, ha="center", fontsize=7)
        ax.errorbar(positions, values, yerr=errors, fmt="o", label=model, capsize=3, markersize=5)
    ax.set_xticks(x, DATASETS); ax.set_xlim(-.5, 2.5); ax.set_ylim(.68, .88)
    ax.set_ylabel("Test AUC (observed mean ± sample SD)"); ax.legend(fontsize=8)
    ax.set_title("Frozen student-disjoint benchmark; incomplete rows remain PENDING")
    for suffix in ["png", "pdf"]:fig.savefig(OUT / f"test_comparison.{suffix}", dpi=180)
    plt.close(fig)
    epoch_path = OUT / "difficulty_epoch_results.csv"
    if epoch_path.exists():
        epochs = pd.read_csv(epoch_path)
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for row_index, dataset in enumerate(["ASSIST2009", "Algebra2005"]):
            for seed, values in epochs[epochs.dataset == dataset].groupby("seed"):
                axes[row_index, 0].plot(values.epoch, values.validation_auc, label=str(seed))
                axes[row_index, 1].plot(values.epoch, values.validation_bce, label=str(seed))
                axes[row_index, 2].plot(values.epoch, values.peak_allocated_bytes / 1024**2, label=str(seed))
            for axis in axes[row_index]:
                axis.set_xlabel("Completed epoch")
                axis.set_title(dataset)
            axes[row_index, 0].set_ylabel("Validation AUC")
            axes[row_index, 1].set_ylabel("Validation BCE")
            axes[row_index, 2].set_ylabel("Peak allocated GPU memory (MiB)")
            axes[row_index, 2].set_ylim(0, 1024)
            axes[row_index, 2].ticklabel_format(axis="y", style="plain", useOffset=False)
            if (epochs.dataset == dataset).any():
                axes[row_index, 0].legend(title="Seed", fontsize=8)
            else:
                axes[row_index, 0].text(.5, .5, "PENDING", ha="center", transform=axes[row_index, 0].transAxes)
        fig.suptitle("Question-difficulty runs: recorded complete epochs")
        for suffix in ["png", "pdf"]:
            fig.savefig(OUT / f"difficulty_training_curves.{suffix}", dpi=180)
        plt.close(fig)
    print(f"Report snapshot: {len(records)} verified model-seed checkpoints; campaign {campaign['status']}.")


if __name__ == "__main__":
    main()
