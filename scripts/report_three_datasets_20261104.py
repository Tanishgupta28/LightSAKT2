"""Measured-only three-dataset release; no training, inference or model tuning."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
from pathlib import Path
import shutil
from statistics import mean, stdev
import numpy as np
import pandas as pd
from scripts.ednet_replacement_data import ROOT, WORK, OUT as EDNET_OUT, DS, write
from scripts.architecture_study import digest
from scripts.replay_ednet_campaign_20261104 import independent_metrics

OUT = ROOT / "reports/professor_three_datasets_20261104"
SEEDS = [42, 123, 2026, 3407, 7777]
DATASETS = ["ASSIST2009", "Algebra2005", DS]
MODELS = ["SAKT", "LightSAKT", "LightSAKT+Difficulty"]
METRICS = ["auc", "accuracy", "f1", "precision", "recall", "bce"]
DISPLAY = {"ASSIST2009": "ASSIST2009", "Algebra2005": "Algebra2005", DS: "EdNet-KT1 (5k students; first/session)"}


def measured_records():
    old = ROOT / "reports/professor_results_20261030/replays"
    records = []
    checks = []
    expected = {}
    for dataset in DATASETS[:2]:
        frame = pd.read_csv(ROOT.parent / f"LightSAKT2/data/processed/{dataset}/interactions.csv", dtype={"student": str})
        for split in ["validation", "test"]:
            expected[dataset, split] = np.concatenate([g.label.to_numpy()[1:] for _, g in frame[frame.split == split].groupby("student", sort=True)])
        for model in MODELS:
            for seed in SEEDS:
                path = old / f"{dataset}-{model}-{seed}.json"
                r = json.loads(path.read_text())
                assert digest(r["checkpoint"]) == r["checkpoint_sha256"]
                for split in ["validation", "test"]:
                    item = r["prediction_artifacts"][split]
                    assert digest(item["path"]) == item["sha256"]
                    with np.load(item["path"]) as arrays:
                        y, p = arrays["target"], arrays["prediction"]
                    assert np.array_equal(y, expected[dataset, split])
                    actual = independent_metrics(y, p)
                    assert max(abs(v - r["cpu_replay"][split][k]) for k, v in actual.items()) < 1e-12
                records.append(r)
                checks.append({"dataset": dataset, "model": model, "seed": seed, "status": "PASS",
                    "existing_checkpoint_and_prediction_hashes": True, "independent_metrics_and_exact_targets": True,
                    "source_replay": str(path), "source_replay_sha256": digest(path), "new_inference": False})
    for path in sorted((EDNET_OUT / "replays").glob("*.json")):
        r = json.loads(path.read_text()); assert r["status"] == "VERIFIED_CPU_REPLAY"
        assert digest(r["checkpoint"]) == r["checkpoint_sha256"]
        records.append(r)
    assert len({(r["dataset"], r["model"], r["seed"]) for r in records}) == len(records)
    write(OUT / "historical_reuse_verification.json", {"status": "PASS", "reused_runs": len(checks), "checks": checks,
        "no_old_training_or_inference_repeated": True})
    return records


def write_tables(records):
    lookup = {(r["dataset"], r["model"], r["seed"]): r for r in records}
    individual, refs = [], []
    for dataset in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                r = lookup.get((dataset, model, seed))
                row = {"dataset": dataset, "dataset_display": DISPLAY[dataset], "model": model, "seed": seed,
                       "status": "VERIFIED" if r else "PENDING"}
                if r:
                    for split in ["validation", "test"]:
                        # Historical native validation logged four metrics;
                        # its already verified CPU replay supplies precision/recall.
                        row.update({f"{split}_{k}": r[split].get(k, r["cpu_replay"][split][k]) for k in METRICS})
                    row.update(parameters=r["parameters"], best_epoch=r["best_epoch"], epochs=r["epochs_trained"],
                        training_seconds=r["training_seconds"], peak_allocated_bytes=r.get("peak_allocated_bytes"),
                        checkpoint_sha256=r["checkpoint_sha256"], run_id=r["run_id"], training_git_commit=r["selected_git_commit"])
                    config_path = OUT / "configurations" / f"{dataset}-{model}-{seed}.json"
                    write(config_path, r["config"])
                    refs.append({"dataset": dataset, "model": model, "seed": seed, "checkpoint": r["checkpoint"],
                        "checkpoint_sha256": r["checkpoint_sha256"], "configuration": str(config_path.relative_to(OUT)),
                        "configuration_sha256": digest(config_path), "processed_sha256": r["processed_sha256"],
                        "split_sha256": r["split_sha256"], "run_id": r["run_id"], "weights_in_archive": False})
                individual.append(row)
    summary = []
    for dataset in DATASETS:
        for model in MODELS:
            group = [r for r in individual if r["dataset"] == dataset and r["model"] == model and r["status"] == "VERIFIED"]
            row = {"dataset": dataset, "model": model, "verified_seeds": len(group), "expected_seeds": 5,
                   "status": "COMPLETE" if len(group) == 5 else "PENDING"}
            for split in ["validation", "test"]:
                for metric in METRICS:
                    values = [r[f"{split}_{metric}"] for r in group]
                    row[f"{split}_{metric}_mean"] = mean(values) if values else None
                    row[f"{split}_{metric}_sd"] = stdev(values) if len(values) > 1 else None
            summary.append(row)
    paired, gains = [], []
    for dataset in DATASETS:
        for comparator in MODELS[:2]:
            group = []
            for seed in SEEDS:
                a, b = lookup.get((dataset, MODELS[2], seed)), lookup.get((dataset, comparator, seed))
                if a and b:
                    row = {"dataset": dataset, "comparator": comparator, "seed": seed,
                           **{f"{s}_auc_delta": a[s]["auc"] - b[s]["auc"] for s in ["validation", "test"]}}
                    group.append(row); paired.append(row)
            row = {"dataset": dataset, "comparator": comparator, "paired_seeds": len(group),
                   "status": "COMPLETE" if len(group) == 5 else "PENDING"}
            for split in ["validation", "test"]:
                v = [r[f"{split}_auc_delta"] for r in group]
                row[f"{split}_auc_delta_mean"] = mean(v) if v else None
                row[f"{split}_auc_delta_sd"] = stdev(v) if len(v) > 1 else None
                row[f"{split}_positive_seed_count"] = sum(x > 0 for x in v)
            gains.append(row)
    for name, rows in [("individual_seed_results", individual), ("summary_results", summary),
                       ("paired_seed_deltas", paired), ("absolute_auc_improvements", gains), ("checkpoint_references", refs)]:
        write(OUT / f"{name}.json", rows)
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    write(OUT / "experiment_provenance.json", {"records": records, "raw_data_or_weights_in_archive": False,
        "historical_results_commit": "c286bcbc", "historical_release_commit": "2103a6e3",
        "replacement_preparation_commit": "5f1b830d3aff3cce146fb5cce215d1e93283cac7",
        "pre_registered_plan_sha256": digest(ROOT / "experiments/ednet_replacement_20261104/plan.json")})
    return individual, summary, gains


def format_metric(row, field):
    value, sd = row.get(field + "_mean"), row.get(field + "_sd")
    return "PENDING" if value is None else f"{value:.6f} ± {sd:.6f}" if sd is not None else f"{value:.6f} (n=1)"


def graph(summary, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 180})
    colors = ["#7b8a9b", "#3478a3", "#c76d2c"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, title in zip(axes.flat, ["validation_auc", "test_auc", "test_accuracy", "test_f1"],
                                 ["Validation AUC", "Held-out test AUC", "Test accuracy", "Test F1 (threshold 0.5)"]):
        for i, model in enumerate(MODELS):
            x, y, sd = [], [], []
            for j, dataset in enumerate(DATASETS):
                row = next(r for r in summary if r["dataset"] == dataset and r["model"] == model)
                if row[metric + "_mean"] is not None:
                    x.append(j + (i - 1) * .23); y.append(row[metric + "_mean"]); sd.append(row[metric + "_sd"] or 0.)
            ax.bar(x, y, width=.22, yerr=sd, capsize=3, label=model, color=colors[i])
        ax.set_xticks(range(3), ["ASSIST2009", "Algebra2005", "EdNet-KT1\n5k, first/session"])
        ax.set_title(title); ax.set_ylim(.5, .95); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, .01))
    fig.suptitle("Matched five seeds; means ± sample SD; fixed student-disjoint splits" if len(records) == 45
                 else "Interim measurements; incomplete runs PENDING; sample SD where available")
    fig.tight_layout(rect=(0, .05, 1, .95)); OUT.joinpath("graphs").mkdir(exist_ok=True)
    for suffix in ["png", "pdf"]: fig.savefig(OUT / f"graphs/three_dataset_comparison.{suffix}")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, dataset in zip(axes, DATASETS):
        for model, color in zip(MODELS, colors):
            group = sorted([r for r in records if r["dataset"] == dataset and r["model"] == model], key=lambda r: SEEDS.index(r["seed"]))
            ax.plot([SEEDS.index(r["seed"]) for r in group], [r["test"]["auc"] for r in group], "o-", label=model, color=color)
        ax.set_title(DISPLAY[dataset].replace(": ", "\n")); ax.set_xticks(range(5), SEEDS, rotation=35)
        ax.set_ylabel("Held-out test AUC"); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); fig.tight_layout()
    for suffix in ["png", "pdf"]: fig.savefig(OUT / f"graphs/individual_seed_test_auc.{suffix}")
    plt.close(fig)


def documentation(individual, summary, gains):
    resource = json.loads((EDNET_OUT / "resource_summary.json").read_text())
    resource_fact = (f"Actual core: {len(resource['completed_runs'])}/15 runs, "
        f"{resource['gpu_window_wall_seconds'] / 60:.2f} admitted window minutes, "
        f"{resource['maximum_allocator_peak_bytes'] / 1024**3:.6f}GiB maximum Torch allocation, "
        f"{max(r['training_seconds'] for r in resource['completed_runs']):.2f}s longest native training invocation. "
        f"Resource-floor/RSS breaches: {len(resource['breach_sample_indices'])}.")
    table = ["| Dataset | Model | Validation AUC | Test AUC | Test accuracy | Test F1 | Seeds |", "|---|---|---|---|---|---|---|"]
    for r in summary:
        table.append(f"| {DISPLAY[r['dataset']]} | {r['model']} | " + " | ".join(format_metric(r, k) for k in ["validation_auc", "test_auc", "test_accuracy", "test_f1"]) + f" | {r['verified_seeds']}/5 |")
    comparison = "\n".join(table)
    gain_table = ["| Dataset | Comparator | Validation AUC gain | Test AUC gain | Positive validation / test seeds |", "|---|---|---|---|---|"]
    for r in gains:
        gain_table.append(f"| {DISPLAY[r['dataset']]} | {r['comparator']} | {format_metric(r,'validation_auc_delta')} | {format_metric(r,'test_auc_delta')} | {r['validation_positive_seed_count']}/{r['paired_seeds']} / {r['test_positive_seed_count']}/{r['paired_seeds']} |")
    individual_table = ["| Dataset | Model | Seed | Validation AUC | Test AUC | Test accuracy | Test F1 |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in individual:
        v = [f"{r[k]:.9f}" if k in r else "PENDING" for k in ["validation_auc", "test_auc", "test_accuracy", "test_f1"]]
        individual_table.append(f"| {DISPLAY[r['dataset']]} | {r['model']} | {r['seed']} | " + " | ".join(v) + " |")
    (OUT / "comparison_table.md").write_text(comparison + "\n")
    (OUT / "individual_seed_table.md").write_text("\n".join(individual_table) + "\n")
    config_table = ["| Dataset | Model | Dimension | Heads | Dropout | Learning rate | Neural parameters |", "|---|---|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for model in MODELS:
            r = next((r for r in individual if r["dataset"] == dataset and r["model"] == model and r["status"] == "VERIFIED"), None)
            if r:
                cfg = json.loads((OUT / "configurations" / f"{dataset}-{model}-{r['seed']}.json").read_text())
                m = cfg["model_config"]
                config_table.append(f"| {DISPLAY[dataset]} | {model} | {m.get('embed_dim',m.get('d'))} | {m.get('num_heads',m.get('num_attn_heads'))} | {m['dropout']} | {cfg['learning_rate']} | {r['parameters']} |")
    complete = sum(r["status"] == "VERIFIED" for r in individual)
    new_complete = sum(r["status"] == "VERIFIED" and r["dataset"] == DS for r in individual)
    headline = f"{complete}/45 core model-seed measurements verified; {new_complete}/15 replacement experiments completed."
    conclusions = []
    for r in gains:
        if r["status"] == "COMPLETE":
            conclusions.append(f"{DISPLAY[r['dataset']]}: modified minus {r['comparator']} mean test AUC = {r['test_auc_delta_mean']:+.6f}; positive in {r['test_positive_seed_count']}/5 matched seeds.")
    report = f'''# LightSAKT + Question Difficulty: three-dataset research results

4 November 2026. {headline}

## Objective and measured findings

Evaluate whether the existing question-success correction improves LightSAKT on three student-disjoint datasets while retaining its causal linear attention. Original ASSIST2009 and Algebra2005 measurements are preserved and reused. ASSIST2015 is replaced with an explicitly reduced EdNet-KT1 protocol approved in principle by the professor. No HiTSKT files or jobs were accessed or changed.

{comparison}

Entries are mean ± **sample SD (ddof=1)** over seeds 42, 123, 2026, 3407 and 7777. Accuracy and F1 in this table are held-out TEST metrics; validation AUC is a separate selection metric. All metrics, including precision, recall, BCE, individual seeds and exact values, are in the paired CSV/JSON files. Incomplete measurements are PENDING; aggregates of partial groups must not be presented as five-seed results.

### Absolute AUC changes: modified LightSAKT minus comparator

{chr(10).join(gain_table)}

{chr(10).join(conclusions)}

The earlier ASSIST2009 seed-42 0.750803537 → 0.781189972 claim concerns validation AUC. It is reproduced by existing checkpoints and CPU replay, not confused with either five-seed mean or test AUC. Differences above are absolute AUC units; paired-difference SD is calculated from matched seeds rather than subtracting group SDs. No publication-readiness, statistical significance or attention-superiority claim follows from these descriptive comparisons.

## Verified architecture and modification

Established original LightSAKT has one normalized causal ELU+1 linear attention block, skill/response embeddings, learned positions, residual layer normalization and a d→d/2→d ReLU FFN. SAKT retains its native causal softmax and d→d→d FFN. “Original” means the previously corrected causal implementation; the preserved historical original_snapshot contained masking/attention/selection defects and its unverified results are excluded. No native source or architecture changed for this replacement.

Modified LightSAKT adds one zero-initialized learned scalar to the readout:
`p = sigmoid(original_logit + alpha * [logit(p_item) - logit(p_skill)])`.
`p_skill=(skill_successes+1)/(skill_count+2)`;
`p_item=(item_successes+10*p_skill)/(item_count+10)`.
Training rates exclude the ENTIRE current student's rows from both aggregates; validation and test use training students only. Unseen items yield exactly zero correction. The raw official correct answer and current user answer are used to construct labels only, never predictors. Alpha=0 preserves the baseline initialization and forward behavior. Attention, depth and native LightSAKT settings remain unchanged.

The fitted item rates are additional information and estimation capacity, even though the neural parameter count grows by only one. They condition on a known item ID that the skill-only baselines do not receive. Improvements do not establish linear-attention superiority.

## Dataset selection and ASSIST2015 replacement

EdNet-KT1 was selected over Junyi because genuine repeated `question_id` values, student sequences, official answer keys and tags were actually obtained from the [publisher's EdNet repository](https://github.com/riiid/ednet). The [EdNet paper](https://arxiv.org/abs/1912.03072) describes the source research dataset. Full archive byte sizes/hashes and actual schema checks appear in DATASET_AND_SPLITS.md. Source metadata has 13,169 unique questions. No synthetic item identifiers or skill IDs were substituted.

The original Junyi study distinguishes exercises containing multiple problems ([Chang, Hsu and Chen, 2015](https://ken77921.github.io/assets/pdf/proj8-education-student%20modeling-paper.pdf)). Individual repeated-question identity was not established from the examined exercise/problem-type metadata. Exercise-level difficulty would be a different labelled protocol. The older DataShop source was inaccessible (HTTP403), and no Junyi payload was needed once authentic EdNet was available. This is not a claim that every Junyi release lacks item IDs.

ASSIST2015's preserved official four-column release has user_id, log_id, sequence_id and correct. Its prior authoritative-source audit found no independent item crosswalk: event IDs and exercise/skill-builder sequences cannot be relabelled as questions. Its five difficulty slots remain historically unavailable, not completed. The old baseline/results/reports are preserved; see ../professor_release_20261031/ASSIST2015_LIMITATION.md. It is excluded from the final three-dataset comparison.

## Dataset protocol and statistics

EdNet is explicitly **5,000 uniformly selected authentic students, first question per solving session**, not the full KT1 benchmark. Selection uses sorted complete ZIP user filenames and NumPy default_rng seed20261104, before reading labels, lengths or model scores. The 5,000 selected ZIP members passed complete-file CRC checks. The full 784,309-user archive was downloaded but the unselected population was not exhaustively decompressed/CRC-tested. No publisher SHA was available; hashes are locally computed fingerprints of the publisher-linked payload.

| Count | EdNet subset |
|---|---:|
| Selected/retained students | 5,000 / 5,000 |
| Students with ≥2 retained observations | 4,816 |
| Raw selected interactions | 721,760 |
| Exact duplicate rows removed | 34 |
| Other questions in same solving session excluded | 158,161 |
| Invalid first-question answers excluded | 96 |
| Missing question metadata | 0 |
| Retained interactions | 563,469 |
| Authentic observed question IDs | 11,893 |
| Train / validation / test students | 3,500 / 500 / 1,000 |
| Train / validation / test interactions | 407,242 / 43,375 / 112,852 |
| Train / validation / test targets | 403,742 / 42,875 / 111,852 |
| Training tagset vocabulary + UNK | 1,451 + 1 |

Student IDs are genuine filename IDs. Chronology uses timestamp then original row order. Exact duplicates are removed; repeated questions in different sessions remain. First-in-session selection occurs before inspecting answer validity and invalid first answers are not replaced. Timestamp ranges must not overlap/tie between sessions; all selected students passed. This prevents other current-session responses from entering the history, but sacrifices bundle coverage and may bias toward first-question positions. It is a pre-registered operational definition, not the official full-question benchmark or proof of wall-clock deployment causality.

Answer correctness is equality between the recorded answer and official key; only a–d are valid (part2 a–c). Full sorted unique question tags form composite skills without row explosion. Vocabulary is fit only on training students; validation/test unseen tags map to UNK0 (1/4 rows respectively). Student split seed42 reuses the existing 70/10/20 function. Each student belongs to exactly one split. The 184 single-observation students remain in the population but have zero next-response targets.

ASSIST2009 retains 274,590 interactions and 2,905/415/831 students; Algebra2005 retains 607,025 interactions and 401/57/116 students. Algebra uses the local student split of the original training subset, not the challenge's official test; target item means lossless (Problem Name, Step Name). Full original counts, fingerprints and identifier semantics are preserved in dataset_metadata.json.

## Training, selection and evaluation

All replacement models use d64, 2 heads, dropout0.1, learning rate0.001, batch64, length100, Adam betas(0.9,0.999), epsilon1e-8, decay0, float32, no AMP/TF32, max50 epochs and patience5. The common compact setting was frozen before EdNet model scores; no new grid search occurred. Historical datasets retain their previously model-specific selected configurations (their original 36-point validation grid). Equal tuning search across all three datasets is not claimed. Per-run configs list exact dimensions, heads, dropout and neural parameters. Algebra sequence length remains200; ASSIST2009 remains100.

{chr(10).join(config_table)}

Native one-overlap chunks preserve every student's n−1 next-response targets, reset context at chunk boundaries and mask padding explicitly. All three models share processed data, student population, split IDs, masks, targets, seed list and budgets. Selection uses STRICT improvement in validation AUC only; ties do not replace the earliest best. Every EdNet config disables test evaluation during training. After completion the frozen selected checkpoint is restored on CPU (batch8) for validation and test, without changing configurations based on test results. Historical baseline flags allowed post-training test scoring; the reported TEST values consistently use their saved CPU replay. Validation values are native selected scores, independently reproduced within AUC/BCE2e-6 and classification1e-4.

Metrics pool all valid interactions globally, not equally weighted students or batch means. Accuracy/F1/precision/recall use threshold0.5 fixed before inference. BCE clips probabilities to [1e−7,1−1e−7]. Independent rank-sum AUC, manual confusion counts and BCE must agree within 1e−12; prediction target arrays must equal concatenated per-student labels[1:] exactly. Dataset/split hashes and the parameter/selection provenance are bound to each checkpoint.

## Leakage, restoration and resource checks

Real EdNet audits verify training-only vocabulary, authentic item/tag alignment, student disjointness, invariance of a student's features to flipping ALL its own labels, invariance to flipping held-out labels and a responsive other-training-student control. All frozen features were independently recomputed. Unseen-item row fractions are 0.000669 validation and 0.001648 test. Six CPU tests verify session preprocessing, invalid-answer policy, padding, causality, finite gradients and zero-alpha equivalence. The optional existing SAKT+Difficulty class also passes the generic EdNet adapter CPU check.

Every new completed latest checkpoint is checked for finite model/Adam state, exact configuration, contiguous epochs, best-validation selection, valid Python/NumPy/Torch/CUDA RNG and a loader RNG matching the exact seeded epoch shuffle stream. Adam step counts match completed batches. Best checkpoints are restored strictly, parameters counted and all saved prediction arrays hashed. Old checkpoint/prediction arrays and independent metrics are reused and rechecked without repeating their training/inference. Checkpoint hashes/configurations, audit outputs and provenance accompany this report; preservation checks cover 490 historical source/report artifacts plus holds and frozen new data/science.

The previously authorized shared MIG allocation was verified immediately before launch and guarded throughout; it was not claimed exclusive. The existing 4GiB allocator cap, 900s per invocation, free GPU/host memory floors, RSS bound, 0.1s batch pacing and 30s step bound remain. Core scope is 15 serial windows, at most13,500s, without automatic retries. Checkpoints are saved every completed epoch. No unrelated process is signalled or changed. Actual timings/peaks are in resource_summary.json; wall time includes deliberate pacing and is not exclusive GPU time or kernel profiling.

{resource_fact}

As an explicit reproducibility check beyond the fifteen unique new checkpoint replays, seed42 was restored again separately for all three model families. All six validation/test prediction arrays reproduced bit for bit on CPU, without overwriting the saved arrays or executing additional GPU inference. See fresh_prediction_reproduction.json.

## Scientific limitations and next comparison

This is **offline student-disjoint** fitting. Item/skill rates use the complete permitted training corpus, including later events of other training students. The explicit timestamp audit finds later training observations in the same item/skill aggregates for 99.6012% of validation targets and 99.5887% of test targets (same-item alone: 96.8% / 95.3930%). These statistics would be unavailable at those historical times in a strict forward-time study. This is a material temporal limitation, not held-out-label contamination under the stated offline protocol; no online-deployment improvement is established. Earlier held-out outcomes may be used as observed causal history for later predictions, identically across models; current/future held-out outcomes do not enter predictors. Within-training self-label leakage is prevented by whole-student exclusion, but a chronological/cross-cohort study remains necessary for deployment claims.

Five initializations share one fixed split per dataset. Their SD captures optimization variation, not independent population/split uncertainty. EdNet's longest validation student contributes 8,194/42,875 targets (19.11%); the longest test student contributes 8,020/111,852 (7.17%). These globally pooled interaction metrics can be strongly influenced by long histories, reinforcing the need for independent cohorts/splits and student-level uncertainty checks. ASSIST2009 validation guided prior design; historical selection exposure remains. EdNet's random subset, composite tags, first/session sampling and untuned compact configuration limit broader generalization. The splits contain unseen students, largely known items: only 0.0669% of validation and 0.1648% of test rows have unseen question IDs. This is permitted item overlap, not evidence of cold-start item performance. The fitted rates are population-dependent success estimates, potentially influenced by student ability and item selection; they are not validated intrinsic difficulty or causal-effect parameters. Released tests must not guide further tuning. These are local controlled protocols and must not be compared directly to unrelated published leaderboard values. Negative or inconsistent results are retained.

The essential feature-matched control is **SAKT + the identical correction**, with the same item statistics, smoothing, own-student exclusion, native SAKT encoder, data, masks, optimizer, seeds, budget and validation selection. Its EdNet five configs and tested CPU adapter are prepared separately; no control GPU runs were charged to the core budget. A distinct resource budget is needed before executing the control campaign. Compare within-encoder gains and both feature-matched models before attributing benefits to attention architecture.

## Reproducibility and artifacts

See REPRODUCIBILITY.md for authoritative download links, exact hashes, reconstruction commands, fixed configs, checkpoint references and archive verification. Machine-readable tables include means/sample SD for validation/test AUC, accuracy, F1, precision, recall and BCE, individual seed metrics, paired deltas and neural parameters. Graphs under graphs/ use actual verified values only. Existing checkpoints/raw data are preserved locally; the portable reporting archive excludes raw data, student-level arrays and model binaries and contains explicit checkpoint references instead. It is not a self-contained model-inference bundle.

## Individual-seed results

{chr(10).join(individual_table)}
'''
    (OUT / "PROFESSOR_REPORT.md").write_text(report)
    summary_text = f'''# Professor progress summary — 4 November 2026

{headline}

ASSIST2015 was replaced with EdNet-KT1 because the preserved ASSIST2015 release lacks independent question identifiers. EdNet has authentic publisher question IDs and answer metadata. We fixed a random 5,000-student subset and retained the first question per solving session to avoid within-bundle response history. All three models use matched student splits, five seeds and validation-selected checkpoints.

{comparison}

{chr(10).join(conclusions)}

The earlier ASSIST2009 0.7508→0.7812 statement is a seed-42 validation result. The table distinguishes five-seed validation and held-out test performance. All reported numbers come from preserved checkpoint/prediction evidence; means use sample SD.

The correction adds training-derived item information and one neural scalar while preserving causal linear attention. Whole-student exclusion removes training self-label contamination; held-out labels do not fit the feature. This remains an offline protocol using complete training-corpus statistics, not demonstrated chronological deployment. EdNet is the explicitly reduced first-question/session subset, not the full KT1 benchmark.

These measurements are appropriate to share as controlled research progress. They do not establish publication readiness or attention superiority. The next key experiment is the separately budgeted, feature-matched SAKT+Difficulty control, followed by independent split/cohort and chronological evaluation. No further tuning should use these released test outcomes.
'''
    (OUT / "PROFESSOR_SUMMARY.md").write_text(summary_text)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = measured_records()
    individual, summary, gains = write_tables(records)
    graph(summary, records); documentation(individual, summary, gains)
    metadata = json.loads((ROOT / "reports/professor_results_20261030/dataset_metadata.json").read_text())
    metadata = [r for r in metadata if r["dataset"] != "ASSIST2015"] + [json.loads((EDNET_OUT / "dataset_statistics.json").read_text())]
    write(OUT / "dataset_metadata.json", metadata)
    for name in ["resource_summary.json", "epoch_results.csv", "checkpoint_state_audit.json", "protected_artifacts_audit.json",
                 "real_data_feature_audit.json", "raw_row_alignment_audit.json", "offline_temporal_availability_audit.json",
                 "population_descriptives.json", "prepared_control_cpu_check.json", "cpu_tests.xml", "verification_notes.json",
                 "fresh_prediction_reproduction.json"]:
        p = EDNET_OUT / name
        if p.exists(): shutil.copy2(p, OUT / name)
    measured = [r for r in individual if r["status"] == "VERIFIED"]
    new = [r for r in records if r["dataset"] == DS]
    campaign = json.loads((EDNET_OUT / "campaign_status.json").read_text())
    audits = json.loads((EDNET_OUT / "checkpoint_state_audit.json").read_text())
    assert len(new) == audits["completed"]
    assert len({r["recorded_training_code_sha256"] for r in new}) <= 1
    for dataset in DATASETS:
        group = [r for r in records if r["dataset"] == dataset]
        assert len({r["processed_sha256"] for r in group}) <= 1 and len({r["split_sha256"] for r in group}) <= 1
        for split in ["validation", "test"]:
            assert len({r["cpu_replay"][split]["n_targets"] for r in group}) <= 1
    if len(measured) == 45:
        assert campaign["status"] == "completed" and len(campaign["completed"]) == 15 and len(new) == 15
        assert all(r["status"] == "COMPLETE" and r["verified_seeds"] == 5 for r in summary)
    write(OUT / "verification_summary.json", {"status": "PASS_COMPLETE" if len(measured) == 45 else "PASS_AVAILABLE_WITH_PENDING",
        "verified_model_seed_results": len(measured), "expected": 45, "historical_training_runs_reused": 30,
        "new_completed_training_runs": len(campaign["completed"]), "new_CPU_checkpoint_replays": len(new),
        "independently_recalculated_validation_test_arrays": 2 * len(measured),
        "new_finite_optimizer_RNG_exact_loader_order_audits": audits["completed"],
        "extra_fresh_CPU_reproductions": "Three seed42 model-family checkpoints, six splits, bitwise-equal predictions; no extra training",
        "unchanged_protected_historical_files": 490, "matched_per_dataset_data_split_masks_targets": True,
        "new_training_test_evaluation_disabled": True, "raw_source_labels_independently_checked": 563469,
        "sample_standard_deviation_ddof": 1, "optional_SAKT_difficulty_controls": "Five CPU-prepared configurations, not GPU-executed",
        "temporal_limit": "Offline full-training-corpus fitting includes later other-training-student observations; no online claim",
        "attention_superiority_established": False, "publication_readiness_claimed": False})
    print(f"GENERATED actual measurements={len(records)}/45, nine summary slots; no training/inference")


if __name__ == "__main__":
    main()
