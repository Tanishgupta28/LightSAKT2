"""Render verified wide-table exports and explicit missing-work documentation."""
from pathlib import Path
from statistics import mean, stdev
import json
import csv
from scripts.build_master_comparison_20261110 import ROOT, OUT, PREP, SEEDS, DS, DATASETS, read, write, csv_file


def main():
    master = read(OUT / "master_comparison.json")
    with (OUT / "individual_seed_results.csv").open(newline="") as stream:
        seeds = [r for r in csv.DictReader(stream) if r["experiment_status"] == "VERIFIED"]
    plan = read(PREP / "plan.json")
    csv_file(OUT / "missing_experiments.csv", plan["entries"])
    configs = []
    for record in read(OUT / "experiment_provenance.json")["completed_records"]:
        c = record["config"]
        configs.append({"dataset": record["dataset"], "model": c["model"], "seed": c["seed"],
            "status": "COMPLETED_VERIFIED", "max_epochs": c["max_epochs"], "patience": c["patience"],
            "seq_len": c["seq_len"], "batch_size": c["batch_size"], "learning_rate": c["learning_rate"],
            "model_config": c["model_config"], "actual_epochs": record["epochs_trained"],
            "best_epoch": record["best_epoch"], "actual_training_seconds": record["training_seconds"],
            "source_configuration": record["config_file"], "checkpoint": record["checkpoint"],
            "implicit_defaults": "output-logit dropout0.5" if c["model"] in ["DKT", "DKT+"] else "native source defaults; see frozen config and source"})
    for entry in plan["entries"]:
        if not entry.get("config"): continue
        c = read(ROOT / entry["config"])
        configs.append({"dataset": entry["dataset"], "model": c["model"], "seed": c["seed"],
            "status": entry["action"], "max_epochs": c["max_epochs"], "patience": c["patience"],
            "seq_len": c["seq_len"], "batch_size": c["batch_size"], "learning_rate": c["learning_rate"],
            "model_config": c["model_config"], "actual_epochs": None, "best_epoch": None, "actual_training_seconds": None,
            "source_configuration": entry["config"], "checkpoint": entry.get("reusable_checkpoint"),
            "implicit_defaults": "MLP dropout0.5; MHA attention dropout0" if c["model"] in ["GKT", "GKT_MHA"] else
            "output-logit dropout0.5" if c["model"] in ["DKT", "DKT+"] else "native source defaults",
            "reason": entry["reason"]})
    csv_file(OUT / "model_configurations.csv", configs)
    lookup = {(r["dataset"], r["model"], int(r["seed"])): r for r in seeds}
    deltas, summary = [], []
    for dataset in DATASETS:
        labels = [r["model"] for r in master if r["dataset"] == dataset and r["experiment_status"] == "VERIFIED_5_SEEDS"
                  and r["model"] != "LightSAKT + Question Difficulty"]
        for comparator in labels:
            local = []
            for seed in SEEDS:
                a = lookup[dataset, "LightSAKT + Question Difficulty", seed]
                b = lookup[dataset, comparator, seed]
                row = {"dataset": dataset, "comparator": comparator, "seed": seed,
                    **{m + "_delta": float(a[m]) - float(b[m]) for m in ["test_auc", "test_accuracy", "test_f1"]}}
                local.append(row); deltas.append(row)
            row = {"dataset": dataset, "comparator": comparator}
            for m in ["test_auc", "test_accuracy", "test_f1"]:
                v = [x[m + "_delta"] for x in local]
                row.update({m + "_delta_mean": mean(v), m + "_delta_std": stdev(v),
                    m + "_positive_seeds": sum(x > 0 for x in v), m + "_negative_seeds": sum(x < 0 for x in v)})
            summary.append(row)
    csv_file(OUT / "paired_test_differences.csv", deltas)
    csv_file(OUT / "paired_test_difference_summary.csv", summary)
    table = ["| Dataset | Model | Seeds | Validation AUC | Test AUC | Test accuracy | Test F1 | Status |",
             "|---|---|---:|---|---|---|---|---|"]
    for r in master:
        values = [f"{r[k + '_mean']:.6f} ± {r[k + '_std']:.6f}" if r[k + "_mean"] is not None else "PENDING"
                  for k in ["validation_auc", "test_auc", "test_accuracy", "test_f1"]]
        table.append(f"| {r['dataset']} | {r['model']} | {r['seed_count']}/5 | " + " | ".join(values) + f" | {r['experiment_status']} |")
    (OUT / "comparison_table.md").write_text("\n".join(table) + "\n")
    missing_table = ["| Dataset | Model | Missing verified seed results | Selected training checkpoints reusable | Preparation |",
                     "|---|---|---:|---:|---|"]
    for r in master:
        if r["experiment_status"] != "PENDING": continue
        missing_table.append(f"| {r['dataset']} | {r['model']} | {len(r['missing_verified_seeds'])} | {r['selected_training_seed_count_available']} | {r['pending_reason']} |")
    inconsistent = [r for r in summary if r["test_f1_negative_seeds"] or r["test_accuracy_negative_seeds"] or r["test_auc_negative_seeds"]]
    findings = "\n".join(f"- {r['dataset']}, versus {r['comparator']}: AUC positive {r['test_auc_positive_seeds']}/5; accuracy {r['test_accuracy_positive_seeds']}/5; F1 {r['test_f1_positive_seeds']}/5 (negative F1 seeds {r['test_f1_negative_seeds']})." for r in inconsistent)
    if not findings: findings = "No negative AUC/accuracy/F1 matched-seed differences against the available complete comparators were found."
    report = f'''# LightSAKT master research comparison — 10 November 2026

**75 existing verified seed records reused; 15 of 27 requested model/dataset groups complete; 12 groups PENDING. No new GPU training was launched.**

This is a complete inventory CSV with explicit missing results, not a claim that all nine models have finished on all datasets. The 45 previously published SAKT/LightSAKT/difficulty measurements are preserved. Thirty previously completed DKT/DKT+/DKVMN seeds on ASSIST2009 and Algebra2005 are added through their existing independent CPU replay evidence, without retraining or new checkpoint prediction inference.

## Master results

{chr(10).join(table)}

Means and sample SD use exactly seeds42,123,2026,3407,7777. Validation AUC is the native validation-selected score; test AUC/accuracy/F1 use existing verified CPU replay. Classification threshold0.5 and globally pooled masked targets are unchanged. Numeric aggregate cells are empty for PENDING groups; their display fields say PENDING. No single-seed value is presented as a five-seed mean.

## Verification and comparable populations

All75 best checkpoints were SHA256-checked, loaded strictly on CPU, checked for finite weights, correct parameter counts, scientific configurations, processed/split hashes and maximum-validation-AUC epochs. Some original seed42 baselines reuse equivalent selected-tuning checkpoints with documented run-name aliases. This does not create additional independent seeds. The existing independent CPU inference cache contains repeated audit entries; it is de-duplicated by run/seed and matched against exact configuration/checkpoint/prediction/metrics hashes, then the latest matching verification timestamp is used, not the best test score.

All120 available saved prediction arrays (90 previously published validation/test arrays plus30 additional baseline test arrays) match the expected original per-student n−1 targets exactly. Independent rank-sum AUC/manual confusion counts/BCE match their saved-array reference metrics within1e−12. Old baseline caches already contain independent CPU validation/test inference; these values are reused. Native GPU test values and CPU replay values can differ in final floating-point digits, are checked within2e−6 and remain separately traceable. Existing historical reports are not rewritten. The previously published45 canonical measurements remain identical in this export.

All models share per-dataset data bytes, fixed student-disjoint70/10/20 partitions (split seed42), training-only vocabulary/UNK and target masks. ASSIST2009 retains274590interactions/4151students, length100. Algebra2005 retains607025interactions/574students, length200; its holdout is a local split of the original training subset, not the official challenge test. Item identity for the correction is authentic problem_id on ASSIST2009 and the lossless (Problem Name,Step Name) tutor-step pair on Algebra.

EdNet is EXACTLY EdNetKT1_5k_first:5000authentic users uniformly selected before labels/length/scores with seed20261104;563469interactions,11893authentic question IDs;student splits3500/500/1000;target counts403742/42875/111852. The first presented question per solving session is selected before answer validation; other current-session answers never enter history. Correctness comes from recorded answers and official answer keys; tags are full sorted composite categories, without interaction explosion. Length100 and the frozen1452-skill input includingUNK remain unchanged. Dataset/split hashes are in every master row. No full-KT1 leaderboard claim.

## Model-specific settings and budgets

All completed models use Adam,betas(.9,.999),epsilon1e−8,weight decay0,float32,noAMP/TF32,max50epochs,patience5,strict validation-AUC checkpoint selection. Actual epochs/runtime, learning rates, dimensions, regularization coefficients and config references are in model_configurations.csv and master training_budget JSON cells. Historical DKT/DKT+/DKVMN/SAKT/LightSAKT configurations came from their existing validation-only searches; new EdNet baseline templates have not undergone an equivalent search. There is no claim of equal hyperparameter search across datasets.

Batch64 applies to the completed five-seed groups. Existing GKT implementations use the documented batch8 override; do not silently change it. DKT/DKT+ retain native output-logit dropout0.5. GKT MLP dropout is0.5 and its MHA module default attention dropout0. All defaults are preserved rather than retuned.

Equal epoch ceilings do not imply equal optimizer-update counts: batch8 GKT makes more updates than batch64 models. Runtime fields are provenance gathered under different device/pacing policies; they are not a matched efficiency benchmark and must not be ranked as such.

“GKT” here is the repository's simplified PAM implementation. **GKT-MHA (implemented variant)** is the repository's MHA subclass, not an established canonical reproduction of the published GKT architecture. The source expressly omits erase-add gating and MLP batch normalization. Both labels must retain this qualification. Core original LightSAKT/SAKT names refer to the established corrected causal implementations; flawed historical snapshots are excluded.

## Missing experiments and reuse opportunities

{chr(10).join(missing_table)}

There are60missing verified seed-result slots. Two ASSIST2009 GKT training checkpoints are already reusable: each is selected by maximum validation AUC from its complete original36-candidate seed42 grid, without test access. Their best validation AUCs are0.764002738 (PAM) and0.767020187 (MHA); these are training-reuse/validation evidence, **not five-seed test results**. Four additional training seeds and final CPU checkpoint evaluation are needed per group. Do not retrain their selected seed42 runs.

Thus at least58additional selected-configuration training runs remain if those two checkpoints are reused. This excludes extra configuration-selection trials: Algebra2005 GKT tuning/selection is incomplete, so ten final-run slots await validation-only configuration selection. Its completed seed42 primary runs use different fixed settings and cannot silently substitute for selected-final runs. Four historical single-seed GKT primary results are preserved separately in excluded_single_seed_primary_evidence.csv, never in the master aggregate cells. Seventy-two completed ASSIST2009 GKT tuning trials are not72independent seeds.

The EdNet gaps are DKT,DKT+,DKVMN,GKT andGKT-MHA:25seed runs. The SAKT+Difficulty control is absent on allthree datasets:15runs. Every control uses the exact native SAKT configuration and same item-feature sidecar as the corresponding completed LightSAKT+Difficulty protocol. Existing SAKTQuestion class is unchanged; the generic frozen-sidecar adapter has CPU tests for allthree datasets.

Fifty reviewed prepared configuration files cover EdNet's25slots, all15controls and the ten ASSIST2009 GKT final slots (including two checkpoint-reuse entries). Algebra's ten GKT slots are explicitly awaiting configuration selection. Preparations use isolated ignored workspaces with checksum-bound symlinks to original data/splits, not copies or new preprocessing. EdNet DKT/DKT+/DKVMN templates transfer the existing ASSIST2009-selected settings without using EdNet test outcomes. EdNet GKT templates retain original primary settings and require memory/runtime feasibility profiling on1452skills. Prepared configs are NOT an execution authorization; every training config disables test evaluation.

## GPU execution blocker

The old exact MIG device is visible and the read-only telemetry check has technical free-memory/host headroom. That does not establish current ownership or a new allocation/budget. The previous EdNet scope finished15/15 invocations; unused wall-time ceiling does not authorize more runs. No new current allocation confirmation or total execution budget has been verified for this expanded campaign. The user was asked for those missing resource details while CPU work continued. GPU_READINESS.json records the observation and distinction. No unrelated process was signalled, and no HiTSKT files/jobs were accessed.

Retain4GiB allocator/900s windows and previous shared safety floors, pacing, step bounds and admission locks if a new scope is granted. GKT on1452skills may require resource profiling and multiple explicitly budgeted exact-resume windows; completion within one900s window is not assumed. Do not bypass holds or launch an old scheduler. No new training or actual-data model prediction inference occurred in this task; CPU tests use tiny synthetic batches to verify integration/causality/padding/backpropagation.

## Findings and methodological limitations

Original LightSAKT does not lead every available baseline: DKT has higher mean test AUC on ASSIST2009 and DKT+ on Algebra2005. Adding the old missing-family measurements therefore changes the breadth of the comparison, not the historical model scores. The item-augmented LightSAKT has higher mean test AUC than the currently complete comparators, but this is an input-information comparison, not attention-architecture superiority. No conclusion can be drawn against PENDING models.

Matched-seed negative/inconsistent classification findings against newly included comparators:

{findings}

Full paired AUC/accuracy/F1 differences and positive/negative seed counts are in paired_test_differences.csv and paired_test_difference_summary.csv. AUC gains and F1 gains are distinct; a positive mean does not imply every seed improved on every metric. No post-hoc significance or general superiority claim is made.

The difficulty readout still adds one scalar times logit(smoothed item success)−logit(smoothed skill success), with item smoothing10, Beta(1,1) skill prior, whole-own-student exclusion in both training aggregates and train-only held-out fitting; unseen items yield0. It supplies item information unavailable to skill-only baselines. No held-out-label/self-label leakage was identified under the documented OFFLINE student-disjoint protocol. Full training-corpus fitting contains later observations of other training students; EdNet's temporal audit flags99.6012%validation and99.5887%test targets. This does not establish chronological deployment performance.

Five initializations use one split per dataset; SD is optimization variability rather than independent cohort uncertainty. EdNet is largely known-item evaluation; long histories are interaction-weighted (largest validation student19.11%of targets). Its first/session subset and composite-tag skills differ from fullEdNet. Historical validation reuse and unequal hyperparameter-search exposure remain. The feature-matched SAKT+Difficulty control, independent-cohort/split replication and chronological evaluation remain necessary. Do not tune against released test outcomes or alter native models to obtain higher results. ASSIST2015 is excluded from this replacement-dataset master and retained historically.

## Files and reproduction

- master_comparison.csv/JSON:27dataset/model rows; separate numeric mean/std for each metric; generic mean/standard_deviation aliases refer ONLY to primary_metric=test_auc.
- individual_seed_results.csv:135prespecified slots,75verified and60PENDING; no missing measurements imputed.
- missing_experiments.csv and experiments/master_comparison_20261110/plan.json:60slots, reuse decisions,50prepared configs and configuration/resource blockers.
- model_configurations.csv, configurations/, experiment_provenance.json: exact settings, runtime/epoch budgets, checkpoints, SHA256 and original verification sources.
- selected_reusable_gkt_checkpoints.json and excluded_single_seed_primary_evidence.csv: explicit training-only reuse versus excluded single-seed evidence.
- verification_summary.json, export_verification.json, cpu_tests.xml and GPU_READINESS.json: audit scope and resource limits.

From the research root, CPU-only regeneration is `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m scripts.build_master_comparison_20261110`, followed by `python -B -m scripts.report_master_comparison_20261110` and `python -B scripts/verify_master_comparison_20261110.py`. The builder loads existing checkpoints and recomputes metrics from saved arrays; it never calls a trainer or executes model prediction forwards. Preserve all source artifacts. New raw data, predictions and weights are not committed; checkpoint paths are local provenance locators. Publication should use only the research branch, fresh ancestry checks, dry-run then normal push; never main or force.
'''
    (OUT / "REPORT.md").write_text(report)
    (OUT / "README.md").write_text('''# Master comparison inventory — 10 November 2026

Start with REPORT.md and master_comparison.csv. All27 requested combinations are
represented;15 have complete verified five-seed results and12 are PENDING.
75existing measured seeds are reused; no completed experiment was retrained.

Numeric metric_mean/metric_std fields are five-seed statistics only. PENDING
numeric cells are empty. Display fields explicitly say PENDING. Generic mean and
standard_deviation columns alias test_auc only; primary_metric states this.
Checkpoint/source/configuration references accompany each verified row.

Read missing_experiments.csv and GPU_READINESS.json before considering execution.
Prepared configurations do not grant allocation or budget. GKT-MHA is explicitly
the implemented variant; both GKT implementations are not canonical reproductions.
Single-seed primary evidence is separate and excluded from master aggregates.

No architectural superiority from added item information is established. Native
models/data/splits/old reports remain unchanged; HiTSKT is on hold and untouched.
''')
    print('REPORT generated; negative/inconsistent paired groups:',len(inconsistent))
    for r in inconsistent: print(r)


if __name__ == "__main__": main()
