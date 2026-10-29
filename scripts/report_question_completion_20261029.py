"""Write final validation report and standalone figures; no training or test use."""
from pathlib import Path
from datetime import datetime
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scripts.architecture_pilot import PROJECT, CONTROL, write_json
from scripts.architecture_study import read_json

REPORT = Path("reports/architecture_study/question_completion_20261029")


def main():
    out = PROJECT / REPORT
    config = read_json(PROJECT / CONTROL / "configs/one_layer_question_difficulty.json")
    result = read_json(PROJECT / "pilot-workspace/results/raw" / config["run_id"] / "metrics.json")
    verification = read_json(out / "completion_verification.json")
    continuity = read_json(out / "state_continuity_verification.json")
    assert verification["status"] == "PASS_COMPLETED_CONTINUATION" and continuity["status"] == "PASS"
    grant = read_json(PROJECT / CONTROL / "gpu_authorization_question_20261029.executed.json")
    terminal = read_json(out / "launch_status.json")
    samples = [json.loads(line) for line in (out / "resource_samples.jsonl").read_text().splitlines()]
    before = read_json(out / "before_resume_records/history.json")
    partial = read_json(PROJECT / "reports/architecture_study/question_pilot_20261028/partial_checkpoint_verification_20261028.json")
    resource = {"total_saved_training_seconds": result["training_seconds"],
                "saved_seconds_before_resume": partial["elapsed_seconds_at_last_saved_epoch"],
                "continuation_training_seconds": result["training_seconds"] - partial["elapsed_seconds_at_last_saved_epoch"],
                "continuation_invocation_seconds": (datetime.fromisoformat(terminal["observed_utc"]) - datetime.fromisoformat(grant["valid_from_utc"])).total_seconds(),
                "peak_allocated_bytes_all_epochs": result["peak_allocated_bytes"],
                "peak_reserved_bytes_all_epochs": result["peak_reserved_bytes"],
                "continuation_min_gpu_free_bytes": min(s["free_bytes"] for s in samples),
                "continuation_min_host_headroom_bytes": min(s["host_headroom_bytes"] for s in samples),
                "continuation_peak_rss_bytes": max(s["process_peak_rss_bytes"] for s in samples),
                "continuation_telemetry_samples": len(samples)}
    write_json(out / "resource_summary.json", resource)
    rows = read_json(out / "comparison.json")["rows"]
    baseline = rows[0]; delta = result["validation"]["auc"] - baseline["validation_auc"]
    labels = ["Original one-layer linear", "Two-layer linear", "Causal softmax", "Recency softmax", "Gated linear", "Question difficulty"]
    table = "\n".join(f"| {name} | {r['validation_auc']:.12f} | {r['validation_f1']:.12f} | {r['validation_accuracy']:.12f} | {r['validation_bce']:.12f} | {r['best_epoch']} | {r['epochs_trained']} |" for name, r in zip(labels, rows))
    text = f'''# Completed question-difficulty pilot — 29 October 2026

**The improvement survived leakage checks, exact continuation and independent
CPU validation replay for the frozen OFFLINE STUDENT-DISJOINT benchmark.**
Final validation AUC{result['validation']['auc']:.12f},delta+{delta:.12f}
(+{delta * 100:.6f}percentage points) over original0.7508035371114405.
One seed and reused validation;not independent training replication or proof of
forward-time online performance or attention-kernel superiority.

| Architecture/input | Validation AUC | F1 | Accuracy | BCE | Best epoch | Total epochs |
|---|---:|---:|---:|---:|---:|---:|
{table}

## Leakage audit and prediction-time boundary

All numerical frozen features exactly match an independent rebuild. Flipping
all real validation labels changes no feature value. Flipping the longest
training student's entire response history changes none of that student's
features. Positive controls show other training students' labels do influence
fitted difficulty,as intended. Full-builder synthetic tests show validation/test
labels and raw correctness do not influence numerical features;test students
are absent from feature files. Whole-file provenance hashes change if input
bytes change,which is not a predictor-value dependency. No real test evaluation,
test feature construction or actual test-label perturbation was performed.

Difficulty is a fitted training-corpus statistic,not a within-student running
estimate. For training rows,ALL own-student outcomes are subtracted from both
question and skill rates,including the current target and future own responses.
For validation rows,all rate estimates use training students only. Training and
validation students are disjoint. Question ID is known metadata,joined one-to-one
by existing student/order keys;raw correctness is not used in that join. Unknown
questions fall back to the skill prior. Original causal input indexing remains
q/r through t predicting response at t+1;observed validation history may be used
just as in original KT evaluation,but future validation outcomes cannot enter
the attention or fitted difficulty feature.

**Temporal limitation:**95.045% of validation attempts have a later-order training
observation for the same problem. order_id is an ordering proxy,not an audited
wall-clock timestamp. Thus features use globally fitted full-training statistics
that need not have existed before each historical event. This is permissible
under the existing retrospective student-disjoint benchmark,where model fitting
also uses the complete training partition. It is NOT a causal forward-time
deployment claim. A chronological evaluation would require a separately frozen
time-aware training/statistics design;we did not silently alter the protocol.

Validation was reused for diagnostic feature choice and early stopping. This
selection exposure remains even though no validation label enters fitted rates.
Fixed-seed replication and feature-matched controls are needed. Input is richer
than the original skill-only model;training-derived per-problem statistics add
information/capacity despite only one additional neural scale. Same split and
evaluation procedure do not mean identical input information.

## Exact continuation verification

Initial execution b24e154a stopped after18saved epochs;resumed from latest.pt
under continuation commitd958cff4. Same run ID/config,not a new initialization.
The first18history entries remain exactly unchanged. Continued epochs19..29,
selected24,ended with bad_epochs5 under original patience5 (max50).

All22 Adam states advanced from step1134 to1827 (63batches/epoch,11new epochs).
The restored loader's full permutation stream produces the final saved loader
RNG exactly;epoch19order SHA256
{continuity['epoch19_index_order_sha256']}.
CPU global RNG matches the expected validation-loader draws;Python/NumPy states
are preserved. CUDA state was present and restored by unchanged engine code;
no extra CUDA audit updates were performed. Source/model/adapter/launcher/observer
identities match the original execution. Hyperparameters,data,feature file,
split,precision and early-stop/checkpoint rules were not changed.

Independent CPU selected-checkpoint replay AUC difference
{verification['absolute_differences']['auc']:.12g};F1/accuracy/precision/recall match
exactly. Full finite model/Adam/RNG/loader state and config/data/feature/grant
bindings verified. All240 protected old files unchanged,including the original
interruption archive and28October reports. Five leakage/pipeline audit CPU tests
and two feature-matched SAKT preparation tests pass. No original scientific
source was edited. Verification performs zero optimizer updates.

## Resource use and artifacts

29epoch cumulative saved training time{resource['total_saved_training_seconds']:.3f}s;
continuation training{resource['continuation_training_seconds']:.3f}s;
new invocation{resource['continuation_invocation_seconds']:.3f}s,below900s.
Peak allocated{result['peak_allocated_bytes']/1024**2:.3f}MiB,reserved
{result['peak_reserved_bytes']/1024**2:.3f}MiB,below4GiB allocator cap.
Minimum continuation host headroom{resource['continuation_min_host_headroom_bytes']/1024**3:.3f}GiB;
GPU free{resource['continuation_min_gpu_free_bytes']/1024**3:.3f}GiB. Shared peer
workload untouched;general holds,pacing and all resource floors retained. New
grant closed_completed. Prior interrupted window remains recorded,not erased.
Cumulative saved training time excludes discarded unsaved work/queue gaps;
shared-device timings are not clean model-throughput comparisons.

Best/latest checkpoint hashes and selected alpha={verification['learned_difficulty_scale']:.9f}
are in completion_verification.json. Run ID:{config['run_id']}.
Checkpoint directory:pilot-workspace/checkpoints/<run ID>/.
Old epoch18 binary archive remains intact;new completed binary/feature/raw-record
archive is indexed by recovery_archive_20261029.json. Feature sidecar is required
for replay and remains local/ignored,not in memory or Git. Both old and new
executed grants are retained. CSV one_layer_question_difficulty_epochs.csv
contains every epoch's validation AUC/F1/accuracy/BCE/precision/recall,online
training metrics,runtime and GPU/RSS peaks. PNG/PDF curves are standalone exports.

CPU replay:CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m scripts.verify_question_completion_20261029
State continuity:python -m scripts.verify_question_state_continuity

## Replication packet — prepared,not launched

experiments/question_replication_20261029/plan.json freezes five seeds
42,123,2026,3407,7777 for LightSAKTQuestion and a feature-matched SAKT control.
Reuse this completed LightSAKT seed42;only four new LightSAKT runs plus five SAKT
controls remain. Both receive exactly the same frozen feature values and one
zero-initialized scale. SAKT control preserves native SAKT forward/attention via
a scoped logit hook;CPU init/dropout/gradient/exact-resume tests pass. Each model
keeps its previously selected settings;feature-matched is not hyperparameter-
identical (SAKT lr0.0005,4heads,dropout0.2;LightSAKT lr0.001,2heads,dropout0.1).
This control can distinguish general information gain from model-family effects.

Proposed SEPARATE budget ceiling:9sequential new invocations,each<=900seconds
and4GiB allocator,total<=8100GPU-seconds (135minutes);same host/GPU safeguards,
no automatic extra windows/retries. This is a conservative ceiling,not a runtime
estimate or approval. Requires explicit campaign budget and a budget-enforcing
seed-general launcher;current pilot launcher intentionally allows seed42 only.
No real-data SAKT control or additional LightSAKT seed was launched.

Report paired validation deltas,mean and sample SD across all five;never best
seed. Five seeds measure initialization variance,not new student-cohort uncertainty.
Keep final test locked until architecture/input selection is frozen and separately
authorized. Algebra2005 needs verified problem/step mapping;current ASSIST2015
four-column release lacks separate problem IDs,no silent input/schema substitution.

Professor update:The audited question-difficulty correction completed29epochs
and reached validation AUC0.781190 versus0.750804 (+0.030386),with exact state
continuity and CPU replay. It is a real one-seed offline validation gain using
richer known-item information. Five-seed and feature-matched SAKT configurations
are prepared;campaign execution and online generalization are not yet established.
'''
    (out / "FINAL_RESULT_20261029.md").write_text(text)
    curves = pd.read_csv(PROJECT / "reports/diagnostics_20261028/architecture_curves.csv")
    new = pd.read_csv(out / "one_layer_question_difficulty_epochs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name, group in curves.groupby("arm", sort=False):
        axes[0].plot(group.epoch, group.validation_auc, label=name.replace("one_layer_", ""), alpha=.65)
    axes[0].plot(new.epoch, new.validation_auc, label="question difficulty, completed", linewidth=2.5)
    axes[0].axvline(18.5, color="gray", linestyle=":", label="resume boundary")
    axes[0].set(xlabel="Epoch", ylabel="Validation AUC", title="Offline student-disjoint, seed 42")
    axes[0].legend(fontsize=7)
    original = curves[curves.arm == "baseline"]
    axes[1].plot(original.epoch, original.validation_bce, label="original validation")
    axes[1].plot(new.epoch, new.validation_bce, label="question validation")
    axes[1].plot(new.epoch, new.training_bce, label="question training (dropout active)", linestyle="--")
    axes[1].set(xlabel="Epoch", ylabel="BCE", title="Same optimizer and checkpoint-selection rule")
    axes[1].legend(fontsize=8)
    for suffix in ["png", "pdf"]:
        fig.savefig(out / f"completion_curves.{suffix}", dpi=180)
    plt.close(fig)
    print(f"Final report written: AUC delta +{delta:.9f};29epochs complete;campaign unlaunched.")


if __name__ == "__main__":
    main()
