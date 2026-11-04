"""Final read-only campaign audit and epoch/resource tables; never trains."""
import json
from pathlib import Path
import pandas as pd
from scripts.ednet_replacement_data import ROOT, WORK, OUT, write
from scripts.architecture_study import digest


def main():
    plan = json.loads((ROOT / "experiments/ednet_replacement_20261104/plan.json").read_text())
    progress = json.loads((OUT / "campaign_status.json").read_text())
    protected = 0
    for mapping in ["protected_sha256", "scientific_code_sha256", "data_binding_sha256", "hold_sha256"]:
        for path, expected in plan[mapping].items():
            assert digest(path) == expected, path
            if mapping == "protected_sha256": protected += 1
    assert len(progress["launched"]) <= 15 and len(set(progress["launched"])) == len(progress["launched"])
    samples = [json.loads(line) for line in (OUT / "resource_samples.jsonl").read_text().splitlines()]
    assert all(s["gpu_uuid"] == plan["gpu_uuid"] for s in samples)
    epochs, completed = [], []
    for entry in plan["entries"]:
        assert digest(entry["config"]) == entry["config_sha256"]
        raw = WORK / "results/raw" / entry["run_id"]
        if not (raw / "metrics.json").exists(): continue
        status = json.loads((raw / "status.json").read_text())
        if status["status"] != "completed": continue
        m = json.loads((raw / "metrics.json").read_text())
        assert m["test"] is None and m["training_seconds"] < 900 and m["peak_allocated_bytes"] <= 4 * 1024**3
        completed.append({"run_id": entry["run_id"], "model": entry["model"], "seed": entry["seed"],
            "training_seconds": m["training_seconds"], "peak_allocated_bytes": m["peak_allocated_bytes"],
            "peak_reserved_bytes": m["peak_reserved_bytes"], "best_epoch": m["best_epoch"], "epochs": m["epochs_trained"]})
        history = json.loads((raw / "history.json").read_text())
        for h in history:
            row = {"run_id": entry["run_id"], "model": entry["model"], "seed": entry["seed"], "epoch": h["epoch"],
                "epoch_seconds": h["epoch_seconds"], "online_training_bce_objective": h["train_objective"]}
            row.update({"validation_" + k: v for k, v in h["validation"].items() if k != "auc_reason"})
            r = h.get("resources", {})
            row.update({k: v for k, v in r.items() if isinstance(v, (int, float))})
            row.update({"online_training_" + k: v for k, v in r.get("online_training_metrics", {}).items() if k != "auc_reason"})
            epochs.append(row)
    breaches = [i for i, s in enumerate(samples) if s["free_bytes"] < 6 * 1024**3 or
        s["host_headroom_bytes"] < 2 * 1024**3 or s["process_peak_rss_bytes"] > 4 * 1024**3]
    consumed = progress.get("admitted_window_seconds", 0)
    assert consumed <= 13500
    if progress["status"] == "completed": assert len(completed) == 15 and not breaches
    summary = {"campaign_status": progress["status"], "gpu_uuid": plan["gpu_uuid"], "completed_runs": completed,
        "gpu_window_wall_seconds": consumed, "new_training_seconds_sum": sum(r["training_seconds"] for r in completed),
        "resource_samples": len(samples), "minimum_shared_gpu_free_bytes": min(s["free_bytes"] for s in samples),
        "minimum_host_headroom_bytes": min(s["host_headroom_bytes"] for s in samples),
        "maximum_worker_rss_bytes": max(s["process_peak_rss_bytes"] for s in samples),
        "maximum_allocator_peak_bytes": max((r["peak_allocated_bytes"] for r in completed), default=None),
        "breach_sample_indices": breaches, "allocator_cap_bytes": 4 * 1024**3, "per_invocation_limit_seconds": 900,
        "shared_not_exclusive": True, "maximum_total_window_seconds": 13500,
        "timing_scope": "Admitted window wall time includes pacing; not exclusive time or kernel-active profiling",
        "unsupported_MIG_utilization": "Existing memory/CPU telemetry fallback; peer ownership unresolved and no signals sent"}
    write(OUT / "resource_summary.json", summary)
    pd.DataFrame(epochs).to_csv(OUT / "epoch_results.csv", index=False)
    write(OUT / "protected_artifacts_audit.json", {"status": "PASS", "unchanged_protected_files": protected,
        "original_models_data_checkpoints_and_reports_preserved": True, "frozen_science_data_configs_and_holds": True})
    print(f"AUDIT PASS {len(completed)}/15 completed; {len(epochs)} epochs; {protected} protected files unchanged")


if __name__ == "__main__":
    main()
