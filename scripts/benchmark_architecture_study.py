"""Bounded synthetic CPU checks/timings, never a real-data training job."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from scripts.architecture_study import ARMS, DATASETS, read_json
from src.evaluation.metrics import masked_bce
from src.models.factory import build_model, predict


def benchmark(root, batch_size=8, steps=3):
    if batch_size < 1 or steps < 1:
        raise ValueError("batch_size and steps must be positive")
    root = Path(root)
    plan = read_json(root / "experiments/architecture_study/plan.json")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    for dataset in DATASETS:
        metadata = plan["datasets"][dataset]
        config = metadata["model_config"]
        for arm in ("original_one_layer_linear", *ARMS):
            torch.manual_seed(42)
            name = "LightSAKT" if arm == "original_one_layer_linear" else "LightSAKTResearch"
            architecture = {} if name == "LightSAKT" else ARMS[arm]
            m = build_model(name, metadata["num_q"], {**config, **architecture}).cpu().train()
            length = config["max_seq_len"]
            g = torch.Generator(device="cpu").manual_seed(123)
            data = [torch.randint(metadata["num_q"], (batch_size, length), generator=g),
                    torch.randint(2, (batch_size, length), generator=g),
                    torch.randint(metadata["num_q"], (batch_size, length), generator=g),
                    torch.randint(2, (batch_size, length), generator=g),
                    torch.ones(batch_size, length, dtype=torch.bool)]
            if batch_size > 1:
                data[-1][-1, length // 2:] = False
            optimizer = torch.optim.Adam(m.parameters(), lr=metadata["learning_rate"])
            seconds = []
            for step in range(steps + 1):
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                p, _ = predict(m, name, data)
                loss = masked_bce(p, data[3], data[-1])
                loss.backward()
                assert torch.isfinite(loss)
                assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in m.parameters())
                optimizer.step()
                if step:
                    seconds.append(time.perf_counter() - started)
            parameters = sum(x.numel() for x in m.parameters())
            # This is parameter/gradient/Adam storage only, not activation or process peak.
            rows.append({"dataset": dataset, "arm": arm, "device": "cpu",
                         "batch_size": batch_size, "seq_len": length, "parameters": parameters,
                         "parameter_gradient_adam_bytes_fp32": parameters * 16,
                         "step_seconds_median": statistics.median(seconds),
                         "step_seconds_samples": seconds, "finite_forward_backward": True})
    return {"python": platform.python_version(), "torch": torch.__version__, "threads": 1,
            "precision": "float32", "data": "synthetic only; no datasets or test records loaded",
            "warmup_steps": 1, "measured_steps": steps,
            "timing_scope": "forward, masked BCE, backward, finite-gradient checks, Adam step",
            "gpu_runtime_inference": "CPU ratios are diagnostic only; no GPU speedup is claimed",
            "rows": rows}


def requirements(root):
    """Scenario estimates from saved GPU baseline timings, not CPU extrapolation."""
    root = Path(root)
    plan = read_json(root / "experiments/architecture_study/plan.json")
    factors = {"two_layer_linear": (1.5, 3.0), "one_layer_softmax": (.5, 2.0),
               "one_layer_recency_softmax": (.5, 2.5)}
    rows = []
    for dataset in DATASETS:
        results = [read_json(root / ref["metrics_path"]) for ref in plan["baseline_references"]
                   if ref["dataset"] == dataset]
        # Seed 42's final result can refer to the selected grid checkpoint.
        epoch_seconds = []
        for result in results:
            history_path = root / "results/raw" / result["run_id"] / "history.json"
            if not history_path.exists():
                physical_run = Path(result["checkpoint"]).parent.name
                history_path = root / "results/raw" / physical_run / "history.json"
            history = read_json(history_path)
            epoch_seconds.extend(row["epoch_seconds"] for row in history)
        per_epoch = statistics.mean(epoch_seconds)
        same_stopping_seconds = sum(r["training_seconds"] for r in results)
        maximum_seconds = per_epoch * 50 * len(results)
        for arm, (lower, upper) in factors.items():
            rows.append({"dataset": dataset, "arm": arm, "runs": len(results),
                         "recorded_baseline_seconds_five_runs": same_stopping_seconds,
                         "recorded_baseline_epoch_seconds_mean": per_epoch,
                         "recorded_baseline_epochs": [r["epochs_trained"] for r in results],
                         "recorded_baseline_peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in results),
                         "assumed_runtime_factor_interval": [lower, upper],
                         "same_stopping_minutes_interval": [same_stopping_seconds * lower / 60,
                                                            same_stopping_seconds * upper / 60],
                         "fifty_epoch_minutes_interval": [maximum_seconds * lower / 60,
                                                          maximum_seconds * upper / 60]})
    totals = {key: [sum(row[key][i] for row in rows) for i in (0, 1)]
              for key in ("same_stopping_minutes_interval", "fifty_epoch_minutes_interval")}
    return {"basis": "Historical selected baseline GPU training/epoch records; serial execution",
            "factors_are_assumptions": True, "gpu_profile_performed": False,
            "excludes": "Startup, data loading, final heldout inference, artifact verification, queue and contention",
            "memory_planning": "Request at least 4 GiB dedicated accelerator memory for a later bounded profile; actual variant peak must be measured before full training.",
            "rows": rows, "all_45_runs_totals": totals}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("reports/architecture_study"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    values = {"training_estimates.json": requirements(args.root),
              "cpu_profile.json": benchmark(args.root, args.batch_size, args.steps)}
    for name, value in values.items():
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print("Synthetic CPU checks complete; no dataset or GPU training started.")
