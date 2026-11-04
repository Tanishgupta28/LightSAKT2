"""Execute only the frozen fifteen-run EdNet comparison, with shared guards."""
import os
import json
import time
import signal
from pathlib import Path
from contextlib import nullcontext
from datetime import datetime, timezone
from scripts.ednet_replacement_data import ROOT, WORK, OUT, DS, write
from scripts.architecture_study import digest
from scripts.architecture_pilot import admission
from scripts.pilot_shared_resources import observe, assess, POLICY

SOURCE = ROOT.parent / "LightSAKT2"
PLAN = ROOT / "experiments/ednet_replacement_20261104/plan.json"


def run(plan):
    if (OUT / "campaign_status.json").exists():
        raise RuntimeError("Existing campaign status: audit before any continuation, never duplicate automatically")
    assert plan["status"] == "FROZEN_REPLACEMENT_CAMPAIGN_20261104"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu_uuid"]
    configs = {entry["config"]: entry["config_sha256"] for entry in plan["entries"]}
    def fast_checks():
        for path, expected in {**plan["hold_sha256"], **configs}.items():
            if digest(path) != expected:
                raise RuntimeError("Preserved hold/frozen config changed")
    def scientific_checks():
        for path, expected in plan["scientific_code_sha256"].items():
            if digest(path) != expected:
                raise RuntimeError("Frozen scientific helper changed")
        for path, expected in plan["data_binding_sha256"].items():
            if digest(path) != expected:
                raise RuntimeError("Frozen dataset/feature/split changed")
    fast_checks(); scientific_checks()
    assess(observe(plan["gpu_uuid"]), admission=True)
    for entry in plan["entries"]:
        assert not (WORK / "results/raw" / entry["run_id"]).exists(), "An existing experiment must not be duplicated"
    import torch
    from src.training import engine
    from src.training.pilot_observer import PilotPause
    from scripts.pilot_metric_extensions import MetricsObserver, record_metrics
    from scripts.professor_data_20261030 import training_features
    torch.set_num_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    capacity = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(POLICY["allocator_cap_bytes"] / capacity, 0)
    consumed = 0.
    progress = {"status": "running", "completed": [], "launched": [], "current": None,
                "plan_sha256": digest(PLAN), "test_evaluation_in_training": False, "owned_pid": os.getpid()}
    write(OUT / "campaign_status.json", progress)
    for entry in plan["entries"]:
        fast_checks(); scientific_checks(); assess(observe(plan["gpu_uuid"]), admission=True)
        if len(progress["launched"]) >= 15 or consumed >= 13500:
            raise RuntimeError("Core invocation/window budget exhausted")
        config = json.loads(Path(entry["config"]).read_text())
        assert config["dataset"] == DS and config["evaluate_test"] is False
        assert config["model"] in {"SAKT", "LightSAKT", "LightSAKTQuestion"}
        started = time.monotonic(); deadline = started + min(900, 13500 - consumed)
        last_sample = 0.
        def guard():
            nonlocal last_sample
            if time.monotonic() >= deadline:
                raise PilotPause("Per-run900s/campaign13500s wall-clock limit reached")
            fast_checks()
            if time.monotonic() - last_sample >= 2:
                sample = observe(plan["gpu_uuid"])
                try:
                    assess(sample)
                except RuntimeError as error:
                    raise PilotPause(str(error)) from error
                with (OUT / "resource_samples.jsonl").open("a") as stream:
                    stream.write(json.dumps({"run_id": config["run_id"], **sample}) + "\n")
                last_sample = time.monotonic()
        observer = MetricsObserver(guard, {"mode": "replacement_ednet_20261104", "stage": DS,
            "launcher_sha256": digest(__file__), "memory_cap_bytes": POLICY["allocator_cap_bytes"],
            "inter_update_pause_seconds": .1, "max_step_seconds": 30,
            "resource_policy_sha256": digest(ROOT / "scripts/pilot_shared_resources.py"),
            "plan_sha256": digest(PLAN), "owned_worker_pid": os.getpid(), "run_id": config["run_id"],
            "authorization": "Current user replacement-dataset five-seed three-model campaign", "shared_gpu": True})
        progress["current"] = config["run_id"]; progress["launched"].append(config["run_id"])
        write(OUT / "campaign_status.json", progress)
        def alarm(signum, frame):
            raise PilotPause("Per-run bounded execution alarm")
        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, max(1, deadline - time.monotonic()))
        try:
            with record_metrics(engine, observer):
                context = training_features(engine, WORK, config) if config["model"] == "LightSAKTQuestion" else nullcontext()
                with context:
                    result = engine.run(config, root=WORK, observer=observer)
            progress["completed"].append({"run_id": config["run_id"], "model": config["model"], "seed": config["seed"],
                "validation_auc": result["validation"]["auc"], "epochs": result["epochs_trained"],
                "best_epoch": result["best_epoch"], "checkpoint_sha256": result["checkpoint_sha256"]})
            print(f"CORE_COMPLETE {config['run_id']} validation_auc={result['validation']['auc']:.9f}", flush=True)
        except BaseException as error:
            progress.update(status="STOPPED_NO_AUTOMATIC_RETRY", reason=f"{type(error).__name__}: {error}")
            write(OUT / "campaign_status.json", progress)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            consumed += time.monotonic() - started
            progress["admitted_window_seconds"] = consumed
            write(OUT / "campaign_status.json", progress)
            torch.cuda.empty_cache()
    progress.update(status="completed", current=None, completed_utc=datetime.now(timezone.utc).isoformat())
    write(OUT / "campaign_status.json", progress)


def main():
    plan = json.loads(PLAN.read_text())
    # Lock the LightSAKT admission BEFORE any CUDA context. No peer job is signaled.
    with admission(SOURCE, WORK):
        run(plan)


if __name__ == "__main__":
    main()
