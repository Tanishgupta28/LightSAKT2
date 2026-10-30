"""Nine scoped sequential runs,never baseline repeats;halt on any unsafe stop."""
import os
import json
import time
import signal
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from scripts.architecture_study import read_json, digest
from scripts.architecture_pilot import PROJECT, SOURCE, write_json, admission
from scripts.pilot_shared_resources import POLICY, observe, assess

PLAN = PROJECT / "experiments/professor_campaign_20261030/plan.json"
OUT = PROJECT / "reports/professor_results_20261030"


def main():
    plan = read_json(PLAN)
    assert plan["status"] == "FROZEN_CURRENT_USER_PHASE3_SCOPE"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu_uuid"]
    assert digest(PROJECT / "scripts/professor_data_20261030.py") == plan["training_adapter_sha256"]
    assert digest(__file__) == plan["training_launcher_sha256"]
    expected = {e["config"]: e["config_sha256"] for e in plan["entries"]}
    def static_checks():
        for path, h in plan["hold_sha256"].items():
            if digest(path) != h:raise RuntimeError("General hold changed;stop scoped campaign")
        for path, h in expected.items():
            if digest(path) != h:raise RuntimeError("Frozen configuration changed")
    static_checks()
    assess(observe(plan["gpu_uuid"]), admission=True)
    import torch
    from src.training import engine
    from src.training.pilot_observer import PilotPause
    from scripts.pilot_metric_extensions import MetricsObserver, record_metrics
    from scripts.professor_data_20261030 import training_features
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    cap = POLICY["allocator_cap_bytes"]; capacity = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(cap / capacity, 0)
    consumed = 0.; launched = 0
    progress = {"status": "running", "completed": [], "launched": [], "new_test_evaluations_in_training": False}
    write_json(OUT / "campaign_status.json", progress)
    for entry in plan["entries"]:
        if entry["action"] == "reuse_completed":continue
        config = read_json(entry["config"]); work = Path(entry["work"])
        directory = work / "results/raw" / config["run_id"]
        if (directory / "status.json").exists():
            raise RuntimeError("Existing run discovered;do not reset or duplicate without an audit")
        static_checks(); assess(observe(plan["gpu_uuid"]), admission=True)
        assert config["evaluate_test"] is False and config["model"] == "LightSAKTQuestion"
        if launched >= plan["maximum_new_runs"] or consumed >= plan["maximum_total_gpu_seconds"]:
            raise RuntimeError("Campaign resource budget exhausted")
        launched += 1; start = time.monotonic(); deadline = start + min(900, plan["maximum_total_gpu_seconds"] - consumed)
        last_sample = 0.
        def guard():
            nonlocal last_sample
            if time.monotonic() >= deadline:raise PilotPause("15-minute or total campaign budget reached")
            static_checks()
            if time.monotonic() - last_sample >= 2:
                sample = observe(plan["gpu_uuid"])
                try:assess(sample)
                except RuntimeError as e:raise PilotPause(str(e)) from e
                with (OUT / "campaign_resource_samples.jsonl").open("a") as stream:
                    stream.write(json.dumps({"run_id": config["run_id"], **sample}) + "\n")
                last_sample = time.monotonic()
        observer = MetricsObserver(guard, {"mode": "professor_campaign_20261030", "stage": entry["dataset"],
                    "launcher_sha256": digest(__file__), "memory_cap_bytes": cap,
                    "inter_update_pause_seconds": .1, "max_step_seconds": 30,
                    "resource_policy_sha256": digest(PROJECT / "scripts/pilot_shared_resources.py"),
                    "plan_sha256": digest(PLAN), "feature_adapter_sha256": plan["training_adapter_sha256"],
                    "owned_worker_pid": os.getpid(), "owned_worker_run_id": config["run_id"], "shared_gpu": True})
        progress["current"] = config["run_id"]; progress["launched"].append(config["run_id"])
        write_json(OUT / "campaign_status.json", progress)
        def alarm(signum, frame):raise PilotPause("Per-run wall-clock alarm reached")
        signal.signal(signal.SIGALRM, alarm); signal.setitimer(signal.ITIMER_REAL, max(1, deadline - time.monotonic()))
        try:
            with admission(SOURCE, work):
                with record_metrics(engine, observer):
                    with training_features(engine, work, config):
                        result = engine.run(config, root=work, observer=observer)
            progress["completed"].append({"run_id": config["run_id"], "validation_auc": result["validation"]["auc"]})
            print(f"CAMPAIGN_COMPLETE {config['run_id']} auc={result['validation']['auc']:.9f}", flush=True)
        except BaseException as error:
            progress.update(status="STOPPED_NO_AUTOMATIC_RETRY", reason=f"{type(error).__name__}: {error}")
            write_json(OUT / "campaign_status.json", progress)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0); consumed += time.monotonic() - start
            progress["gpu_window_seconds_consumed"] = consumed
            write_json(OUT / "campaign_status.json", progress)
            torch.cuda.empty_cache()
    progress.update(status="completed", current=None, completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(OUT / "campaign_status.json", progress)


if __name__ == "__main__":
    main()
