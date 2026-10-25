"""One-dataset, seed-42 architecture pilot; fail closed before CUDA on a hold.

prepare/reference are CPU-only. run is explicitly gated and launches at most
two sequential runs in one stage. No scheduler, recovery or test hook is called.
"""
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from scripts.architecture_study import ARMS, digest, read_json


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = Path("/workspace/LightSAKT2")
PILOT = Path("reports/architecture_study/pilot")
CONTROL = Path("experiments/architecture_study/pilot")
DATASET = "ASSIST2009"
SEED = 42
STAGES = {"depth": ("baseline", "two_layer_linear"),
          "attention": ("one_layer_softmax", "one_layer_recency_softmax")}


class PilotBlocked(RuntimeError):
    pass


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_holds(source=SOURCE, project=PROJECT, grant=None, stage="depth"):
    exception = (grant or {}).get("hold_exception", {})
    for root in dict.fromkeys((Path(source).resolve(), Path(project).resolve())):
        marker = root / "experiments/USER_TRAINING_HOLD.json"
        if marker.exists():
            permitted = False
            try:
                permitted = (stage == "depth" and exception["stage"] == stage
                             and exception["dataset"] == DATASET and exception["seed"] == SEED
                             and exception["user_authorization_evidence"]
                             and exception["hold_marker_sha256"][str(marker)] == digest(marker))
                for arm in STAGES[stage]:
                    config_path = Path(project) / CONTROL / "configs" / f"{arm}.json"
                    permitted = permitted and exception["configuration_sha256"][arm] == digest(config_path)
                end = datetime.fromisoformat(grant["valid_until_utc"].replace("Z", "+00:00"))
                permitted = permitted and datetime.now(timezone.utc) < end
            except (KeyError, TypeError, ValueError, OSError):
                permitted = False
            if not permitted:
                raise PilotBlocked(f"Active training hold: {marker}; no matching bounded exception")


def validate_inputs(source, project=PROJECT):
    source, project = Path(source), Path(project)
    plan = read_json(project / "experiments/architecture_study/plan.json")
    metadata = plan["datasets"][DATASET]
    stats = read_json(source / "reports" / f"{DATASET}_statistics.json")
    for key, relative in (("processed_sha256", f"data/processed/{DATASET}/interactions.csv"),
                          ("split_sha256", f"data/splits/{DATASET}.json")):
        if digest(source / relative) != metadata[key] or stats[key] != metadata[key]:
            raise PilotBlocked(f"Dataset/split hash mismatch: {relative}")
    if digest(source / "configs/protocol.json") != plan["protocol_sha256"]:
        raise PilotBlocked("Protocol hash changed")
    selected = metadata["selected_grid_run"]
    config = read_json(source / "configs/runs" / f"{selected}.json")
    if digest(source / "configs/runs" / f"{selected}.json") != metadata["selected_config_sha256"]:
        raise PilotBlocked("Selected configuration changed")
    assignments = read_json(source / "data/splits" / f"{DATASET}.json")
    sets = [set(assignments[k]) for k in ("train", "validation", "test")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise PilotBlocked("Student overlap")
    ref = next(r for r in plan["baseline_references"] if r["dataset"] == DATASET and r["seed"] == SEED)
    if digest(source / ref["checkpoint"]) != ref["checkpoint_sha256"]:
        raise PilotBlocked("Baseline checkpoint hash changed")
    return plan, metadata, config, ref


def prepare(source=SOURCE, project=PROJECT):
    source, project = Path(source), Path(project)
    plan, metadata, selected, ref = validate_inputs(source, project)
    configurations = {}
    for arm in ("baseline", *ARMS):
        config = copy.deepcopy(selected)
        config.update(seed=SEED, group="architecture_pilot", device="cuda", evaluate_test=False,
                      run_id=f"{DATASET}-LightSAKT-pilot-{arm}-{SEED}-v1")
        if arm != "baseline":
            config["model"] = "LightSAKTResearch"
            config["model_config"].update(ARMS[arm])
        config["pilot"] = {"arm": arm, "processed_sha256": metadata["processed_sha256"],
                           "split_sha256": metadata["split_sha256"], "baseline_reference": ref["run_id"]}
        path = project / CONTROL / "configs" / f"{arm}.json"
        if path.exists() and read_json(path) != config:
            raise PilotBlocked(f"Refusing to change an existing pilot configuration: {path}")
        write_json(path, config)
        configurations[arm] = config
    record = {"dataset": DATASET, "seed": SEED, "stages": STAGES,
              "choice": "Smallest selected width (64), L=100, shortest recorded baseline runtime; depth can be screened with low resource use.",
              "configurations": configurations, "selection_metric": "validation_auc",
              "max_epochs_each": selected["max_epochs"], "patience": selected["patience"],
              "wall_time_limit_seconds_per_invocation": 900, "memory_limit_bytes": 4 * 1024 ** 3,
              "minimum_free_bytes": 1024 ** 3, "maximum_sequential_runs_per_stage": 2,
              "launches_jobs": False, "test_evaluation": False,
              "source_root": str(source.resolve()), "study_plan_sha256": digest(project / "experiments/architecture_study/plan.json"),
              "baseline_checkpoint": ref["checkpoint"], "baseline_checkpoint_sha256": ref["checkpoint_sha256"],
              "saved_baseline_validation_auc": ref["validation_auc"],
              "baseline_retraining_tolerance": .001,
              "success_rule": "Screening signal: validation-AUC delta >= 0.001 over the fresh same-seed baseline, matching baseline replay and all correctness/resource gates; not statistical significance.",
              "promotion": "Evaluate qualifying arms on all five original seeds and all three datasets; negative one-seed results remain preliminary."}
    write_json(project / CONTROL / "pilot.json", record)
    return record


def authorization(path, stage, now=None):
    if path is None or not Path(path).is_file():
        raise PilotBlocked("No documented GPU execution authorization supplied")
    try:
        record = read_json(path)
    except (OSError, ValueError) as error:
        raise PilotBlocked("Cannot read a valid GPU execution authorization") from error
    now = datetime.now(timezone.utc) if now is None else now
    try:
        start = datetime.fromisoformat(record["valid_from_utc"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(record["valid_until_utc"].replace("Z", "+00:00"))
        valid = (record["status"] == "authorized" and record["project"] == "LightSAKT2"
                 and record["dataset"] == DATASET and record["seed"] == SEED
                 and stage in record["stages"] and record["sharing"] in ("shared", "dedicated")
                 and record["gpu_uuid"].startswith(("GPU-", "MIG-"))
                 and record["allocation_reference"] and record["authorized_by"]
                 and start.tzinfo is not None and end.tzinfo is not None and start <= now < end)
    except (KeyError, ValueError, TypeError, AttributeError):
        valid = False
    if not valid:
        raise PilotBlocked("GPU authorization is absent, expired, or outside this pilot's scope")
    return record


@contextmanager
def admission(source, work):
    # Use the original project's lock so the isolated checkout cannot evade it.
    lock_path = Path(source) / "experiments/gpu_admission.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PilotBlocked("Another LightSAKT2 job holds GPU admission; no worker was started")
        Path(work).mkdir(parents=True, exist_ok=True)
        with (Path(work) / "pilot.lock").open("a") as local:
            try:
                fcntl.flock(local, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise PilotBlocked("This pilot already has an active worker")
            try:
                yield
            finally:
                fcntl.flock(local, fcntl.LOCK_UN)
                fcntl.flock(lock, fcntl.LOCK_UN)


def stage_inputs(source, work):
    source, work = Path(source).resolve(), Path(work).resolve()
    if work == source or source in work.parents:
        raise PilotBlocked("Pilot output must be outside the original checkout")
    for relative in (f"data/processed/{DATASET}/interactions.csv", f"data/splits/{DATASET}.json",
                     f"reports/{DATASET}_statistics.json"):
        destination = work / relative; destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if not destination.is_symlink() or destination.resolve() != source / relative:
                raise PilotBlocked(f"Unexpected staged input: {destination}")
        else:
            destination.symlink_to(source / relative)


def write_epochs(history, destination, measured=True):
    rows = []
    for row in history:
        resources = row.get("resources", {})
        rows.append({"epoch": row["epoch"], "validation_auc": row["validation"]["auc"],
                     "validation_accuracy": row["validation"].get("accuracy"),
                     "validation_f1": row["validation"]["f1"], "validation_bce": row["validation"]["bce"],
                     "training_bce": row["train_objective"], "epoch_seconds": row["epoch_seconds"],
                     "peak_allocated_bytes": resources.get("peak_allocated_bytes"),
                     "peak_reserved_bytes": resources.get("peak_reserved_bytes"),
                     "process_peak_rss_bytes": resources.get("process_peak_rss_bytes"),
                     "evidence": "new_pilot" if measured else "historical_baseline"})
    if rows:
        destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)


def reference(source=SOURCE, project=PROJECT):
    # CPU replay is allowed while training is held. Hide CUDA before torch import.
    if "torch" in __import__("sys").modules:
        raise PilotBlocked("Run CPU reference in a fresh process with CUDA hidden")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from torch.utils.data import DataLoader
    from src.datasets.sequences import Sequences
    from src.models.factory import build_model
    from src.training.engine import evaluate
    source, project = Path(source), Path(project)
    _, metadata, selected, ref = validate_inputs(source, project)
    torch.set_num_threads(1)
    started = time.perf_counter()
    checkpoint = torch.load(source / ref["checkpoint"], map_location="cpu", weights_only=False)
    if (checkpoint["processed_sha256"] != metadata["processed_sha256"]
            or checkpoint["split_sha256"] != metadata["split_sha256"]
            or checkpoint["config"]["model_config"] != selected["model_config"]):
        raise PilotBlocked("Checkpoint metadata mismatch")
    model = build_model("LightSAKT", metadata["num_q"], selected["model_config"]).cpu()
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset = Sequences(source / f"data/processed/{DATASET}/interactions.csv", "validation", selected["seq_len"])
    validation, _, _ = evaluate(model, "LightSAKT", DataLoader(dataset, batch_size=selected["batch_size"]), "cpu")
    difference = abs(validation["auc"] - ref["validation_auc"])
    import resource
    result = {"dataset": DATASET, "seed": SEED, "device": "cpu", "inference_only": True,
              "split": "validation", "checkpoint": ref["checkpoint"],
              "checkpoint_sha256": digest(source / ref["checkpoint"]), "checkpoint_epoch": checkpoint["epoch"],
              "validation": validation, "saved_validation_auc": ref["validation_auc"],
              "auc_absolute_difference": difference, "auc_tolerance": 2e-6,
              "matches_saved_validation_auc": difference <= 2e-6,
              "elapsed_seconds": time.perf_counter() - started,
              "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "test_loader_constructed": False, "test_predictions_accessed": False,
              "processed_sha256": metadata["processed_sha256"], "split_sha256": metadata["split_sha256"]}
    result.update(torch_version=torch.__version__, python_version=__import__("platform").python_version(),
                  model_source_sha256=digest(project / "src/models/light_sakt.py"),
                  attention_source_sha256=digest(project / "src/attention/linear.py"))
    write_json(project / PILOT / "baseline_validation_replay.json", result)
    if difference > 2e-6:
        raise PilotBlocked("Saved baseline validation AUC did not replay within tolerance")
    history_run = Path(ref["checkpoint"]).parent.name
    write_epochs(read_json(source / "results/raw" / history_run / "history.json"),
                 project / PILOT / "historical_baseline_epochs.csv", measured=False)
    return result


def summarize(work, record, project=PROJECT):
    rows = []
    for arm, config in record["configurations"].items():
        directory = Path(work) / "results/raw" / config["run_id"]
        row = {"arm": arm, "status": "not_run", "validation_auc": None}
        if (directory / "history.json").exists():
            write_epochs(read_json(directory / "history.json"), Path(project) / PILOT / f"{arm}_epochs.csv")
        if (directory / "status.json").exists():
            row["status"] = read_json(directory / "status.json")["status"]
        if row["status"] == "completed":
            result = read_json(directory / "metrics.json")
            if (read_json(directory / "config.json") != config or result.get("test") is not None
                    or any(result[key] != config[key] for key in ("run_id", "dataset", "seed", "model"))):
                raise PilotBlocked("Pilot result/config mismatch or test data access")
            auc = result["validation"]["auc"]
            if auc is None or not math.isfinite(auc) or not 0 <= auc <= 1:
                raise PilotBlocked("Invalid pilot validation AUC")
            for key in ("processed_sha256", "split_sha256"):
                if result[key] != config["pilot"][key]:
                    raise PilotBlocked("Pilot result data binding mismatch")
            if digest(result["checkpoint"]) != result["checkpoint_sha256"]:
                raise PilotBlocked("Pilot checkpoint hash mismatch")
            row.update(validation_auc=result["validation"]["auc"], validation_f1=result["validation"]["f1"],
                       validation_accuracy=result["validation"].get("accuracy"),
                       validation_bce=result["validation"]["bce"], best_epoch=result["best_epoch"],
                       epochs_trained=result["epochs_trained"], training_seconds=result["training_seconds"],
                       peak_allocated_bytes=result["peak_allocated_bytes"], peak_reserved_bytes=result["peak_reserved_bytes"],
                       checkpoint=result["checkpoint"], checkpoint_sha256=result["checkpoint_sha256"])
        rows.append(row)
    baseline = rows[0]["validation_auc"]
    baseline_reproduced = (abs(baseline - record["saved_baseline_validation_auc"]) <= record["baseline_retraining_tolerance"]
                           if baseline is not None else None)
    for row in rows:
        row["validation_auc_delta"] = row["validation_auc"] - baseline if baseline is not None and row["validation_auc"] is not None else None
        row["screening_signal"] = (row["validation_auc_delta"] >= .001 and baseline_reproduced
                                    and row.get("peak_allocated_bytes") is not None
                                    and row["peak_allocated_bytes"] <= record["memory_limit_bytes"]
                                    ) if row["validation_auc_delta"] is not None else None
    result = {"dataset": DATASET, "seed": SEED, "test_evaluation": False, "rows": rows,
              "baseline_retraining_within_tolerance": baseline_reproduced,
              "interpretation": "One-seed validation screening only; resource/replay checks and five seeds required before a performance claim."}
    write_json(Path(project) / PILOT / "comparison.json", result)
    return result


def execute(stage, source, work, authorization_path, project=PROJECT):
    # Keep every hold/authorization/config check before importing torch or probing CUDA.
    source, project, work = Path(source).resolve(), Path(project).resolve(), Path(work).resolve()
    # The general hold remains active; only a user-authorized depth exception
    # bound to both marker hashes and exact pilot configurations can pass it.
    provisional = read_json(authorization_path) if authorization_path and Path(authorization_path).is_file() else None
    check_holds(source, project, provisional, stage)
    if work == source or source in work.parents:
        raise PilotBlocked("Pilot output must be outside the original checkout")
    grant = authorization(authorization_path, stage)
    authorization_digest = digest(authorization_path)
    record = prepare(source, project)
    if stage == "attention":
        existing = summarize(work, record, project)
        if any(row["status"] != "completed" for row in existing["rows"][:2]):
            raise PilotBlocked("Complete the one-layer versus two-layer stage first")
        if not existing["baseline_retraining_within_tolerance"]:
            raise PilotBlocked("Investigate baseline retraining discrepancy before the attention stage")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != grant["gpu_uuid"]:
        raise PilotBlocked("CUDA_VISIBLE_DEVICES must name exactly the authorized device UUID")
    deadline = time.monotonic() + record["wall_time_limit_seconds_per_invocation"]
    from src.training.pilot_observer import PilotObserver, PilotPause
    from scripts.pilot_shared_resources import POLICY, assess, observe
    with admission(source, work):
        check_holds(source, project, grant, stage)
        stage_inputs(source, work)
        for _ in range(2):
            sample = observe(grant["gpu_uuid"])
            assess(sample, admission=True)
            write_json(project / PILOT / "resource_admission.json", {"sample": sample, "policy": POLICY})
            time.sleep(.2)
        import torch
        from src.training.engine import run
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise PilotBlocked("Exactly one authorized CUDA device must be visible")
        capacity = torch.cuda.get_device_properties(0).total_memory
        cap = record["memory_limit_bytes"]; floor = POLICY["runtime_free_bytes"]
        free, _ = torch.cuda.mem_get_info(0)
        if capacity < cap or free < cap + floor:
            raise PilotBlocked("Insufficient free GPU memory for the 4 GiB cap plus 1 GiB reserve")
        torch.cuda.set_per_process_memory_fraction(cap / capacity, 0)
        last_probe = 0.
        def guard():
            nonlocal last_probe
            try:
                check_holds(source, project, grant, stage)
            except PilotBlocked as error:
                raise PilotPause(str(error)) from error
            now = time.monotonic()
            if now >= deadline:
                raise PilotPause("Pilot wall-time limit reached; only this worker is stopping")
            if now - last_probe >= 2:
                try:
                    authorization(authorization_path, stage)
                    if digest(authorization_path) != authorization_digest:
                        raise PilotBlocked("GPU authorization changed during the pilot")
                except PilotBlocked as error:
                    raise PilotPause(str(error)) from error
                remaining, _ = torch.cuda.mem_get_info(0)
                if remaining < floor:
                    raise PilotPause("Shared GPU free-memory reserve breached")
                try:
                    sample = observe(grant["gpu_uuid"])
                    assess(sample)
                except RuntimeError as error:
                    raise PilotPause(str(error)) from error
                with (project / PILOT / "resource_samples.jsonl").open("a") as stream:
                    stream.write(json.dumps(sample) + "\n")
                last_probe = now
        observer = PilotObserver(guard, {"stage": stage, "gpu_authorization_sha256": authorization_digest,
                                         "launcher_sha256": digest(__file__), "memory_cap_bytes": cap,
                                         "inter_update_pause_seconds": POLICY["inter_update_pause_seconds"],
                                         "max_step_seconds": POLICY["max_step_seconds"],
                                         "resource_policy_sha256": digest(Path(__file__).with_name("pilot_shared_resources.py")),
                                         "shared_gpu": grant["sharing"] == "shared"})
        invoked = False
        try:
            for arm in STAGES[stage]:
                guard()
                invoked = True
                write_json(project / PILOT / "launch_status.json", {
                    "status": "running", "stage": stage, "current_arm": arm,
                    "trainer_invoked": True, "gpu_uuid": grant["gpu_uuid"],
                    "observed_utc": datetime.now(timezone.utc).isoformat()})
                run(record["configurations"][arm], root=work, observer=observer)
                summarize(work, record, project)
                # Reclaim only this process's unused cache between sequential models.
                torch.cuda.empty_cache()
        except BaseException as error:
            write_json(project / PILOT / "launch_status.json", {
                "status": "interrupted", "stage": stage, "trainer_invoked": invoked,
                "reason": f"{type(error).__name__}: {error}", "automatic_retry": False,
                "observed_utc": datetime.now(timezone.utc).isoformat()})
            if isinstance(error, PilotBlocked):
                error.training_started = invoked
            raise
        finally:
            summarize(work, record, project)
    write_json(project / PILOT / "launch_status.json", {
        "status": "completed", "stage": stage, "trainer_invoked": invoked,
        "automatic_retry": False, "observed_utc": datetime.now(timezone.utc).isoformat()})
    return summarize(work, record, project)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "reference", "run"))
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--work-root", type=Path, default=PROJECT / "pilot-workspace")
    parser.add_argument("--stage", choices=tuple(STAGES), default="depth")
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            result = prepare(args.source_root)
        elif args.action == "reference":
            result = reference(args.source_root)
        else:
            result = execute(args.stage, args.source_root, args.work_root, args.authorization)
        if args.action == "prepare":
            print(f"Prepared ASSIST2009 seed 42: depth stage has 2 sequential runs; attention stage has 2. No jobs submitted.")
        else:
            print(json.dumps(result, indent=2))
    except PilotBlocked as error:
        record = {"status": "blocked", "reason": str(error), "stage": args.stage,
                  "training_started": getattr(error, "training_started", False), "automatic_retry": False,
                  "observed_utc": datetime.now(timezone.utc).isoformat()}
        write_json(PROJECT / PILOT / "launch_status.json", record)
        print(json.dumps(record, indent=2))
        raise SystemExit(2)
