import builtins
import copy
import fcntl
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import torch

from scripts.architecture_pilot import (PilotBlocked, admission, authorization, check_holds,
                                       execute, prepare, stage_inputs, summarize, write_epochs)
from src.training.engine import run
from src.training.pilot_observer import PilotObserver, PilotPause
from test_training import fixture


def test_both_original_and_worktree_holds_are_enforced_before_torch(tmp_path, monkeypatch):
    source, project, work = [tmp_path / name for name in ("source", "project", "work")]
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        assert name != "torch", "The hold must block execution before torch import/CUDA probing"
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    for root in (source, project):
        marker = root / "experiments/USER_TRAINING_HOLD.json"
        marker.parent.mkdir(parents=True)
        marker.write_text("malformed markers also block")
        with pytest.raises(PilotBlocked, match="hold"):
            execute("depth", source, work, None, project)
        assert not work.exists()
        marker.unlink()


def grant(tmp_path):
    now = datetime.now(timezone.utc)
    record = {"status": "authorized", "project": "LightSAKT2", "dataset": "ASSIST2009", "seed": 42,
              "stages": ["depth", "attention"], "sharing": "shared", "gpu_uuid": "MIG-test-only",
              "valid_from_utc": (now - timedelta(minutes=1)).isoformat(),
              "valid_until_utc": (now + timedelta(minutes=1)).isoformat(),
              "authorized_by": "synthetic test fixture", "allocation_reference": "not a real authorization"}
    path = tmp_path / "grant.json"; path.write_text(json.dumps(record))
    return path, record


@pytest.mark.parametrize("change", ({"status": "unverified"}, {"dataset": "ASSIST2015"},
                                   {"seed": 123}, {"stages": ["attention"]}, {"gpu_uuid": "0"},
                                   {"sharing": "unknown"}, {"authorized_by": ""},
                                   {"valid_until_utc": "2000-01-01T00:00:00Z"},
                                   {"valid_from_utc": "2020-01-01T00:00:00"}))
def test_gpu_grant_rejects_wrong_scope_expiry_or_missing_timezone(tmp_path, change):
    path, record = grant(tmp_path)
    record.update(change); path.write_text(json.dumps(record))
    with pytest.raises(PilotBlocked):
        authorization(path, "depth")


def test_valid_scoped_shared_grant_is_read_without_cuda(tmp_path):
    path, record = grant(tmp_path)
    assert authorization(path, "depth") == record
    with pytest.raises(PilotBlocked):
        authorization(None, "depth")


def test_original_admission_lock_does_not_wait_or_displace_another_worker(tmp_path):
    source = tmp_path / "source"; (source / "experiments").mkdir(parents=True)
    with (source / "experiments/gpu_admission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(PilotBlocked, match="Another"):
            with admission(source, tmp_path / "work"):
                pytest.fail("Must not enter while another worker owns admission")


def test_pilot_writes_only_isolated_input_links(tmp_path):
    source, work = tmp_path / "source", tmp_path / "work"
    for relative in ("data/processed/ASSIST2009/interactions.csv", "data/splits/ASSIST2009.json",
                     "reports/ASSIST2009_statistics.json"):
        path = source / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("unchanged")
    stage_inputs(source, work); stage_inputs(source, work)
    for path in work.rglob("*"):
        if path.is_file():
            assert path.is_symlink() and path.read_text() == "unchanged"
    with pytest.raises(PilotBlocked, match="outside"):
        stage_inputs(source, source / "bad")


@pytest.mark.parametrize("model,layers", (("LightSAKT", None), ("LightSAKTResearch", 2)))
def test_observer_preserves_training_updates_metrics_and_exact_resume(tmp_path, model, layers):
    roots = [tmp_path / name for name in ("plain", "observed", "resumed")]
    configs = [fixture(root) for root in roots]
    for config in configs:
        config["model"] = model
        if layers:
            config["model_config"].update(num_layers=layers, attention_type="linear")
    plain = run(configs[0], roots[0])
    observed = run(configs[1], roots[1], observer=PilotObserver())
    run(configs[2], roots[2], stop_after_epoch=1, observer=PilotObserver(
        provenance={"mode": "cpu_synthetic_test", "gpu_authorization_sha256": "old lease"}))
    resumed = run(configs[2], roots[2], observer=PilotObserver(
        provenance={"mode": "cpu_synthetic_test", "gpu_authorization_sha256": "renewed lease"}))
    assert plain["validation"] == observed["validation"] == resumed["validation"]
    states = [torch.load(root / "checkpoints/toy/latest.pt", weights_only=False) for root in roots]
    for key, value in states[0]["model"].items():
        for state in states[1:]:
            torch.testing.assert_close(value, state["model"][key], atol=0, rtol=0)
    for row in states[1]["history"]:
        assert row["resources"]["process_peak_rss_bytes"] > 0
        assert row["resources"]["peak_allocated_bytes"] is None
    with pytest.raises(ValueError, match="observer"):
        config = copy.deepcopy(configs[2]); config["max_epochs"] = 3
        # Completed runs correctly return existing metrics; use the interrupted checkpoint.
        status_path = roots[2] / "results/raw/toy/status.json"
        status_path.write_text(json.dumps({"status": "interrupted"}))
        run(config, roots[2], observer=PilotObserver(provenance={"mode": "changed"}))


def test_cooperative_pause_retains_last_complete_epoch(tmp_path):
    config = fixture(tmp_path)
    class PausingObserver(PilotObserver):
        def start_epoch(self, epoch, device):
            if epoch == 2:
                raise PilotPause("Synthetic resource/hold stop")
            super().start_epoch(epoch, device)
    with pytest.raises(PilotPause):
        run(config, tmp_path, observer=PausingObserver())
    state = torch.load(tmp_path / "checkpoints/toy/latest.pt", weights_only=False)
    assert state["epoch"] == 1 and "optimizer" in state and "loader_rng" in state and "rng" in state
    status = json.loads((tmp_path / "results/raw/toy/status.json").read_text())
    assert status["status"] == "interrupted"
    assert "Synthetic" in status["failure_reason"]


def test_pilot_prepares_one_dataset_seed_and_equal_budgets_without_submitting(tmp_path):
    project = tmp_path / "project"
    (project / "experiments/architecture_study").mkdir(parents=True)
    original = Path("experiments/architecture_study/plan.json")
    (project / original).write_bytes(original.read_bytes())
    record = prepare("/workspace/LightSAKT2", project)
    assert record["maximum_sequential_runs_per_stage"] == 2
    assert record["launches_jobs"] is False and record["test_evaluation"] is False
    assert len(record["configurations"]) == 4
    for config in record["configurations"].values():
        assert config["dataset"] == "ASSIST2009" and config["seed"] == 42
        assert config["max_epochs"] == 50 and config["patience"] == 5
        assert config["learning_rate"] == .001 and config["batch_size"] == 64
        assert config["evaluate_test"] is False


def test_epoch_export_labels_historical_and_new_memory_scope(tmp_path):
    path = tmp_path / "epochs.csv"
    rows = [{"epoch": 1, "train_objective": .6, "validation": {"auc": .7, "f1": .8, "bce": .5},
             "epoch_seconds": 1.2}]
    write_epochs(rows, path, measured=False)
    assert "historical_baseline" in path.read_text()
    assert "validation_auc" in path.read_text() and "training_bce" in path.read_text()


def test_observed_run_peak_includes_earlier_epoch_peaks(tmp_path):
    class SyntheticPeakObserver(PilotObserver):
        def end_epoch(self, epoch, device):
            record = super().end_epoch(epoch, device)
            record.update(peak_allocated_bytes=100 if epoch == 1 else 50,
                          peak_reserved_bytes=200 if epoch == 1 else 75)
            return record
    result = run(fixture(tmp_path), tmp_path, observer=SyntheticPeakObserver())
    assert result["peak_allocated_bytes"] == 100 and result["peak_reserved_bytes"] == 200


@pytest.mark.parametrize("relocated", [False, True])
def test_comparison_uses_fresh_validation_pair_and_requires_baseline_reproduction(tmp_path, relocated):
    import hashlib
    configurations = {}
    for arm, auc in (("baseline", .75), ("two_layer_linear", .752)):
        config = {"run_id": arm, "dataset": "ASSIST2009", "seed": 42, "model": "LightSAKTResearch",
                  "pilot": {"processed_sha256": "data", "split_sha256": "split"}}
        configurations[arm] = config
        directory = tmp_path / "work/results/raw" / arm; directory.mkdir(parents=True)
        checkpoint = directory / "synthetic_checkpoint.pt"; checkpoint.write_bytes(b"synthetic unit test only")
        result = {**{key: config[key] for key in ("run_id", "dataset", "seed", "model")},
                  "validation": {"auc": auc, "f1": .8, "bce": .5}, "test": None,
                  "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  "processed_sha256": "data", "split_sha256": "split", "best_epoch": 1,
                  "epochs_trained": 1, "training_seconds": 1, "peak_allocated_bytes": 100,
                  "peak_reserved_bytes": 200}
        for name, value in (("config.json", config), ("status.json", {"status": "completed"}), ("metrics.json", result)):
            (directory / name).write_text(json.dumps(value))
        if relocated:
            restored = tmp_path / "work/checkpoints" / arm / "best.pt"
            restored.parent.mkdir(parents=True)
            checkpoint.rename(restored)
    record = {"configurations": configurations, "saved_baseline_validation_auc": .75,
              "baseline_retraining_tolerance": .001, "memory_limit_bytes": 1000}
    comparison = summarize(tmp_path / "work", record, tmp_path / "project")
    assert comparison["rows"][1]["validation_auc_delta"] == pytest.approx(.002)
    assert comparison["rows"][1]["screening_signal"] is True
    if relocated:
        restored.write_bytes(b"wrong restored bytes")
        with pytest.raises(PilotBlocked, match="checkpoint hash"):
            summarize(tmp_path / "work", record, tmp_path / "project")
        restored.write_bytes(b"synthetic unit test only")
    record["saved_baseline_validation_auc"] = .8
    assert summarize(tmp_path / "work", record, tmp_path / "project")["rows"][1]["screening_signal"] is False
    result["test"] = {"auc": .99}; (directory / "metrics.json").write_text(json.dumps(result))
    with pytest.raises(PilotBlocked, match="test"):
        summarize(tmp_path / "work", record, tmp_path / "project")


def test_scoped_hold_release_cannot_enable_other_stages_or_modified_configs(tmp_path):
    from scripts.architecture_pilot import CONTROL, digest
    from scripts.user_training_hold import require_training_enabled
    source, project = tmp_path / "source", tmp_path / "project"
    markers = []
    for root in (source, project):
        marker = root / "experiments/USER_TRAINING_HOLD.json"
        marker.parent.mkdir(parents=True)
        marker.write_text('{"status":"USER_TRAINING_HOLD_ACTIVE"}')
        markers.append(marker)
    configs = {}
    for arm in ("baseline", "two_layer_linear"):
        path = project / CONTROL / "configs" / f"{arm}.json"
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text('{"frozen":true}')
        configs[arm] = digest(path)
    _, record = grant(tmp_path)
    record["hold_exception"] = {"stage": "depth", "dataset": "ASSIST2009", "seed": 42,
                                "user_authorization_evidence": "synthetic test fixture only",
                                "hold_marker_sha256": {str(p): digest(p) for p in markers},
                                "configuration_sha256": configs}
    check_holds(source, project, record, "depth")
    with pytest.raises(SystemExit):
        require_training_enabled(markers[0])  # General training remains blocked.
    with pytest.raises(PilotBlocked):
        check_holds(source, project, record, "attention")
    (project / CONTROL / "configs/baseline.json").write_text('{"changed":true}')
    with pytest.raises(PilotBlocked):
        check_holds(source, project, record, "depth")


def test_shared_resources_enforce_pressure_limits_and_log_unsupported_mig():
    from scripts.pilot_shared_resources import GIB, assess
    sample = {"free_bytes": 20 * GIB, "host_headroom_bytes": 10 * GIB,
              "process_peak_rss_bytes": GIB, "utilization_status": "unsupported_mig",
              "utilization_gpu_percent": None}
    assert "fallback" in assess(sample, admission=True)
    with pytest.raises(RuntimeError, match="free-memory"):
        assess({**sample, "free_bytes": 5 * GIB})
    with pytest.raises(RuntimeError, match="Host memory"):
        assess({**sample, "host_headroom_bytes": GIB})
    with pytest.raises(RuntimeError, match="busy"):
        assess({**sample, "utilization_status": "ok", "utilization_gpu_percent": 90}, admission=True)
