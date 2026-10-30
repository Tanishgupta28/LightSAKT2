"""Freeze all scientific settings before the newly authorized test release."""
import copy
import json
from pathlib import Path
from scripts.architecture_study import digest, read_json
from scripts.architecture_pilot import PROJECT, SOURCE, CONTROL, write_json
from scripts.pilot_shared_resources import POLICY, observe, assess

OUT = PROJECT / "reports/professor_results_20261030"
PLAN = PROJECT / "experiments/professor_campaign_20261030"


def main():
    PLAN.mkdir(exist_ok=False)
    specs = read_json(OUT / "dataset_feature_audit.json")
    entries = []
    for ds in ["ASSIST2009", "Algebra2005"]:
        stats = read_json(SOURCE / f"reports/{ds}_statistics.json")
        assert digest(SOURCE / stats["raw_path"]) == stats["raw_sha256"]
        selected = read_json(SOURCE / f"experiments/selections/{ds}-LightSAKT.json")["selected_run_id"]
        original = read_json(SOURCE / "configs/runs" / f"{selected}.json")
        for seed in [42, 123, 2026, 3407, 7777]:
            config = copy.deepcopy(original)
            config.update(model="LightSAKTQuestion", group="question_replication", seed=seed, device="cuda", evaluate_test=False,
                          run_id=f"{ds}-LightSAKTQuestion-replication-{seed}-v1")
            config["feature_path"] = specs[ds]["train_validation_feature_path"]
            config["question_difficulty"] = {k: specs[ds][k] for k in ["algorithm", "smoothing", "processed_sha256", "split_sha256", "raw_sha256"]}
            config["question_difficulty"]["feature_sha256"] = specs[ds]["train_validation_feature_sha256"]
            config["question_identity"] = specs[ds]["question_identity"]
            work = PROJECT / f".tools/professor_campaign_20261030/workspaces/{ds}"
            reuse = ds == "ASSIST2009" and seed == 42
            if reuse:
                config = read_json(PROJECT / CONTROL / "configs/one_layer_question_difficulty.json")
                work = PROJECT / "pilot-workspace"
            path = PLAN / "configs" / f"{ds}-{seed}.json"
            write_json(path, config)
            entries.append({"dataset": ds, "seed": seed, "model": "LightSAKTQuestion", "run_id": config["run_id"],
                            "config": str(path), "config_sha256": digest(path), "work": str(work),
                            "action": "reuse_completed" if reuse else "authorized_new_run"})
    protected = []
    for ds in ["ASSIST2009", "ASSIST2015", "Algebra2005"]:
        for model in ["LightSAKT", "SAKT"]:
            for seed in [42, 123, 2026, 3407, 7777]:
                directory = SOURCE / f"results/raw/{ds}-{model}-final-{seed}"
                result = read_json(directory / "metrics.json")
                checkpoint = Path(result["checkpoint"])
                if not checkpoint.is_absolute():
                    checkpoint = SOURCE / checkpoint
                assert digest(checkpoint) == result["checkpoint_sha256"]
                protected += [checkpoint] + [p for p in directory.iterdir() if p.is_file()]
        protected += [SOURCE / f"data/processed/{ds}/interactions.csv", SOURCE / f"data/splits/{ds}.json"]
    protected += [p for p in (PROJECT / "pilot-workspace/checkpoints").rglob("*.pt")]
    protected += [p for p in (PROJECT / "pilot-workspace/results/raw").rglob("*") if p.is_file()]
    protected += list((PROJECT / "src").rglob("*.py"))
    for folder in ["question_pilot_20261028", "question_completion_20261029"]:
        protected += [p for p in (PROJECT / "reports/architecture_study" / folder).rglob("*") if p.is_file()]
    holds = {}
    for root in [SOURCE, PROJECT]:
        for name in ["USER_TRAINING_HOLD.json", "PAUSED_BY_USER.json"]:
            path = root / "experiments" / name
            if path.exists():
                holds[str(path)] = digest(path); protected.append(path)
    sample = observe("MIG-d2fcfd79-720e-5218-82c9-0a132be68920"); assessment = assess(sample, admission=True)
    plan = {"status": "FROZEN_CURRENT_USER_PHASE3_SCOPE", "source_request_date": "2026-10-30",
            "authorization": "Current user asks to execute missing multi-seed comparisons where resources and existing shared permissions pass; earlier preparation-only scope superseded for these nine necessary runs.",
            "architecture_frozen": "Existing LightSAKTQuestion unchanged,one scalar on exact same difficulty formula; no new model feature.",
            "test_release": "Current user explicitly requests test metrics for frozen validation-selected checkpoints; no test tuning or post-test changes.",
            "entries": entries, "feature_specs": specs,
            "blocked": {"ASSIST2015-LightSAKTQuestion": "PENDING: independent question identity absent in available four-column source; no substitution or dataset change."},
            "gpu_uuid": sample["gpu_uuid"], "shared_policy": POLICY,
            "maximum_new_runs": 9, "maximum_total_gpu_seconds": 8100,
            "window_seconds_per_run": 900, "concurrency": 1, "automatic_retry": False,
            "allocation": {"type": "existing intentional shared MIG permission", "exclusive_ownership_claim": False,
                           "peer_pid_ownership": "unresolved and not assumed ours;never signal,stop or modify peers", "read_only_admission": sample, "assessment": assessment},
            "hold_sha256": holds, "configuration_sha256": {e["config"]: e["config_sha256"] for e in entries},
            "protected_sha256": {str(p): digest(p) for p in protected},
            "training_adapter_sha256": digest(PROJECT / "scripts/professor_data_20261030.py"),
            "training_launcher_sha256": digest(PROJECT / "scripts/run_professor_campaign_20261030.py")}
    write_json(PLAN / "plan.json", plan)
    print("Frozen nine new runs,one reused modified seed42,thirty existing baselines;ASSIST2015 modified blocked by missing item metadata.")


if __name__ == "__main__":
    main()
