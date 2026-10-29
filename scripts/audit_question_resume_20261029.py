"""Read-only real-data leakage/state audit before an exact epoch-18 resume."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import hashlib
import json
import random
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.architecture_pilot import PROJECT, SOURCE, CONTROL, prepare_question
from scripts.architecture_study import digest, read_json
from scripts.question_difficulty_inputs import difficulty_features, QuestionDifficultySequences
from scripts.pilot_metric_extensions import MetricsObserver

OUT = PROJECT / "reports/architecture_study/question_completion_20261029"


def main():
    torch.set_num_threads(1); OUT.mkdir(parents=True, exist_ok=True)
    work = PROJECT / "pilot-workspace"
    config = prepare_question(SOURCE, work, PROJECT, persist=False)["configurations"]["one_layer_question_difficulty"]
    directory = work / "results/raw" / config["run_id"]
    assert read_json(directory / "status.json")["status"] == "interrupted"
    assert read_json(directory / "config.json") == config
    old = read_json(PROJECT / "reports/architecture_study/question_pilot_20261028/partial_checkpoint_verification_20261028.json")
    checkpoint_dir = work / "checkpoints" / config["run_id"]
    assert digest(checkpoint_dir / "latest.pt") == old["latest_checkpoint_sha256"]
    assert digest(checkpoint_dir / "best.pt") == old["best_checkpoint_sha256"]
    latest = torch.load(checkpoint_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert latest["epoch"] == latest["best_epoch"] == 18 and latest["bad_epochs"] == 0
    assert latest["config"] == config and config["evaluate_test"] is False
    assert latest["history"] == read_json(directory / "history.json")
    assert all(torch.isfinite(x).all() for x in latest["model"].values())
    for state in latest["optimizer"]["state"].values():
        assert all(not isinstance(x, torch.Tensor) or torch.isfinite(x).all() for x in state.values())
    opt = latest["optimizer"]["param_groups"][0]
    assert opt["lr"] == config["learning_rate"] and tuple(opt["betas"]) == tuple(config["betas"])
    assert opt["eps"] == config["epsilon"] and opt["weight_decay"] == config["weight_decay"]
    code_hash = hashlib.sha256(b"".join(str(p).encode() + p.read_bytes()
        for p in sorted(Path("src").rglob("*.py")) if "visualization" not in p.parts)).hexdigest()
    assert code_hash == latest["environment"]["training_code_sha256"]
    # Ensure the non-src adapter and launcher exactly match their execution revision.
    for name in ["scripts/question_difficulty_inputs.py", "scripts/architecture_pilot.py",
                 "scripts/pilot_metric_extensions.py", "scripts/pilot_shared_resources.py"]:
        historical = subprocess.check_output(["git", "show", f"b24e154a:{name}"])
        assert historical == (PROJECT / name).read_bytes(), name
    identity = latest["environment"]["observer"]["resume_identity"]
    assert identity["launcher_sha256"] == digest(PROJECT / "scripts/architecture_pilot.py")
    assert identity["evaluation_extension_sha256"] == digest(PROJECT / "scripts/pilot_metric_extensions.py")
    local_rng = random.Random(); local_rng.setstate(latest["rng"]["python"])
    local_np = np.random.RandomState(); local_np.set_state(latest["rng"]["numpy"])
    local_torch = torch.Generator(); local_torch.set_state(latest["rng"]["torch"])
    assert len(latest["rng"]["cuda"]) == 1 and latest["rng"]["cuda"][0].dtype == torch.uint8
    feature_path = work / "data/auxiliary/ASSIST2009-question-difficulty.json"
    assert digest(feature_path) == config["question_difficulty"]["feature_sha256"]
    frozen = read_json(feature_path)
    frame = pd.read_csv(SOURCE / "data/processed/ASSIST2009/interactions.csv", dtype={"student": str})
    frame = frame[frame.split.isin(["train", "validation"])].copy()
    train_students = set(frame.loc[frame.split == "train", "student"])
    validation_students = set(frame.loc[frame.split == "validation", "student"])
    assert not train_students & validation_students
    assert set(frozen["students"]) == train_students | validation_students
    raw = pd.read_csv(SOURCE / "data/raw/ASSIST2009.csv", encoding="latin1",
                      usecols=["user_id", "order_id", "problem_id"], dtype={"user_id": str})
    raw = raw.rename(columns={"user_id": "student", "order_id": "order"})
    raw = raw.merge(frame[["student", "order"]], how="inner", on=["student", "order"], validate="many_to_one")
    assert raw.groupby(["student", "order"]).problem_id.nunique().max() == 1
    frame = frame.merge(raw.drop_duplicates(["student", "order"]), on=["student", "order"], validate="one_to_one")
    recomputed = difficulty_features(frame)
    for student, g in frame.assign(difficulty=recomputed).groupby("student"):
        np.testing.assert_array_equal(g.difficulty.to_numpy(), frozen["students"][student]["values"])
    changed = frame.copy(); mask = changed.split == "validation"
    changed.loc[mask, "label"] = 1 - changed.loc[mask, "label"]
    np.testing.assert_array_equal(recomputed, difficulty_features(changed))
    # Perturb the entire longest training student's labels, including future ones.
    selected_student = frame[frame.split == "train"].groupby("student").size().idxmax()
    own = frame.student.to_numpy() == selected_student
    changed = frame.copy(); changed.loc[own, "label"] = 1 - changed.loc[own, "label"]
    np.testing.assert_array_equal(recomputed[own], difficulty_features(changed)[own])
    tr = frame[frame.split == "train"]; va = frame[frame.split == "validation"]
    latest_train_order = tr.groupby("problem_id").order.max()
    later = va.problem_id.map(latest_train_order) > va.order
    # order_id is an ordering proxy, not an audited wall-clock timestamp.
    timing = {"validation_fraction_with_later_training_problem_observation_order_proxy": float(later.mean()),
              "training_order_range": [int(tr.order.min()), int(tr.order.max())],
              "validation_order_range": [int(va.order.min()), int(va.order.max())],
              "interpretation": "Full-training-partition fitted statistics, valid for offline student-disjoint evaluation; not a forward-time deployment claim. Baseline fitting also uses the complete training partition. order_id is only an ordering proxy."}
    dataset = QuestionDifficultySequences(SOURCE / "data/processed/ASSIST2009/interactions.csv",
                                          "train", 100, feature_path)
    def next_order():
        generator = torch.Generator(); generator.set_state(latest["loader_rng"])
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, generator=generator)
        iterator = iter(loader)  # Includes the original loader's base-seed consumption.
        return [list(batch) for batch in iterator._sampler_iter]
    batches = next_order(); assert batches == next_order()
    flattened = [i for batch in batches for i in batch]
    assert sorted(flattened) == list(range(len(dataset)))
    order_hash = hashlib.sha256(np.asarray(flattened, dtype=np.int64).tobytes()).hexdigest()
    params = latest["environment"]
    assert str(torch.__version__) == params["torch"]
    result = {"status": "PASS_FOR_OFFLINE_STUDENT_DISJOINT_PROTOCOL", "scientifically_valid_to_resume": True,
              "validation_label_perturbation": "all numerical features invariant",
              "own_training_student_all_labels_perturbation": "own features invariant",
              "feature_reconstruction": "all stored numerical features exactly match recomputation",
              "test_policy": "No actual test dataset, predictions, statistics or labels analyzed; full-builder synthetic test-label perturbation is separate.",
              "timing": timing, "checkpoint_epoch": 18, "best_epoch": 18, "bad_epochs": 0,
              "latest_checkpoint_sha256": digest(checkpoint_dir / "latest.pt"),
              "best_checkpoint_sha256": digest(checkpoint_dir / "best.pt"),
              "training_code_sha256": code_hash, "feature_sha256": digest(feature_path),
              "next_epoch": 19, "next_epoch_train_chunks": len(dataset), "next_epoch_batches": len(batches),
              "next_epoch_index_order_sha256": order_hash, "next_epoch_first_16_indices": flattened[:16],
              "rng_states": "Python/NumPy/CPU Torch states reconstruct; one CUDA state present, restored only after GPU authorization",
              "optimizer": "finite full Adam state, unchanged learning rate/betas/epsilon/weight decay",
              "observer_resume_identity": identity, "original_environment": {k: params[k] for k in ["python", "torch", "cuda", "gpu"]}}
    (OUT / "leakage_and_resume_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"audit": result["status"], "next_epoch": 19, "order_hash": order_hash, "timing": timing}, indent=2))


if __name__ == "__main__":
    main()
