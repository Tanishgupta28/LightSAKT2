"""Verify Adam counters, loader permutation stream and CPU RNG across resume."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import io
import hashlib
import pickle
import tarfile
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.architecture_pilot import PROJECT, SOURCE, CONTROL, write_json
from scripts.architecture_study import read_json
from scripts.question_difficulty_inputs import QuestionDifficultySequences


def main():
    torch.set_num_threads(1)
    out = PROJECT / "reports/architecture_study/question_completion_20261029"
    config = read_json(PROJECT / CONTROL / "configs/one_layer_question_difficulty.json")
    archive = read_json(PROJECT / "reports/architecture_study/question_pilot_20261028/recovery_archive_20261028.json")
    name = f"pilot-workspace/checkpoints/{config['run_id']}/latest.pt"
    with tarfile.open(archive["archive"]) as tar:
        payload = tar.extractfile(name).read()
    assert hashlib.sha256(payload).hexdigest() == archive["files_sha256"][name]
    before = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    after = torch.load(PROJECT / name, map_location="cpu", weights_only=False)
    assert before["epoch"] == 18 and after["epoch"] > 18
    resumed_epochs = after["epoch"] - before["epoch"]
    dataset = QuestionDifficultySequences(SOURCE / "data/processed/ASSIST2009/interactions.csv",
        "train", 100, PROJECT / "pilot-workspace/data/auxiliary/ASSIST2009-question-difficulty.json")
    batch_count = (len(dataset) + 63) // 64
    states_before = before["optimizer"]["state"]; states_after = after["optimizer"]["state"]
    assert states_before.keys() == states_after.keys()
    for key in states_before:
        assert int(states_before[key]["step"]) == batch_count * 18
        assert int(states_after[key]["step"]) == batch_count * after["epoch"]
    generator = torch.Generator(); generator.set_state(before["loader_rng"])
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, generator=generator)
    first_order = None
    for _ in range(resumed_epochs):
        iterator = iter(loader)
        order = [index for batch in iterator._sampler_iter for index in batch]
        if first_order is None:
            first_order = hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes()).hexdigest()
    assert torch.equal(generator.get_state(), after["loader_rng"])
    audit = read_json(out / "leakage_and_resume_audit.json")
    assert first_order == audit["next_epoch_index_order_sha256"]
    # CUDA model dropout consumes CUDA RNG. The CPU global stream consumes one
    # base-seed draw per epoch for the unshuffled validation DataLoader.
    cpu = torch.Generator(); cpu.set_state(before["rng"]["torch"])
    for _ in range(resumed_epochs):
        torch.empty((), dtype=torch.int64).random_(generator=cpu)
    assert torch.equal(cpu.get_state(), after["rng"]["torch"])
    assert pickle.dumps(before["rng"]["python"]) == pickle.dumps(after["rng"]["python"])
    assert before["rng"]["numpy"][0] == after["rng"]["numpy"][0]
    np.testing.assert_array_equal(before["rng"]["numpy"][1], after["rng"]["numpy"][1])
    assert before["rng"]["numpy"][2:] == after["rng"]["numpy"][2:]
    result = {"status": "PASS", "resumed_epochs": resumed_epochs, "batches_per_epoch": batch_count,
              "optimizer_states_checked": len(states_before),
              "adam_step_before": batch_count * 18, "adam_step_after": batch_count * after["epoch"],
              "loader_rng_after_matches_exact_permutation_stream": True,
              "epoch19_index_order_sha256": first_order, "cpu_global_rng_after_matches_expected_draws": True,
              "python_numpy_rng_states_preserved": True,
              "cuda_rng": "Original saved state present/restored by unchanged engine; no additional CUDA work performed in this audit.",
              "optimizer_updates_performed": 0, "test_evaluation": False}
    write_json(out / "state_continuity_verification.json", result)
    print("PASS: Adam counters, epoch19 permutation stream and saved CPU/Python/NumPy RNG continuity.")


if __name__ == "__main__":
    main()
