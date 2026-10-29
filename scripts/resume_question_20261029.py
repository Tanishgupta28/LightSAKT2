"""Resume the audited epoch-18 run without altering its launcher identity."""
import argparse
from pathlib import Path
from scripts import architecture_pilot as pilot
from scripts.architecture_study import digest, read_json

REPORT = Path("reports/architecture_study/question_completion_20261029")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    root = pilot.PROJECT; work = root / "pilot-workspace"
    audit = read_json(root / REPORT / "leakage_and_resume_audit.json")
    assert audit["scientifically_valid_to_resume"] is True
    config = read_json(root / pilot.CONTROL / "configs/one_layer_question_difficulty.json")
    checkpoint = work / "checkpoints" / config["run_id"] / "latest.pt"
    if digest(checkpoint) != audit["latest_checkpoint_sha256"]:
        raise pilot.PilotBlocked("Epoch-18 checkpoint changed; do not start or restart")
    assert read_json(work / "results/raw" / config["run_id"] / "status.json")["status"] == "interrupted"
    # Only output paths change. Original launcher bytes, stage, observer identity,
    # training engine, model and scientific configuration remain identical.
    pilot.SINGLE_ATTENTION_STAGES["question"] = REPORT
    pilot.execute("question", pilot.SOURCE, work, args.authorization, project=root)


if __name__ == "__main__":
    main()
