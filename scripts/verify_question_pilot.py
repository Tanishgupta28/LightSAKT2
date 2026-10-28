"""Independent validation-only CPU replay of the question-feature pilot."""
from scripts.architecture_pilot import QUESTION_PILOT
from scripts.verify_softmax_pilot import verify

if __name__ == "__main__":
    verify("one_layer_question_difficulty", QUESTION_PILOT,
           "gpu_authorization_question_20261028.executed.json")
