"""Independent CPU validation replay of the bounded recency pilot."""
from scripts.architecture_pilot import RECENCY_PILOT
from scripts.verify_softmax_pilot import verify


if __name__ == "__main__":
    verify("one_layer_recency_softmax", RECENCY_PILOT,
           "gpu_authorization_recency_20261027.executed.json")
