"""Independent CPU validation replay for the single gated-linear pilot."""
from scripts.architecture_pilot import GATING_PILOT
from scripts.verify_softmax_pilot import verify


if __name__ == "__main__":
    verify("one_layer_gated_linear", GATING_PILOT,
           "gpu_authorization_gating_20261027.executed.json")
