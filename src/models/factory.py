from src.models.light_sakt import LightSAKT
from src.models.sakt import SAKT
from src.models.dkt import DKT
from src.models.dkt_plus import DKTPlus
from src.models.dkvmn import DKVMN
from src.models.gkt import PAM, MHA


def build_model(name, num_q, config):
    constructors = {"LightSAKT": LightSAKT, "SAKT": SAKT, "DKT": DKT,
                    "DKT+": DKTPlus, "DKVMN": DKVMN, "GKT": PAM, "GKT_MHA": MHA}
    return constructors[name](num_q=num_q, **config)


def predict(model, name, batch):
    q, r, qry, target, valid = batch
    if name in {"DKT", "DKT+", "GKT", "GKT_MHA"}:
        out = model(q, r, valid)
        all_p = out[0] if isinstance(out, tuple) else out
        return all_p.gather(-1, qry.masked_fill(~valid, 0).unsqueeze(-1)).squeeze(-1), all_p
    return model(q, r, qry, valid)[0], None
