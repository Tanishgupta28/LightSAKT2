# import os
# import argparse
# import json
# import pickle
# import torch
# import sys
# import itertools
# import copy
# from torch.utils.data import DataLoader, random_split
# from torch.optim import SGD, Adam

# # Add parent directory to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from data_loaders.assist2009 import ASSIST2009
# from data_loaders.assist2018 import ASSIST2018
# from data_loaders.assist2015 import ASSIST2015
# from data_loaders.algebra2005 import Algebra2005
# from data_loaders.statics2011 import Statics2011
# from models.dkt import DKT
# from models.dkt_plus import DKTPlus
# from models.dkvmn import DKVMN
# from models.sakt import SAKT
# from models.gkt import PAM, MHA
# from models.utils import collate_fn


# def train_and_evaluate(model_name, dataset_name, config):
#     train_config = config["train_config"]
#     seq_len = train_config["seq_len"]
#     train_ratio = train_config["train_ratio"]
#     batch_size = train_config["batch_size"]

#     # ✅ Load dataset
#     if dataset_name == "ASSIST2009":
#         dataset = ASSIST2009(seq_len)
#     elif dataset_name == "ASSIST2018":
#         dataset = ASSIST2018(seq_len)
#     elif dataset_name == "ASSIST2015":
#         dataset = ASSIST2015(seq_len)
#     elif dataset_name == "Algebra2005":
#         dataset = Algebra2005(seq_len)
#     elif dataset_name == "Statics2011":
#         dataset = Statics2011(seq_len)
#     else:
#         raise ValueError(f"Unknown dataset: {dataset_name}")

#     # ✅ Split into train/test
#     train_size = int(len(dataset) * train_ratio)
#     test_size = len(dataset) - train_size

#     generator = torch.Generator(device="cpu").manual_seed(42)

#     indices = torch.randperm(len(dataset), device="cpu")  # force CPU
#     train_indices = indices[:train_size]
#     test_indices = indices[train_size:]

#     train_dataset = torch.utils.data.Subset(dataset, train_indices)
#     test_dataset = torch.utils.data.Subset(dataset, test_indices)

#     # ✅ Data loaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

#     # ✅ Print info
#     print(f"📊 Train dataset size: {len(train_dataset)}")
#     print(f"📊 Test dataset size: {len(test_dataset)}")

#     print("\n🔹 Example from train dataset:")
#     for q, r, mask in train_loader:   # 🔹 FIXED: unpack 3 values
#         print("Q shape:", q.shape)
#         print("R shape:", r.shape)
#         print("Mask shape:", mask.shape)
#         print("Q sample:", q[0][:10])  # print first 10 tokens of first sample
#         print("R sample:", r[0][:10])
#         print("Mask sample:", mask[0][:10])
#         break

#     print("\n🔹 Example from test dataset:")
#     for q, r, mask in test_loader:   # 🔹 FIXED: unpack 3 values
#         print("Q shape:", q.shape)
#         print("R shape:", r.shape)
#         print("Mask shape:", mask.shape)
#         print("Q sample:", q[0][:10])
#         print("R sample:", r[0][:10])
#         print("Mask sample:", mask[0][:10])
#         break

#     return None, None, None  # not training, so return dummies


# def main(model_name, dataset_name, mode, config_path="./config.json"):
#     with open(config_path) as f:
#         config = json.load(f)

#     # Just call train_and_evaluate, no training
#     _, _, _ = train_and_evaluate(model_name, dataset_name, config)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_name", type=str, default="sakt",
#                         help="Model to train: [dkt, dkt+, dkvmn, sakt, gkt]")
#     parser.add_argument("--dataset_name", type=str, default="ASSIST2009",
#                         help="Dataset name")
#     parser.add_argument("--mode", type=str, default="train", choices=["train", "grid_search"])
#     parser.add_argument("--config", type=str, default="./config.json",
#                         help="Path to config file")

#     args = parser.parse_args()

#     print(f"🚀 Loading {args.model_name} on {args.dataset_name}")
#     main(args.model_name, args.dataset_name, args.mode, args.config)




import os
import argparse
import json
import torch
import sys
from torch.utils.data import DataLoader

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loaders.assist2009 import ASSIST2009
from data_loaders.assist2018 import ASSIST2018
from data_loaders.assist2015 import ASSIST2015
from data_loaders.algebra2005 import Algebra2005
from data_loaders.statics2011 import Statics2011
from models.dkt import DKT
from models.dkt_plus import DKTPlus
from models.dkvmn import DKVMN
from models.sakt import SAKT
from models.gkt import PAM, MHA
from models.utils import collate_fn


def train_and_evaluate(model_name, dataset_name, config):
    train_config = config["train_config"]
    seq_len = train_config["seq_len"]
    train_ratio = train_config["train_ratio"]
    batch_size = train_config["batch_size"]

    # ✅ Load dataset
    if dataset_name == "ASSIST2009":
        dataset = ASSIST2009(seq_len)
    elif dataset_name == "ASSIST2018":
        dataset = ASSIST2018(seq_len)
    elif dataset_name == "ASSIST2015":
        dataset = ASSIST2015(seq_len)
    elif dataset_name == "Algebra2005":
        dataset = Algebra2005(seq_len)
    elif dataset_name == "Statics2011":
        dataset = Statics2011(seq_len)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # ✅ Split into train/test
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    indices = torch.randperm(len(dataset), device="cpu")  # force CPU
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # ✅ Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # ✅ Print info
    print(f"📊 Train dataset size: {len(train_dataset)}")
    print(f"📊 Test dataset size: {len(test_dataset)}")

    # 🔹 Example from train dataset
    print("\n🔹 Example from train dataset:")
    for batch in train_loader:
        if isinstance(batch, (list, tuple)):
            print(f"Batch length: {len(batch)}")
            for i, item in enumerate(batch):
                print(f" Item {i} shape:", item.shape if torch.is_tensor(item) else type(item))
            # unpack safely
            q, r, *others = batch
            print("Q sample:", q[0][:10])
            print("R sample:", r[0][:10])
            if others:
                for j, extra in enumerate(others):
                    if torch.is_tensor(extra):
                        print(f" Extra {j} sample:", extra[0][:10])
        else:
            print("Unexpected batch type:", type(batch))
        break

    # 🔹 Example from test dataset
    print("\n🔹 Example from test dataset:")
    for batch in test_loader:
        if isinstance(batch, (list, tuple)):
            print(f"Batch length: {len(batch)}")
            for i, item in enumerate(batch):
                print(f" Item {i} shape:", item.shape if torch.is_tensor(item) else type(item))
        else:
            print("Unexpected batch type:", type(batch))
        break

    return None, None, None  # not training, so return dummies


def main(model_name, dataset_name, mode, config_path="./config.json"):
    with open(config_path) as f:
        config = json.load(f)

    # Just call train_and_evaluate, no training
    _, _, _ = train_and_evaluate(model_name, dataset_name, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="sakt",
                        help="Model to train: [dkt, dkt+, dkvmn, sakt, gkt]")
    parser.add_argument("--dataset_name", type=str, default="ASSIST2009",
                        help="Dataset name")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "grid_search"])
    parser.add_argument("--config", type=str, default="./config.json",
                        help="Path to config file")

    args = parser.parse_args()

    print(f"🚀 Loading {args.model_name} on {args.dataset_name}")
    main(args.model_name, args.dataset_name, args.mode, args.config)

