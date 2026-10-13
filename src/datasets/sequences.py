import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Sequences(Dataset):
    """One-overlap chunks preserve all n-1 next-response targets; context resets per chunk."""
    def __init__(self, path, split, seq_len):
        self.seq_len = seq_len
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        frame = pd.read_csv(path, dtype={"student": str})
        frame = frame[frame.split == split]
        self.samples, self.identities = [], []
        self.n_targets = 0
        for student, rows in frame.groupby("student", sort=True):
            q, r = rows.q.to_numpy(), rows.label.to_numpy()
            for start in range(0, len(q)-1, seq_len):
                count = min(seq_len, len(q)-1-start)
                arrays = [np.zeros(seq_len, dtype=np.int64) for _ in range(4)]
                arrays[0][:count] = q[start:start+count]
                arrays[1][:count] = r[start:start+count]
                arrays[2][:count] = q[start+1:start+count+1]
                arrays[3][:count] = r[start+1:start+count+1]
                valid = np.arange(seq_len) < count
                self.samples.append(tuple(torch.from_numpy(a) for a in arrays)+(torch.from_numpy(valid),))
                self.identities.append({"student": student, "start": start, "n_targets": count})
                self.n_targets += count

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
