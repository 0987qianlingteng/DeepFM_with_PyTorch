import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CONTINUOUS_FIELDS = 13
CONTINUOUS_CLIP = np.array([20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50], dtype=np.float32)


class CriteoDataset(Dataset):
    """
    Processed Criteo-style dataset.
    Format:
    - train file: 39 features + 1 label (last column)
    - test file: 39 features
    """

    def __init__(self, root, train=True, train_file="train.txt", test_file="test.txt"):
        self.root = root
        self.train = train
        self.train_file = train_file
        self.test_file = test_file

        if not os.path.isdir(root):
            raise RuntimeError(f"Dataset root not found: {root}")

        path = os.path.join(root, self.train_file if train else self.test_file)
        if not os.path.exists(path):
            raise RuntimeError(f"Dataset file not found: {path}")

        # Keep header=None to avoid accidental type/column parsing.
        data = pd.read_csv(path, header=None).fillna(0.0).values

        if train:
            self.features = data[:, :-1].astype(np.float32)
            self.target = data[:, -1].astype(np.float32)
        else:
            self.features = data.astype(np.float32)
            self.target = None

    def __len__(self):
        return len(self.features)

    def _build_xi_xv(self, row):
        # Continuous fields: use shared index 0, value is normalized continuous value.
        cont = row[:CONTINUOUS_FIELDS]
        cont = np.maximum(cont, 0.0)
        cont = np.minimum(cont, CONTINUOUS_CLIP)
        cont = np.log1p(cont).astype(np.float32)

        # Categorical fields: index comes from encoded id, value is 1.0.
        cate = row[CONTINUOUS_FIELDS:].astype(np.int64)

        xi_cont = np.zeros(CONTINUOUS_FIELDS, dtype=np.int64)
        xi = np.concatenate([xi_cont, cate], axis=0)

        xv_cate = np.ones_like(cate, dtype=np.float32)
        xv = np.concatenate([cont, xv_cate], axis=0)
        return xi, xv

    def __getitem__(self, idx):
        row = self.features[idx]
        xi, xv = self._build_xi_xv(row)
        xi_t = torch.from_numpy(xi).long()
        xv_t = torch.from_numpy(xv).float()
        if self.train:
            y_t = torch.tensor(self.target[idx], dtype=torch.float32)
            return xi_t, xv_t, y_t
        return xi_t, xv_t

    def get_feature_pair_by_id(self, sample_id):
        """Get (Xi, Xv) tensor pair for a given sample id."""
        if sample_id < 0 or sample_id >= len(self.features):
            raise IndexError(f"sample_id {sample_id} out of range [0, {len(self.features) - 1}]")
        xi, xv = self._build_xi_xv(self.features[sample_id])
        return torch.from_numpy(xi).long(), torch.from_numpy(xv).float()
