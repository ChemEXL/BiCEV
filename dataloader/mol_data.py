# Parts of the code were adapted from https://github.com/insilicomedicine/BiAAE

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MoleculeDataset(Dataset):
    def __init__(self, filename="data/zinc_dataset_250k.csv", train=True, seed=777):
        data = pd.read_csv(filename).dropna()
        self.smiles = data.values[:, 1]
        
        np.random.seed(seed)
        
        train_test_perm = np.random.permutation(self.smiles.shape[0])
        
        if train:
            self.smiles = self.smiles[train_test_perm[:225000]]
        else:
            self.smiles = self.smiles[train_test_perm[225000:]]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], torch.ones(1)