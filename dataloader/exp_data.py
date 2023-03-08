# Parts of the code were adapted from https://github.com/insilicomedicine/BiAAE

import pickle
import numpy as np
import pandas as pd
import random
import cmapPy.pandasGEXpress.parse as parse
import torch
from torch.utils.data.dataset import Dataset as TorchDataset


class ExpressionDataSet:
    def __init__(self, seed=0):
        
        with open('data/landmark_gene_row_ids.pickle', 'rb') as handle:
            landmark_gene_row_ids = pickle.load(handle)

        gene_file = 'data/GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx'
        self.gene_exp_gct = parse.parse(gene_file, rid=landmark_gene_row_ids)
        self.seed = seed
        self._load_genes()
        self._load_experiments()

    def _load_genes(self):
        self.gene_exp = self.gene_exp_gct.data_df
        
    def _load_experiments(self):
        self.experiments = pd.read_csv("data/GSE70138_sig_smiles.csv")
        self.experiments = self.experiments[self.experiments['sig_id'].isin(self.gene_exp.columns)]


class ExpressionSampler(TorchDataset):
    def __init__(self, exp_dataset, test_set=None): 

        r'''
        :param exp_dataset: common ExpressionDataSet for train\validation\test samplers
        :param test_set: If test_set is None: use all the data. test_set == False or 0 corresponds to the training set,
            test_set == True or 1 corresponds to the validation set, test_set == 2 corresponds to the test set
        '''
        self.dataset = exp_dataset
        self.gene_exp = self.dataset.gene_exp

        np.random.seed(self.dataset.seed)
        self.experiments = self.dataset.experiments.copy()
        
        self.sig_ids = self.experiments.sig_id
        
        # train or validation or test set
        if test_set is not None:  
            np.random.shuffle(self.sig_ids)
            division_point = self.sig_ids.shape[0] // 5
            if not test_set:    # training set
                self.sig_ids = self.sig_ids[2 * division_point:]
            elif test_set > 1:  # test set
                self.sig_ids = self.sig_ids[
                                division_point:2 * division_point]
            else:               # validation set
                self.sig_ids = self.sig_ids[:division_point]
    
        self.experiments = self.experiments[self.experiments['sig_id'].isin(self.sig_ids.values)].reset_index(drop=True)
        self.smiles = self.experiments.canonical_smiles.values

    def __len__(self):
        return self.experiments.shape[0]

    def __getitem__(self, idx):
        null = 1
        while null != 0:
        # 1. random index of signature
            index = idx

        # 2. get signature name
            sig = self.experiments.sig_id[index]

        # 3. match signature with smiles
            smiles = self.experiments[self.experiments['sig_id']==sig].canonical_smiles.values[0]
            exp = torch.from_numpy(self.gene_exp[sig].values)
            if smiles=="-666":
                null = 1
                idx = random.randint(0, len(self.experiments)-1)
            else:
                null = 0

        # 4. return gene expression and smiles
        return smiles, exp
