from model.cla_component import CompoundEncoder, CompoundDecoder
from dataloader.mol_data import MoleculeDataset
from utils import tanimoto_calc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from rdkit import Chem
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

class CLA(pl.LightningModule):
    def __init__(self, ref_smiles=None):
        super(CLA, self).__init__()
        self.z_dim = 44

        self.enc = CompoundEncoder(out_dim=2 * self.z_dim)
        self.dec = CompoundDecoder(in_dim=self.z_dim)
        
        # kl loss parameter
        self.beta = 0.01
        self.stat = []
        self.epoch = 0
        if ref_smiles == None:
            ref = pd.read_csv("data/zinc_dataset_250k.csv")
            self.ref_smiles = set(ref.smiles.values)
        else:
            self.ref_smiles = ref_smiles


    @staticmethod
    def sample_repar_z(means, logvar):
        return means + torch.randn_like(means) * torch.exp(0.5 * logvar)


    @staticmethod
    def kl_div(means_q, logvar_q, means_p=None, logvar_p=None):
        if means_p is None: 
            return -0.5 * torch.mean(
                torch.sum(1 + logvar_q - means_q.pow(2) - logvar_q.exp(),
                          dim=-1))
        else:
            return -0.5 * torch.mean(
                torch.sum(1 - logvar_p + logvar_q -
                          (means_q.pow(2) + logvar_q.exp()) * (-logvar_p).exp(),
                          dim=-1))


    def training_step(self, batch, batch_nb):
        # pair of objects
        x, _ = batch

        # compute proposal distributions (get mean and variance from data)
        p_z_means, p_z_logvar = torch.split(self.enc(x), self.z_dim, -1)
        
        # sample z (sample new distribution ref to data distribution)
        z_sample = self.sample_repar_z(p_z_means, p_z_logvar)

        # kl divergence (measure distance of proposal distribution)
        kl = self.kl_div(p_z_means, p_z_logvar)

        # reconstrunction loss 
        x_by_z_logprob = self.dec.get_log_prob(x=x, z=z_sample).mean()

        loss = (-x_by_z_logprob + self.beta * kl)

        return {'loss': loss,
                'log': {
                    'x_by_z_logprob': x_by_z_logprob,
                    'kl': kl}
                }
    

    def validation_step(self, batch, batch_nb):
        x, _ = batch
        
        # compute proposal distributions
        p_z_means, p_z_logvar = torch.split(self.enc(x), self.z_dim, -1)
        
        # sample z
        z_sample = self.sample_repar_z(p_z_means, p_z_logvar)
        
        # sample molecules
        sampled_sm = self.dec.sample(z_sample)

        # calculate validity, uniqueness, similarity, and novelty of generated molecules in each batch
        valid = len([s for s in sampled_sm if Chem.MolFromSmiles(s) is not None]) / len(sampled_sm)
        unique = len(np.unique(sampled_sm)) / len(sampled_sm)
        similar = [tanimoto_calc(x_ob,s) for (x_ob, s) in zip(x, sampled_sm) if (Chem.MolFromSmiles(s) is not None)]
        similar_avg = np.mean(similar)
        novelty = 1-(len(set(sampled_sm).intersection(self.set_ref)) / len(sampled_sm))

        return {'valid': valid, 'unique': unique, 'novelty': novelty, 
                'similar': similar_avg, 'smiles': sampled_sm, 'ref_smi':x}
        
    
    def validation_epoch_end(self, outputs):
        val_stats = {}

        self.output = outputs

        # Calculate average of validity, uniqueness, similarity, and novelty 
        val_stats['val_valid'] = np.array([x['valid'] for x in outputs]).mean()
        val_stats['val_unique'] = np.array([x['unique'] for x in outputs]).mean()
        val_stats['val_novelty'] = np.array([x['novelty'] for x in outputs]).mean()
        val_stats['val_similar'] = np.array([x['similar'] for x in outputs]).mean()
        val_stats['val_smiles'] = outputs[0]['smiles']
                
        self.stat.append(val_stats)
            
        # save model weight
        torch.save(self.enc.state_dict(), 'weight/pretrain_cla_enc_ep'+str(self.epoch)+'.ckpt')
        torch.save(self.dec.state_dict(), 'weight/pretrain_cla_dec_ep'+str(self.epoch)+'.ckpt')

        self.epoch = self.epoch + 1

        return {'log': val_stats}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)


    def train_dataloader(self):
        return DataLoader(MoleculeDataset(train=True),
                          batch_size=256, shuffle=True, num_workers=10)


    def val_dataloader(self):
        return DataLoader(MoleculeDataset(train=False),
                          batch_size=256, shuffle=True, num_workers=10)
