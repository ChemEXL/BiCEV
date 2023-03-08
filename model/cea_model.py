# Parts of the code were adapted from https://github.com/insilicomedicine/BiAAE

from model.cla_component import CompoundEncoder, CompoundDecoder
from model.cea_component import FinetunedEncoder, FinetunedDecoder, ExpressionEncoder, ExpressionDecoder
from dataloader.exp_data import ExpressionDataSet, ExpressionSampler
from utils import tanimoto_calc, fcd_metrics

import pickle
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from rdkit import Chem

class BiCEV(pl.LightningModule):
    def __init__(self, cla_enc_weight=None, cla_dec_weight=None, ref_smiles=None):
        super(BiCEV, self).__init__()
        
        self.z_dim = 20
        self.joint_dim = 10

        self.loss_rec_lambda_x = 20
        self.loss_rec_lambda_y = 0.1
        self.loss_shared_lambda = 2
        self.loss_kl_beta = 0.05

        self.test_stats = []
        self.stat = []
        self.epoch = 0

        if ref_smiles == None:
            ref = pd.read_csv("data/zinc_dataset_250k.csv")
            self.ref_smiles = set(ref.smiles.values)
        else:
            self.ref_smiles = ref_smiles
        
        cmp_enc = CompoundEncoder(out_dim=88)
        if cla_enc_weight != None:
            cmp_enc.load_state_dict(torch.load(cla_enc_weight, map_location='cuda:0'))
        self.enc_x = FinetunedEncoder(cmp_enc, out_dim=2*self.z_dim)
        self.enc_y = ExpressionEncoder(out_dim=2*self.z_dim)

        cmp_dec = CompoundDecoder(in_dim=44)
        if cla_dec_weight != None:  
            cmp_dec.load_state_dict(torch.load(cla_dec_weight, map_location='cuda:0'))
        self.dec_x = FinetunedDecoder(cmp_dec, in_dim=self.z_dim)
        self.dec_y = ExpressionDecoder(in_dim=self.z_dim)
    
    
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
     
        
    # A method that generates a molecule given desired expressions.
    def sample(self, y):
        
        # encode expression, then split outputs into exclusive parts and shared parts
        p_zs_y_means, p_zs_y_logvar = torch.split(self.enc_y(y), self.z_dim, -1)
        p_z_y_means, p_s_y_means = torch.split(p_zs_y_means, self.z_dim - self.joint_dim, -1)
        p_z_y_logvar, p_s_y_logvar = torch.split(p_zs_y_logvar, self.z_dim - self.joint_dim, -1)
        
        # sample z from distribution of exclusive parts
        # sample s from distribution of shared parts
        z_y_sample = self.sample_repar_z(p_z_y_means, p_z_y_logvar)
        s_y_sample = self.sample_repar_z(p_s_y_means, p_s_y_logvar)
        z_x_sample = torch.randn_like(z_y_sample)

        sampled_x = self.dec_x.sample(torch.cat((z_x_sample, s_y_sample), 1))
        
        return sampled_x

    
    def training_step(self, batch, batch_nb):
        # pair of (smiles, expression)
        x, y = batch
        
        # encode expression, then split outputs into exclusive parts and shared parts
        p_zs_x_means, p_zs_x_logvar = torch.split(self.enc_x(x), self.z_dim, -1)
        p_z_x_means, p_s_x_means = torch.split(p_zs_x_means, self.z_dim - self.joint_dim, -1)
        p_z_x_logvar, p_s_x_logvar = torch.split(p_zs_x_logvar, self.z_dim - self.joint_dim, -1)
        
        p_zs_y_means, p_zs_y_logvar = torch.split(self.enc_y(y), self.z_dim, -1)
        p_z_y_means, p_s_y_means = torch.split(p_zs_y_means, self.z_dim - self.joint_dim, -1)
        p_z_y_logvar, p_s_y_logvar = torch.split(p_zs_y_logvar, self.z_dim - self.joint_dim, -1)
        
        # sample z from distribution of exclusive parts
        # sample s from distribution of shared parts
        z_x_sample = self.sample_repar_z(p_z_x_means, p_z_x_logvar)
        s_x_sample = self.sample_repar_z(p_s_x_means, p_s_x_logvar)
        z_y_sample = self.sample_repar_z(p_z_y_means, p_z_y_logvar)
        s_y_sample = self.sample_repar_z(p_s_y_means, p_s_y_logvar)
        
        # KL loss
        z_x_kl = self.kl_div(p_z_x_means, p_z_x_logvar)
        z_y_kl = self.kl_div(p_z_y_means, p_z_y_logvar)
        loss_kl = self.loss_kl_beta * (z_x_kl + z_y_kl)
        
        # Reconstruction loss
        x_z_logprob = self.dec_x.get_log_prob(x, torch.cat((z_x_sample, s_x_sample), -1))
        y_z_logprob = self.dec_y.get_log_prob(y, torch.cat((z_y_sample, s_y_sample), -1))
        loss_rec = -(self.loss_rec_lambda_x * x_z_logprob + self.loss_rec_lambda_y * y_z_logprob)

        # Shared loss
        loss_shared =  torch.norm(s_x_sample - s_y_sample, p=2, dim=-1).mean() * self.loss_shared_lambda

        loss = loss_rec + loss_kl + loss_shared
        
        return {'loss': loss,
                'log': {
                    'loss_rec': loss_rec, 
                    'loss_kl': loss_kl,
                    'loss_shared': loss_shared
                    }
                }
        
    def validation_step(self, batch, batch_nb):
            
        # random index of expressions
        index = random.sample(range(len(batch[0])),20)
        
        # duplicate pairs of smiles and expressions are used to investigate 
        # the model's ability to generate unique outputs for identical expression profiles 
        smiles_ref = []
        for i in index:
            for n in range(20):
                smiles_ref.append(batch[0][i]) 
        smiles_batch = tuple(smiles_ref)

        exp_ref = []
        for i in index:
            for n in range(20):
                exp_ref.append(batch[1][i]) 
        exp_batch = torch.stack(exp_ref)

        new_batch = [smiles_batch,exp_batch]

        # sample molecules based on a given set of expressions
        x, y = new_batch
        sampled_sm = self.sample(y)
                            
        # validate generated molecules using criteria
        # including validity, uniqueness, novelty, and similarity score
        valid = len([s for s in sampled_sm[0] if Chem.MolFromSmiles(s) is not None]) / len(sampled_sm[0])
        unique = len(np.unique(sampled_sm[0])) / len(sampled_sm[0])
        set_gen = set(sampled_sm[0])
        novelty = 1-(len(set_gen.intersection(self.ref_smiles)) / len(sampled_sm[0]))
        similarity_score = []
        for i in range(len(x)):
            similarity_score.append(tanimoto_calc(x[i],sampled_sm[0][i]))
        
        return {'valid': valid, 'unique': unique, 'similar': similarity_score, 
                'novelty':novelty, 'smiles': sampled_sm, 'ref_smi':x}


    def validation_epoch_end(self, outputs):
        val_stats = {}

        self.output = outputs

        # calculate average of each criteria, including validity, uniqueness, similarity, and novelty
        val_stats['val_valid'] = np.array([x['valid'] for x in outputs]).mean()
        val_stats['val_unique'] = np.array([x['unique'] for x in outputs]).mean()
        val_stats['val_sim'] = np.array([x['similar'] for x in outputs]).mean()
        val_stats['val_novelty'] = np.array([x['novelty'] for x in outputs]).mean()
        val_stats['val_smiles'] = outputs[0]['smiles']
        val_stats['val_fcd'] = fcd_metrics([s for s in outputs[0]['smiles'][0] if s != "" and Chem.MolFromSmiles(s) is not None], outputs[0]['ref_smi'])

        self.stat.append(val_stats)

        # save stat
        with open('BiCEV_validate_result.pickle', 'wb') as handle:
            pickle.dump(self.stat, handle)

        return {'log': val_stats}
    

    def test_step(self, batch, batch_nb):

        # random index of expressions
        index = random.sample(range(len(batch[0])),20)

        # duplicate pairs of smiles and expressions are used to investigate 
        # the model's ability to generate unique outputs for identical expression profiles 
        smiles_ref = []
        for i in index:
            for n in range(20):
                smiles_ref.append(batch[0][i]) 
        smiles_batch = tuple(smiles_ref)

        exp_ref = []
        for i in index:
            for n in range(20):
                exp_ref.append(batch[1][i]) 
        exp_batch = torch.stack(exp_ref)

        new_batch = [smiles_batch,exp_batch]

        # sample molecules based on a given set of expressions
        x, y = new_batch
        sampled_sm = self.sample(y)

        # validate generated molecules using criteria
        # including validity, uniqueness, novelty, and similarity score
        valid = len([s for s in sampled_sm[0] if Chem.MolFromSmiles(s) is not None]) / len(sampled_sm[0])
        unique = len(np.unique(sampled_sm[0])) / len(sampled_sm[0])
        set_gen = set(sampled_sm[0])
        novelty = 1-(len(set_gen.intersection(self.ref_smiles)) / len(sampled_sm[0]))
        similarity_score = []
        for i in range(len(x)):
            similarity_score.append(tanimoto_calc(x[i],sampled_sm[0][i]))

        return {'valid': valid, 'unique': unique, 'similar': similarity_score,
                'novelty':novelty, 'smiles': sampled_sm, 'ref_smi': x}


    def test_epoch_end(self, outputs):
        self.output = outputs
        test_stats = {}

        # calculate average of each criteria, including validity, uniqueness, similarity, novelty, and fcd
        test_stats['test_valid'] = np.array([x['valid'] for x in outputs]).mean()
        test_stats['test_unique'] = np.array([x['unique'] for x in outputs]).mean()
        test_stats['test_sim'] = np.array([x['similar'] for x in outputs]).mean()
        test_stats['test_novelty'] = np.array([x['novelty'] for x in outputs]).mean()
        test_stats['test_fcd'] = fcd_metrics([s for s in self.output[0]['smiles'][0] if Chem.MolFromSmiles(s) is not None], self.output[0]['ref_smi']) 
        test_stats['test_smiles'] = outputs[0]['smiles']

        self.test_stats.append(test_stats)

        # save test stat
        with open('BiCEV_test_result.pickle', 'wb') as handle:
            pickle.dump(self.test_stats, handle)

        return {'log': test_stats}

    
    def predict_from_single_expression(self, gene_exp, amount):
        """
        Generate molecules from a given expressions
        gene_exp: an expression as a input
        amount: the amount of molecules to generate
        """
        gene_exp = torch.tensor(gene_exp,device='cuda:0')
        exp_batch = self.duplicate_expr(gene_exp, amount)
        
        return self.sample(exp_batch.float())


    def predict_from_multiple_expressions(self, Z_a, Z_b, amount):
        """
        Generate molecules from given two expressions
        Z_a: the first expression to combine
        Z_b: the second expression to combine
        amount: the amount of molecules to generate
        """ 
        Z_ab = Z_a + Z_b - np.multiply(Z_a,Z_b)
        Z_ab = torch.tensor(Z_ab,device='cuda:0')
        exp_batch = self.duplicate_expr(Z_ab, amount)
        
        return [self.sample(exp_batch) for n in range(1)]
    
    
    def duplicate_expr(self,gene_exp, amount):
        """
        Duplicate gene expression with specific amount
        gene expressions: the gene expression that will be duplicated
        amount: the number of times to duplicate the gene expression
        """ 
        exp_ref = []
        for n in range(amount):
            exp_ref.append(gene_exp) 
        return torch.stack(exp_ref)

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


    def train_dataloader(self):
        dataset = ExpressionSampler(ExpressionDataSet(), test_set=0)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True,
                                drop_last=False)
        return dataloader


    def val_dataloader(self):
        dataset = ExpressionSampler(ExpressionDataSet(), test_set=1)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True,
                                drop_last=False)
        return dataloader


    def test_dataloader(self):
        dataset = ExpressionSampler(ExpressionDataSet(), test_set=2)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True,
                                drop_last=False)
        return dataloader
        
        