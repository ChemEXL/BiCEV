# Parts of the code were adapted from Shayakhmetov, R.; Kuznetsov, M.; Zhebrak, A.; Kadurin, A.; Nikolenko, S.; Aliper, A.; Polykovskiy, D. Molecular Generation for Desired Transcriptome Changes With Adversarial Autoencoders. Frontiers in Pharmacology 2020, 11.

import torch.nn.functional as F
from torch import nn 

class ExpressionEncoder(nn.Module):
    def __init__(self, out_dim):
        super(ExpressionEncoder, self).__init__()
        
        self.out_dim = out_dim
        
        self.exp_fc = nn.Sequential(
            nn.Linear(978, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, out_dim)
        )
        
    def forward(self, batch):
        return self.exp_fc(batch)


class ExpressionDecoder(nn.Module):
    def __init__(self, in_dim):
        super(ExpressionDecoder, self).__init__()
        
        self.in_dim = in_dim
        
        self.expr_fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 978),
        )
        
    def forward(self, z):
        return self.expr_fc(z)
    
    def get_log_prob(self, x, z):

        diff_pred = self.forward(z)
        diff = x
        return -6 * nn.MSELoss()(diff_pred, diff)
    
    def sample(self, z):
        return self.forward(z)

# For wraping up compound encoder in order to finetune
class FinetunedEncoder(nn.Module):
    def __init__(self, fr_enc, out_dim):
        super().__init__()
        self.fr_enc = fr_enc
        
        self.out_dim = out_dim
        
        self.step_counter = 0
        
        self.new_fc = nn.Sequential(nn.Linear(fr_enc.out_dim, 256),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, out_dim))
                               
        for p in self.fr_enc.parameters():
            p.requires_grad = False
                
        for p in self.fr_enc.final_mlp.parameters():
            p.requires_grad = True
            
        self.parameters = nn.ParameterList(self.new_fc.parameters())
        
    def forward(self, x):
        self.step_counter += 1
        
        return self.new_fc(self.fr_enc(x))

# For wraping up compound decoder in order to finetune
class FinetunedDecoder(nn.Module):
    def __init__(self, fr_dec, in_dim):
        super().__init__()
        
        self.fr_dec = fr_dec
        self.in_dim = in_dim
        
        self.new_fc = nn.Sequential(nn.Linear(in_dim, 128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128, 512),
                                    nn.LeakyReLU(),
                                    nn.Linear(512, fr_dec.in_dim))
            
        self.step_counter = 0
 
        for p in self.fr_dec.parameters():
            p.requires_grad = False
                
        for p in self.fr_dec.lat_fc.parameters():
            p.requires_grad = True

        self.parameters = nn.ParameterList(self.new_fc.parameters())
        
    def get_log_prob(self, x, z):        
        return self.fr_dec.get_log_prob(x, self.new_fc(z))
    
    def sample(self, z):
        return [self.fr_dec.sample(self.new_fc(z)) for i in range(10)]