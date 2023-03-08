import torch
import torch.nn.functional as F
from torch import nn
from utils import get_vocab_size, encode, decode


class CompoundEncoder(nn.Module):
    def __init__(self, out_dim, hidden_size=256, num_layers=3):
        super(CompoundEncoder, self).__init__()

        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embs = nn.Embedding(get_vocab_size(), hidden_size)
        self.rnn = nn.LSTM(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers)

        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, out_dim))


    def forward(self, sm_list):

        tokens, lens = encode(sm_list)
        to_feed = tokens.transpose(1, 0).to(self.embs.weight.device)
        outputs = self.rnn(self.embs(to_feed))[0]
        self.outputs = outputs
        outputs = outputs[lens, torch.arange(len(lens))]

        return self.final_mlp(outputs)


class CompoundDecoder(nn.Module):
    def __init__(self, in_dim, hidden_size=256, num_layers=3):
        super(CompoundDecoder, self).__init__()

        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embs = nn.Embedding(get_vocab_size(), hidden_size)
        self.rnn = nn.LSTM(input_size=hidden_size + in_dim,
                          hidden_size=hidden_size,
                          num_layers=num_layers)

        self.lat_fc = nn.Linear(in_dim, hidden_size)
        self.out_fc = nn.Linear(hidden_size, get_vocab_size())


    def get_log_prob(self, x, z):

        tokens, lens = encode(x)
        x = tokens.transpose(1, 0).to(self.embs.weight.device)

        x_emb = self.embs(x)

        z_0 = z.unsqueeze(0).repeat(x_emb.shape[0], 1, 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)

        h_0 = self.lat_fc(z)
        c_0 = self.lat_fc(z)
        h_0 = h_0.unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
        c_0 = c_0.unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
        
        output, _ = self.rnn(x_input, (h_0,c_0))
        y = self.out_fc(output)
         
        recon_loss = -F.cross_entropy(
            y.transpose(1, 0)[:, :-1].contiguous().view(-1, y.size(-1)),
            x.transpose(1, 0)[:, 1:].contiguous().view(-1),
            ignore_index=0,
            reduction='none'
        )
        
        recon_loss = (recon_loss.view(x.shape[1], -1).sum(dim=-1) / torch.tensor(lens).to(z_0.device).float())
        recon_loss = recon_loss.mean()
        
        return recon_loss
    

    def sample(self, z):
        with torch.no_grad():
            n_batch = z.shape[0]
            z_0 = z.unsqueeze(0)

            max_len = 100
            
            for j in range(10):
                h = self.lat_fc(z)
                h = h.unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
                c = self.lat_fc(z)
                c = c.unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
                w = torch.tensor(1, device=z.device).repeat(n_batch)
                x = torch.tensor(0, device=z.device).repeat(n_batch, max_len)
                x[:, 0] = 1
                end_pads = torch.tensor([max_len], device=z.device).repeat(
                    n_batch)
                eos_mask = torch.zeros(n_batch, dtype=torch.uint8,
                                       device=z.device).bool()

            for i in range(1, max_len):
                x_emb = self.embs(w).unsqueeze(0)

                x_input = torch.cat([x_emb, z_0], dim=-1)
                self.x_input = x_input
                self.h = h
                o, (h, c) = self.rnn(x_input, (h, c))
                y = self.out_fc(o.squeeze(1))
                y = F.softmax(y, dim=-1)

                w = torch.max(y[0], dim=-1)[1]
                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = ~eos_mask & (w == 2)
                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask

            new_x = []
            for i in range(x.size(0)):
                new_x += decode(x[i, 1:end_pads[i]].unsqueeze(0))
            return new_x
