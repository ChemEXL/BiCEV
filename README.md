# De Novo Design of Molecules with Multi-action Potential from Differential Gene Expression using Variational Autoencoder


This repository contains the code to implement BiCEV (Bidirectional Compound-Expression Variational Autoencoder) for generating molecules from given gene expression data.


### Additional Files

Download link: https://kmutt.me/bicev_data

* Compound dataset and gene expression dataset (including GSE70138, gene-knockdown profiles, and combined signatures of synergistic drug pairs) can be loaded via `data` folder. 
* Weights of CLA and BiCEV model are also provided in `weight` folder.


### Training model
#### Pretraining Chemical Language Autoencoder Model (CLA)

```
import pytorch_lightning as pl
from model.cla_model import CLA

cla_model = CLA()
cla_trainer = pl.Trainer(max_epochs=20)
cla_trainer.fit(cla_model)
```

#### Training BiCEV 

```
import pytorch_lightning as pl
from model.cea_model import BiCEV

model = BiCEV(cla_enc_weight='weight/cla_encoder_weight.ckpt',
                cla_dec_weight='weight/cla_decoder_weight.ckpt')
trainer = pl.Trainer(gpus=[0], max_epochs=1)
trainer.fit(model)
```
