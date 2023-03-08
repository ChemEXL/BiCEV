# De Novo Design of Molecules with Multi-action Potential from Differential Gene Expression using Variational Autoencoder


This repository contains the code to implement BiCEV (Bidirectional Compound-Expression Variational Autoencoder) for generating molecules from given gene expression data.


### Additional Files

* The compound dataset and gene expression dataset (including GSE70138, gene-knockdown profiles, and combined signatures of synergistic drug pairs) can be  downloaded from `dataset` folder in this link: https://kmutt.me/bicev_data.
* Please copy the downloaded dataset to `data` folder to implement.
* Weights of the CLA and BiCEV models are also provided in the `weights` folder in the same link.


### Installation
BiCEV depends on NumPy, Pandas, PyTorch, PyTorch Lightning, RDKit-pypi, cmapPy, and fcd. 
You can install these dependencies using the following command:

`pip install -r requirements.txt`



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

model = BiCEV(cla_enc_weight='cla_encoder_weight.ckpt',
                cla_dec_weight='cla_decoder_weight.ckpt')
trainer = pl.Trainer(gpus=[0], max_epochs=1)
trainer.fit(model)
```
