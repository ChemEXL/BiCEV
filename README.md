# De Novo Design of Molecules with Multi-action Potential from Differential Gene Expression using Variational Autoencoder


This repository contains the code to implement BiCEV (Bidirectional Compound-Expression Variational Autoencoder) for generating molecules from given gene expression data.


### Additional Files


* Compound and gene expression datasets (including GSE70138, gene-knockdown profiles, and combined signatures of synergistic drug pairs) can be downloaded from the `dataset` folder in this link: https://kmutt.me/bicev_data.
* Please copy the downloaded datasets to the `data` folder to implement.
* The weights of CLA and BiCEV models are also provided  in the same link in the `weights` folder.


### Installation
BiCEV implementation relies on NumPy, Pandas, PyTorch, PyTorch Lightning, RDKit-pypi, cmapPy, and fcd. 
You may install these dependencies using the following command (recommend for cuda version 12.0):

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

---
### References
* ZINC15: Sterling and Irwin, J. Chem. Inf. Model, 2015 http://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00559. 
* CMAP LINCS 2020: https://clue.io/data/CMap2020#LINCS2020
* LINCS L1000 GSE70138: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138
* Subramanian A, et al. A Next Generation Connectivity Map: L1000 Platform And The First 1,000,000 Profiles. Cell. 2017/12/1. 171(6):1437–1452.
* Shayakhmetov, R.; Kuznetsov, M.; Zhebrak, A.; Kadurin, A.; Nikolenko, S.; Aliper, A.; Polykovskiy, D. Molecular Generation for Desired Transcriptome Changes With Adversarial Autoencoders. Frontiers in Pharmacology 2020, 11.