# De Novo Design of Molecules with Multi-action Potential from Differential Gene Expression using Variational Autoencoder
---

The modulating effect of therapeutics on gene transcription is well reported and has been intensively studied for both clinical and research purposes. One aspect that recently came to light of the utility of drug-induced transcriptional changes is in de novo molecular design, which highlights the possibility to phenotype-match the molecules being designed to an expression signature of interest. In our work, we build an autoencoder-based generative model, BiCEV, around this concept. Our generative autoencoder can demonstrably generate a set of novel molecules with impressive validity (96%), uniqueness (98%), and internal diversity (0.77). We have made further attempts to validate the BiCEV model by implementing it on gene-knockdown profiles and combined signatures of synergistic drug pairs. From these attempts, the collective quality of the resulting structures is consistently high.


This repository contains the code for implement BiCEV(Bidirectional Compound-Expression Variational Autoencoder) to generated molecules from given gene expression.

### Additional Files
* Compound dataset and gene expression dataset (including GSE70138, gene-knockdown profiles, and combined signatures of synergistic drug pairs) can be loaded via `data` folder this link: https://kmutt.me/bicev_data
* Weights of CLA and BiCEV model are also provided in `weight` folder in the same link 



### Training model
#### Pretraining Chemical Language Autoencoder Model (CLA)

`cla_model = CLA()`

`cla_trainer = pl.Trainer(max_epochs=20)`

`cla_trainer.fit(cla_model)`


#### Training BiCEV 

`model = BiCEV(cla_enc_weight='weight/cla_encoder_weight.ckpt',`
`                cla_dec_weight='weight/cla_decoder_weight.ckpt')`

`trainer = pl.Trainer(gpus=[0], max_epochs=1)`

`trainer.fit(model)`
