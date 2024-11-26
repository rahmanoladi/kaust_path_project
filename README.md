# Introduction
This Project explores the importance of incorporating path information in GNN-style models in the arena of molecular graphs for the task of modelling Quantitative Structure Activity Relationship (QSAR). Our experiments harnessed six popular datasets from the Moleculenet suite of datasets, namely FreeSolv, ESOL, Lipophilicity, BACE, BBBP and ClinTox. For breadth, we explored four different GNN-style models: Graphormer, DeeperGCN, Mix-Hop, as well as a novel model of ours which we term T-Hop. Further, we considered the scenario in which the node and edge features of the molecular graphs are deliberatly corrupted with calibrated levels of Gaussian noise, to see whether or not path information can compensate for the added noise. While collation and interpretation of experimental results is ongoing, the current picture appears to be that, in some cases, path information and addition of noise can respectively improve model performance; in other cases we observed that incorporating path information and adding noise to features actually mars model performance.

# Requirements
The programs in this repo were written using Python 3.9.18, Pytorch 2.2.1, DGL 2.4.0 and DGLLife 0.3.2. You should install the above packages in your environment before running the programs in this repo.

# Sample Runs


