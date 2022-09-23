Gp2 Developability Modeling, DevRep2, Lead Author: Alexander Golinski

Contact Information golin010@umn.edu

Hackel Lab University of Minnesota
Martiniani Lab New York University

This project contains Python3 scripts used to predict the yield of Gp2 paratope variants utilizing high-throughput developability assays via transfer learning. 

Required packages are included in the conda_package_list.txt. One can also install the conda environment used to train/test the models using awg_gpu.yml. 

File descriptions: model_module.py - base model class that defines how to cross-validate, test, and evaluate model performances. submodels_module.py - subclasses that modify the model inputs/outputs and datasets for model evaluation. model_architectures.py - describes the hyperparameters and construction of the possible model architectures.load_format_data.py - helper functions to format the data from the pickeled DataFrames to useful inputs for model evaluations.

Folder descriptions: /aaindex/ - scripts that compare DevRep embeddings to AAIndex physicochemical properties 

/plots/ - scripts that compare proto-DevRep architectures and generate performance statistics displayed in the manuscript

/alternative models/ - scripts that train and compare performance of the alternative models displayed in the manuscript, namely training regimes using either simulated HT assays or recombinant yields

/sampling/ - scripts that both conduct nested sampling/simulated annealing on DevRep embeddings to generate statistical mechanical properties of the developabililty landscape and then cluster and sample these embeddings for in vitro testing


