# Firing Rate Model of V1, Layer 5

This model of the V1, layer 5 includes Cortico-Cortical (CC), Cortico-Subcortical (CS), PV and SST neurons. Plasticity is not implemented yet. In this model, we observe orientation and direction selectivity.

### Requirements

Use ```conda env create -f network.yml``` to unpack the conda environment. Or download with ```conda install <module>``` the requirements manually. After that, activate the environment with ```conda activate network```

### Run simulation

Use ```python3 ds_os_model.py <put config here>``` where the second argument is the config you want to use. For instance use ```configs.test_config```

### Some hints
- the file ds_os_model is used to find suitable hyperparameters. If you want to just run one simulation delete the last parallelization part
- the model was implemented to measure direction and orientation selectivity. You might want to change the input (in Implementation/helper.py) and evaluation metrics according to your project
- it's advised to plot the activity over time to check the simulation
