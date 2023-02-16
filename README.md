# Firing Rate Model of V1, Layer 5

This model of the V1, layer 5 includes Cortico-Cortical (CC), Cortico-Subcortical (CS), PV and SST neurons. Plasticity is not implemented yet. In this model, we observe orientation and direction selectivity.

## 1. Requirements

Use ```conda env create -f network.yml``` to unpack the conda environment. Or download with ```conda install <module>``` the requirements manually. After that, activate the environment with ```conda activate network```

## 2. Run simulation

Use ```python3 ds_os_model.py <put config here>``` where the second argument is the config you want to use. For instance use ```configs.test_config```

> Change: in order to incorporate hyperparameter tuning, the ```configs/debug_config``` is directly used as the input configuration. For tuning just change the configuration file firectly.

## 3. Network workflow

![L5_neuron_rate_model](https://user-images.githubusercontent.com/91852421/218749790-0954ceca-c649-4af5-a94d-543c83df0cd7.png)


## 4. Input parameters

The input parameters in this network can be sectioned into 3 parts:

### 4.1 Parameters for network configuration:

General configurations considering how the network updates neuron weights and activities are:

* ```learning_rule```: learning rule of the neuron weight update
* ```nonlinearity_rule```: activation function for activity changing rate calculation
* ```update_function```: function to calculate activity changing rate (in time scope of 1/```tau```)
* ```integrator```: function to update activity and weights (in time scope of ```delta_t```)
* ```N```: Number of each type of neurons in the system
* ```w_noise```:synaptic weight noise

Meanwhile, we also allow flexible training of selected neurons and patterns:

* ```neurons```: selecting which types of neurons are learning (updating weights)
* ```phase_list```: indicating how the neurons are being trained temporally (alternating training modes)

Additionally, to train the model to our deserable configuration, we need to set the target weight and initial weights, target weights are based on [this paper](https://www.science.org/doi/epdf/10.1126/science.abj5861)

* ```w_target```: the target presy-postsyn weight extracted from the paper. row: presyn, col: postsyn
* Yet in our model, the target weight cannot be directly compared as we also introduce the probability of two types of neurons forming a synapse, as well as the matrix scaling, the eventual weight configuration to be compared should is set to be another parameter ```w_compare```
* ```w_initial```: the initial weights for the network input, the initial weight is sampled using guassian distribution with mean as target weight and std as  0.25. The randomization is only done to neurons being trained indicated by ```neurons```, the rest of the weights are set to be the target weight. 

### 4.2 Parameters for training time:

* Total length and training steps are defined by ```(Ttau*tau)/delta_t```
* ```tau```: reverse of neuron activity updating scale 
* ```tau_learn```: reverse of weight updating scale
* ```tau_threshold```: reverse of threshold updating scale

### 4.3 Parameters for stimulus input:

* ```steady_input```: Indicating if the input of those parameters are steady or flunctrating. if ```steady```, then ```temporalF``` is not used.
* ```amplitude```: Input limit of the stimulus parameters
* ```spatialF```, ```temporalF```, ```spatialPhase```: Parameters for the stimulus input eventually. If ```temporalF``` changes, the period to take weight/activity mean, as well as the training parameters in ```training_till_convergence``` section of the network should be changed.


## 5. Output

![sim_dic_structure](https://user-images.githubusercontent.com/91852421/219392866-1532b666-5819-4b24-92ca-5c3b56bb8b12.png)


For training of the network on single instance, the simulator returns a dictionary with the following returning values:

* ```weight_df```: a dataframe showing the synaptic weights from each unique pair of simulation+radian combination. 
* ```selectivity_df```: a dataframe showing the ```mean```, ```the mean of std in each simluation```, and ```the std of mean in each simulation``` for direction selectivity, oriental selectivity, oriental selectivty paper, and activity.
* ```summary_info```: a summarized information for  ```os_rel```, ```ds_rel```, ```os_p_rel```, ```nan_counter```, ```not_eq_counter```, and ```activity_off```. 
* ```meta_data```: configuration of the network

If ```visualization_mode``` is activated, then the network will also plot the following graphs:

* the os, ds, os_p, and activity of different neurons before and after training
* the weight of synapses after training
* the change in activity over time 
* the change in weight over time with target weights marked
* the eventual distribution of neuron activities

meanwhile, it also updates the dictionary with ```loss_value```, which reflects the loss function value, as well as the data for weight and activity plotting. 


## 6. Parameter tuning 

Current method for parameter tuning: Bayesian parameter inference. (Potentially varational bayesian inference in the future) 

After finding the best parameters, the system will save all the tuning parameter attempts into a csv file along with their respective loss function value. And the system will run again on the optimal value set to see the performance.

