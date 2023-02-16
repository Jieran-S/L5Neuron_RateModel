import numpy as np
np.random.seed(50)

####### meta data parameters #######

sim_number = 5
jobs_number = 4
name_sim = 'Stable_activity'

########## Time parameters ##########
# Simulation size/Total steps: (Ttau*tau)/delta_t = 250/0.02 = 12500.0
delta_t = 0.01
number_steps_before_learning = 500

"""
Simulation time hyperparameters: 
-   Ttau*tau:       Total simulation duration
-   tau:            reverse of neuron activity updating scale 
-   tau_learn:      reverse of weight updating scale
-   tau_threshold:  reverse of threshold updating scale

Relationship:   Tau > Tau_threshold > Tau_learn  
                delta_t sufficiently small
"""
tau = 0.1
Ttau = 250
tau_learn = 1997.467    # 1963.0376975536815
tau_threshold = 2.7922382927606537

########## Network parameters ##########
# Training rule specification
rule_list = ['None', 'BCM', 'Slide_BCM', 'Oja', 'Cov']     
learning_rule = rule_list[2]

# Training neuron specification
neurons_list = ['excit_only', 'inhibit_only', 'all']
neurons = neurons_list[2]

# Training pattern specification:
phase_patterns = {
    "No_Alter": np.repeat(['all'], 10),
    "CC_CS_Alter": np.repeat(['CS','CC','Rest'], 10),
    "All_Alter": np.repeat(['CS','CC','PV','SST','Rest'], 10)
}

phase_key = list(phase_patterns)[0]
phase_list = phase_patterns[phase_key]                # training all neurons at all time
 
# target synaptic weight matrix of CS, CC, PV and SST neurons 
# Campognola 2022 PSP amplitude: https://www.science.org/doi/epdf/10.1126/science.abj5861 
# Row: pre-syn; Col: post-syn

w_target = np.array( [[0.27, 0, 1.01, 0.05],
                       [0.19, 0.24, 0.48, 0.09],
                       [-0.32, -0.52, -0.47, -0.44],
                       [-0.19, -0.11, -0.18, -0.19]])
# Considering the w_target * probability / largest eigenvalue(for matrix scaling), 
# we need to transform the target weight as below
w_compare = np.array([  [ 0.01459867,  0.        ,  0.06143608,  0.00388622],
                        [ 0.00577864,  0.00486622,  0.03568564,  0.00790761],
                        [-0.04649947, -0.06677541, -0.07941407, -0.02081662],
                        [-0.0333877 , -0.00483243, -0.01764006, -0.00642071]])

# connection probability matrix of CS, CC, PV and SST neurons (presyn, postsyn)
prob = np.array([ [0.16, 0,    0.18, 0.23],
                  [0.09, 0.06, 0.22, 0.26],
                  [0.43, 0.38, 0.5,  0.14],
                  [0.52, 0.13, 0.29, 0.1 ]])

w_initial = np.abs(w_target)

# When only training CC and CS
if neurons == 'excit_only':
    train_neuron_type = [0,1]
elif neurons == 'inhibit_only':
    train_neuron_type = [2,3]
else:
    train_neuron_type = [0,1,2,3]

for i in train_neuron_type:
    for j in train_neuron_type:
        w_initial[i,j] = abs(np.random.normal(w_target[i,j], scale= 0.1*abs(w_target[i,j]))) 

w_initial[-2:,] *= -1

# number of CS, CC, PV and SST neurons 
N = np.array([45, 275, 46, 34])

# synaptic weight noise
w_noise = 0.03 

########## Activation function ##########
nonlinearity_rule = 'supralinear'
gamma = 1
# delta activity generation function
update_function = 'version_normal'
# activity and weight update function
integrator = 'forward_euler'
 
########## Input ##########

# radian of input
degree = [0, 90, 180, 270]

# TODO: If rotational input. (steady_input as random)
# steady_input = np.random.choice([0,1], size=(4,), replace=True)

# Input form and amplitude distributionInput_negative()
steady_input = [0,0,0,0] #[1,1,1,1]
amplitude = [6,5,10,5]
# [17.34637650447213, 3.485229955567148, 6.685986709907562, 3.7992341006167583]
spatialF = 10
temporalF = 50
spatialPhase = 1
"""
If change temporalF, need to change:
-   convergence check method, 
-   mean for plotting peroid, 
-   convergence input_step, 
-   break condition for step limit
"""

########## Hyperparameter tuning ##########
tuning = False

# steady loss function parameter. If not tuning the network for steady performance, no need to use this parameter
Max_act = 15
