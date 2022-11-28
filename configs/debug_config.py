import numpy as np
np.random.seed(50)

'''
Debug configuration, fewer neuron simluated. Testing the parameter searching funciton 
and the hyperparameter tuning toolkit
'''

####### simulation parameters #######
sim_number = 5
# For parallel computing
jobs_number = 4
name_sim = 'Stable_activity'
update_function = 'version_normal'
integrator = 'forward_euler'

# Simulation size config.
# With this delta_t, 2500 can steadily converge the simluation before learning
# Total steps: (Ttau*tau)/delta_t = 250/0.02 = 12500.0
delta_t = 0.01
number_steps_before_learning = 500

# Learning rule hyperparameter
# Ttau*tau: Total simulation duration
# tau: reverse of neuron activity updating scale 
# tau_learn: reverse of weight updating scale
# tau_threshold: reverse of threshold updating scale
# Relationship: Tau > Tau_threshold > Tau_learn (?Threshold updating quicker?), 
#               and delta_t sufficiently small

tau = 0.1
Ttau = 250
tau_learn = 1997.467    # 1963.0376975536815
tau_threshold = 2.1     # 2.7922382927606537

####### Network parameters #######
# Training rule specification
rule_list = ['None','BCM','Slide_BCM']      #'Simple_test'
learning_rule = rule_list[2]

# Training neuron specification
neurons_list = ['excit_only', 'inhibit_only', 'all']
neurons = neurons_list[0]

# Training pattern specification:
# phase_list = np.array([1,2])    # placeholder without training
phase_list =  np.repeat(['CS','CC','Rest'], 10)     #specific pattern


'''
# Change the weight into random input
W_1 = np.random.rand(2,4)
W_2 = - np.random.rand(2,4)
w_initial = np.concatenate([W_1, W_2], axis=0)
'''

# Extract the initial random value from the guassian distribution
w_initial = np.array( [[0.27, 0, 1.01, 0.05],
                       [0.19, 0.24, 0.48, 0.09],
                       [-0.32, -0.52, -0.47, -0.44],
                       [-0.19, -0.11, -0.18, -0.19]])
w_intiial_CCCS = np.array([ [0.27, 0    ],
                            [0.19, 0.24,]])
for i in range(2):
    for j in range(2):
        w_initial[i,j] = abs(np.random.normal(w_intiial_CCCS[i,j], scale= 0.25)) 

# synaptic strength matrix of CS, CC, PV and SST neurons 
# (rows are the presyn cell)
# Campognola 2022 PSP amplitude
# Row: pre-syn; Col: post-syn
w_target = np.array( [[0.27, 0, 1.01, 0.05],
                       [0.19, 0.24, 0.48, 0.09],
                       [-0.32, -0.52, -0.47, -0.44],
                       [-0.19, -0.11, -0.18, -0.19]])

W_compare = np.array([  [ 0.01459867,  0.        ,  0.06143608,  0.00388622],
                        [ 0.00577864,  0.00486622,  0.03568564,  0.00790761],
                        [-0.04649947, -0.06677541, -0.07941407, -0.02081662],
                        [-0.0333877 , -0.00483243, -0.01764006, -0.00642071]])


# Campognola 2022 PSP amplitude: https://www.science.org/doi/epdf/10.1126/science.abj5861 
# connection probability matrix of CS, CC, PV and SST neurons (rows are the presyn cell)
prob = np.array([[0.16, 0, 0.18, 0.23],
                  [0.09, 0.06, 0.22, 0.26],
                  [0.43, 0.38, 0.5, 0.14],
                  [0.52, 0.13, 0.29, 0.1]])

# number of CS, CC, PV and SST neurons 
N = np.array([45, 275, 46, 34])

w_noise = 0.03 # synaptic weight noise

####### Activation function #######
nonlinearity_rule = 'supralinear'
gamma = 1
 
####### Input #######
degree = 0 

'''
# In case of the future when there is rotation.
steady_input = np.random.choice([0,1], size=(4,), replace=True)
'''
# Currently we kept all the same steady input
steady_input = [0,0,0,0] #[1,1,1,1]
amplitude = [6,5,10,5]

# Hyperparameters for testing
domain = np.linspace(0, 20, num= 100)
# amplitude = np.random.choice(domain, size=(4,), replace=True) 
spatialF = 10
# If change temporalF, need to change:
# convergence check method, 
# mean for plotting peroid, 
# convergence input_step, 
# break condition for step limit
temporalF = 50
spatialPhase = 1

# loss function parameter
Max_act = 15
tuning = True
############# parameter not in use #################
