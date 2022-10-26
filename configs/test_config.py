import numpy as np
np.random.seed(42)

####### simulation parameters #######
sim_number = 10
jobs_number = 4
name_sim ='Test_Evaluation'
update_function = 'version_normal'
integrator = 'forward_euler'

# Simulation Config 
delta_t = 0.02
number_steps_before_learning = 500

# Learning rule hyperparameter
# tau: BCM and activity update time scale 
# Ttau: Total time simulation 
# tau_threshold: Total time for 
tau = 0.1
Ttau = 2000
tau_threshold = 1000

####### Network parameters #######
learnlist = ['None','BCM','Slide_BCM']      #'Simple_test'
learning_rule = learnlist[1]
 
# synaptic strength matrix of CS, CC, PV and SST neurons (rows are the presyn cell)
# Campognola 2022 PSP amplitude
'''
w_initial = np.array( [[0.27, 0, 1.01, 0.05],
                       [0.19, 0.24, 0.48, 0.09],
                       [-0.32, -0.52, -0.47, -0.44],
                       [-0.19, -0.11, -0.18, -0.19]])
'''

# Change the weight into random input
W_1 = np.random.rand(2,4)
W_2 = - np.random.rand(2,4)
w_initial = np.concatenate([W_1, W_2], axis=0)

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

# range of the input is unlimited (Change in the scope) 
# Input from another a neuron to another neuron 
input_cs_steady = [0]
input_cc_steady = [0]
input_pv_steady = [1]
input_sst_steady = [1]
input_cs_amplitude = 1
input_cc_amplitude = 1
input_pv_amplitude = 2
input_sst_amplitude = 2
spatialF = 1
temporalF = 1
spatialPhase = 1


# cc_cs_weight = [0.19,0,0.0625,0.125,0.25,0.5,1] #np.arange(0,1,0.02)