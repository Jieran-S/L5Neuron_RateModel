import numpy as np

####### simulation parameters #######
sim_number = 5
jobs_number = 4
name_sim ='os_ds_'
update_function = 'version_normal'
integrator = 'forward_euler'
delta_t = 0.01
tau = 0.1
Ttau = 2500

####### Network parameters #######
learning_rule = 'none'

# synaptic strength matrix of CS, CC, PV and SST neurons (rows are the presyn cell)
# Campognola 2022 PSP amplitude
w_initial = np.array([[0.27, 0, 1.01, 0.05],
                       [0.19, 0.24, 0.48, 0.09],
                       [-0.32, -0.52, -0.47, -0.44],
                       [-0.19, -0.11, -0.18, -0.19]])

# Campognola 2022 PSP amplitude
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
# [0.0625,0.125,0.25,0.5,1,2,4,8,16]
input_cs = [2]  # bar means moving bar input
input_cc = ['bar']
input_pv = [0.9]
input_sst = [0.9]
degree = [0, 90, 180, 270]
spatialF = [1]
temporalF = [1]
spatialPhase = [1]
amplitude = [1]
steady_input = 0