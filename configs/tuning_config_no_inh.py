import numpy as np

####### simulation parameters #######
sim_number = 50
jobs_number = 800
name_sim ='no_inh_negative_inp'
update_function = 'version_normal'
integrator = 'forward_euler'
delta_t = 0.01
tau = 0.1
Ttau = 300

####### Network parameters #######
learning_rule = 'none'

# synaptic strength matrix of CS, CC, PV and SST neurons (rows are the presyn cell)
# Campognola 2022 PSP amplitude
w_initial = np.array([[0.27, 0, 0,0],
                       [0.19, 0.24, 0,0],
                       [0,0,0,0],
                       [0,0,0,0]])

# Campognola 2022 PSP amplitude
# connection probability matrix of CS, CC, PV and SST neurons (rows are the presyn cell)
prob = np.array([[0.16, 0, 0,0],
                  [0.09, 0.06, 0,0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]])

# number of CS, CC, PV and SST neurons
N = np.array([45, 275, 46, 34])

w_noise = 0.03 # synaptic weight noise

####### Activation function #######
nonlinearity_rule = 'supralinear'
gamma = 1

####### Input #######
degree = [0, 90, 180, 270]

input_cs_steady = [0,1]
input_cc_steady = [0,1]
input_pv_steady = [0]
input_sst_steady = [0]
input_cs_amplitude = [0,0.125,0.25,0.5,1,2,4]
input_cc_amplitude = [0,0.125,0.25,0.5,1,2,4]
input_pv_amplitude = [0]
input_sst_amplitude = [0]
sst_vary = [1]
pv_vary = [1]

spatialF = [1]
temporalF = [1]
spatialPhase = [1]
cc_cs_weight = [0.19,0,0.0625,0.125,0.25,0.5,1] #np.arange(0,1,0.02)