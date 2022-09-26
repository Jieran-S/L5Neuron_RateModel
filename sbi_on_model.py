from os.path import abspath
import sys
sys.path.append(abspath(''))
import numpy as np
import math
import os

import Implementation.network_model as nm
from Implementation.helper import distributionInput_sbi, generate_connectivity, \
    calculate_selectivity_sbi
import configs.test_config_sbi as p

def run_simulation(params):
    [input_cs_steady,input_cc_steady,input_pv_steady,input_sst_steady,
                input_cs_amplitude,input_cc_amplitude,input_pv_amplitude,input_sst_amplitude,
                spatialF,temporalF,spatialPhase] = params.tolist()

    # network parameters
    N = p.N
    prob = p.prob
    w_initial = p.w_initial
    w_noise = p.w_noise

    # input parameters
    amplitude = [input_cs_amplitude,input_cc_amplitude,input_pv_amplitude,input_sst_amplitude]
    steady_input = [input_cs_steady,input_cc_steady,input_pv_steady,input_sst_steady]

    # prepare different orientation inputs
    degree = p.degree
    radians = []
    for i in degree:
        radians.append(math.radians(i))

    # Evaluation metrics
    nan_counter, not_eq_counter = 0, 0
    os_rel, ds_rel, ds_paper_rel = 0,0,0
    os_mean_all, ds_mean_all, ds_paper_mean_all, n_rel_all = [], [], [], []

    ################## iterate through different initialisations ##################
    for sim in range(p.sim_number):
        title_folder = 'data/figures/sbi'+str(sim)
        if not(os.path.exists(title_folder)):
            os.mkdir(title_folder)

        # weights
        W_rec = generate_connectivity(N, prob, w_initial, w_noise)

        # eye matrix
        num_neurons = W_rec.shape[0]
        W_project_initial = np.eye(num_neurons)

        # initial activity
        initial_values = np.random.uniform(low=0, high=1, size=(sum(N),))

        activity_data = []
        success = 0

        ################## iterate through different inputs ##################
        for g in radians:
            # build network here
            Sn = nm.SimpleNetwork(W_rec,W_project=W_project_initial,nonlinearity_rule=p.nonlinearity_rule,
                                  integrator=p.integrator,delta_t=p.delta_t,tau=p.tau,Ttau=p.Ttau,
                                  update_function = p.update_function,learning_rule = p.learning_rule,
                                  gamma = p.gamma)

            # define inputs
            inputs = distributionInput_sbi(spatialF=spatialF,temporalF=temporalF,orientation=g,
                            spatialPhase=spatialPhase,amplitude=amplitude,T=Sn.tsteps,steady_input=steady_input,N = N)

            # run
            activity, w = Sn.run(inputs, initial_values)
            activity = np.asarray(activity)

            # check nan
            if np.isnan(activity[-1]).all():
                nan_counter += 1
                break

            # check equilibrium
            a1 = activity[-2000:-1000, :]
            a2 = activity[-1000:, :]
            mean1 = np.mean(a1, axis=0)
            mean2 = np.mean(a2, axis=0)
            check_eq = np.sum(np.where(mean1 - mean2 < 0.05, np.zeros(np.sum(N)), 1))
            if check_eq > 0:
                not_eq_counter += 1
                break
            if g == radians[-1]:
                success = 1
            activity_data.append(activity)
        activity = np.array(activity_data)

        if success:
            activity_cs = np.mean(activity[:, -1500:, :N[0]], axis=1)
            activity_cc = np.mean(activity[:, -1500:, sum(N[:1]):sum(N[:2])], axis=1)
            activity_all = [activity_cs, activity_cc]

            # calculate proportion of reliable CS and CC cells (at least one active)
            n_rel = []
            for popu in range(2):
                for neuron in range(N[popu]):
                    n_on = 0
                    for stim in range(4):
                        if activity_all[popu][stim, neuron] > 0.0001:
                            n_on += 1
                n_rel.append(n_on/N[popu])
            n_rel_all.append(n_rel)

            os_mean, ds_mean, ds_paper_mean= calculate_selectivity_sbi(activity_all)
            os_mean_all.append(os_mean)
            ds_mean_all.append(ds_mean)
            ds_paper_mean_all.append(ds_paper_mean)

    # calculate mean of orientation and direction selectivity
    if os_mean_all != []:
        n_rel_data = np.mean(np.array(n_rel_all),axis=0)
        os_mean_data = np.mean(np.array(os_mean_all),axis=0)
        ds_mean_data = np.mean(np.array(ds_mean_all),axis=0)
        ds_paper_mean_data = np.mean(np.array(ds_paper_mean_all), axis=0)

        if np.abs((os_mean_data[0] - os_mean_data[1])) > 0.00001:
            os_rel = (os_mean_data[0] - os_mean_data[1])/(os_mean_data[0] + os_mean_data[1])
        if np.abs((ds_mean_data[0] - ds_mean_data[1])) > 0.00001:
            ds_rel = (ds_mean_data[0] - ds_mean_data[1]) / (ds_mean_data[0] + ds_mean_data[1])
        if np.abs((ds_paper_mean_data[0] - ds_paper_mean_data[1])) > 0.00001:
            ds_paper_rel = \
                (ds_paper_mean_data[0] - ds_paper_mean_data[1]) / (ds_paper_mean_data[0] + ds_paper_mean_data[1])
    else:
        n_rel_data, os_mean_data, ds_mean_data, ds_paper_mean_data = \
            [0,0], [0,0], [0,0], [0,0]
            #[None, None], [None, None], [None, None], [None, None]

    # collect results here
    row = []
    selectivity_data = [os_mean_data, ds_mean_data, ds_paper_mean_data, n_rel_data]
    for selectivity_data_i in selectivity_data:
        for d in selectivity_data_i:
            row.append(d)
    row = row + [os_rel,ds_rel,ds_paper_rel, nan_counter/p.sim_number, not_eq_counter/p.sim_number]
    row = np.array(row)
    print(row)
    return row

############### prepare csv file ###############
#print(run_simulation(input_cs_steady=1,input_cc_steady=0,input_pv_steady=1,input_sst_steady=1,
#                input_cs_amplitude=2,input_cc_amplitude=1,input_pv_amplitude=0.9,input_sst_amplitude=0.9,
#                spatialF=1,temporalF=1,spatialPhase=1))