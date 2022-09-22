from os.path import abspath
import sys
sys.path.append(abspath(''))
import numpy as np
from datetime import datetime
import time
import math
import importlib
import os

import Implementation.network_model as nm
from Implementation.helper import distributionInput, generate_connectivity, \
    calculate_selectivity, plot_activity
if len(sys.argv) != 0:
    p = importlib.import_module(sys.argv[1])
else:
    import test_config as p

np.random.seed(42)

def run_simulation(input_cs, input_cc, input_pv, input_sst,
                   spatialF,temporalF,spatialPhase,amplitude,
                   start_time,title):

    # network parameters
    N = p.N
    prob = p.prob
    w_initial = p.w_initial
    w_noise = p.w_noise

    # prepare different orientation inputs
    degree = p.degree
    radians = []
    for i in degree:
        radians.append(math.radians(i))

    # Evaluation metrics
    nan_counter, not_eq_counter = 0, 0
    activity_off = [0,0,0,0]
    os_rel, ds_rel, ds_paper_rel = None, None, None
    os_mean_all, os_std_all, ds_mean_all, ds_std_all, ds_paper_mean_all, ds_paper_std_all, a_mean_all, a_std_all = \
        [], [], [], [], [], [], [], []

    ################## iterate through different initialisations ##################
    for sim in range(p.sim_number):
        # Folder for figures
        title_i = title + str(sim)
        path = 'data/figures/' + title_i
        os.mkdir(path)

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
            print('#######################')
            # build network here
            Sn = nm.SimpleNetwork(W_rec,
                                  W_project=W_project_initial,
                                  nonlinearity_rule=p.nonlinearity_rule,
                                  integrator=p.integrator,
                                  delta_t=p.delta_t,
                                  tau=p.tau,
                                  Ttau=p.Ttau,
                                  update_function = p.update_function,
                                  learning_rule = p.learning_rule,
                                  gamma = p.gamma)

            # define inputs
            inputs = distributionInput(spatialF=spatialF,
                                       temporalF=temporalF,
                                       orientation=g,
                                       spatialPhase=spatialPhase,
                                       amplitude=amplitude,
                                       T=Sn.tsteps,
                                       steady=p.steady_input,
                                       input_cs = input_cs,
                                       input_cc = input_cc,
                                       input_pv = input_pv,
                                       input_sst = input_sst,
                                       N = N)

            # run
            activity, w = Sn.run(inputs, initial_values)
            activity = np.asarray(activity)

            # check nan
            if np.isnan(activity[-1]).all():
                nan_counter += 1
                break

            # check equilibrium
            a1 = activity[-3000:-1500, :]
            a2 = activity[-1500:, :]
            mean1 = np.mean(a1, axis=0)
            mean2 = np.mean(a2, axis=0)
            check_eq = np.sum(np.where(mean1 - mean2 < 0.05, np.zeros(np.sum(N)), 1))
            if check_eq > 0:
                not_eq_counter += 1
                print('Not eq')
            else:
                print('Eq')

            if g == radians[-1]:
                success = 1
            activity_data.append(activity)

        # plot activity and split activity for each population
        activity = np.array(activity_data)
        print('activity shape', activity.shape)
        plot_activity(activity, N, title_i)


        if success:
            # mean and std of activity
            a_mean = [np.mean(activity[:, -1000:, :N[0]]),
                          np.mean(activity[:, -1000:, sum(N[:1]):sum(N[:2])]),
                          np.mean(activity[:, -1000:, sum(N[:2]):sum(N[:3])]),
                          np.mean(activity[:, -1000:, sum(N[:3]):sum(N)])]
            a_std = [np.std(activity[:, -1000:, :N[0]]),
                         np.std(activity[:, -1000:, sum(N[:1]):sum(N[:2])]),
                         np.std(activity[:, -1000:, sum(N[:2]):sum(N[:3])]),
                         np.std(activity[:, -1000:, sum(N[:3]):sum(N)])]

            a_mean_all.append(a_mean)
            a_std_all.append(a_std)

            # use only reliable cells

            activity_cs = np.mean(activity[:, -1000:, :N[0]], axis=1)
            activity_cc = np.mean(activity[:, -1000:, sum(N[:1]):sum(N[:2])], axis=1)
            activity_pv = np.mean(activity[:, -1000:, sum(N[:2]):sum(N[:3])], axis=1)
            activity_sst = np.mean(activity[:, -1000:, sum(N[:3]):sum(N)], axis=1)
            activity_not_reliable = [activity_cs, activity_cc, activity_pv, activity_sst]

            activity_popu = []
            for popu in range(len(N)):
                reliable_cells = []
                for neuron in range(N[popu]):
                    not_reliable = 0
                    for stim in range(4):
                        if activity_not_reliable[popu][stim, neuron] < 0.0001:
                            not_reliable += 1
                    if not_reliable != 4:
                        reliable_cells.append(activity_not_reliable[popu][:, neuron])
                reliable_cells = np.array(reliable_cells).T
                if len(reliable_cells)>0:
                    activity_popu.append(reliable_cells)
                else:
                    activity_off[popu] += 1
            if len(activity_popu) == 4:
                os_mean, os_std, ds_mean, ds_std, ds_paper_mean, ds_paper_std = calculate_selectivity(activity_popu)
                os_mean_all.append(os_mean)
                os_std_all.append(os_std)
                ds_mean_all.append(ds_mean)
                ds_std_all.append(ds_std)
                ds_paper_mean_all.append(ds_paper_mean)
                ds_paper_std_all.append(ds_paper_std)

    # calculate mean of orientation and direction selectivity
    if os_mean_all != []:
        a_mean_data = np.mean(np.array(a_mean_all), axis=0)
        a_std_data = np.std(np.array(a_mean_all), axis=0)
        a_std_sim_data = np.mean(np.array(a_std_all), axis=0)

        os_mean_data = np.mean(np.array(os_mean_all),axis=0)
        os_std_data = np.std(np.array(os_mean_all), axis=0)
        os_std_sim_data = np.mean(np.array(os_std_all), axis=0)

        ds_mean_data = np.mean(np.array(ds_mean_all), axis=0)
        ds_std_data = np.std(np.array(ds_mean_all), axis=0)
        ds_std_sim_data = np.mean(np.array(ds_std_all), axis=0)

        ds_paper_mean_data = np.mean(np.array(ds_paper_mean_all), axis=0)
        ds_paper_std_data = np.std(np.array(ds_paper_mean_all), axis=0)
        ds_paper_std_sim_data = np.mean(np.array(ds_paper_std_all), axis=0)

        if os_mean_data[1] > 0.00001 and ds_mean_data[1] > 0.00001:
            os_rel = 1 - os_mean_data[0] / os_mean_data[1]
            ds_rel = 1 - ds_mean_data[0] / ds_mean_data[1]
            ds_paper_rel = 1 - ds_paper_mean_data[0] / ds_paper_mean_data[1]
    else:
        os_mean_data, os_std_data, ds_mean_data, ds_std_data, ds_paper_mean_data, ds_paper_std_data = \
            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
        os_std_sim_data, ds_std_sim_data, ds_paper_std_sim_data = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
        a_mean_data, a_std_data, a_std_sim_data = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]

    # collect results here
    row = [input_cc, input_cs,input_pv, input_sst,
           spatialF,temporalF,spatialPhase,amplitude,
           nan_counter,not_eq_counter,activity_off]
    selectivity_data = [os_mean_data, os_std_data, os_std_sim_data,
                        ds_mean_data, ds_std_data, ds_std_sim_data,
                        ds_paper_mean_data, ds_paper_std_data, ds_paper_std_sim_data,
                        a_mean_data, a_std_data, a_std_sim_data]
    for selectivity_data_i in selectivity_data:
        for d in selectivity_data_i:
            row.append(d)
    row = row + [os_rel,ds_rel,ds_paper_rel,time.time() - start_time]
    return row

############### prepare csv file ###############
for input_cs in p.input_cs:
    for input_cc in p.input_cc:
        for input_pv in p.input_pv:
            for input_sst in p.input_sst:
                for spatialF in p.spatialF:
                    for temporalF in p.temporalF:
                        for spatialPhase in p.spatialPhase:
                            for amplitude in p.amplitude:

                                now = datetime.now() # current date and time
                                title = now.strftime("%m:%d:%Y_%H:%M:%S")
                                start_time = time.time()

                                run_simulation(input_cs, input_cc, input_pv, input_sst,
                                                spatialF,temporalF,spatialPhase,amplitude,
                                                start_time,title)