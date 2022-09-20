from os.path import abspath
import sys
sys.path.append(abspath(''))
import numpy as np
from datetime import datetime
import csv
from joblib import Parallel, delayed
import time
import math
import importlib

import Implementation.network_model as nm
from Implementation.helper import distributionInput, generate_connectivity, calculate_selectivity
if len(sys.argv) != 0:
    p = importlib.import_module(sys.argv[1])
else:
    import test_config as p

np.random.seed(42)

def run_simulation(input_cs, input_cc, input_pv, input_sst,
                   spatialF,temporalF,spatialPhase,amplitude,start_time,title):

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
    nan_counter = 0
    not_eq_counter = 0
    activity_off = [0,0,0,0]
    os_rel = None
    ds_rel = None
    os_mean_all, os_std_all, ds_mean_all, ds_std_all = [], [], [], []
    a_mean_all, a_std_all = [], []

    for sim in range(p.sim_number):
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
            a1 = activity[1000:2000, :]
            a2 = activity[2000:3000, :]
            mean1 = np.mean(a1, axis=0)
            mean2 = np.mean(a2, axis=0)
            check_eq = np.sum(np.where(mean1 - mean2 < 0.001, np.zeros(np.sum(N)), 1))
            if check_eq > 0:
                not_eq_counter += 1
                break

            if g == radians[-1]:
                success = 1
            activity_data.append(activity)

        activity = np.array(activity_data)

        if success:
            # activity
            a_mean = [np.mean(activity[:, 1000:, :N[0]]),
                          np.mean(activity[:, 1000:, sum(N[:1]):sum(N[:2])]),
                          np.mean(activity[:, 1000:, sum(N[:2]):sum(N[:3])]),
                          np.mean(activity[:, 1000:, sum(N[:3]):sum(N)])]
            a_std = [np.std(activity[:, 1000:, :N[0]]),
                         np.std(activity[:, 1000:, sum(N[:1]):sum(N[:2])]),
                         np.std(activity[:, 1000:, sum(N[:2]):sum(N[:3])]),
                         np.std(activity[:, 1000:, sum(N[:3]):sum(N)])]

            a_mean_all.append(a_mean)
            a_std_all.append(a_std)

            # only reliable cells
            activity_cs = np.mean(activity[:, 1000:, :N[0]], axis=1)
            activity_cc = np.mean(activity[:, 1000:, sum(N[:1]):sum(N[:2])], axis=1)
            activity_pv = np.mean(activity[:, 1000:, sum(N[:2]):sum(N[:3])], axis=1)
            activity_sst = np.mean(activity[:, 1000:, sum(N[:3]):sum(N)], axis=1)

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
                os_mean, os_std, ds_mean, ds_std = calculate_selectivity(activity_popu)
                os_mean_all.append(os_mean)
                os_std_all.append(os_std)
                ds_mean_all.append(ds_mean)
                ds_std_all.append(ds_std)


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
        if os_mean_data[1] > 0.00001 and ds_mean_data[1] > 0.00001:
            os_rel = 1 - os_mean_data[0] / os_mean_data[1]
            ds_rel = 1 - ds_mean_data[0] / ds_mean_data[1]
    else:
        os_mean_data, os_std_data, ds_mean_data, ds_std_data = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
        os_std_sim_data, ds_std_sim_data = [0, 0, 0, 0], [0, 0, 0, 0]
        a_mean_data, a_std_data, a_std_sim_data = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]

    # collect results here
    row = [input_cc, input_cs,input_pv, input_sst,
           spatialF,temporalF,spatialPhase,amplitude,
           nan_counter,not_eq_counter,activity_off,
           os_mean_data[0],os_mean_data[1],os_mean_data[2],os_mean_data[3],
           os_std_data[0],os_std_data[1],os_std_data[2],os_std_data[3],
           os_std_sim_data[0],os_std_sim_data[1],os_std_sim_data[2],os_std_sim_data[3],
           ds_mean_data[0],ds_mean_data[1],ds_mean_data[2],ds_mean_data[3],
           ds_std_data[0],ds_std_data[1],ds_std_data[2],ds_std_data[3],
           ds_std_sim_data[0],ds_std_sim_data[1],ds_std_sim_data[2],ds_std_sim_data[3],
           a_mean_data[0], a_mean_data[1], a_mean_data[2], a_mean_data[3],
           a_std_data[0], a_std_data[1], a_std_data[2], a_std_data[3],
           a_std_sim_data[0], a_std_sim_data[1], a_std_sim_data[2], a_std_sim_data[3],
           os_rel,ds_rel,
           time.time() - start_time]

    # write into csv file
    with open(title, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)

############### prepare csv file ###############

now = datetime.now() # current date and time
time_id = now.strftime("%m:%d:%Y_%H:%M:%S")
title = 'data/' + p.name_sim + time_id + '.csv'

row = ['input_cs', 'input_cc','input_pv', 'input_sst',
        'spatialF','temporalF','spatialPhase','amplitude',
        'nan_counter','not_eq_counter','activity_off',
        'os_mean1','os_mean2','os_mean3','os_mean4',
        'os_std1','os_std2','os_std3','os_std4',
        'os_std_sim1','os_std_sim2','os_std_sim3','os_std_sim4',
        'ds_mean1','ds_mean2','ds_mean3','ds_mean4',
        'ds_std1','ds_std2','ds_std3','ds_std4',
        'ds_std_sim1','ds_std_sim2','ds_std_sim3','ds_std_sim4',
        'a_mean1','a_mean2','a_mean3','a_mean4',
        'a_std1','a_std2','a_std3','a_std4',
        'a_std_sim1','a_std_sim2','a_std_sim3','a_std_sim4',
        'os rel','ds rel',
        'time']

f = open(title, 'w')
writer = csv.writer(f)
writer.writerow(row)
f.close()

############### start simulation ###############

start_time = time.time()

# use joblib to parallelize simulations with different parameter values
Parallel(n_jobs=p.jobs_number)(delayed(run_simulation)(input_cs, input_cc, input_pv, input_sst,
                                                       spatialF,temporalF,spatialPhase,amplitude,start_time,title)
                    for input_cs in p.input_cs
                    for input_cc in p.input_cc
                    for input_pv in p.input_pv
                    for input_sst in p.input_sst
                    for spatialF in p.spatialF
                    for temporalF in p.temporalF
                    for spatialPhase in p.spatialPhase
                    for amplitude in p.amplitude)