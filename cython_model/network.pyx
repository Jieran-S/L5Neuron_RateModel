STUFF = "Hi"  # this is an old hack to help Cython compile

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

import numpy as np
cimport numpy as np
cimport cython
import copy
DTYPE = np.double
FTYPE = np.float
ITYPE = np.int
ctypedef np.double_t DTYPE_t
ctypedef np.float_t FTYPE_t
ctypedef np.int_t ITYPE_t

from libc cimport math
from cpython cimport bool
from libc.math cimport log, sqrt, exp
from libc.stdlib cimport rand, RAND_MAX
from cython.parallel import prange

np.random.seed(42)

def run_simulation(input_cs_steady, input_cc_steady, input_pv_steady, input_sst_steady,
    input_cs_amplitude, input_cc_amplitude, input_pv_amplitude, input_sst_amplitude,
    spatialF, temporalF, spatialPhase,start_time,title):

    # network parameters
    cdef np.ndarray[ITYPE_t, ndim=1] N = p.N
    cdef np.ndarray[FTYPE_t, ndim=2] prob = p.prob
    cdef np.ndarray[FTYPE_t, ndim=2] w_initial = p.w_initial
    cdef float w_noise = p.w_noise

    # input parameters
    cdef np.ndarray[FTYPE_t, ndim=1] amplitude = \
        np.array([input_cs_amplitude, input_cc_amplitude, input_pv_amplitude, input_sst_amplitude])
    cdef np.ndarray[FTYPE_t, ndim=1] steady_input = \
        np.array([input_cs_steady, input_cc_steady, input_pv_steady, input_sst_steady])

    # prepare different orientation inputs
    cdef np.ndarray[FTYPE_t, ndim=1] radians = np.radians(p.degree)

    # Evaluation metrics
    cdef int nan_counter = 0
    cdef int not_eq_counter = 0
    cdef np.ndarray[FTYPE_t, ndim=1] activity_off = [0,0,0,0]
    cdef int tsteps = int((p.Ttau * p.tau) / p.delta_t)
    cdef int sim_number = p.sim_number
    cdef np.ndarray[FTYPE_t, ndim=1] os_mean_all = np.zeros(sim_number, dtype= FTYPE)
    cdef np.ndarray[FTYPE_t, ndim=1] os_std_all = np.zeros(sim_number, dtype= FTYPE)
    cdef np.ndarray[FTYPE_t, ndim=1] ds_mean_all = np.zeros(sim_number, dtype= FTYPE)
    cdef np.ndarray[FTYPE_t, ndim=1] ds_std_all = np.zeros(sim_number, dtype= FTYPE)
    cdef np.ndarray[FTYPE_t, ndim=1] os_paper_mean_all = np.zeros(sim_number, dtype= FTYPE)
    cdef np.ndarray[FTYPE_t, ndim=1] os_paper_std_all = np.zeros(sim_number, dtype= FTYPE)
    cdef np.ndarray[FTYPE_t, ndim=2] a_mean_all = np.zeros((sim_number,4), dtype= FTYPE)
    cdef np.ndarray[FTYPE_t, ndim=2] a_std_all = np.zeros((sim_number,4), dtype= FTYPE)
    cdef np.ndarray[FTYPE_t, ndim=2] W_rec
    cdef int num_neurons
    cdef np.ndarray[FTYPE_t, ndim=2] W_project_initial
    cdef np.ndarray[FTYPE_t, ndim=1] initial_values
    cdef np.ndarray[FTYPE_t, ndim=2] activity_data
    cdef int success
    cdef int g
    cdef np.ndarray[FTYPE_t, ndim=1] inputs
    cdef np.ndarray[FTYPE_t, ndim=2] w
    cdef np.ndarray[FTYPE_t, ndim=2] a1
    cdef np.ndarray[FTYPE_t, ndim=2] a2
    cdef np.ndarray[FTYPE_t, ndim=1] mean1
    cdef np.ndarray[FTYPE_t, ndim=1] mean2
    cdef int check_eq
    cdef np.ndarray[FTYPE_t, ndim=1] a_mean_data = np.zeros(4)
    cdef np.ndarray[FTYPE_t, ndim=1] a_std_data = np.zeros(4)
    cdef np.ndarray[FTYPE_t, ndim=1] a_std_sim_data = np.zeros(4)

    cdef np.ndarray[FTYPE_t, ndim=1] os_mean_data = np.zeros(4)
    cdef np.ndarray[FTYPE_t, ndim=1] os_std_data = np.zeros(4)
    cdef np.ndarray[FTYPE_t, ndim=1] os_std_sim_data = np.zeros(4)

    cdef np.ndarray[FTYPE_t, ndim=1] ds_mean_data = np.zeros(4)
    cdef np.ndarray[FTYPE_t, ndim=1] ds_std_data = np.zeros(4)
    cdef np.ndarray[FTYPE_t, ndim=1] ds_std_sim_data = np.zeros(4)

    cdef np.ndarray[FTYPE_t, ndim=1] os_paper_mean_data = np.zeros(4)
    cdef np.ndarray[FTYPE_t, ndim=1] os_paper_std_data = np.zeros(4)
    cdef np.ndarray[FTYPE_t, ndim=1] os_paper_std_sim_data = np.zeros(4)

    cdef float os_rel = 0
    cdef float ds_rel = 0
    cdef float os_paper_rel = 0

    cdef np.ndarray[FTYPE_t, ndim=1] res

    cdef np.ndarray[FTYPE_t, ndim=1] selectivity_data_i
    cdef int d_i
    cdef np.ndarray[FTYPE_t, ndim=1] row = np.zeros(66)

    ################## iterate through different initialisations ##################
    cdef int sim
    for sim in prange(sim_number, nogil=True):
        # weights
        W_rec = generate_connectivity(N, prob, w_initial, w_noise)

        # eye matrix
        num_neurons = W_rec.shape[0]
        W_project_initial = np.eye(num_neurons)

        # initial activity
        initial_values = np.random.uniform(low=0, high=1, size=(sum(N),))

        activity_data = np.zeros((len(radians),tsteps), dtype= FTYPE)
        success = 0

        ################## iterate through different inputs ##################
        for g in radians:

            # build network here
            Sn = nm.SimpleNetwork(W_rec, W_project=W_project_initial, nonlinearity_rule=p.nonlinearity_rule,
                                  integrator=p.integrator, delta_t=p.delta_t, tau=p.tau, Ttau=p.Ttau,
                                  update_function=p.update_function, learning_rule=p.learning_rule,
                                  gamma=p.gamma)

            # define inputs
            inputs = distributionInput(spatialF=spatialF, temporalF=temporalF,
                                            orientation=g, spatialPhase=spatialPhase, amplitude=amplitude,
                                            T=Sn.tsteps,steady_input=steady_input, N=N)

            # run

            activity_data[g,:], w = Sn.run(inputs, initial_values)

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

        if success:
            # mean and std of activity
            a_mean_all[sim,:] = np.array([np.mean(activity[:, -1500:, :N[0]]),
                                          np.mean(activity[:, -1500:, np.sum(N[:1]):np.sum(N[:2])]),
                                          np.mean(activity[:, -1500:, np.sum(N[:2]):np.sum(N[:3])]),
                                          np.mean(activity[:, -1500:, np.sum(N[:3]):np.sum(N)])])

            a_std_all[sim, :] = np.array([np.std(activity[:, -1500:, :N[0]]),
                                           np.std(activity[:, -1500:, np.sum(N[:1]):np.sum(N[:2])]),
                                           np.std(activity[:, -1500:, np.sum(N[:2]):np.sum(N[:3])]),
                                           np.std(activity[:, -1500:, np.sum(N[:3]):np.sum(N)])])
            """
            # use only reliable cells
            activity_cs = np.mean(activity[:, -1500:, :N[0]], axis=1)
            activity_cc = np.mean(activity[:, -1500:, sum(N[:1]):sum(N[:2])], axis=1)
            activity_pv = np.mean(activity[:, -1500:, sum(N[:2]):sum(N[:3])], axis=1)
            activity_sst = np.mean(activity[:, -1500:, sum(N[:3]):sum(N)], axis=1)



            activity_not_reliable = [activity_cs, activity_cc, activity_pv, activity_sst]

            for popu in prange(len(N), nogil=True):
                if a_mean_all[sim,popu] < 0.0001:
                    activity_off[popu] += 1

            if np.sum(activity_off) == 0:
                res = calculate_selectivity(activity_popu)
                os_mean_all[sim] = res[0]
                os_std_all[sim] = res[1]
                ds_mean_all[sim] = res[2]
                ds_std_all[sim] = res[3]
                os_paper_mean_all[sim] = res[4]
                os_paper_std_all[sim] =res[5]

    # calculate mean of orientation and direction selectivity
    if np.sum(os_mean_all) != 0:
        a_mean_data = np.mean(np.array(a_mean_all), axis=0)
        a_std_data = np.std(np.array(a_mean_all), axis=0)
        a_std_sim_data = np.mean(np.array(a_std_all), axis=0)

        os_mean_data = np.mean(np.array(os_mean_all),axis=0)
        os_std_data = np.std(np.array(os_mean_all), axis=0)
        os_std_sim_data = np.mean(np.array(os_std_all), axis=0)

        ds_mean_data = np.mean(np.array(ds_mean_all), axis=0)
        ds_std_data = np.std(np.array(ds_mean_all), axis=0)
        ds_std_sim_data = np.mean(np.array(ds_std_all), axis=0)

        os_paper_mean_data = np.mean(np.array(os_paper_mean_all), axis=0)
        os_paper_std_data = np.std(np.array(os_paper_mean_all), axis=0)
        os_paper_std_sim_data = np.mean(np.array(os_paper_std_all), axis=0)

        if os_mean_data[1] > 0.00001 and ds_mean_data[1] > 0.00001:
            os_rel = 1 - os_mean_data[0] / os_mean_data[1]
            ds_rel = 1 - ds_mean_data[0] / ds_mean_data[1]
            os_paper_rel = 1 - os_paper_mean_data[0] / os_paper_mean_data[1]

    # collect results here
    row[:10] = np.array([input_cc, input_cs,input_pv, input_sst,
           spatialF,temporalF,spatialPhase,amplitude,
           nan_counter,not_eq_counter])#,activity_off])
    for selectivity_data_i in [os_mean_data, os_std_data, os_std_sim_data,
                    ds_mean_data, ds_std_data, ds_std_sim_data,
                    os_paper_mean_data, os_paper_std_data, os_paper_std_sim_data,
                    a_mean_data, a_std_data, a_std_sim_data]:
        for d_i in range(len(selectivity_data_i)):
            row[11+d_i] = selectivity_data_i[d_i]
    row[-4:] = [os_rel,ds_rel,os_paper_rel,time.time() - start_time]

    # write into csv file
    with open(title, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(list(row))"""

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
        'os_paper_mean1','os_paper_mean2','os_paper_mean3','os_paper_mean4',
        'os_paper_std1','os_paper_std2','os_paper_std3','os_paper_std4',
        'os_paper_std_sim1','os_paper_std_sim2','os_paper_std_sim3','os_paper_std_sim4',
        'a_mean1','a_mean2','a_mean3','a_mean4',
        'a_std1','a_std2','a_std3','a_std4',
        'a_std_sim1','a_std_sim2','a_std_sim3','a_std_sim4',
        'os rel','ds rel','os_paper_rel',
        'time']

f = open(title, 'w')
writer = csv.writer(f)
writer.writerow(row)
f.close()

############### start simulation ###############

start_time = time.time()

run_simulation(input_cs_steady=1,input_cc_steady=0,input_pv_steady=1,input_sst_steady=1,
               input_cs_amplitude=2,input_cc_amplitude=1,input_pv_amplitude=0.9,input_sst_amplitude=0.9,
               spatialF=1,temporalF=1,spatialPhase=1,start_time=start_time,title=title)