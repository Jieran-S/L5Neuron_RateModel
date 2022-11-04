#%%
# Path and file save environment
from os.path import abspath
from pathlib import Path  
import os
import sys
from datetime import datetime
import csv
import math

from sympy import N
sys.path.append(abspath(''))

# Simulation, hyperparameter tuning
import numpy as np
import pandas as pd
import hyperopt 
from joblib import Parallel, delayed

# Model config and parameter
import importlib
import Implementation.network_model as nm
import Implementation.helper as helper
# if len(sys.argv) != 0:
#     p = importlib.import_module(sys.argv[1])
# else:
import configs.debug_config as p

np.random.seed(42)

'''
# Parameter not in need so far
 input_cs_steady, input_cc_steady, input_pv_steady, input_sst_steady,
 input_cs_amplitude, input_cc_amplitude, input_pv_amplitude, input_sst_amplitude, 
'''
def run_simulation(Amplitude, Steady_input, spatialF, temporalF, spatialPhase, 
                    learning_rule, number_steps_before_learning, Ttau, evaluation_mode):
    """
    not_before = 0
    if not(input_cs_steady==0 and input_cc_steady==0 and input_pv_steady==0 and input_sst_steady==0):
        if not(input_cs_steady == 0 and input_cc_steady == 0 and input_pv_steady == 0 and input_sst_steady == 1):
            if not(input_cs_steady == 0 and input_cc_steady == 0 and input_pv_steady == 1 and input_sst_steady == 0):
                if not(input_cs_steady == 0 and input_cc_steady == 0 and input_pv_steady == 1 and input_sst_steady == 1):
                    if not (input_cs_steady == 0 and input_cc_steady == 1 and input_pv_steady == 0 and input_sst_steady == 0):
                        not_before = 1
    """

    # network parameters
    N = p.N
    prob = p.prob
    w_initial = p.w_initial
    # w_initial[1,0] = cc_cs_weight
    w_noise = p.w_noise

    # input parameters (parameter for tuning)
    # amplitude = [input_cs_amplitude, input_cc_amplitude, input_pv_amplitude, input_sst_amplitude]
    # steady_input = [input_cs_steady, input_cc_steady, input_pv_steady, input_sst_steady]
    amplitude = Amplitude
    steady_input = Steady_input

    # Evaluation metrics
    nan_counter, not_eq_counter = 0, 0
    activity_off = [0,0,0,0]
    os_rel, ds_rel, os_paper_rel = None, None, None
    os_mean_all, os_std_all, ds_mean_all, ds_std_all, os_paper_mean_all, os_paper_std_all, a_mean_all, a_std_all = \
        [], [], [], [], [], [], [], []

    ################## iterate through different initialisations ##################
    activity_data = []
    weights_data = []
    for sim in range(p.sim_number):

        w_initial = np.concatenate([np.random.rand(4,2), -np.random.rand(4,2)], axis=1)
        # Generating an synaptic matrix that returns the synaptic connections
        W_rec = helper.generate_connectivity(N, prob, w_initial, w_noise)
        W_rec = W_rec/max(np.linalg.eigvals(W_rec).real)

        # randomized weight beginning input
        num_neurons = W_rec.shape[0]
        W_project_initial = np.eye(num_neurons)

        # initial activity
        initial_values = np.random.uniform(low=0, high=1, size=(sum(N),)) 

        success = 0
        a_data = np.cos(np.random.uniform(0, np.pi, (np.sum(N),)))
        b_data = np.sin(np.random.uniform(0, np.pi, (np.sum(N),)))

        ################## iterate through different inputs ##################
        # Change the orientation value? taking g=1 or sth
        # for g in radians:
            # build network here
        g = p.degree
        Sn = nm.SimpleNetwork(W_rec, W_project=W_project_initial, nonlinearity_rule=p.nonlinearity_rule,
                                integrator=p.integrator, delta_t=p.delta_t, tau=p.tau, Ttau=Ttau, 
                                number_steps_before_learning = number_steps_before_learning, 
                                update_function=p.update_function, learning_rule=learning_rule,
                                gamma=p.gamma)
        # define inputs
        inputs = helper.distributionInput_negative(a_data=a_data, b_data=b_data,
                                    spatialF=spatialF, temporalF=temporalF, orientation=g,
                                    spatialPhase=spatialPhase, amplitude=amplitude, T=Sn.tsteps,
                                    steady_input=steady_input, N=N)

        # run
        activity, weights, steps = Sn.run(inputs, initial_values, simulate_till_converge = True)
        
        # Slice the matrix within time scale to 1/2
        # Taking only the last tsteps information, not taking the tuning part in concern 
        activity = np.asarray(activity)[-Sn.tsteps::2]
        weights = np.asarray(weights)[-Sn.tsteps::2]

        # check nan
        if np.isnan(activity[-1]).all():
            nan_counter += 1
            print(f'nan exist in sim:{sim}: ', np.count_nonzero(~np.isnan(activity[-1])))
            # assign the value such that it is plotable
            activity[-1][np.isnan(activity[-1])] = 1 
        if evaluation_mode == True:
            # check equilibrium
            a1 = activity[-100:-50, :]
            a2 = activity[-50:, :]
            mean1 = np.mean(a1, axis=0)
            mean2 = np.mean(a2, axis=0)
            check_eq = np.sum(np.where(mean1 - mean2 < 0.05, np.zeros(np.sum(N)), 1))
            if check_eq > 0:
                not_eq_counter += 1
                print(f'Simulation {sim} not converged: {int(check_eq)} neurons, {steps} steps')
            else: 
                print(f'activity {sim} converges. {steps} steps')

        # Sanity check
        # print(f'weight shape: {weights.shape}, sim: {sim}')
        weights_data.append(weights)
        activity_data.append(activity)
    
    activity = np.asarray(activity_data)
    weights = np.asarray(weights_data)

    # Insert if success part here

    if evaluation_mode == True:

        # Create folder and pathways for data storage
        now = datetime.now() # current date and time
        DateFolder = now.strftime('%m_%d')
        if os.path.exists(f'data/{DateFolder}') == False:
            os.makedirs(f'data/{DateFolder}')
        time_id = now.strftime("%m%d_%H:%M")
        csvtitle = f'data/{DateFolder}/{p.name_sim}_{time_id}_{p.learning_rule}.csv'

        # plot randomly 2 simulation graph
        for isim in np.random.choice(np.arange(sim), 2):
            helper.plot_activity(activity, N, f'data/{DateFolder}',sim = isim, learningrule= learning_rule, Ttau = Ttau)
            helper.plot_weights(weights, N, f'data/{DateFolder}', sim = isim, learningrule= learning_rule, Ttau= Ttau)

        # Evaluation metric 
        (Tvar, Svar, Smean, Avar) = helper.sim_eva(weights=weights, activity=activity, N=N)

        # foming csv files
        mat_header = ['CS-CS', 'CS-CC', 'CS-PV', 'CS-SST',
                        'CC-CS', 'CC-CC', 'CC-PV', 'CC-SST',
                        'PV-CS', 'PV-CC', 'PV-PV', 'PV-SST',
                        'SST-CS', 'SST-CC', 'SST-PV', 'SST-SST']
        sim_char_vec = np.char.mod('%d', np.arange(Tvar.shape[0]))
        list1 = ['Tvar_' + x for x in sim_char_vec]
        list2 = ['Svar_' + x for x in sim_char_vec]
        list3 = ['Mean_' + x for x in sim_char_vec]
        list4 = ['Avar_' + x for x in sim_char_vec]
        indlist = np.concatenate([list1, list2, list3, list4])
        csv_mat = np.concatenate([Tvar, Svar, Smean, np.repeat(Avar, 4, axis=1).reshape(5,4,4)], axis=0).reshape(Tvar.shape[0]*4, -1)
        csv_df = pd.DataFrame(csv_mat, columns=mat_header, index = indlist)

        # saving csv files
        filepath = Path(csvtitle)  
        filepath.parent.mkdir(parents=True, exist_ok=True)
        csv_df.to_csv(filepath,  float_format='%.3f')

        return (activity, weights)
    else:
        return (activity, weights)

def objective(params):
    amplitude = [params['cc'], params['cs'], params['pv'], params['sst']]
    (activity, weights) = run_simulation( Amplitude= amplitude,
                Steady_input= p.steady_input,
                spatialF=p.spatialF,
                temporalF=p.temporalF,
                spatialPhase=p.spatialPhase,
                learning_rule= p.learning_rule, 
                number_steps_before_learning =p.number_steps_before_learning, 
                Ttau =p.Ttau,
                evaluation_mode=False)
    
    (Tvar, Svar, Smean, Avar) = helper.sim_eva(weights=weights, activity= activity, N=p.N)
    lossval = helper.lossfun(Tvar, Svar, Smean, Avar, 
                                Activity= activity, 
                                w_target=p.w_target, 
                                MaxAct=p.Max_act)
    return lossval

#%%
############### start simulation ###############
if __name__ == "__main__":
    
    # inputing all tunable parameters from the test.config
    
    (activity, weights) = run_simulation( Amplitude= p.amplitude,
                    Steady_input= p.steady_input,
                    spatialF=p.spatialF,
                    temporalF=p.temporalF,
                    spatialPhase=p.spatialPhase,
                    learning_rule= p.learning_rule, 
                    number_steps_before_learning =p.number_steps_before_learning, 
                    Ttau =p.Ttau,
                    evaluation_mode=True)
    
  #%%  Hyper parameter tuning
               
    # Hyperpot attempt
    if p.tuning == True: 
        # define domain
        space = {
            'cs': hyperopt.hp.uniform('cs', 0, 20),
            'cc': hyperopt.hp.uniform('cc', 0, 20),
            'pv': hyperopt.hp.uniform('pv', 0, 20),
            'sst':hyperopt.hp.uniform('sst', 0, 20)
        }

        # define algorithm
        algo_tpe = hyperopt.tpe.suggest
        algo_rand = hyperopt.rand.suggest

        # define history
        trails = hyperopt.Trials()

        # Start hyperparameter search
        tpe_best = hyperopt.fmin(fn=objective, space=space, algo=algo_tpe, trials=trails, 
                        max_evals=20)
#%% Tuning results visualization

        # Printing out results
        print('Minimum loss attained with TPE:    {:.4f}'.format(trails.best_trial['result']['loss']))
        print('\nNumber of trials needed to attain minimum with TPE:    {}'.format(trails.best_trial['misc']['idxs']['cc'][0]))
        print('\nBest input set: {}'.format(tpe_best)) 

        # Saving the results into a csv file: still have to see here
        tpe_results = pd.DataFrame({'loss': [x['loss'] for x in trails.results], 
                                    'iteration': trails.idxs_vals[0]['cc'],
                                    'cs': trails.idxs_vals[1]['cs'],
                                    'cc': trails.idxs_vals[1]['cc'],
                                    'pv': trails.idxs_vals[1]['pv'],
                                    'sst': trails.idxs_vals[1]['sst'],}).fillna(method='ffill')
        # plot and visualize the value trace

        # put the best parameter back and see the results
        Best_amplitude = [tpe_best['cs'], tpe_best['cc'], tpe_best['pv'], tpe_best['sst']]
        run_simulation( Amplitude= Best_amplitude,
                        Steady_input= p.steady_input,
                        spatialF=p.spatialF,
                        temporalF=p.temporalF,
                        spatialPhase=p.spatialPhase,
                        learning_rule= p.learning_rule, 
                        number_steps_before_learning =p.number_steps_before_learning, 
                        Ttau =p.Ttau,
                        evaluation_mode=True)

    # Spearmint attempt

    """
input_cs_steady=0,
                input_cc_steady=0,
                input_pv_steady=1,
                input_sst_steady=1,
                input_cs_amplitude=p.input_cs_amplitude,
                input_cc_amplitude=p.input_cc_amplitude,
                input_pv_amplitude=p.input_pv_amplitude,
                input_sst_amplitude=p.input_sst_amplitude,

# use joblib to parallelize simulations with different parameter values

Parallel(n_jobs=p.jobs_number)(delayed(run_simulation)(input_cs_steady, input_cc_steady, input_pv_steady,
                                                       input_sst_steady,input_cs_amplitude, input_cc_amplitude,
                                                       input_pv_amplitude, input_sst_amplitude,cc_cs_weight,
                                                       spatialF,temporalF, spatialPhase,start_time,title)
                    for input_cs_steady in p.input_cs_steady
                    for input_cc_steady in p.input_cc_steady
                    for input_pv_steady in p.input_pv_steady
                    for input_sst_steady in p.input_sst_steady
                    for input_cs_amplitude in p.input_cs_amplitude
                    for input_cc_amplitude in p.input_cc_amplitude
                    for input_pv_amplitude in p.input_pv_amplitude
                    for input_sst_amplitude in p.input_sst_amplitude
                    for cc_cs_weight in p.cc_cs_weight
                    for spatialF in p.spatialF
                    for temporalF in p.temporalF
                    for spatialPhase in p.spatialPhase)
"""