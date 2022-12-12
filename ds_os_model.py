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
import pickle

# Model config and parameter
import importlib
import Implementation.network_model as nm
import Implementation.helper as helper
import Implementation.visualization as vis
# if len(sys.argv) != 0:
#     p = importlib.import_module(sys.argv[1])
# else:
import configs.debug_config as p

np.random.seed(42)

'''
 Parameter not in need so far
 input_cs_steady, input_cc_steady, input_pv_steady, input_sst_steady,
 input_cs_amplitude, input_cc_amplitude, input_pv_amplitude, input_sst_amplitude, 
'''
def run_simulation(Amplitude, Steady_input, spatialF, temporalF, spatialPhase, 
                    learning_rule, number_steps_before_learning, Ttau, evaluation_mode,
                    tau, 
                    tau_learn, 
                    tau_threshold):
    """
    not_before = 0
    if not(input_cs_steady==0 and input_cc_steady==0 and input_pv_steady==0 and input_sst_steady==0):
        if not(input_cs_steady == 0 and input_cc_steady == 0 and input_pv_steady == 0 and input_sst_steady == 1):
            if not(input_cs_steady == 0 and input_cc_steady == 0 and input_pv_steady == 1 and input_sst_steady == 0):
                if not(input_cs_steady == 0 and input_cc_steady == 0 and input_pv_steady == 1 and input_sst_steady == 1):
                    if not (input_cs_steady == 0 and input_cc_steady == 1 and input_pv_steady == 0 and input_sst_steady == 0):
                        not_before = 1
    """

    ########## network parameters ##########
    N = p.N
    prob = p.prob
    w_target = p.w_target
    w_noise = p.w_noise

    ########## Input tuning parameters ##########
    amplitude = Amplitude
    steady_input = Steady_input

    ########## directional variation input ##########
    degree = p.degree
    radians = []
    for i in degree:
        radians.append(math.radians(i))
    
    ########## selectivity evaluation ##########
    nan_counter, not_eq_counter = 0, 0
    activity_off = [0,0,0,0]
    os_rel, ds_rel, os_paper_rel = None, None, None
    os_mean_all, os_std_all, ds_mean_all, ds_std_all, os_paper_mean_all, os_paper_std_all, a_mean_all, a_std_all = \
        [], [], [], [], [], [], [], []

    ################## iterate through different initialisations ##################
    all_activity_data = []
    all_weights_data = []
    for sim in range(p.sim_number):
        
        activity_data = []
        weights_data = []

        ########## weight initialization ##########
        w_initial = np.abs(w_target)
        if p.neurons == 'excit_only':
            train_neuron_type = [0,1]
        elif p.neurons == 'inhibit_only':
            train_neuron_type = [2,3]
        else:
            train_neuron_type = [0,1,2,3]

        for i in train_neuron_type:
            for j in train_neuron_type:
                w_initial[i,j] = abs(np.random.normal(w_target[i,j], scale= 0.1*abs(w_target[i,j]))) 

        # TODO:Trying total ramdomization
        # w_initial = np.random.uniform(low=0, high=1, size=(4,4))
        w_initial[-2:,] *= -1
        
        ########## network initialization ##########
        W_rec = helper.generate_connectivity(N, prob, w_initial, w_noise)
        W_rec = W_rec/max(np.linalg.eigvals(W_rec).real)

        # weight matrix format initialization
        W_project_initial = np.eye(W_rec.shape[0])

        ########## activity initialization ##########
        initial_values = np.random.uniform(low=0, high=1, size=(np.sum(N),)) 

        ########## stimulus input initialization ##########
        length = np.random.uniform(0, 1, (np.sum(N),))
        angle = np.pi * np.random.uniform(0, 2, (np.sum(N),))
        a_data = np.sqrt(length) * np.cos(angle)
        b_data = np.sqrt(length) * np.sin(angle)

        # selectivity measurement       
        success = 0

        ################## iterate through different inputs ##################
        for deg,g in enumerate(radians):
            # build network here
            Sn = nm.SimpleNetwork(W_rec, W_project=W_project_initial, nonlinearity_rule=p.nonlinearity_rule,
                                    integrator=p.integrator, delta_t=p.delta_t, number_steps_before_learning = number_steps_before_learning, 
                                    update_function=p.update_function, learning_rule=learning_rule,
                                    gamma=p.gamma, N = p.N, 
                                    neurons= p.neurons, 
                                    #parameters to tune
                                    tau=tau, 
                                    Ttau=Ttau, 
                                    tau_learn=tau_learn, 
                                    tau_threshold=tau_threshold,
                                    phase_list=p.phase_list,
                                    degree = degree[deg])
            
            # define inputs
            inputs = helper.distributionInput_negative(a_data=a_data, b_data=b_data,
                                        spatialF=spatialF, temporalF=temporalF, orientation=g,
                                        spatialPhase=spatialPhase, amplitude=amplitude, T=Sn.tsteps,
                                        steady_input=steady_input, N=N)

            # run simulation
            activity, weights, steps = Sn.run(inputs, initial_values, simulate_till_converge = True)
            
            # OPTIONAL: Slice the matrix within time scale to 1/2
            # OPTIONAL: Taking only the last tsteps information 
            activity = np.asarray(activity)
            weights = np.asarray(weights)

            # check nan
            if np.isnan(activity[-1]).all():
                if evaluation_mode == True:
                    print(f'nan exist in sim:{sim}: ', np.count_nonzero(~np.isnan(activity[-1])))
                # assign the value such that it is plotable
                activity[-1][np.isnan(activity[-1])] = 1 
                nan_counter += 1 
                # break

            # for hyperparameter tuning turn evaluation_mode to False
            # check equilibrium
            a1 = activity[-50:-25, :]
            a2 = activity[-25:, :]
            mean1 = np.mean(a1, axis=0)
            mean2 = np.mean(a2, axis=0)
            check_eq = np.sum(np.where(mean1 - mean2 < 0.05, np.zeros(np.sum(N)), 1))
            if check_eq > 0:
                not_eq_counter += 1
                # break
                if evaluation_mode == True:
                    ...
                    # print(f'Simulation {sim}, degree {degree[deg]} not converged: {int(check_eq)} neurons, {steps} steps')
            elif evaluation_mode == True: 
                # print(f'Simulation {sim}, degree {degree[deg]} converged. {steps} steps')
                ...

            if g == radians[-1]:
                success = 1
                all_activity_data.append(activity)
                all_weights_data.append(weights) 
            
            weights_data.append(weights)
            activity_data.append(activity)
        
        activity = np.asarray(activity_data)    # activity: (radians, timesteps, neurons)
        weights = np.asarray(weights_data)      # weight:   (radians, timesteps, postsyn, presyn)
        
        # TODO: Only attach the mean activities and change the plot activity function
        # all_activity_data.append(activity)      # all_activity_data:    (simulation, radians, timesteps, neurons)
        # all_weights_data.append(weights)        # weight:               (simulation, radians, timesteps, postsyn, presyn))

        ################## selectivity evaluation ##################
        if success: 
            (a_mean, a_std, activity_not_reliable) = helper.dir_eva(activity, N)
            # activity_not_reliable: a list of 4, each of which: (radians, neuron_number)
            
            # mean and std for different neuron types in one simulation
            a_mean_all.append(a_mean)
            a_std_all.append(a_std)
            
            activity_popu = []
            
            """
            iterating over all neuron type, and mean neuron activities wrt different orientation,
            returning a list of 4, each containing only selective neurons activities 
            (len(radians), all_selective_neurons)

            if one type of neuron show no activities at all, add it to activity_off
            """
            for popu in range(len(N)):
                reliable_cells = []

                # iterating through all neuron
                for neuron in range(N[popu]):
                    not_reliable = 0
                    
                    # Test if neuron is active in different stimulus
                    for rad in range(4):
                        if activity_not_reliable[popu][rad, neuron] < 0.0001:
                            not_reliable += 1
                    
                    # Only append neuron if active in at least one direction
                    if not_reliable != 4:
                        reliable_cells.append(activity_not_reliable[popu][:, neuron])
                
                # reliable_cell: (len(radians), all_selective_neurons )
                reliable_cells = np.array(reliable_cells).T

                if len(reliable_cells)>0:
                    activity_popu.append(reliable_cells)
                else:
                    activity_off[popu] += 1 

            if len(activity_popu) == 4:
                # returning the selectivities of the 4 neuron types
                os_mean, os_std, ds_mean, ds_std, os_paper_mean, os_paper_std = helper.calculate_selectivity(activity_popu)
                os_mean_all.append(os_mean)
                os_std_all.append(os_std)
                ds_mean_all.append(ds_mean)
                ds_std_all.append(ds_std)
                os_paper_mean_all.append(os_paper_mean)
                os_paper_std_all.append(os_paper_std)

    # storage of only 0-degree activities
    activity =  np.asarray(all_activity_data)           # activity:    (simulation, radians, timesteps, neurons)
    weights  =  np.asarray(all_weights_data)            # weights:     (simulation, radians, timesteps, postsyn, presyn))

    ################## selectivity evaluation over all simulations ##################
    # all vectors: (sim, neuron type (4))
    if os_mean_all != []:
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
            os_rel = (os_mean_data[0] - os_mean_data[1]) / (os_mean_data[0] + os_mean_data[1])
            ds_rel = (ds_mean_data[0] - ds_mean_data[1]) / (ds_mean_data[0] + ds_mean_data[1])
            os_paper_rel = (os_paper_mean_data[0] - os_paper_mean_data[1]) / (
                        os_paper_mean_data[0] + os_paper_mean_data[1])
    else:
        os_mean_data = os_std_data = ds_mean_data = ds_std_data = os_paper_mean_data = \
        os_paper_std_data = os_std_sim_data = ds_std_sim_data = os_paper_std_sim_data = \
        [0,0,0,0]

    # lists of 4, each repesenting a neuron type
    a_mean_data = np.mean(np.array(a_mean_all), axis=0)
    a_std_data = np.std(np.array(a_mean_all), axis=0)
    a_std_sim_data = np.mean(np.array(a_std_all), axis=0)

    ################## Information and evaluation data storage ##################

    # weight config evaluation
    weight_mat = Sn.weight_eva(weights=weights)
    
    mat_header = ['CS-CS', 'CS-CC', 'CS-PV', 'CS-SST',
                    'CC-CS', 'CC-CC', 'CC-PV', 'CC-SST',
                    'PV-CS', 'PV-CC', 'PV-PV', 'PV-SST',
                    'SST-CS', 'SST-CC', 'SST-PV', 'SST-SST']
    
    indlist = [ 'Mean_weight',  'Std_mean_W',     'Mean_std_W',      
                'Mean_Wdel',    'Std_mean_Wdel',  'Mean_std_Wdel']

    weight_df = pd.DataFrame(weight_mat, columns=mat_header, index = indlist)

    # Selectivity matrix over simulation
    selectivity_header = ['CS', 'CC', 'PV', 'SST']
    selectivity_ind = [ 'Mean_act', 'Std_of_mean',  'Mean_std',      # a_mean_data, a_std_data, a_std_sim_data
                        'Mean_of_OS',  'Std_mean_OS',  'Mean_std_OS', 
                        'Mean_of_DS',  'Std_mean_DS',  'Mean_std_DS', 
                        'Mean_of_OS_p','Std_mean_OS_p','Mean_std_OS_p', 
                        ]
    selectivity_mat = np.concatenate((  a_mean_data,    a_std_data,     a_std_sim_data, 
                                        os_mean_data,   os_std_data,    os_std_sim_data, 
                                        ds_mean_data,   ds_std_data,    ds_std_sim_data,
                                        os_paper_mean_data, os_paper_std_data, os_paper_std_sim_data
                                        )).reshape(-1,4)
    selectivity_df = pd.DataFrame(selectivity_mat, index=selectivity_ind, columns=selectivity_header)

    # final summarization figure
    summary_info = {"Real_OS": os_rel, "Real_DS": ds_rel, "Real_OS_paper": os_paper_rel, 
                    'nan_counter': nan_counter,'not_eq_counter': not_eq_counter, 'activity_off': activity_off}

    # basic configuration dataframe 
    meta_data_header = [   'CC_input', 'CS_input', 'PV_input', 'SST_input', 
                        'tau', 'tau_learn', 'tau_threshold', 
                        "learning_rule", "training_mode", "training_pattern"]

    meta_data_mat = p.amplitude + \
                [Sn.tau] + [Sn.tau_learn] + [Sn.tau_threshold] + \
                [p.learning_rule] + [p.neurons] + [p.phase_key]

    meta_data = pd.DataFrame(meta_data_mat, index= meta_data_header, columns = ['value'])

    # returned product: A dictionary storing all relevant information
    sim_dic = {
        "weight_df":        weight_df,
        "selectivity_df":   selectivity_df,
        "summary_info":     summary_info,
        "meta_data":        meta_data,
        # "weights":          weights, 
        # "activity":         activity,
    }
    
    # plotting the simulation graphs
    if evaluation_mode == True:
        
        # Adding on the loss value form objective function
        sim_dic['loss_value'] = helper.lossfun(sim_dic, config = p)

        # plot activity and weight results
        if p.sim_number <6:
            for isim in range(p.sim_number):
                Sn.plot_activity(activity=activity, sim=isim, saving=False)
                Sn.plot_weights(weights=weights, sim=isim, saving=False)
        else:
            choice = np.random.choice(np.arange(p.sim_number), 2, replace=False)
            for isim in choice:
                Sn.plot_activity(activity=activity, sim=isim, saving=False)
                Sn.plot_weights(weights=weights, sim=isim, saving=False)
        
        # saving the dictionary files
        pkltitle = helper.create_data_dir(config=p)
        filepath = Path(pkltitle)  
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(sim_dic, f)

        # To load the dictionary:
        # with open("path/to/pickle.pkl", 'rb') as f:
        #    loaded_dict = pickle.load(f)

    return sim_dic

def objective(params):
    amplitude = [params['cc'], params['cs'], params['pv'], params['sst']]
    sim_dic = run_simulation( 
                Amplitude= amplitude,
                Steady_input= p.steady_input,
                spatialF=p.spatialF,
                temporalF=p.temporalF,
                spatialPhase=p.spatialPhase,
                learning_rule= p.learning_rule, 
                number_steps_before_learning =p.number_steps_before_learning, 
                Ttau =p.Ttau,
                tau = p.tau, 
                tau_learn = p.tau_learn, 
                tau_threshold=p.tau_threshold,
                evaluation_mode=False) 
    
    # return the loss function value
    lossval = helper.lossfun(sim_dic, config=p)
    return lossval

# loss function to reach a stable configuration for the 
def stable_sim_objective(params): 
    tau_learn, tau_threshold_fac =  params['tau_learn'], params['tau_threshold_fac']
    Sim_dic = run_simulation( 
                Amplitude= p.amplitude,
                Steady_input= p.steady_input,
                spatialF=p.spatialF,
                temporalF=p.temporalF,
                spatialPhase=p.spatialPhase,
                learning_rule= p.learning_rule, 
                number_steps_before_learning =p.number_steps_before_learning, 
                Ttau =p.Ttau,
                tau = p.tau, 
                tau_learn = tau_learn, 
                tau_threshold=tau_threshold_fac*tau_learn,
                evaluation_mode=False) 
    
    # TODO: Change the stable simulation loss function input if needed in the future
    # lossval = helper.Stable_sim_loss(activity=activity, Max_act=20)
    return ...

#%% Simulation started once for trail

if __name__ == "__main__":
    
    # inputing all tunable parameters from the test.config and first run and visualize
    sim_dic = run_simulation( Amplitude= p.amplitude,
                    Steady_input= p.steady_input,
                    spatialF=p.spatialF,
                    temporalF=p.temporalF,
                    spatialPhase=p.spatialPhase,
                    learning_rule= p.learning_rule, 
                    number_steps_before_learning =p.number_steps_before_learning, 
                    Ttau =p.Ttau,
                    tau=p.tau,
                    tau_learn=p.tau_learn,
                    tau_threshold=p.tau_threshold,
                    evaluation_mode=True,)

#%%  Hyper parameter tuning
if __name__ == "__main__":            
    # Hyperpot attempt
    if p.tuning == True: 
        # define domain
        '''
        space = {
            'cs': hyperopt.hp.uniform('cs', 0, 20),
            'cc': hyperopt.hp.uniform('cc', 0, 20),
            'pv': hyperopt.hp.uniform('pv', 0, 20),
            'sst':hyperopt.hp.uniform('sst', 0, 20)
        }
        '''

        space = {
            'tau_learn': hyperopt.hp.uniform('tau_learn', 1000, 2000),
            'tau_threshold_fac': hyperopt.hp.uniform('tau_threshold_fac', 0.001, 1),
        }        

        # define algorithm
        algo_tpe = hyperopt.tpe.suggest
        algo_rand = hyperopt.rand.suggest

        # define history
        trails = hyperopt.Trials()

        # Start hyperparameter search
        tpe_best = hyperopt.fmin(fn=stable_sim_objective, space=space, algo=algo_tpe, trials=trails, 
                        max_evals=50)
#%% Tuning results visualization

        # Printing out results
        print('Minimum loss attained with TPE:    {:.4f}'.format(trails.best_trial['result']['loss']))
        print('\nNumber of trials needed to attain minimum with TPE: {}'
                    .format(trails.best_trial['misc']['idxs']['tau_learn'][0]))
        print('\nBest input set: {}'.format(tpe_best)) 

        # Saving the results into a csv file: still have to see here
        """
        tpe_results = pd.DataFrame({'loss': [x['loss'] for x in trails.results], 
                                    'iteration': trails.idxs_vals[0]['cc'],
                                    'cs': trails.idxs_vals[1]['cs'],
                                    'cc': trails.idxs_vals[1]['cc'],
                                    'pv': trails.idxs_vals[1]['pv'],
                                    'sst': trails.idxs_vals[1]['sst'],}).fillna(method='ffill')
        """
        tpe_results = pd.DataFrame({'loss': [x['loss'] for x in trails.results], 
                                    'iteration': trails.idxs_vals[0]['tau_learn'],
                                    'tau_learn': trails.idxs_vals[1]['tau_learn'],
                                    'tau_threshold': trails.idxs_vals[1]['tau_threshold_fac'],}).fillna(method='ffill')
        # plot and visualize the value trace
        now = datetime.now() # current date and time
        DateFolder = now.strftime('%m_%d')
        if os.path.exists(f'data/{DateFolder}') == False:
            os.makedirs(f'data/{DateFolder}')
        time_id = now.strftime("%m%d_%H:%M")
        Tuning_Title = f'data/{DateFolder}/{p.name_sim}_{time_id}_{p.learning_rule}_TuningResults.csv'

        filepath = Path(Tuning_Title)  
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # saving the hyperparameter tuning profile
        tpe_results.to_csv(filepath, float_format='%.3f')
        
        # plotting the hyperparameter tuning profile 


        # put the best parameter back and see the results
        # Best_amplitude = [tpe_best['cs'], tpe_best['cc'], tpe_best['pv'], tpe_best['sst']]
        (Best_act, Best_weight, _ )= run_simulation( 
                Amplitude= p.amplitude,
                Steady_input= p.steady_input,
                spatialF=p.spatialF,
                temporalF=p.temporalF,
                spatialPhase=p.spatialPhase,
                learning_rule= p.learning_rule, 
                number_steps_before_learning =p.number_steps_before_learning, 
                Ttau =p.Ttau,
                tau = p.tau, 
                tau_learn = tpe_best["tau_learn"], 
                tau_threshold=tpe_best["tau_threshold_fac"]*tpe_best["tau_learn"],
                evaluation_mode=False) 
        
        #Saving the data after running
        with open(f'data/{DateFolder}/Tuning_Tau_Activity_weight.npy', 'wb') as f:
            np.save(f, Best_act)
            np.save(f, Best_weight)
        
    # TODO: Spearmint tuning package