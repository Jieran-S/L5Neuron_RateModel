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

# Simulation, visualization and hyperparameter tuning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import hyperopt 
from joblib import Parallel, delayed

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
'''
def run_simulation(Amplitude, Steady_input, spatialF, temporalF, spatialPhase, 
                    learning_rule, Ttau, visualization_mode,
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
    activity_off = np.array([0,0,0,0])
    os_mean_all, os_std_all, ds_mean_all, ds_std_all, os_paper_mean_all, os_paper_std_all, a_mean_all, a_std_all = \
        [], [], [], [], [], [], [], []

    ########## data storage structure ##########
    """
    all_activity/weights_plot_list: store all activities/weights data across simulations for plotting
    weights_eva_list: store last 50 steps of weights across simulation for evaluation
    ini_weights_list: store initial weights info across all simulation for evaluation
    """
    activity_plot_list = []     # (sim, radians, timestep/n, neurons)
    weights_plot_list = []      # (sim * radians, timestep/n, postsyn, presyn)
    weights_eva_list = []       # (sim * radians, 60, postsyn, presyn)
    ini_weights_list = []           # (sim * radians, postsyn, presyn)

    ################## iterate through different initialisations ##################
    for sim in range(p.sim_number):

        # store activities per simulation for intra-simulation evaluation
        activity_eval_sim = []  # (radians, timestep/n, neurons)

        ########## weight initialization ##########
        w_initial = np.abs(w_target)
        train_neuron_type = [[0,1],[2,3],[0,1,2,3]][p.neurons_list.index(p.neurons)]

        for i in train_neuron_type:
            for j in train_neuron_type:
                w_initial[i,j] = abs(np.random.normal(w_target[i,j], scale= 0.1*abs(w_target[i,j]))) 

        # TODO:Trying total ramdomization in the future 
        # w_initial = np.random.uniform(low=0, high=1, size=(4,4))
        w_initial[-2:,] *= -1
        
        ########## activity initialization ##########
        initial_values = np.random.uniform(low=0, high=1, size=(np.sum(N),)) 

        ########## stimulus input initialization ##########
        length = np.random.uniform(0, 1, (np.sum(N),))
        angle = np.pi * np.random.uniform(0, 2, (np.sum(N),))
        a_data = np.sqrt(length) * np.cos(angle)
        b_data = np.sqrt(length) * np.sin(angle)

        # selectivity measurement       
        success = 0

        ########## network initialization ##########
        W_rec = helper.generate_connectivity(N, prob, w_initial, w_noise)
        W_rec = W_rec/max(np.linalg.eigvals(W_rec).real)

        # weight matrix format initialization
        W_project_initial = np.eye(W_rec.shape[0])

        # build the network
        Sn = nm.SimpleNetwork(W_rec, W_project=W_project_initial, nonlinearity_rule=p.nonlinearity_rule,
                                integrator=p.integrator, delta_t=p.delta_t, 
                                update_function=p.update_function, 
                                number_steps_before_learning = p.number_steps_before_learning, 
                                gamma=p.gamma, N = p.N, 
                                neurons= p.neurons, 
                                phase_list=p.phase_list,
                                #parameters to tune
                                learning_rule=learning_rule,
                                tau=tau, 
                                Ttau=Ttau, 
                                tau_learn=tau_learn, 
                                tau_threshold=tau_threshold)

        ####################### iterate through different inputs #######################
        for deg,g in enumerate(radians):
            
            # define inputs
            inputs = helper.distributionInput_negative(a_data=a_data, b_data=b_data, orientation=g, T=Sn.tsteps,N=N,
                                                        # Tuning parameters below
                                                        spatialF=spatialF, 
                                                        temporalF=temporalF,
                                                        spatialPhase=spatialPhase,
                                                        amplitude=amplitude,
                                                        steady_input=steady_input,   )

            # run simulation: raw_activity:(tstep, neurons); raw_weights:(tstep, postsyn, presyn)
            raw_activity, raw_weights = Sn.run(inputs, initial_values, simulate_till_converge = True)
            
            ############ data quality checking ############

            # mean period of input, change according to termporalF
            n = 25
            # for non-steady input, take only mean for downstream analysis
            activity_mean = np.mean(np.asarray(raw_activity).reshape(-1,n, raw_activity.shape[-1]), axis = 1)
            weights_mean  = np.mean(np.asarray(raw_weights).reshape(-1, n, raw_weights.shape[-2], raw_weights.shape[-1]), axis = 1)

            # check nan
            if np.isnan(activity_mean[-1]).all():
                if visualization_mode:
                    print(f'nan exist in sim:{sim}: ', np.count_nonzero(~np.isnan(activity_mean[-1])))
                # assign the value such that it is plotable
                activity_mean[-1][np.isnan(activity_mean[-1])] = 1 
                nan_counter += 1 
                # break

            # check equilibrium
            mean1 = np.mean(activity_mean[-10:-5, :], axis=0)
            mean2 = np.mean(activity_mean[-5:, :], axis=0)
            check_eq = np.sum(np.where(mean1 - mean2 < 0.05, np.zeros(np.sum(N)), 1))
            if check_eq > 0:
                not_eq_counter += 1
                # break
                if visualization_mode:
                    ...
                    # print(f'Simulation {sim}, degree {degree[deg]} not converged: {int(check_eq)} neurons, {steps} steps')
            elif visualization_mode: 
                # print(f'Simulation {sim}, degree {degree[deg]} converged. {steps} steps')
                ...
            
            # check if the simulation can be entered criteria
            if g == radians[-1]:
                success = 1
            
            ############ data storage for different purposes ############
            # for evaluation 
            weights_eval = weights_mean[-50:]             # only the last 25 data points are needed
            activity_eval = activity_mean[int(-600/n):]   # only the last 500/n data is needed
            
            activity_eval_sim.append(activity_eval)
            weights_eva_list.append(weights_eval)        
            ini_weights_list.append(W_rec)          
            
            # for visualization (only for visualization_mode)
            if visualization_mode:
                # Slicing the timesteps, leaving only 50 of them for plotting
                plot_steps = 50
                plot_interval = activity_mean.shape[0]//(plot_steps-1) - 1 
                plot_begin    = activity_mean.shape[0]%(plot_steps-1)
                activity_plot = activity_mean[plot_begin::plot_interval]
                weights_plot  = weights_mean[plot_begin::plot_interval]

        # ------ radians loop ends here ------
        activity_eval_sim = np.asarray(activity_eval_sim)           # activity_eval_sim:    (radians, 600/n, neurons)
        
        if visualization_mode:
            activity_plot_list.append(np.asarray(activity_plot))        # activity_plot_list:    (simulation, radians, plot_steps, neurons)
            weights_plot_list.append(np.asarray(weights_plot))

        ################## Intrasimulation selectivity evaluation ##################
        if success: 
            # activity_not_reliable: a list of 4, each of which: (radians, neuron_number)
            a_mean, a_std, activity_not_reliable = Sn.activity_eva(activity_eval_sim, n)
            
            # mean and std for different neuron types in one simulation
            a_mean_all.append(a_mean)
            a_std_all.append(a_std)
            
            activity_popu, activity_off_sim = Sn.selectivity_eva_intrasim(activity_not_reliable)
            activity_off += activity_off_sim

            if len(activity_popu) == 4:
                # returning the selectivities of the 4 neuron types
                os_mean, os_std, ds_mean, ds_std, os_paper_mean, os_paper_std = helper.calculate_selectivity(activity_popu)
                os_mean_all.append(os_mean)
                os_std_all.append(os_std)
                ds_mean_all.append(ds_mean)
                ds_std_all.append(ds_std)
                os_paper_mean_all.append(os_paper_mean)
                os_paper_std_all.append(os_paper_std)

    # ---------------- simulations end here ----------------
    ################## evaluation over all simulations ##################

    selectivity_data, rel_data = Sn.selectivity_eva_all(  os_mean_all, os_std_all, 
                                                ds_mean_all, ds_std_all, 
                                                os_paper_mean_all, os_paper_std_all,
                                                a_mean_all,  a_std_all)

    # weight config evaluation
    weights_eval_all  =  np.asarray(weights_eva_list)            # weights:     (sim * radians, 50, postsyn, presyn))
    weight_data = Sn.weight_eva(weights=weights_eval_all, 
                                initial_weights=np.asarry(ini_weights_list))
    
    ################## Information and evaluation data storage ##################
    
    # weight evaluation metric
    weight_col = ['CS-CS', 'CS-CC', 'CS-PV', 'CS-SST',
                    'CC-CS', 'CC-CC', 'CC-PV', 'CC-SST',
                    'PV-CS', 'PV-CC', 'PV-PV', 'PV-SST',
                    'SST-CS', 'SST-CC', 'SST-PV', 'SST-SST']
    weight_ind = [  'Mean_weight',  'Std_mean_W',     'Mean_std_W',      
                    'Mean_Wdel',    'Std_mean_Wdel',  'Mean_std_Wdel']
    weight_df = pd.DataFrame(weight_data, columns=weight_col, index = weight_ind)

    # Selectivity matrix over simulation
    selectivity_col = ['CS', 'CC', 'PV', 'SST']
    selectivity_ind = [ 'Mean_act', 'Std_of_mean',  'Mean_std',      # a_mean_data, a_std_data, a_std_sim_data
                        'Mean_of_OS',  'Std_mean_OS',  'Mean_std_OS', 
                        'Mean_of_DS',  'Std_mean_DS',  'Mean_std_DS', 
                        'Mean_of_OS_p','Std_mean_OS_p','Mean_std_OS_p']
    selectivity_df = pd.DataFrame(selectivity_data, index=selectivity_ind, columns=selectivity_col)

    # summary info
    summary_info = {
        "Real_OS": rel_data[0], 
        "Real_DS": rel_data[1], 
        "Real_OS_paper": rel_data[2], 
        'nan_counter': nan_counter,
        'not_eq_counter': not_eq_counter, 
        'activity_off': activity_off}

    # basic configuration  
    meta_data_ind = ['CC_input', 'CS_input', 'PV_input', 'SST_input', 
                        'tau', 'tau_learn', 'tau_threshold', 
                        "learning_rule", "training_mode", "training_pattern"]
    meta_data_data = p.amplitude + \
                [Sn.tau] + [Sn.tau_learn] + [Sn.tau_threshold] + \
                [p.learning_rule] + [p.neurons] + [p.phase_key]
    meta_data = pd.DataFrame(meta_data_data, index= meta_data_ind, columns = ['value'])

    # returned: A dictionary storing all relevant information
    sim_dic = {
        "weight_df":        weight_df,
        "selectivity_df":   selectivity_df,
        "summary_info":     summary_info,
        "meta_data":        meta_data,
    }

    ################## visualization under visualization_mode ##################
    # plotting the simulation graphs
    if visualization_mode:
        # processing plotting data for visualization
        activity_plot = np.asarray(activity_plot_list)
        weights_plot =  np.asarray(weights_plot_list)

        # updating the data storing dictionary
        sim_dic.update({
            'loss_value':       helper.lossfun(sim_dic, config = p), 
            'activity_plot':    activity_plot,
            'weights_plot':     weights_plot
        })
        
        color_list = ['blue', 'salmon', 'lightseagreen', 'mediumorchid']
        DateFolder, time_id = helper.create_data_dir(config=p)

        ##### bar plot for weight change and final weight value #####
        fig_w, ax_w = plt.subplots(2,1)
        x_pos_w = np.arange(weight_df.shape[0])
        title_list_w = ['Mean weight','$\delta$ weight']
        for i in range(2):
            ax_w[i].bar(x_pos_w, weight_df.iloc[3*i,], 
                        yerr = weight_df.loc[3*i+2,], color = np.repeat(color_list, 4),
                        align='center', alpha=0.5, ecolor='black')
            ax_w[i].set_ylabel(title_list_w[i])
            ax_w[i].set_xticks(x_pos_w)
            ax_w[i].set_xticklabels(weight_df.columns)
            ax_w[i].set_title(title_list_w[i])
            ax_w[i].yaxis.grid(True)
        
        fig_s.set_size_inches(20, 12, forward=True)
        fig_w.tight_layout(pad=1.0)
        fig_w.suptitle(f"Weight by {p.learning_rule}")
        fig_w.show()
        # fig_w.savefig(f'data/{DateFolder}/{time_id}_{p.learning_rule}_Wei.png', dpi=100)

        ##### bar plot for activity, os, ds and os_paper #####
        fig_s, ax_s = plt.subplots(2,2)

        x_pos_s = np.arange(selectivity_df.shape[0])
        title_list_s = ['Mean activity',                'Orientational selectivity (OS)',
                        'Directional selectivity (DS)', 'Orientational selectivity_p (OS_p)']
        
        for i in range(4):
            axs = ax_s.flatten()[i]
            axs.bar(x_pos_s, selectivity_df.iloc[3*i,], 
                    yerr = selectivity_df.loc[3*i+2,], color = color_list,
                    align='center', alpha=0.5, ecolor='black')
            axs.set_ylabel(title_list_s[i])
            axs.set_xticks(x_pos_s)
            axs.set_xticklabels(selectivity_df.columns)
            axs.set_title(title_list_s[i])
            axs.yaxis.grid(True)
        
        fig_s.set_size_inches(20, 22, forward=True)
        fig_s.tight_layout(pad=1.0)
        fig_s.suptitle(f"Activity by {p.learning_rule}")
        fig_s.show()
        # fig_s.savefig(f'data/{DateFolder}/{time_id}_{p.learning_rule}_Act_OS.png', dpi=100)
        
        ##### TODO: activity and weight plot with error bar #####

        # Just do a dot plot and line them for the final product

        ##### TODO: activity and weight distribution #####
        activity_dis = activity_plot[:, -1]
        weights_dis = weights_plot[:, -1]


        '''
        if p.sim_number <6:
            for isim in range(p.sim_number):
                Sn.plot_activity(activity=activity, sim=isim, saving=False)
                Sn.plot_weights(weights=weights, sim=isim, saving=False)
        else:
            choice = np.random.choice(np.arange(p.sim_number), 2, replace=False)
            for isim in choice:
                Sn.plot_activity(activity=activity, sim=isim, saving=False)
                Sn.plot_weights(weights=weights, sim=isim, saving=False)
        '''

        ######## saving the results ########
        filepath = Path(f'data/{DateFolder}/{p.name_sim}_{time_id}_{p.learning_rule}.pkl')  
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(sim_dic, f)
        
        """
        To load the dictionary:
        with open("path/to/pickle.pkl", 'rb') as f:
           loaded_dict = pickle.load(f)
        """
       

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
                Ttau =p.Ttau,
                tau = p.tau, 
                tau_learn = p.tau_learn, 
                tau_threshold=p.tau_threshold,
                visualization_mode=False) 
    
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
                Ttau =p.Ttau,
                tau = p.tau, 
                tau_learn = tau_learn, 
                tau_threshold=tau_threshold_fac*tau_learn,
                visualization_mode=False) 
    
    # FIXME: Change the stable simulation loss function input if needed in the future
    # lossval = helper.Stable_sim_loss(activity=activity, Max_act=20)
    return ...

#%% Simulation with default parameter for visualization

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
                    visualization_mode=True,)

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
                visualization_mode=False) 
        
        #Saving the data after running
        with open(f'data/{DateFolder}/Tuning_Tau_Activity_weight.npy', 'wb') as f:
            np.save(f, Best_act)
            np.save(f, Best_weight)
        
    # TODO: Spearmint tuning package