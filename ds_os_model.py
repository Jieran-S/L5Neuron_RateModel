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
import seaborn as sns
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

def run_simulation(Amplitude, Steady_input, spatialF, temporalF, spatialPhase, 
                    learning_rule, Ttau, visualization_mode, neurons, phase_list,
                    tau, 
                    tau_learn, 
                    tau_threshold):
    '''
    All parameter inputs to the function is tunable parameters from the system. 

    @Amplitude: a list of 4, input stimulus to CS, CC, PV and SST
    @teady_input: a list of 4, indicating which of the 4 types of neurons receive steady input
    @spatialF, @temporalF, @spatialPhase: parameter to generate input distribution
    @learning_rule: which learning rule will be applied 
    @Ttau, @tau, @tau_learn, @tau_threshold: time-related parameters for training
    @neurons, @phase_list: the neurons and training patterns for the system 
    @isualization_mode: if we plot the results
    '''
   
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

    os_mean_all_bl, os_std_all_bl, ds_mean_all_bl, ds_std_all_bl, os_paper_mean_all_bl, os_paper_std_all_bl, a_mean_all_bl, a_std_all_bl = \
    [], [], [], [], [], [], [], []
    ########## data storage structure ##########
    """
    all_activity/weights_plot_list: store all activities/weights data across simulations for plotting
    weights_eva_list: store last 50 steps of weights across simulation for evaluation
    ini_weights_list: store initial weights info across all simulation for evaluation
    """
    activity_plot_list = []     # (sim, radians, 50, neurons)
    weights_plot_list = []      # (sim*radians, 50, postsyn, presyn)
    weights_eva_list = []       # (sim*radians, 60, postsyn, presyn)
    ini_weights_list = []       # (sim*radians, postsyn, presyn)

    ################## iterate through different initialisations ##################
    for sim in range(p.sim_number):
        """
        store activities per simulation for intra-simulation evaluation
        sim: for each simulation
        bl: activity value before learning
        plot: for plotting, keep 50 time steps
        """
        activity_eval_sim = []  # (radians, timestep/n, neurons)
        activity_eval_bl = []   # (radians, 5, neurons)
        activity_plot_sim = []  # (radians, 50, neurons)

        ########## weight initialization ##########
        w_initial = np.abs(w_target)
        train_neuron_type = [[0,1],[2,3],[0,1,2,3]][p.neurons_list.index(p.neurons)]

        for i in train_neuron_type:
            for j in train_neuron_type:
                w_initial[i,j] = abs(np.random.normal(w_target[i,j], scale= 0.1*abs(w_target[i,j]))) 

        # TODO:Trying uniformed distribution randomization in the future 
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
                                neurons= neurons, 
                                phase_list=phase_list,
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
            raw_activity, raw_weights = Sn.run(inputs, initial_values, simulate_till_converge = False)
            raw_activity = np.asarray(raw_activity)
            raw_weights  = np.asarray(raw_weights)
            ############ data quality checking ############

            # !!! mean period of input, change according to termporalF
            n = 25
            # for non-steady input, take only mean for downstream analysis
            activity_mean = np.mean(raw_activity.reshape(-1,n, raw_activity.shape[-1]), axis = 1)
            weights_mean  = np.mean(raw_weights.reshape(-1, n, raw_weights.shape[-2], raw_weights.shape[-1]), axis = 1)

            # check nan
            if np.isnan(activity_mean[-1]).all():
                if visualization_mode:
                    print(f'nan exist in sim:{sim}: ', np.count_nonzero(~np.isnan(activity_mean[-1])))
                # assign the value such that it is plotable
                activity_mean[-1][np.isnan(activity_mean[-1])] = 0 
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
            # Learning time may change if we have learning_till_converge == T
            LearningTime = int(max((Sn.number_steps_before_learning - (Sn.step- Sn.tsteps))*Sn.delta_t/Sn.tau,0)/n)
            activity_before_learn = activity_mean[max(LearningTime-5, 0):LearningTime]
            activity_eval_sim.append(activity_eval)
            activity_eval_bl.append(activity_before_learn)
            weights_eva_list.append(weights_eval)        
            ini_weights_list.append(W_rec)          
            
            if visualization_mode: # For visualization only
                # Slicing the timesteps, leaving only 50 of them for plotting
                plot_steps = 50
                plot_interval = activity_mean.shape[0]//(plot_steps-1)
                plot_begin    = activity_mean.shape[0]%(plot_steps-1) - 1

                activity_plot_sim.append(activity_mean[plot_begin::plot_interval])     # appending the activity of each set of 4 orientations
                weights_plot_list.append(weights_mean[plot_begin::plot_interval])      # appending all possible weights

        # ------ radians loop ends here ------
        activity_eval_sim = np.asarray(activity_eval_sim)           # activity_eval_sim:    (radians, 600/n, neurons)
        activity_eval_bl = np.asarray(activity_eval_bl)             # activity_eval_bl:     (radians, 5, neurons)

        if visualization_mode:
            activity_plot_list.append(np.asarray(activity_plot_sim))        # activity_plot_list: (simulation, radians, 50, neurons)

        ################## Intrasimulation selectivity evaluation ##################
        if success: 
            # activity_not_reliable: a list of 4, each of which: (radians, neuron_number)
            a_mean, a_std, activity_not_reliable = Sn.activity_eva(activity_eval_sim, n)
            mean_bl, std_bl, activiity_not_reliable_bl = Sn.activity_eva(activity_eval_bl, n, 5*n)

            # mean and std for different neuron types in one simulation
            a_mean_all.append(a_mean)
            a_mean_all_bl.append(mean_bl)
            a_std_all.append(a_std)
            a_std_all_bl.append(std_bl)
            
            activity_popu, activity_off_sim = Sn.selectivity_eva_intrasim(activity_not_reliable)
            activity_popu_bl, _ = Sn.selectivity_eva_intrasim(activiity_not_reliable_bl)
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
                # do the same for before learning (bl)
                os_mean, os_std, ds_mean, ds_std, os_paper_mean, os_paper_std = helper.calculate_selectivity(activity_popu_bl)
                os_mean_all_bl.append(os_mean)
                os_std_all_bl.append(os_std)
                ds_mean_all_bl.append(ds_mean)
                ds_std_all_bl.append(ds_std)
                os_paper_mean_all_bl.append(os_paper_mean)
                os_paper_std_all_bl.append(os_paper_std)

    # ---------------- simulations end here ----------------
    ################## evaluation over all simulations ##################
    
    # selectivity dataframe, the mean, mean_std, std_mean of os, ds, os_p and activity
    selectivity_data, rel_data = Sn.selectivity_eva_all(os_mean_all, os_std_all, 
                                                        ds_mean_all, ds_std_all, 
                                                        os_paper_mean_all, os_paper_std_all,
                                                        a_mean_all,  a_std_all)
    selectivity_data_bl, _ = Sn.selectivity_eva_all(os_mean_all_bl, os_std_all_bl,
                                                            ds_mean_all_bl, ds_std_all_bl,
                                                            os_paper_mean_all_bl, os_paper_std_all_bl,
                                                            a_mean_all_bl, a_std_all_bl)
    # weight config evaluation
    weights_eval_all  =  np.asarray(weights_eva_list)            # weights:     (sim * radians, 50, postsyn, presyn))
    weight_data = Sn.weight_eva(weights=weights_eval_all, 
                                initial_weights=np.asarray(ini_weights_list))
    
    ################## Information and evaluation data storage ##################
    
    # weight evaluation metric, Postsyn-Presyn
    weight_col = ['CS-CS', 'CS-CC', 'CS-PV', 'CS-SST',
                    'CC-CS', 'CC-CC', 'CC-PV', 'CC-SST',
                    'PV-CS', 'PV-CC', 'PV-PV', 'PV-SST',
                    'SST-CS', 'SST-CC', 'SST-PV', 'SST-SST']
    weight_ind = [  'Mean_weight',  'Std_mean_W',     'Mean_std_W',      
                    'Mean_Wini',    'Std_mean_Wini',  'Mean_std_Wini',
                    'Mean_Wdel',    'Std_mean_Wdel',  'Mean_std_Wdel']
    weight_df = pd.DataFrame(weight_data, columns=weight_col, index = weight_ind)

    # Selectivity matrix over simulation
    selectivity_col = ['CS', 'CC', 'PV', 'SST']
    selectivity_ind = [ 'Mean_act', 'Std_of_mean',  'Mean_std',      # a_mean_data, a_std_data, a_std_sim_data
                        'Mean_of_OS',  'Std_mean_OS',  'Mean_std_OS', 
                        'Mean_of_DS',  'Std_mean_DS',  'Mean_std_DS', 
                        'Mean_of_OS_p','Std_mean_OS_p','Mean_std_OS_p']
    selectivity_df = pd.DataFrame(selectivity_data, index=selectivity_ind, columns=selectivity_col)
    selectivity_bl_df = pd.DataFrame(selectivity_data_bl, index=selectivity_ind, columns=selectivity_col)
    all_selectivity = pd.concat([selectivity_df, selectivity_bl_df], keys=['after', 'before'], names=["Condition", "Property"])
    # summary info
    summary_info = {
        "Real_OS": rel_data[0], 
        "Real_DS": rel_data[1], 
        "Real_OS_paper": rel_data[2], 
        'nan_counter': nan_counter,
        'not_eq_counter': not_eq_counter, 
        'activity_off': activity_off}

    # basic configuration in meta-data df
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
        "selectivity_df":   all_selectivity,
        "summary_info":     summary_info,
        "meta_data":        meta_data,
    }
    
    ################## visualization under visualization_mode ##################
    # plot all potential graphs for information extraction
    if visualization_mode:
        fig_size = (10,11)
        # color_list = ['blue', 'salmon', 'lightseagreen', 'mediumorchid']
        # line_col    = ['darkblue','deeppink','seagreen','fuchsia']
        color_list = ["#FF0000", "#00A08A","#F2AD00","#5BBCD6"]
        line_col    = ["#800000",'#02401B','#B3672B','#046C9A']

        DateFolder, time_id = helper.create_data_dir(config=p)
        saving = False

        ########## bar plot for weight change and final weight value ##########
        vis.weights_barplot(weight_df, line_col = line_col, 
                        color_list = color_list, learning_rule = learning_rule,
                        config = p, saving = saving)

        ########## bar plot for activity, os, ds and os_paper ##########
        vis.selectivity_barplot(selectivity_df, selectivity_bl_df,
                                fig_size = fig_size, color_list = color_list, learning_rule = learning_rule,
                                config = p, saving = saving)

        ########## activity plot with error bar ##########
        neuron_list = ['CS','CC','PV','SST']

        # prepare the data
        mean_act_sim = np.mean(np.asarray(activity_plot_list), axis = 0)     # (radian, 50 , neurons)
        act_list    = [ mean_act_sim[:, :, :N[0]],                  # CS 
                        mean_act_sim[:, :, sum(N[:1]):sum(N[:2])],  # CC 
                        mean_act_sim[:, :, sum(N[:2]):sum(N[:3])],  # PV 
                        mean_act_sim[:, :, sum(N[:3]):sum(N)] ]     # SST
        mean_act_neuron = np.asarray([np.mean(act, axis = -1) for act in act_list])     # (neuron types, radians, 50)
        error_act_neuron= np.asarray([np.nanstd(act, axis = -1)  for act in act_list])     # (neuron types, radians, 50)
        act_plot_dic = {
            "mean_act": mean_act_neuron,
            "std_act": error_act_neuron,
            "structure": "(neuron_types, radians, plot_steps(50))"
        }

        # start the plot
        vis.activity_plot(act_plot_dic, 
                        color_list = color_list, fig_size = fig_size, learning_rule = learning_rule,
                        neuron_list = neuron_list, line_col = line_col,
                        config = p, saving = saving)

        ########## weight plot with error bar ##########
        # prepare the data
        mean_weights_sim = np.mean(np.asarray(weights_plot_list), axis = 0)  #(50, postsyn, presyn)
        mean_weights = std_weights = np.empty((plot_steps, 4, 4))   
        weights_vector = [  mean_weights_sim[:, :N[0], :],                  #CS
                            mean_weights_sim[:, sum(N[:1]):sum(N[:2]), :],  #CC
                            mean_weights_sim[:, sum(N[:2]):sum(N[:3]), :],  #PV
                            mean_weights_sim[:, sum(N[:3]):sum(N), :]]      #SST
        for ind,wei in enumerate(weights_vector):
            prewei_vec = [  wei[:, :, :N[0]],
                            wei[:, :, sum(N[:1]):sum(N[:2])],
                            wei[:, :, sum(N[:2]):sum(N[:3])],
                            wei[:, :, sum(N[:3]):sum(N)]]
            # assigning the presynaptic neurons to corresponding columns
            mean_weights[:, ind, :] = np.array([np.mean(x, axis =(1,2)) for x in prewei_vec]).T
            std_weights[:, ind, :] = np.array([np.nanstd(x, axis =(1,2)) for x in prewei_vec]).T
        wei_plot_dic = {
            "mean_weights": mean_weights,
            "std_weights": std_weights,
            "structure": "(plot_steps(50), postsyn, presyn)"
        }

        vis.weights_plot(wei_plot_dic, 
                        color_list = color_list, fig_size = fig_size, learning_rule = learning_rule,
                        neuron_list = neuron_list, line_col = line_col,
                        config = p, saving = saving)        

        ##### Activity distribution #####
        # TODO: Do we need to plot the weight distribution by the end of training?
        # weights_dis = np.mean(mean_weights_sim[-5:,:, :], axis = 0) # (presyn, postsyn)
        
        # data processing: a dataframe. col: neuron types, row: neuron response sorted by orientations
        activity_dis = [np.mean(x[:, -5:, :], axis = 1) for x in act_list]   # a list of 4, each (4*N[i],) followed by the 
        act_ser_list = []
        for act in activity_dis:
            sim_char_vec = np.char.mod('%03d', np.arange(act.shape[1]))
            # double comprehension: first iterate outer then inner loop
            indexlist = [f'{m:03}' + x for m in p.degree for x in sim_char_vec]    
            act_ser_list.append(pd.Series(act.flatten(), index = indexlist)) 

        activity_df = pd.DataFrame({ "CS": act_ser_list[0],"CC": act_ser_list[1],
                                    "PV": act_ser_list[2],"SST":act_ser_list[3]})
        activity_df['Degree'] = np.repeat(p.degree, int(activity_df.shape[0]/4))

        histo_color_list = ['salmon', 'lightseagreen', 'mediumorchid','blue']
        # histo_color_list = ["#5BBCD6","#00A08A","#F2AD00","#FF0000"]
        vis.activity_histogram(activity_df, 
                            color_list = histo_color_list,
                            fig_size = fig_size, learning_rule = learning_rule,
                            config = p, saving = saving)       
 
         ##### TODO: Correlation between the mean weights and its activity #####

        # updating the data storing dictionary
        sim_dic.update({
            'loss_value':       helper.lossfun(sim_dic, config = p), 
            'activity_plot':    act_plot_dic,
            'weights_plot':     wei_plot_dic,
            'activity_hist':    activity_df
        })
    
        ######## saving the results ########
        filepath = Path(f'data/{DateFolder}/{p.name_sim}_{time_id}_{p.learning_rule}.pkl')  
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(sim_dic, f)
        '''
        To load the dictionary:
        with open("path/to/pickle.pkl", 'rb') as f:
           loaded_dict = pickle.load(f)
        '''
       
    return sim_dic, activity_plot_list, weights_plot_list

def objective(params):
    """
    For hyperparameter tuning: 
    Define a parameter space, using the randomly selected parameter value to run the network,
    then we return the loss function value for the system to find the best parameter pairs.
    """
    amplitude = [params['cc'], params['cs'], params['pv'], params['sst']]
    tau_learn, tau_threshold_fac =  params['tau_learn'], params['tau_threshold_fac']
    sim_dic, _, _ = run_simulation( 
                Amplitude= amplitude,
                Steady_input= p.steady_input,
                spatialF=p.spatialF,
                temporalF=p.temporalF,
                spatialPhase=p.spatialPhase,
                learning_rule= p.learning_rule, 
                neurons=p.neurons,
                phase_list=p.phase_list,
                Ttau =p.Ttau,
                tau = p.tau, 
                tau_learn = tau_learn, 
                tau_threshold=tau_threshold_fac*tau_learn,
                visualization_mode=False) 
    
    # return the loss function value
    lossval = helper.lossfun(sim_dic, config=p)
    return lossval


#%% Simulation with default parameter for visualization
if __name__ == "__main__":
    if p.tuning != True:
        # inputing all tunable parameters from the test.config and first run and visualize
        sim_dic, activity_plot_list, weights_plot_list = run_simulation( Amplitude= p.amplitude,
                        Steady_input= p.steady_input,
                        spatialF=p.spatialF,
                        temporalF=p.temporalF,
                        spatialPhase=p.spatialPhase,
                        learning_rule= p.learning_rule, 
                        phase_list=p.phase_list,
                        neurons=p.neurons,
                        Ttau =p.Ttau,
                        tau=p.tau,
                        tau_learn=p.tau_learn,
                        tau_threshold=p.tau_threshold,
                        visualization_mode=True,)
#%% Simulation selectivity overiew
if __name__ == "__main__":
    if p.tuning != True:
        
        Dic_list = []
        for rule in p.rule_list[1:]:
            sim_dic, _, _ = run_simulation( Amplitude= p.amplitude,
                                Steady_input= p.steady_input,
                                spatialF=p.spatialF,
                                temporalF=p.temporalF,
                                spatialPhase=p.spatialPhase,
                                learning_rule= rule, 
                                phase_list=p.phase_list,
                                neurons=p.neurons,
                                Ttau =p.Ttau,
                                tau=p.tau,
                                tau_learn=p.tau_learn,
                                tau_threshold=p.tau_threshold,
                                visualization_mode=True,)
            Dic_list.append(sim_dic)
        
        fig_size = (15,17)
        color_list = ["#FF0000", "#00A08A","#F2AD00","#5BBCD6"]
        line_col    = ["#800000",'#02401B','#B3672B','#046C9A']
        saving = True
        DateFolder, time_id = helper.create_data_dir(config=p)

        title_list_s = ['Orientational selectivity (OS)',
                        'Directional selectivity (DS)', 'Orientational selectivity_p (OS_p)']

        learn_list = ['BCM','slide_BCM', 'oja','Cov']

        fig_s, ax_s = plt.subplots(3,1, figsize=fig_size)
        bar_width = 0.45
        x_pos_i = [0,1,2,3,5,6,7,8,10,11,12,13,15,16,17,18]
        x_pos_w = [x + 0.45 for x in x_pos_i]

        for i in range(3):
            axs = ax_s.flatten()[i]
            for ind, selectivityDF in enumerate(Dic_list):
                selectivity_df_bl = selectivityDF['selectivity_df'].loc['before']
                selectivity_df = selectivityDF['selectivity_df'].loc['after']
                axs.bar(x_pos_i[ind::4], list(selectivity_df_bl.iloc[3*(i+1),]), color = color_list[ind],
                        align='center', alpha=0.5, label = f'{learn_list[ind]}_before', width = bar_width)
                axs.bar(x_pos_w[ind::4], list(selectivity_df.iloc[3*(i+1),]), color = line_col[ind],
                        align='center', alpha=0.5, label = f'{learn_list[ind]}_after', width = bar_width)
                axs.errorbar(x_pos_i[ind::4], list(selectivity_df_bl.iloc[3*(i+1),]),yerr = list(selectivity_df_bl.iloc[3*(i+1)+2,]),
                            fmt = 'none', capsize = 5, ecolor = 'gray')
                axs.errorbar(x_pos_w[ind::4], list(selectivity_df.iloc[3*(i+1),]),yerr = list(selectivity_df.iloc[3*(i+1)+2,]),
                    fmt = 'none', capsize = 5, ecolor = 'gray')

            axs.set_xticks(np.mean([x_pos_i[::4],x_pos_i[1::4],x_pos_i[2::4],x_pos_i[3::4]], axis = 0) + 0.5*bar_width)
            axs.set_xticklabels(["CC","CS","PV","SST"])
            axs.set_ylabel(title_list_s[i])
            axs.set_ylim(bottom = -0.05, top = 1)
            axs.set_title(f"{title_list_s[i]}")
            axs.yaxis.grid(True) 
            axs.margins(x=0.02)
            axs.legend()   
            
        fig_s.tight_layout(pad=1.0)
        # fig_s.show()
        if saving:
            fig_s.savefig(f'data/{DateFolder}/{time_id}_selectivity_all.png', dpi=100)


#%%  Hyper parameter tuning
if __name__ == "__main__":            
    # using Hyperpot for tuning
    if p.tuning == True: 
        
        # define parameter space
        space = {
            'cs': hyperopt.hp.uniform('cs', 0, 20),
            'cc': hyperopt.hp.uniform('cc', 0, 20),
            'pv': hyperopt.hp.uniform('pv', 0, 20),
            'sst':hyperopt.hp.uniform('sst', 0, 20),
            'tau_learn': hyperopt.hp.uniform('tau_learn', 1000, 2000),
            'tau_threshold_fac': hyperopt.hp.uniform('tau_threshold_fac', 0.001, 1),
        }        

        # define algorithms
        algo_tpe = hyperopt.tpe.suggest
        algo_rand = hyperopt.rand.suggest

        # define history (for saving tuning results)
        trails = hyperopt.Trials()

        # Start hyperparameter search
        tpe_best = hyperopt.fmin(fn=objective, space=space, algo=algo_tpe, trials=trails, 
                                # define maximal round for tuning duration
                                max_evals=300)
#%% Tuning results visualization
        # Printing out results
        print('Minimum loss attained with TPE:    {:.4f}'.format(trails.best_trial['result']['loss']))
        print('\nNumber of trials needed to attain minimum with TPE: {}'
                    .format(trails.best_trial['misc']['idxs']['tau_learn'][0]))
        print('\nBest input set: {}'.format(tpe_best)) 

        # Saving the results into a csv file: still have to see here
        tpe_results = pd.DataFrame({'loss': [x['loss'] for x in trails.results], 
                                    'iteration': trails.idxs_vals[0]['cc'],
                                    'cs': trails.idxs_vals[1]['cs'],
                                    'cc': trails.idxs_vals[1]['cc'],
                                    'pv': trails.idxs_vals[1]['pv'],
                                    'sst': trails.idxs_vals[1]['sst'],
                                    'tau_learn': trails.idxs_vals[1]['tau_learn'],
                                    'tau_threshold': trails.idxs_vals[1]['tau_threshold_fac'],}).fillna(method='ffill')
        
        # save the tuning results
        DateFolder, time_id = helper.create_data_dir(config=p)
        filepath = Path(f'data/{DateFolder}/{p.name_sim}_{time_id}_{p.learning_rule}_TuningResults.csv')
        tpe_results.to_csv(filepath, float_format='%.3f')
        
        # TODO: plotting the hyperparameter tuning profile 

        # plot the best attempt results to see how it works
        Best_amplitude = [tpe_best['cs'], tpe_best['cc'], tpe_best['pv'], tpe_best['sst']]
        
        sim_dic_best, activity_plot_list, weights_plot_list = run_simulation( Amplitude= Best_amplitude,
                        Steady_input= p.steady_input,
                        spatialF=p.spatialF,
                        temporalF=p.temporalF,
                        spatialPhase=p.spatialPhase,
                        learning_rule= p.learning_rule, 
                        phase_list=p.phase_list,
                        neurons=p.neurons,
                        Ttau =p.Ttau,
                        tau=p.tau,
                        tau_learn=tpe_best['tau_learn'],
                        tau_threshold=tpe_best['tau_learn']*tpe_best['tau_threshold_fac'],
                        visualization_mode=True,)
        
    # TODO: Spearmint tuning package