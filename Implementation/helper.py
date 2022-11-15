import enum
import os
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import configs.debug_config

# remove top and right axis from plots
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False 
 
np.random.seed(42)

def distributionInput(a_data,b_data,spatialF, temporalF, orientation, spatialPhase, amplitude, T, steady_input, N):
    """
    Generates a moving bar as input to CS, CC, PV, SST. 
    Using the function from textbook theoretical neurosciences. 
    Turning the image into stimulus by converting the difference between that pixel over time and the 
    difference between the pixel and the overall backaground level of luminance. 
    Output: Stimulus to the L5 neuron
    Steady_input: making a check to make sure the input is steady.
    """
    
    i = 0
    inputs_p_all = []
    N_indices = [[0, N[0]], [sum(N[:1]), sum(N[:2])], [sum(N[:2]), sum(N[:3])], [sum(N[:3]), sum(N)]]
    for popu in N_indices:
        inputs_p = []

        if steady_input[i] > 0.5:
            for t in range(T):
                inputs_p.append(amplitude[i] * np.cos(
                    spatialF * a_data[popu[0]:popu[1]] * np.cos(orientation) +
                    spatialF * b_data[popu[0]:popu[1]] * np.sin(orientation) - spatialPhase)
                                       * np.cos(temporalF) + amplitude[i])
            inputs_p = np.array(inputs_p)
        else:
            for t in range(T):
                inputs_p.append(amplitude[i] * np.cos(
                    spatialF * a_data[popu[0]:popu[1]] * np.cos(orientation) +
                    spatialF * b_data[popu[0]:popu[1]] * np.sin(orientation) - spatialPhase)
                                       * np.cos(temporalF * t) + amplitude[i])
            inputs_p = np.array(inputs_p)
        i += 1
        inputs_p_all.append(inputs_p)

    inputs = np.concatenate((inputs_p_all), axis=1)

    return (inputs)

def distributionInput_negative(a_data,b_data,spatialF, temporalF, orientation, spatialPhase, amplitude, T, steady_input, N):
    """
    Generates a moving bar as input to CS, CC, PV, SST.
    Return an array A of size [T(t-step), N(population)]
    a,b_data: Input axis coordinate for different cells. drawn from random cos/sin distribution
    """
    i = 0
    # output list with all time-steps + neuron values
    inputs_p_all = []
    # Section the index with different type of neurons
    N_indices = [[0, N[0]],                 #CS
                [sum(N[:1]), sum(N[:2])],   #CC
                [sum(N[:2]), sum(N[:3])],   #PV
                [sum(N[:3]), sum(N)]]       #SST
    
    for popu in N_indices:
        inputs_p = []

        # input = A cos(K x cos(theta) + K y sin(theta) - phi) * cos(w t)
        if steady_input[i] > 0.5:
            for t in range(T): # all time steps
                inputs_p.append(amplitude[i] * np.cos(
                    spatialF * a_data[popu[0]:popu[1]] * np.cos(orientation) +
                    spatialF * b_data[popu[0]:popu[1]] * np.sin(orientation) - spatialPhase)
                                       * np.cos(temporalF))
            inputs_p = np.array(inputs_p)
        else:
            for t in range(T):
                inputs_p.append(amplitude[i] * np.cos(
                    spatialF * a_data[popu[0]:popu[1]] * np.cos(orientation) +
                    spatialF * b_data[popu[0]:popu[1]] * np.sin(orientation) - spatialPhase)
                                       * np.cos(temporalF * t))
            inputs_p = np.array(inputs_p)

        i += 1
        inputs_p_all.append(inputs_p)

    inputs = np.concatenate((inputs_p_all), axis=1)

    return (inputs)

def create_synapses(N_pre, N_post, prob, same_population=False):
    """
    Create random connections between two groups or within one group.
    :param N_pre: number of neurons in pre group
    :param N_post: number of neurons in post group
    :param prob:   connectivity (probability for connecting, synaptic strength)
    :param same_population: whether to allow autapses (connection of neuron to itself if pre = post population)
    :return: 2xN_con array of connection indices (pre-post pairs)
    """

    # indegree: the expectation of how many pre neurons will form synapses with post neurons
    indegree = int(np.round(prob * N_pre))  # no variance of indegree = fixed indegree

    i = np.array([], dtype=int)
    j = np.array([], dtype=int)

    for n in range(N_post):

        if same_population:  
            # if autapses are disabled, remove index of present post neuron from pre options
            opts = np.delete(np.arange(N_pre, dtype=int), n)
        else:
            opts = np.arange(N_pre, dtype=int)

        # Randomly select the expected number of neurons from pre to connect with post neurons
        pre = np.random.choice(opts, indegree, replace=False)

        # add connection indices to list pre -> indexes for pre-neurons; post -> index of the post neuron
        i = np.hstack((i, pre))  
        j = np.hstack((j, np.repeat(n, indegree)))

    # return a 2x(N_post*indegree) array, representing the connection between each pair of pre-post neurons
    return np.array([i, j])

def normal_distr_weights(weight, w_noise):
    """
    Generate weights with normally distributed weight noise.
    weight: Experimental tested weight in the literature. The target 
    situation that we want to achieve.
    """

    if weight > 0:
        weight = np.abs(np.random.normal(weight, w_noise))
    elif weight < 0:
        weight = -np.abs(np.random.normal(weight, w_noise))
    return (weight)

def generate_connectivity(N, p, w_initial, w_noise):
    """
    Generates a connectivity matrix where rows are postsyn. neuron and columns are presynaptic neuron.
    p: Probability matrix 4x4 (connection probability matrix of CS, CC, PV and SST neurons (rows are the presyn cell))
    w_initial: the target weight to achieve according to Campognola 2022 PSP amplitude. Presented 
               in terms of synaptic strength matrix (sequared, 4x4) where rows are presynaptic cells
    w_noise: degree of noise introduced to the synpatic connection
    N: 1x4 vector representing the number of CS, CC, PV and SST neurons
    """
    # Individual neuron matrix
    W_rec = np.zeros((np.sum(N), np.sum(N)))

    # iterate through all cell population
    for pre in range(p.shape[0]):
        for post in range(p.shape[0]):
            same_population = False
            # For neurons of the same types, considering the stimulation among themselves
            if pre == post:
                same_population = True
            # Forming the connection index between different types of neurons
            con = create_synapses(N[pre], N[post], p[pre][post], same_population)

            # Go through every column -> aka go through all the synpases
            for i in range(con.shape[1]):
                # identifying the corresponding pre/post neurons of the synpase
                pre_neuron = con[0][i]
                post_neuron = con[1][i]
                
                # region: How much should we skip for the target population
                step_pre = np.sum(N[:pre])
                step_post = np.sum(N[:post])

                # Adding the connectivity between the two neurons in the final matrix
                W_rec[step_pre + pre_neuron][step_post + post_neuron] = normal_distr_weights(w_initial[pre][post],
                                                                                             w_noise)
    # Why tranposed? 
    return (W_rec.T)

def calculate_selectivity_sbi(activity_popu):
    """
    Calculate mean and std of selectivity.

    """

    os_mean_data = []  # orientation selectivity
    ds_mean_data = []  # directions selectivity
    os_paper_mean_data = []  # directions selectivity calculated as in paper

    for population in range(len(activity_popu)):
        preferred_orientation = np.argmax(activity_popu[population], axis=0)

        os, ds, os_paper = [], [], []

        for neuron in range(activity_popu[population].shape[1]):
            s_max_index = preferred_orientation[neuron]

            # activity of preferred stimulus
            s_pref = activity_popu[population][s_max_index][neuron]

            # activity of preferred orientation in both directions
            s_pref_orient = np.mean([activity_popu[population][s_max_index][neuron],
                                    activity_popu[population][(s_max_index + 2) % 4][neuron]])

            # activity of orthogonal stimulus
            s_orth = np.mean([activity_popu[population][(s_max_index + 1) % 4][neuron],
                             activity_popu[population][(s_max_index + 3) % 4][neuron]])

            # activity of opposite stimulus
            s_oppo = activity_popu[population][(s_max_index + 2) % 4][neuron]

            os.append(np.abs(s_pref_orient - s_orth) / (s_pref_orient + s_orth))
            ds.append(np.abs(s_pref - s_oppo) / (s_pref + s_oppo))
            os_paper.append(np.abs(s_pref - s_orth) / (s_pref + s_orth))

        os_mean_data.append(np.mean(os))
        ds_mean_data.append(np.mean(ds))
        os_paper_mean_data.append(np.mean(os_paper))

    return (os_mean_data,ds_mean_data,os_paper_mean_data)

def calculate_selectivity(activity_popu):
    """
    Calculate mean and std of selectivity.
    No need to be used in my project.
    """

    os_mean_data = []  # orientation selectivity
    os_std_data = []
    ds_mean_data = []  # directions selectivity
    ds_std_data = []
    os_paper_mean_data = []  # directions selectivity calculated as in paper
    os_paper_std_data = []

    for population in range(4):
        preferred_orientation = np.argmax(activity_popu[population], axis=0)

        os, ds, os_paper = [], [], []
        preferred_orientation_freq = [0, 0, 0, 0]

        for neuron in range(activity_popu[population].shape[1]):
            s_max_index = preferred_orientation[neuron]
            preferred_orientation_freq[s_max_index] += 1

            # activity of preferred stimulus
            s_pref = activity_popu[population][s_max_index][neuron]

            # activity of preferred orientation in both directions
            s_pref_orient = np.mean([activity_popu[population][s_max_index][neuron],
                                    activity_popu[population][(s_max_index + 2) % 4][neuron]])

            # activity of orthogonal stimulus
            s_orth = np.mean([activity_popu[population][(s_max_index + 1) % 4][neuron],
                             activity_popu[population][(s_max_index + 3) % 4][neuron]])

            # activity of opposite stimulus
            s_oppo = activity_popu[population][(s_max_index + 2) % 4][neuron]

            os.append((s_pref_orient - s_orth) / (s_pref_orient + s_orth))
            ds.append((s_pref - s_oppo) / (s_pref + s_oppo))
            os_paper.append((s_pref - s_orth) / (s_pref + s_orth))

        os_mean_data.append(np.mean(os))
        os_std_data.append(np.std(os))
        ds_mean_data.append(np.mean(ds))
        ds_std_data.append(np.std(ds))
        os_paper_mean_data.append(np.mean(os_paper))
        os_paper_std_data.append(np.std(os_paper))

    return (os_mean_data, os_std_data,ds_mean_data,ds_std_data,os_paper_mean_data,os_paper_std_data)

def plot_activity(activity, N,sim, learningrule, Ttau):

    '''
    activity: 3d matrix with infomraiton on the activation of different neurons
    '''

    if len(activity) == 0:
        return(0)
    # Extract the connectivity data for each cell population? 
    activity = activity[sim]
    activity_cs = activity[:, :N[0]]
    activity_cc = activity[:, sum(N[:1]):sum(N[:2])]
    activity_pv = activity[:, sum(N[:2]):sum(N[:3])]
    activity_sst = activity[:, sum(N[:3]):sum(N)]
    activity_vec = [activity_cs, activity_cc, activity_pv, activity_sst]
    namelist = ['CS', 'CC', 'PV', 'SST']

    fig,axes = plt.subplots(2,2)
    for ind, act in enumerate(activity_vec):
        axs = axes.flatten()[ind]
        for i in range(act.shape[1]):
            axs.plot(np.linspace(0, Ttau, act.shape[0]), act[:,i],c='grey',alpha=0.5)
        axs.set_title(namelist[ind])

    fig.tight_layout(pad=2.0)

    now = datetime.now() # current date and time
    DateFolder = now.strftime('%m_%d')
    if os.path.exists(f'data/{DateFolder}') == False:
        os.makedirs(f'data/{DateFolder}')
    time_id = now.strftime("%m%d_%H:%M")

    time_id = datetime.now().strftime("%m%d_%H:%M")
    title_save = f'data/{DateFolder}/{learningrule}_{sim}_{time_id}_act.png'
    # fig.savefig(title_save)

def plot_weights(weights, N, saving, sim, learningrule, Ttau):

    '''
    weights: a 3D matrix (4D if all simulation taken into account): Tstep x N(post-syn) x N(pre-syn)
    '''
    weights = weights[sim]
    weight_cs = weights[:, :N[0], :]
    weight_cc = weights[:, sum(N[:1]):sum(N[:2]), :]
    weight_pv = weights[:, sum(N[:2]):sum(N[:3]), :]
    weight_sst = weights[:, sum(N[:3]):sum(N), :]
    weights_vector = [weight_cs, weight_cc, weight_pv, weight_sst]

    fig, axes = plt.subplots(2,2)
    for j,wei in enumerate(weights_vector):
        # return the average weight from one cell to a specific responses
        w_to_cs = np.mean(wei[:, :, :N[0]], axis=-1)
        w_to_cc = np.mean(wei[:, :, sum(N[:1]):sum(N[:2])], axis= -1 )
        w_to_pv = np.mean(wei[:, :, sum(N[:2]):sum(N[:3])], axis= -1 )
        w_to_sst = np.mean(wei[:, :, sum(N[:3]):sum(N)], axis= -1 )
        x_length = w_to_cc.shape[0]
        # see full colortable: https://matplotlib.org/3.1.0/gallery/color/named_colors.html
        color_list = ['blue', 'salmon', 'lightseagreen', 'mediumorchid']
        label_list = ['cs pre', 'cc pre', 'pv pre', 'sst pre']
        # Specify the graph
        axs = axes.flatten()[j]
        # Different weight type
        for ind,plotwei in enumerate([w_to_cs,w_to_cc, w_to_pv, w_to_sst]):
            # Different cell numbers
            for i in range(plotwei.shape[1]):
                axs.plot(np.linspace(0, Ttau, x_length), plotwei[:, i], c = color_list[ind], label = label_list[ind], alpha = 0.5)
        
        name = ['CS', 'CC', 'PV','SST']
        axs.set_title(f"postsyn(col):{name[j]}")
    
    # Set legend content and location
    handles, labels = axs.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),loc = 'lower center', ncol = 4, bbox_to_anchor=(0.5, 0))

    #save graph
    fig.tight_layout(pad=2.0)

    now = datetime.now() # current date and time
    DateFolder = now.strftime('%m_%d')
    if os.path.exists(f'data/{DateFolder}') == False:
        os.makedirs(f'data/{DateFolder}')
    time_id = now.strftime("%m%d_%H:%M")

    time_id = datetime.now().strftime("%m%d_%H:%M")
    title_save = f'data/{DateFolder}/{learningrule}_{sim}_{time_id}_weight.png'
    if saving == True:
        fig.savefig(title_save)

def sim_eva(weights, activity, N):
    '''
    weight: 4D matrix (sim, Tstep, N(post-syn), N(pre-syn))
    activity: 3D matrix (sim, Tstep, N)

    Tvar: time-series average variance for individual weights
    Svar: neuron-wide weight variance for all different neurons
    Smean: mean value for weight the final product
    Avar: Variance of activity across neuron
    '''
    Tvar = Svar = Smean = np.empty([weights.shape[0], 4, 4])
    Avar = np.empty([activity.shape[0],4])

    for sim in range(weights.shape[0]):
        weights_vec = weights[sim]
        # weights for different post-syn
        weight_cs = weights_vec[:, :N[0], :]
        weight_cc = weights_vec[:, sum(N[:1]):sum(N[:2]), :]
        weight_pv = weights_vec[:, sum(N[:2]):sum(N[:3]), :]
        weight_sst = weights_vec[:, sum(N[:3]):sum(N), :]
        weights_vector = [weight_cs, weight_cc, weight_pv, weight_sst]

        #Activity variance
        Act_vec = np.mean(activity[sim, -20:, :], axis = 0)
        Avar[sim, :] = [np.var(Act_vec[:N[0]]), 
                        np.var(Act_vec[sum(N[:1]):sum(N[:2])]),
                        np.var(Act_vec[sum(N[:2]):sum(N[:3])]),
                        np.var(Act_vec[sum(N[:3]):sum(N)]) ]

        # row-based post-syn situation
        for j, wei in enumerate(weights_vector):

            # Time-series variance: time-based variance for the last 400 steps
            mat_tvar = np.var(wei[-400:, :, :], axis= 0)
            # Taking the average of all same value (Post-pre same condition)
            Tvar[sim, j, : ] = [np.mean(mat_tvar[:, :N[0]]),
                            np.mean(mat_tvar[:, sum(N[:1]):sum(N[:2])]),
                            np.mean(mat_tvar[:, sum(N[:2]):sum(N[:3])]),
                            np.mean(mat_tvar[:, sum(N[:3]):sum(N)])]

            # Taking the average of the last 10 value to denoise
            wei = np.mean(wei[-10:, :, :], axis=0)

            # neuron-wise variance (Same condition directly calculate variance)
            Svar[sim, j, : ] = [np.var(wei[:, :N[0]]),
                            np.var(wei[:, sum(N[:1]):sum(N[:2])]),
                            np.var(wei[:, sum(N[:2]):sum(N[:3])]),
                            np.var(wei[:, sum(N[:3]):sum(N)]) ]

            # mean value for expression
            Smean[sim, j, : ] = [np.mean(wei[:, :N[0]]),
                            np.mean(wei[:, sum(N[:1]):sum(N[:2])]),
                            np.mean(wei[:, sum(N[:2]):sum(N[:3])]),
                            np.mean(wei[:, sum(N[:3]):sum(N)]) ]

    # Average over all simulations
    return (Tvar, Svar, Smean, Avar)

## For purely debuging purpose
def find_weights(weights, N):
    def get_sim_weights(weights_vec):
        weight_cs = weights_vec[:, :N[0], :]
        weight_cc = weights_vec[:, sum(N[:1]):sum(N[:2]), :]
        weight_pv = weights_vec[:, sum(N[:2]):sum(N[:3]), :]
        weight_sst = weights_vec[:, sum(N[:3]):sum(N), :]
        weights_vector = [weight_cs, weight_cc, weight_pv, weight_sst]
        weight_eva = np.empty((4,4))
        for j, wei in enumerate(weights_vector):
            wei = np.mean(wei[-10:, :, :], axis=0)
            weight_eva[j, :] = [np.mean(wei[:, :N[0]]),
                            np.mean(wei[:, sum(N[:1]):sum(N[:2])]),
                            np.mean(wei[:, sum(N[:2]):sum(N[:3])]),
                            np.mean(wei[:, sum(N[:3]):sum(N)]) ]
        return weight_eva

    if len(weights.shape) == 3:
        return get_sim_weights(weights)
    elif len(weights.shape) == 4: 
        Smean = np.empty([weights.shape[0], 4, 4])
        for sim in range(weights.shape[0]):
            Smean[sim, :, :] = get_sim_weights(weights_vec=weights[sim])
        return Smean 
    else:
        print('dimension wrong (must be 3d or 4d)')



def lossfun(Smean, Tvar, Svar, Avar, Activity, w_target, prob, MaxAct):
    '''
    Smean, Tvar, Svar, Avar: results from evaluation metic
    Activity: Activity matrix to calcualte the out-of-range distribution (with panelty)
    w_initial: designated final outcome 
    sim: no. of simulation

    Returning the RMSE of the eventual results. The evaluation metric consists of the 
    mean euclidean distance between the final value and the designated target, plus the 
    variance value in terms of both time and among nuerons
    '''
    
    # average time variance over simulation and taking sum
    # Svar_sum = np.sum(np.mean(Svar, axis = 0))
    Tvar_sum = np.sum(np.mean(Tvar, axis = 0))
    Avar_mean = np.mean(Avar) # Just take the average activity across differen type is enough
    Reg_factor = 0.1

    # Taking the root mean sAquare log error(RMSLE) panelty for out-of-range activity
    Activity = abs(np.mean(Activity[:, -20:, :], axis = 1).flatten() - 0.5*MaxAct)
    Aor = np.log1p(Activity) - np.log1p(0.5*MaxAct)
    if len(Aor[Aor>0]) == 0: 
        Aor_rmsle = 0 
    else:
        Aor_rmsle = np.sqrt(np.mean(np.square(Aor[Aor > 0])))

    # mean euclidean distance, w_target need to be transposed as the row should mean post-syn neurons
    eigenval = 2.959173309858104
    Smean_flat = Smean.reshape(Smean.shape[0],-1)
    w_target_flat = (w_target*prob/eigenval).T.flatten()
    rmse = np.sqrt(np.mean(np.square(Smean_flat - w_target_flat)))
    
    return abs(100*rmse + Aor_rmsle - Reg_factor*(Avar_mean - Tvar_sum))
    
def Stable_sim_loss(activity, Max_act = 20): 
    '''
    evaluating the amount of capping happening in the curent situation
    '''
    residual_value = activity.flatten() - Max_act
    Total_Capped = len(residual_value > -0.01)
    No_Neuron_Capped = np.sum(np.any(activity-Max_act>-0.01, axis = 1))

    return No_Neuron_Capped * Total_Capped