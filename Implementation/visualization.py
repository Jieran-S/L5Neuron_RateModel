import os 
import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime


def plot_activity(activity, config, sim, degree = 0, saving = False):
    '''
    input:
    activity: (simulation, degree, time-step, neurons)
    config: input configuration file
    sim: which simulation to plot
    degree: the index of the degree to plot (1-4)
    saving: if save the plot into the file directory
    ----
    Output:
    A figure of 4 axes showing the activities of CS, CC, PV, SST
    wrt simulation time steps. mean activity value is marked in yellow.

    Noted: if changing the TemporalF input in configuration, change the
    period of taking mean accordingly.
    '''

    N = config.N
    learningrule = config.learning_rule
    Ttau = config.Ttau

    if len(activity) == 0:
        return(0)
    # Extract the connectivity data for each cell population? 
    activity = activity[sim, degree, :, :]
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
            n = 25
            # axs.plot(np.arange(act.shape[0])*5, act[:,i],c='grey',alpha=0.5)
            axs.plot(np.linspace(0, Ttau, act.shape[0]), act[:,i],c='grey',alpha=0.3)

        for i in range(act.shape[1]):
            mean_act = np.average(act[:,i].reshape(-1, n), axis=1)
            axs.plot(np.linspace(0, Ttau, act.shape[0]), np.repeat(mean_act, n),c='orange',alpha=0.8)

        axs.set_title(namelist[ind])

    fig.tight_layout(pad=2.0)

    now = datetime.now() # current date and time
    DateFolder = now.strftime('%m_%d')
    if os.path.exists(f'data/{DateFolder}') == False:
        os.makedirs(f'data/{DateFolder}')
        
    if saving == True:
        time_id = now.strftime("%m%d_%H:%M")

        time_id = datetime.now().strftime("%m%d_%H:%M")
        title_save = f'data/{DateFolder}/{learningrule}_{sim}_{time_id}_act.png'
        fig.savefig(title_save)

def plot_weights(weights, config, sim, degree = 0, saving = False ):

    '''
    input:
    weights: (simulation, degree, time-step, post syn, pre syn)
    config: input configuration file
    sim: which simulation to plot
    degree: the index of the degree to plot (1-4)
    saving: if save the plot into the file directory
    ----
    Output:
    A figure of 4 axes showing the weight where CS, CC, PV and SST are as post-synpatic neurons
    In each axe, there are 4 colors marking the pre-synpatic neuron types, and from different 
    color we can tell the syanaptic strengths over different pre-post synpatic neuron combination

    the vertical line signifies the time learning starts, and the horizontal line is the target weight,
    each target weight is chosen as roughly the same color form the synaptic color to allow easy referencing.
    '''

    N = config.N
    Ttau = config.Ttau
    learningrule = config.learning_rule

    weights = weights[sim, degree, :, :, :]
    weight_cs = weights[:, :N[0], :]
    weight_cc = weights[:, sum(N[:1]):sum(N[:2]), :]
    # weight_pv = weights[:, sum(N[:2]):sum(N[:3]), :]
    # weight_sst = weights[:, sum(N[:3]):sum(N), :]
    weights_vector = [weight_cs, weight_cc] #, weight_pv, weight_sst]

    fig, axes = plt.subplots(1,2)
    for j,wei in enumerate(weights_vector):
        # return the average weight from one cell to a specific responses
        w_to_cs = np.mean(wei[:, :, :N[0]], axis=-1)
        w_to_cc = np.mean(wei[:, :, sum(N[:1]):sum(N[:2])], axis= -1 )
        # w_to_pv = np.mean(wei[:, :, sum(N[:2]):sum(N[:3])], axis= -1 )
        # w_to_sst = np.mean(wei[:, :, sum(N[:3]):sum(N)], axis= -1 )
       
        x_length = w_to_cc.shape[0]
        # see full colortable: https://matplotlib.org/3.1.0/gallery/color/named_colors.html
        color_list = ['blue', 'salmon', 'lightseagreen', 'mediumorchid']
        label_list = ['cs pre', 'cc pre', 'pv pre', 'sst pre']
        
        # Specify the graph
        axs = axes.flatten()[j]
        
        # Different weight type
        for ind,plotwei in enumerate([w_to_cs,w_to_cc]): #, w_to_pv, w_to_sst]):
            # Different cell numbers
            for i in range(plotwei.shape[1]):
                # axs.plot(np.arange(x_length)*5, plotwei[:, i], c = color_list[ind], label = label_list[ind], alpha = 0.5)
                axs.plot(np.linspace(0, Ttau, x_length), 
                         plotwei[:, i], 
                         c = color_list[ind], 
                         label = label_list[ind], 
                         alpha = 0.5)
        
        name = ['CS', 'CC'] #, 'PV','SST']
        axs.set_title(f"postsyn(col):{name[j]}")
    
    # Set legend content and location
    handles, labels = axs.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend( by_label.values(), 
                by_label.keys(),
                loc = 'lower center', 
                ncol = 4, bbox_to_anchor=(0.5, 0))

    #save graph
    fig.tight_layout(pad=2.0)

    now = datetime.now() # current date and time
    DateFolder = now.strftime('%m_%d')
    if os.path.exists(f'data/{DateFolder}') == False:
        os.makedirs(f'data/{DateFolder}')

    if saving == True:
        time_id = now.strftime("%m%d_%H:%M")

        time_id = datetime.now().strftime("%m%d_%H:%M")
        title_save = f'data/{DateFolder}/{learningrule}_{sim}_{time_id}_weight.png'
        fig.savefig(title_save)

def plot_barplot(dataframe, keyword, config):
    """
    Plot the bar plot for different types of data
    """
    ...

def plot_activity_summary(activity, config):
    ...

def plot_distribution(input, config):
    ...