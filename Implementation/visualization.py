import os 
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import Implementation.helper as helper

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

def weights_barplot(weight_df, **kwargs):
    """
    Plot the bar plot for different types of data
    """
    color_list = kwargs['color_list']
    p = kwargs['config']
    saving = kwargs['saving']
    learning_rule = kwargs['learning_rule']
    DateFolder, time_id = helper.create_data_dir(config=p)


    fig_w, ax_w = plt.subplots(1,2, figsize=(15, 5))
    x_pos_w = np.arange(weight_df.shape[1])
    title_list_w = ['Mean weight','$\Delta$weight']
    for i in range(2):
        ax_w[i].bar(x_pos_w, list(weight_df.iloc[3*i,]), 
                    yerr = list(weight_df.iloc[3*i+2,]), color = np.repeat(color_list, 4),
                    align='center', alpha=0.5)
        ax_w[i].errorbar(x_pos_w, list(weight_df.iloc[3*i,]), yerr = list(weight_df.iloc[3*i+2,]),
                    fmt = '-o', capsize = 10, ecolor = 'black')
        ax_w[i].set_ylabel(title_list_w[i])
        ax_w[i].set_xticks(x_pos_w)
        ax_w[i].set_xticklabels(weight_df.columns)
        ax_w[i].set_title(title_list_w[i])
        ax_w[i].yaxis.grid(True)
        ax_w[i].margins(x=0.02)
        plt.setp(ax_w[i].get_xticklabels(), rotation=30, horizontalalignment='right')

    fig_w.tight_layout(pad=1.0)
    fig_w.suptitle(f"Weight by {learning_rule}", y = 1)
    fig_w.show()
    if saving:
        fig_w.savefig(f'data/{DateFolder}/{time_id}_{learning_rule}_wei.png', dpi=100)

def selectivity_barplot(selectivity_df, selectivity_df_bl, **kwargs):

    fig_size = kwargs['fig_size']
    color_list = kwargs['color_list']
    p = kwargs['config']
    saving = kwargs['saving']
    learning_rule = kwargs['learning_rule']
    DateFolder, time_id = helper.create_data_dir(config=p)

    fig_s, ax_s = plt.subplots(2,2, figsize=fig_size)
    bar_width = 0.6
    x_pos_s = np.arange(selectivity_df.shape[1])
    x_pos_b = [x + 0.2 for x in x_pos_s]
    title_list_s = ['Mean activity',                'Orientational selectivity (OS)',
                    'Directional selectivity (DS)', 'Orientational selectivity_p (OS_p)']
    
    for i in range(4):
        axs = ax_s.flatten()[i]
        axs.bar(x_pos_s, list(selectivity_df_bl.iloc[3*i,]), color = color_list[0],
                align='center', alpha=0.5, label = 'before', width = bar_width)
        axs.bar(x_pos_b, list(selectivity_df.iloc[3*i,]), color = color_list[1],
                align='center', alpha=0.5, label = 'after', width = bar_width)
        axs.errorbar(x_pos_s, list(selectivity_df_bl.iloc[3*i,]),yerr = list(selectivity_df_bl.iloc[3*i+2,]),
                    fmt = '-o', capsize = 10, ecolor = 'black')
        axs.errorbar(x_pos_b, list(selectivity_df.iloc[3*i,]),yerr = list(selectivity_df.iloc[3*i+2,]),
            fmt = '-o', capsize = 10, ecolor = 'black')         
        axs.set_xticks([x + 0.5*bar_width for x in x_pos_s])
        axs.set_xticklabels(selectivity_df.columns)
        axs.set_ylabel(title_list_s[i])
        axs.set_ylim(bottom = -0.1, top = None)
        axs.set_title(title_list_s[i])
        axs.yaxis.grid(True) 
        axs.margins(x=0.02)
        axs.legend()
    
    fig_s.tight_layout(pad=1.0)
    fig_s.suptitle(f"Activity by {learning_rule}", y = 1)
    fig_s.show()
    if saving: 
        fig_s.savefig(f'data/{DateFolder}/{time_id}_{learning_rule}_act_OS.png', dpi=100)

def activity_plot(act_plot_dic, **kwargs):

    fig_size = kwargs['fig_size']
    color_list = kwargs['color_list']
    neuron_list = kwargs['neuron_list']
    line_col = kwargs['line_col']
    p = kwargs['config']
    learning_rule = kwargs['learning_rule']
    saving = kwargs['saving']


    DateFolder, time_id = helper.create_data_dir(config=p)
    mean_act_neuron = act_plot_dic['mean_act']
    error_act_neuron = act_plot_dic['std_act']

    # start the plot
    fig_ap, ax_ap = plt.subplots(2,2, figsize=fig_size)    # 4 input orientations
    x_plot = np.linspace(0, p.Ttau, mean_act_neuron.shape[-1])
    
    for i in range(4):  
        axs_ap = ax_ap.flatten()[i]
        act = mean_act_neuron[:,i]   # activities of same radian (neurontypes, 50)
        err = error_act_neuron[:,i]  # std of the same radian

        for j in range(act.shape[0]):
            axs_ap.plot(x_plot, act[j], 
                        color = line_col[j], label = neuron_list[j])
            axs_ap.fill_between(x_plot, act[j]-err[j], act[j]+err[j],
                                alpha=0.2, facecolor= color_list[j])
        axs_ap.margins(x=0)
        axs_ap.set_ylabel("Firing rates")
        axs_ap.set_xlabel("Time steps")
        axs_ap.set_title(f"Degree: {p.degree[i]}")

    handles_a, labels_a = axs_ap.get_legend_handles_labels()
    by_label_a = dict(zip(labels_a, handles_a))
    fig_ap.legend( by_label_a.values(), 
                by_label_a.keys(),
                loc = 'lower center', 
                ncol = 4, bbox_to_anchor=(0.5, -0.02))
    fig_ap.tight_layout(pad=1.0)
    fig_ap.suptitle(f"Activity by {learning_rule}", y=1)
    fig_ap.show()
    if saving: 
        fig_ap.savefig(f'data/{DateFolder}/{time_id}_{learning_rule}_plot_act.png', dpi=100)

def weights_plot(wei_plot_dic, **kwargs):

    fig_size = kwargs['fig_size']
    color_list = kwargs['color_list']
    neuron_list = kwargs['neuron_list']
    line_col = kwargs['line_col']
    p = kwargs['config']
    learning_rule = kwargs['learning_rule']
    saving = kwargs['saving']

    DateFolder, time_id = helper.create_data_dir(config=p)
    mean_weights = wei_plot_dic['mean_weights']
    std_weights = wei_plot_dic['std_weights']
    x_plot = np.linspace(0, p.Ttau, mean_weights.shape[0])

        # starting the plot
    fig_wp, ax_wp = plt.subplots(2,2, figsize=fig_size)    # 4 post-synaptic neurons
    label_list_w = ['cs pre', 'cc pre', 'pv pre', 'sst pre']

    for i in range(4):           # 4 postsyn neurons
        axs_wp = ax_wp.flatten()[i]
        weis = mean_weights[:,i]   # postsynaptic neurons, (50, presyn)
        err = std_weights[:,i]    

        for j in range(weis.shape[-1]):     
            axs_wp.fill_between(x_plot, weis[:,j]-err[:,j], weis[:,j]+err[:,j],
                                alpha=0.2, facecolor= color_list[j])
            axs_wp.plot(x_plot, weis[:,j], 
                        color = line_col[j], label = label_list_w[j])
            # plotting the reference line
            axs_wp.axhline(y = p.w_compare[i,j], linewidth = 2, 
                        color = color_list[j], linestyle = '--')

        axs_wp.margins(x=0)
        axs_wp.set_ylabel("weights")
        axs_wp.set_xlabel("Time steps")
        axs_wp.set_title(f"postsyn(col):{neuron_list[i]}")
    
    handles, labels = axs_wp.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig_wp.legend( by_label.values(), 
                by_label.keys(),
                loc = 'lower center', 
                ncol = 4, bbox_to_anchor=(0.5, -0.02))
    fig_wp.tight_layout(pad=1.0)
    fig_wp.suptitle(f"Weights by {learning_rule}", y = 1)
    fig_wp.show()
    if saving:
        fig_wp.savefig(f'data/{DateFolder}/{time_id}_{learning_rule}_plot_wei.png', dpi=100)

def activity_histogram(activity_df, **kwargs):

    fig_size = kwargs['fig_size']
    color_list = kwargs['color_list']
    p = kwargs['config']
    learning_rule = kwargs['learning_rule']
    saving = kwargs['saving']

    DateFolder, time_id = helper.create_data_dir(config=p)
    fig_ad, ax_ad = plt.subplots(2, 2, figsize=fig_size, gridspec_kw=dict(width_ratios=[1, 1]))
    for i in range(4):
        axs_ad = ax_ad.flatten()[i]
        sns.histplot(activity_df, x = activity_df.columns[i], hue='Degree', kde=True, 
                    stat="density", fill = True, alpha = 0.2, 
                    palette = color_list, multiple="layer",
                    common_norm=False, ax = ax_ad.flatten()[i])
        
        axs_ad.margins(x=0.02)
        axs_ad.set_xlabel("Firing rate")
        axs_ad.set_title(f"{activity_df.columns[i]}")
    fig_ad.tight_layout(pad=1.0)
    fig_ad.suptitle(f"Activity distribution by {learning_rule}", verticalalignment = 'top', y = 1)
    fig_ad.show()
    if saving:
        fig_ad.savefig(f'data/{DateFolder}/{time_id}_{learning_rule}_dis_act.png', dpi=100)