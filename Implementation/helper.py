import numpy as np
np.random.seed(42)
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

# remove top and right axis from plots
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

def distributionInput(spatialF, temporalF, orientation, spatialPhase, amplitude, T, steady_input,
                      input_cs, input_cc, input_pv, input_sst, N):
    """
    Generates a moving bar as input to CS, CC, PV, SST.

    """

    a_data = np.cos(np.random.uniform(0, np.pi, (np.sum(N),)))
    b_data = np.sin(np.random.uniform(0, np.pi, (np.sum(N),)))

    inputs = []
    for t in range(T):
        inputs.append(np.abs(amplitude * np.cos(
            spatialF * a_data * np.cos(orientation) + spatialF * b_data * np.sin(
                orientation) - spatialPhase) * np.cos(
            temporalF * t)))

    inputs = np.array(inputs)

    # static input (if neurons don't receive moving bar input)
    if input_cs != 'bar':
        inputs[:, :N[0]] = input_cs
    if input_cc != 'bar':
        inputs[:, N[0]:sum(N[:2])] = input_cc
    if input_pv != 'bar':
        inputs[:, sum(N[:2]):sum(N[:3])] = input_pv
    if input_sst != 'bar':
        inputs[:, sum(N[:3]):] = input_sst

    return (inputs)


def distributionInput_sbi(spatialF, temporalF, orientation, spatialPhase, amplitude, T, steady_input, N):
    """
    Generates a moving bar as input to CS, CC, PV, SST.

    """

    a_data = np.cos(np.random.uniform(0, np.pi, (np.sum(N),)))
    b_data = np.sin(np.random.uniform(0, np.pi, (np.sum(N),)))

    i = 0
    inputs_p_all = []
    N_indices = [[0, N[0]], [sum(N[:1]), sum(N[:2])], [sum(N[:2]), sum(N[:3])], [sum(N[:3]), sum(N)]]
    for popu in N_indices:
        inputs_p = []

        if steady_input[i] < 0.5:
            inputs_p = np.ones((T, N[i])) * amplitude[i]
        else:
            for t in range(T):
                inputs_p.append(np.abs(amplitude[i] * np.cos(
                    spatialF * a_data[popu[0]:popu[1]] * np.cos(orientation) +
                    spatialF * b_data[popu[0]:popu[1]] * np.sin(orientation) - spatialPhase)
                                       * np.cos(temporalF * t)))
            inputs_p = np.array(inputs_p)

        i += 1
        inputs_p_all.append(inputs_p)

    inputs = np.concatenate((inputs_p_all), axis=1)

    return (inputs)

def create_synapses(N_pre, N_post, c, same_population=False):
    """
    Create random connections between two groups or within one group.
    :param N_pre: number of neurons in pre group
    :param N_post: number of neurons in post group
    :param c:   connectivity
    :param same_population: whether to allow autapses (connection of neuron to itself if pre = post population)
    :return: 2xN_con array of connection indices (pre-post pairs)
    """

    indegree = int(np.round(c * N_pre))  # no variance of indegree = fixed indegree

    i = np.array([], dtype=int)
    j = np.array([], dtype=int)

    for n in range(N_post):

        if same_population:  # if autapses are disabled, remove index of present post neuron from pre options
            opts = np.delete(np.arange(N_pre, dtype=int), n)
        else:
            opts = np.arange(N_pre, dtype=int)

        pre = np.random.choice(opts, indegree, replace=False)

        # add connection indices to list
        i = np.hstack((i, pre))
        j = np.hstack((j, np.repeat(n, indegree)))

    return np.array([i, j])

def normal_distr_weights(weight, w_noise):
    """
    Generate weights with normally distributed weight noise.

    """

    if weight > 0:
        weight = np.abs(np.random.normal(weight, w_noise))
    elif weight < 0:
        weight = -np.abs(np.random.normal(weight, w_noise))
    return (weight)


def generate_connectivity(N, p, w_initial, w_noise):
    """
    Generates a connectivity matrix where rows are postsyn. neuron and columns are presynaptic neuron.

    """

    W_rec = np.zeros((np.sum(N), np.sum(N)))

    for pre in range(p.shape[0]):
        for post in range(p.shape[0]):
            same_population = False
            if pre == post:
                same_population = True
            con = create_synapses(N[pre], N[post], p[pre][post], same_population)
            for i in range(con.shape[1]):
                pre_neuron = con[0][i]
                post_neuron = con[1][i]
                step_pre = np.sum(N[:pre])
                step_post = np.sum(N[:post])
                W_rec[step_pre + pre_neuron][step_post + post_neuron] = normal_distr_weights(w_initial[pre][post],
                                                                                             w_noise)
    return (W_rec.T)

def calculate_selectivity_sbi(activity_popu):
    """
    Calculate mean and std of selectivity.

    """

    os_mean_data = []  # orientation selectivity
    ds_mean_data = []  # directions selectivity
    ds_paper_mean_data = []  # directions selectivity calculated as in paper

    for population in range(len(activity_popu)):
        preferred_orientation = np.argmax(activity_popu[population], axis=0)

        os, ds, ds_paper = [], [], []
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

            os.append((s_pref - s_orth) / (s_pref + s_orth))
            ds.append((s_pref_orient - s_oppo) / (s_pref_orient + s_oppo))
            ds_paper.append((s_pref - s_oppo) / (s_pref + s_oppo))

        os_mean_data.append(np.mean(os))
        ds_mean_data.append(np.mean(ds))
        ds_paper_mean_data.append(np.mean(ds_paper))

    return (os_mean_data,ds_mean_data,ds_paper_mean_data)

def calculate_selectivity(activity_popu):
    """
    Calculate mean and std of selectivity.

    """

    os_mean_data = []  # orientation selectivity
    os_std_data = []
    ds_mean_data = []  # directions selectivity
    ds_std_data = []
    ds_paper_mean_data = []  # directions selectivity calculated as in paper
    ds_paper_std_data = []

    for population in range(4):
        preferred_orientation = np.argmax(activity_popu[population], axis=0)

        os, ds, ds_paper = [], [], []
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

            os.append((s_pref - s_orth) / (s_pref + s_orth))
            ds.append((s_pref_orient - s_oppo) / (s_pref_orient + s_oppo))
            ds_paper.append((s_pref - s_oppo) / (s_pref + s_oppo))

        os_mean_data.append(np.mean(os))
        os_std_data.append(np.std(os))
        ds_mean_data.append(np.mean(ds))
        ds_std_data.append(np.std(ds))
        ds_paper_mean_data.append(np.mean(ds_paper))
        ds_paper_std_data.append(np.std(ds_paper))

    return (os_mean_data, os_std_data,ds_mean_data,ds_std_data,ds_paper_mean_data,ds_paper_std_data)

def plot_activity(activity, N, title):
    if len(activity) == 0:
        return(0)
    activity_cs = activity[:, :, :N[0]]
    activity_cc = activity[:, :, sum(N[:1]):sum(N[:2])]
    activity_pv = activity[:, :, sum(N[:2]):sum(N[:3])]
    activity_sst = activity[:, :, sum(N[:3]):sum(N)]

    for g in range(activity_cs.shape[0]): # degrees
        fig,axs = plt.subplots()
        for i in range(activity_cs.shape[2]):
            plt.plot(range(activity_cs.shape[1]),activity_cs[g,:,i],c='grey',alpha=0.5)
        plt.title('CS')
        title_save = title+ '/' + str(g)+ '_CS.png'
        fig.savefig(title_save)

        fig, axs = plt.subplots()
        for i in range(activity_cc.shape[2]):
            plt.plot(range(activity_cc.shape[1]), activity_cc[g,:,i], c='grey', alpha=0.5)
        plt.title('CC')
        title_save = title + '/' + str(g) + '_CC.png'
        fig.savefig(title_save)

        fig, axs = plt.subplots()
        for i in range(activity_pv.shape[2]):
            plt.plot(range(activity_pv.shape[1]), activity_pv[g,:,i], c='grey', alpha=0.5)
        plt.title('PV')
        title_save = title + '/' + str(g) + '_PV.png'
        fig.savefig(title_save)

        fig, axs = plt.subplots()
        for i in range(activity_sst.shape[2]):
            plt.plot(range(activity_sst.shape[1]), activity_sst[g,:,i], c='grey', alpha=0.5)
        plt.title('SST')
        title_save = title + '/' + str(g) + '_SST.png'
        fig.savefig(title_save)

