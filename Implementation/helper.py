import numpy as np
np.random.seed(42)
import math


def distributionInput(spatialF, temporalF, orientation, spatialPhase, amplitude, T, steady,
                      input_cs, input_cc, input_pv, input_sst, N):
    """
    Generates a moving bar as input to CS, CC, PV, SST.

    """

    a_data = np.cos(np.random.uniform(0, np.pi, (np.sum(N),)))
    b_data = np.sin(np.random.uniform(0, np.pi, (np.sum(N),)))

    inputs = []

    if not (steady):
        for t in range(T):
            inputs.append(np.abs(amplitude * np.cos(
                spatialF * a_data * np.cos(orientation) + spatialF * b_data * np.sin(
                    orientation) - spatialPhase) * np.cos(
                temporalF * t)))
    else:
        for t in range(T):
            inputs.append(np.abs(
                amplitude * np.cos(spatialF * a_data * np.cos(orientation) + spatialF * b_data * np.sin(orientation))))

    inputs = np.array(inputs)

    if input_cs != 'bar':
        inputs[:, :N[0]] = input_cs
    if input_cc != 'bar':
        inputs[:, N[0]:sum(N[:2])] = input_cc
    if input_pv != 'bar':
        inputs[:, sum(N[:2]):sum(N[:3])] = input_pv
    if input_sst != 'bar':
        inputs[:, sum(N[:3]):] = input_sst

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