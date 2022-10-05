from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
import numpy as np
import torch
import matplotlib as mpl
from sbi_on_model import run_simulation
from HH_helper_functions import calculate_summary_statistics
from HH_helper_functions import HHsimulator
from HH_helper_functions import syn_current
import time
import pickle

def run_HH_model(params):

    params = np.asarray(params)

    # input current, time step
    I, t_on, t_off, dt, t, A_soma = syn_current()

    t = np.arange(0, len(I), 1) * dt

    # initial voltage
    V0 = -70

    states = HHsimulator(V0, params.reshape(1, -1), dt, t, I)

    return dict(data=states.reshape(-1), time=t, dt=dt, I=I.reshape(-1))

def simulation_wrapper(params):
    """
    Returns summary statistics from conductance values in `params`.

    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """
    obs = run_HH_model(params)
    summstats = torch.as_tensor(calculate_summary_statistics(obs))
    return summstats

with open('data/my_inference.pkl', 'rb') as f:
    posterior = pickle.load(f)

observation_summary_statistics = [1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 0, 0]

# true parameters and respective labels
true_params = np.array([50.0, 5.0])
labels_params = [r"$g_{Na}$", r"$g_{K}$"]
observation_trace = run_HH_model(true_params)
observation_summary_statistics = calculate_summary_statistics(observation_trace)

samples = posterior.sample((10000,), x=observation_summary_statistics)

limits = [[0, 1], [0, 1], [0, 1], [0, 1],
          [0, 6], [0, 6], [0, 6], [0, 6],
          [0, 2], [0, 2], [0, 2]]

limits = [[0.5, 80], [1e-4, 15.0]]
fig, axes = analysis.pairplot(
    samples,
    limits=limits,
    figsize=(5, 5),
    points_offdiag={"markersize": 6},
    points_colors="r",
);

fig.savefig('data/figures/sbi2.png')