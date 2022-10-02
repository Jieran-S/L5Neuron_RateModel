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

# remove top and right axis from plots
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

start_time = time.time()

def simulation_wrapper(params):
    """
    Returns summary statistics from conductance values in `params`.

    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """
    summstats = torch.as_tensor(run_simulation(params))
    return summstats

def run_HH_model(params):

    params = np.asarray(params)

    # input current, time step
    I, t_on, t_off, dt, t, A_soma = syn_current()

    t = np.arange(0, len(I), 1) * dt

    # initial voltage
    V0 = -70

    states = HHsimulator(V0, params.reshape(1, -1), dt, t, I)

    return dict(data=states.reshape(-1), time=t, dt=dt, I=I.reshape(-1))

def simulation_wrapper2(params):
    """
    Returns summary statistics from conductance values in `params`.

    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """
    obs = run_HH_model(params)
    summstats = torch.as_tensor(calculate_summary_statistics(obs))
    return summstats


prior_min = [0,0,0,0,0,0,0,0,0,0,0] # np.ones((11,))*1e-4
prior_max = [1,1,1,1,6,6,6,6,2,2,2]

prior = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
)

posterior = infer(
    simulation_wrapper, prior, method="SNPE", num_simulations=300, num_workers=4
)

print('#### t1',time.time()-start_time)

observation_summary_statistics = [1,1,1,1,1,1,1,1,-1,1,1,0,0]

samples = posterior.sample((10000,), x=observation_summary_statistics)

print('#### t2',time.time()-start_time)

limits = [[0,1],[0,1],[0,1],[0,1],
          [0,6],[0,6],[0,6],[0,6],
          [0,2],[0,2],[0,2]]
fig, axes = analysis.pairplot(
    samples,
    limits=limits,
    figsize=(5, 5),
    points_offdiag={"markersize": 6},
    points_colors="r",
);

fig.savefig('data/figures/sbi.png')

print('#### t3',time.time()-start_time)