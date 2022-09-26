from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
import numpy as np
import torch
import matplotlib as mpl
from sbi_on_model import run_simulation

# remove top and right axis from plots
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

def simulation_wrapper(params):
    """
    Returns summary statistics from conductance values in `params`.

    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """
    summstats = torch.as_tensor(run_simulation(params))
    return summstats

prior_min = [0,0,0,0,0,0,0,0,0,0,0]
prior_max = [1,1,1,1,6,6,6,6,2,2,2]

prior = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
)

posterior = infer(
    simulation_wrapper, prior, method="SNPE", num_simulations=2, num_workers=4
)


observation_summary_statistics = [1,1,1,1,1,1,1,1,1,1,1,0,0]
samples = posterior.sample((10000,), x=observation_summary_statistics)

limits = [[0,1],[0,1],[0,1],[0,1],
          [0,6],[0,6],[0,6],[0,6],
          [0,2],[0,2],[0,2]]
fig, axes = analysis.pairplot(
    samples,
    limits=limits,
    ticks=limits,
    figsize=(5, 5),
    points=true_params,
    points_offdiag={"markersize": 6},
    points_colors="r",
);

fig.savefig('data/figures/sbi.png')