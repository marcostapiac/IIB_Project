import sys

import matplotlib.pyplot as plt

sys.path.extend(['../Classes', '../Plotter'])
from ClassLevyJumpProcesses import GIGSubordinator
from ClassNormalVarianceMeanProcesses import NormalVarianceMeanProcess
from ClassPlotter import TimeSeriesPlotter
import numpy as np
from tqdm import tqdm


def path_simulation(t1=0.0, t2=1.00, num_obs=100, num_epochs=2000, subordinator_truncation=0.0, mu=0.0, mu_W=0.0,
                    var_W=1.0, delta=0.8, gamma=1.0,
                    lambd=-0.1, nProcesses=10):
    time_ax = np.linspace(t1, t2, num_obs)
    ax = plt.subplot()
    sub = GIGSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, delta=delta, gamma=gamma, lambd=lambd)
    process = NormalVarianceMeanProcess(mu, mu_W, var_W, sub)
    title = "Sample GH Paths with $\\mu, \\mu_{W}, \\sigma_{W}, \\delta, \\gamma, \\lambda = " + str(mu) + " ," + str(
        mu_W) + " ," + str(var_W) + " ," + str(delta) + " ," + str(gamma) + " ," + str(
        lambd) + " ," + "$ at time $t = 1.0$"
    plotter = TimeSeriesPlotter(time_ax, "", xlabel="Time", ylabel="Position", plottitle=title, ax=ax)
    for _ in tqdm(range(nProcesses)):
        plotter.set_y1(process.generate_path())
        plotter.plot()
    plt.show()


path_simulation(nProcesses=10)
