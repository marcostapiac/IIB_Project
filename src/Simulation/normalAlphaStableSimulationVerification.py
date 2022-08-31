import sys

import matplotlib.pyplot as plt

sys.path.extend(['../Classes', '../Plotter'])
from ClassLevyJumpProcesses import AlphaStableSubordinator
from ClassNormalVarianceMeanProcesses import NormalVarianceMeanProcess
from ClassPlotter import QQPlotter
import numpy as np
from tqdm import tqdm


def verify_simulation(t1=0.0, t2=1.0, num_obs=100, num_epochs=2000, subordinator_truncation=0.0, mu=0.0, mu_W=0.0,
                             var_W=1.0, alpha=0.9, gamma=1.0, nProcesses=10):
    sub = AlphaStableSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, alpha=alpha, gamma=gamma)
    process = NormalVarianceMeanProcess(mu=mu, mu_W=mu_W, var_W=var_W, subordinator=sub)
    proc_vals = []
    for _ in tqdm(range(nProcesses)):
        proc_vals.append(np.sum(process.generate_jumps()))
    rvs = process.marginal_samples(nProcesses, t2 - t1)
    title = "Q-Q Plot for Normal Alpha Stable Process with $\\mu, \\mu_{W}, \\sigma_{w}, \\alpha, \\gamma = " + str(
        mu) + " ," + str(mu_W) + " ," + str(var_W) + " ," + str(alpha) + " ," + str(
        (t2 - t1) * gamma) + "$ at time $t = T_{horizon}$"
    plotter = QQPlotter(rvs, proc_vals, ylabel="Marginal Random Variables of Normal Alpha Stable Process", log=False,
                        plottitle=title)
    plotter.plot()
    plt.show()


verify_simulation(nProcesses=100000)
