import sys

import matplotlib.pyplot as plt

sys.path.extend(['../Classes', '../Plotter'])
from ClassLevyJumpProcesses import GIGSubordinator
from ClassNormalVarianceMeanProcesses import NormalVarianceMeanProcess
from ClassPlotter import QQPlotter
import numpy as np
from tqdm import tqdm


def verification_simulation(t1=0.0, t2=1.0, num_obs=100, num_epochs=2000, subordinator_truncation=0.0, mu=0.0, mu_W=0.0, var_W= 1.0, delta=0.8,
                            gamma=1.0,
                            lambd=-0.1, nProcesses=10000):
    sub = GIGSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, delta=delta, gamma=gamma, lambd=lambd)
    process = NormalVarianceMeanProcess(mu=mu, mu_W=mu_W, var_W=var_W, subordinator=sub)
    title = "Q-Q Plot of GH Paths with $\\mu, \\mu_{W}, \\sigma_{W}, \\delta, \\gamma, \\lambda = "+ str(mu) + " ,"+ str(mu_W) + " ,"+ str(var_W) + " ," + str(delta) + " ," + str(gamma) + " ," + str(
        lambd) + " ," + "$ at time $t = T_{horizon}$"
    proc_vals = []
    for _ in tqdm(range(nProcesses)):
        proc_vals.append(np.sum(process.generate_jumps(t2 - t1)))
    rvs = sub.marginal_samples(nProcesses, t2 - t1)
    plotter = QQPlotter(rvs, proc_vals, ylabel="Marginal Random Variables of Generalised Hyperbolic Process", log=False, plottitle=title)
    plotter.plot()

    plt.show()


verification_simulation(nProcesses=10000)
