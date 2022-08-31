import sys

import matplotlib.pyplot as plt

sys.path.extend(['../Classes', '../Plotter'])
from ClassLevyJumpProcesses import GIGSubordinator
from ClassPlotter import QQPlotter
import numpy as np
from tqdm import tqdm


def verification_simulation(t1=0.0, t2=1.00, num_obs=100, num_epochs=2000, subordinator_truncation=0.0, delta=2.0,
                            gamma=1.5,
                            lambd=0.4, nProcesses=100000):
    sub = GIGSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, delta=delta, gamma=gamma, lambd=lambd)
    title = "Q-Q Plot of GIG Paths with $\\delta, \\gamma, \\lambda = " + str(delta) + " ," + str(gamma) + " ," + str(
        lambd) + "$ at time $t = T_{horizon}$"
    gigVals = []
    for _ in tqdm(range(nProcesses)):
        sub.generate_jumps(None)  # GIG does NOT take epochs
        gigVals.append(np.sum(sub.get_jump_sizes()))

    rvs = sub.marginal_samples(nProcesses, t2 - t1)
    plotter = QQPlotter(rvs, gigVals, ylabel="Marginal Random Variables of GIG Subordinator", log=True, plottitle=title)
    plotter.plot()

    plt.show()


verification_simulation(nProcesses=10000)
