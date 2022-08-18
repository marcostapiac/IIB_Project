import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import gamma as gammaDist

sys.path.extend(['../Classes', '../Plotter'])
from ClassLevyJumpProcesses import TemperedStableSubordinator
from ClassPlotter import QQPlotter
import numpy as np


def verify_simulation(t1=0.0, t2=2.0, num_obs=100, num_epochs=1000, subordinator_truncation=0.0, kappa=0.5,
                         delta=2.0, gamma=1.26,
                         nProcesses=10000):
    sub = TemperedStableSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, kappa=kappa, delta=delta,
                                       gamma=gamma)
    vals = []
    for _ in tqdm(range(nProcesses)):
        sub.generate_jumps(sub.generate_epochs())
        vals.append(np.sum(sub.get_jump_sizes()))
    rvs = sub.marginal_samples(nProcesses, t2 - t1)
    title = "Q-Q Plot for Tempered Stable Process with $\\kappa, \\delta, \\gamma = " + str(kappa) + " ," + str(
        (t2 - t1) * delta) + " ," + str(gamma) + " $"
    plotter = QQPlotter(rvs, vals, ylabel="Marginal Random Variables of Tempered Stable Process", log=False,
                        plottitle=title)
    plotter.plot()
    plt.show()


verify_simulation(nProcesses=1000000)
