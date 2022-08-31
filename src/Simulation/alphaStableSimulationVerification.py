import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.extend(['../Classes', '../Plotter'])
from ClassLevyJumpProcesses import AlphaStableSubordinator
from ClassPlotter import QQPlotter
import numpy as np


def verify_simulation(t1=0.0, t2=5.0, num_obs=100, num_epochs=1000, subordinator_truncation=0.0, alpha=0.9,  gamma=2.0, nProcesses=10000):
    sub = AlphaStableSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, alpha=alpha, gamma=gamma)
    vals = []
    for _ in tqdm(range(nProcesses)):
        sub.generate_jumps(sub.generate_epochs())
        vals.append(np.sum(sub.get_jump_sizes()))
    rvs = sub.marginal_samples(nProcesses, t2 - t1)
    title = "Q-Q Plot for Alpha Stable Subordinator with $\\alpha, \\gamma = " + str(alpha) + " ," + str((t2-t1)*gamma) + " $"

    plotter = QQPlotter(rvs, vals, ylabel="Marginal Random Variables of Alpha Stable Subordinator", log=True,
                        plottitle=title)
    plotter.plot()
    plt.show()


verify_simulation(nProcesses=100000)
