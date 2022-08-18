import sys

import matplotlib.pyplot as plt

sys.path.extend(['../Classes', '../Plotter'])
from ClassLevyJumpProcesses import AlphaStableSubordinator
from ClassPlotter import TimeSeriesPlotter
import numpy as np


def path_simulation(t1=0.0, t2=10.0, num_obs=100, num_epochs=2000, subordinator_truncation=0.0, alpha=0.8, gamma=1.0, nProcesses=10):
    time_ax = np.linspace(t1, t2, num_obs)
    ax = plt.subplot()
    sub = AlphaStableSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, alpha=alpha,
                                  gamma=gamma)
    title = "Sample Alpha Stable Paths with $\\alpha, \\gamma = " + str(alpha) + " ," + str(gamma) + " ," + "$ at time $t = 1.0$"
    plotter = TimeSeriesPlotter(time_ax, "", xlabel="Time", ylabel="Position", plottitle=title, ax=ax)
    for i in range(nProcesses):
        plotter.set_y1(sub.generate_path())
        plotter.plot()
    plt.show()


path_simulation(nProcesses=10)
