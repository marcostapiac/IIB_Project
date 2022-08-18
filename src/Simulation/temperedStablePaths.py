import sys

import matplotlib.pyplot as plt

sys.path.extend(['../Classes', '../Plotter'])
from ClassLevyJumpProcesses import TemperedStableSubordinator
from ClassPlotter import TimeSeriesPlotter
import numpy as np


def path_simulation(t1=0.0, t2=10.0, num_obs=100, num_epochs=2000, subordinator_truncation=0.0, kappa=0.7, delta=1.5,  gamma=1.0, nProcesses=10):
    time_ax = np.linspace(t1, t2, num_obs)
    ax = plt.subplot()
    g_sub = TemperedStableSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, kappa=kappa, delta=delta, gamma=gamma)
    title = "Sample Tempered Stable Paths with $\\kappa, \\delta, \\gamma = " + str(kappa) + " ," + str(delta) + " ," + str(gamma) + " $ at time $t = 1$"
    plotter = TimeSeriesPlotter(time_ax, "", xlabel="Time", ylabel="Position", plottitle=title, ax=ax)
    for i in range(nProcesses):
        plotter.set_y1(g_sub.generate_path())
        plotter.plot()
    plt.show()


path_simulation(nProcesses=10)
