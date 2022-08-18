import sys

import matplotlib.pyplot as plt

sys.path.extend(['../Classes', '../Plotter'])
from ClassLevyJumpProcesses import GammaSubordinator
from ClassNormalVarianceMeanProcesses import NormalVarianceMeanProcess
from ClassPlotter import TimeSeriesPlotter
import numpy as np


def path_simulation(t1=0.0, t2=10.0, num_obs=100, num_epochs=2000, subordinator_truncation=0.0, mu=0., mu_W=0.0, var_W=1.0, nu=1.0, gamma=2.0, nProcesses=10):
    time_ax = np.linspace(t1, t2, num_obs)
    ax = plt.subplot()
    sub = GammaSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, nu=nu,
                                  gamma=gamma)
    process = NormalVarianceMeanProcess(mu=mu, mu_W=mu_W, var_W=var_W, subordinator=sub)
    title = "Sample Normal Alpha Stable Paths with $\\mu, \\mu_{W}, \\sigma_{w}, \\nu, \\gamma = "+ str(mu) + " ,"+ str(mu_W) + " ,"+ str(var_W) + " ," + str(nu) + " ," + str(gamma) + " ," + "$ at time $t = 1.0$"
    plotter = TimeSeriesPlotter(time_ax, "", xlabel="Time", ylabel="Position", plottitle=title, ax=ax)
    for _ in range(nProcesses):
        plotter.set_y1(process.generate_path())
        plotter.plot()
    plt.show()


path_simulation(nProcesses=10)
