import sys

import matplotlib.pyplot as plt

sys.path.extend(['../Classes', '../Plotter'])
from ClassLevyJumpProcesses import TemperedStableSubordinator
from ClassNormalVarianceMeanProcesses import NormalVarianceMeanProcess
from ClassPlotter import TimeSeriesPlotter
import numpy as np


def path_simulation(t1=0.0, t2=10.0, num_obs=100, num_epochs=2000, subordinator_truncation=0.0, mu=0.0, mu_W=0.0, var_W=1.0,kappa=0.7, delta=1.5,  gamma=1.0, nProcesses=10):
    time_ax = np.linspace(t1, t2, num_obs)
    fig, ax = plt.subplots()
    sub = TemperedStableSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, kappa=kappa, delta=delta, gamma=gamma)
    process = NormalVarianceMeanProcess(mu=mu, mu_W=mu_W, var_W=var_W, subordinator=sub)
    title = "Sample Normal Tempered Stable Paths with with $\\mu, \\mu_{W}, \\sigma_{w}, \\kappa, \\delta, \\gamma = "+ str(mu) + " ,"+ str(mu_W) + " ,"+ str(var_W) + " ," +  str(kappa) + " ," + str(delta) + " ," + str(gamma) + " $ at time $t = 1$"
    plotter = TimeSeriesPlotter(time_ax, "", xlabel="Time", ylabel="Position", plottitle=title, fig=fig, ax=ax)
    for _ in range(nProcesses):
        plotter.set_y1(process.generate_path())
        plotter.plot()
    plt.show()


path_simulation(nProcesses=10)
