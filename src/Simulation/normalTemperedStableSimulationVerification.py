import sys

sys.path.extend(['../Classes', '../Plotter'])
from ClassLevyJumpProcesses import TemperedStableSubordinator
from ClassNormalVarianceMeanProcesses import NormalVarianceMeanProcess
from ClassPlotter import QQPlotter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def verify_simulation(t1=0.0, t2=1.0, num_obs=100, num_epochs=2000, subordinator_truncation=0.0, mu=0., mu_W=0.0,
                      var_W=1.0, kappa=0.8, delta=0.7, gamma=1.0, nProcesses=10):
    sub = TemperedStableSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, kappa=kappa, delta=delta,
                                     gamma=gamma)
    process = NormalVarianceMeanProcess(mu=mu, mu_W=mu_W, var_W=var_W, subordinator=sub)
    proc_vals = []
    for _ in tqdm(range(nProcesses)):
        proc_vals.append(np.sum(process.generate_jumps()))
    rvs = process.marginal_samples(nProcesses, t2 - t1)
    title = "Q-Q Plot for Normal Tempered Stable  with $\\mu, \\mu_{W}, \\sigma_{w}, \\kappa,  \\delta, \\gamma = " + str(
        mu) + " ," + str(mu_W) + " ," + str(var_W) + " ," + str(kappa) + " ," + str(
        (t2 - t1) * delta) + " ," + str(gamma) + "$ at time $t = T_{horizon}$"
    plotter = QQPlotter(rvs, proc_vals, ylabel="Marginal Random Variables of Normal Tempered Stable Process", log=False,
                        plottitle=title)
    plotter.plot()
    plt.show()


verify_simulation(mu=0.0, mu_W=0.0, var_W=1.0, nProcesses=100000)
