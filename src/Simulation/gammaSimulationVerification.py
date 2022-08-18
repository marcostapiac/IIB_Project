import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import gamma as gammaDist

sys.path.extend(['../Classes', '../Plotter'])
from ClassLevyJumpProcesses import GammaSubordinator
from ClassPlotter import QQPlotter, HistogramPlotter
import numpy as np


def verify_simulation(t1=0.0, t2=10., num_obs=100, num_epochs=1000, subordinator_truncation=0.0, nu=2.0,
                            gamma=1.0,
                            nProcesses=10000):
    g_sub = GammaSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, nu=nu, gamma=gamma)
    gvals = []
    for _ in tqdm(range(nProcesses)):
        g_sub.generate_jumps(g_sub.generate_epochs())
        gvals.append(np.sum(g_sub.get_jump_sizes()))
    rvs = g_sub.marginal_samples(nProcesses, t2 - t1)
    title = "Q-Q Plot for Gamma Process with $\\nu, \\gamma = " + str((t2 - t1) * nu) + " ," + str(gamma) + " $"
    plotter = QQPlotter(rvs, gvals, ylabel="Marginal Random Variables of Gamma Process", log=False, plottitle=title)
    plotter.plot()

    linspace = np.linspace(min(gvals), max(gvals), len(gvals))
    pdfVs = gammaDist.pdf(linspace, (t2 - t1) * nu, loc=0, scale=2 / (gamma ** 2))
    title = "Histogram of " + str(nProcesses) + " Gamma process paths with $\\nu, \\gamma = " + str((t2 - t1) * nu) + "," + str(gamma) + " $ "
    plotter = HistogramPlotter(gvals, xlabel="Jump Sizes", ylabel="Probability Density", plottitle = title, plotlabel="Gamma PDF at $t= T_{horizon}$", bins=100, pdf_values=pdfVs)
    plotter.plot()
    plt.show()


verify_simulation(nProcesses=100000)
