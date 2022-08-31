import sys

sys.path.extend(['../Classes', '../Plotter'])
from ClassLevyJumpProcesses import GammaSubordinator
from ClassNormalVarianceMeanProcesses import NormalVarianceMeanProcess
from ClassPlotter import HistogramPlotter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import matplotlib

# matplotlib.use("pgf")
matplotlib.rcParams.update({
    # "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 10.5,
    # "pgf.preamble": "\n".join([
    #    r"\usepackage[utf8]{inputenc}\DeclareUnicodeCharacter{2212}{-}",
    # ])
})


def verify_CLT(t1=0.0, t2=1.0, num_obs=200, num_epochs=2000, subordinator_truncation=0.0, mu=0., mu_W=0.0,
               var_W=1.0, nu=1.0, gamma=2.0, nProcesses=10):
    sub = GammaSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation=subordinator_truncation, nu=nu,
                            gamma=gamma)
    process = NormalVarianceMeanProcess(mu=mu, mu_W=mu_W, var_W=var_W, subordinator=sub)
    proc_vals = []
    for _ in tqdm(range(nProcesses)):
        proc_vals.append(np.sum(process.generate_small_jumps()))
    # TODO: INVESTIGATE
    proc_vals = (proc_vals - np.mean(proc_vals)) / np.std(proc_vals)
    pdf_vals = norm.pdf(np.linspace(norm.ppf(0.00001), norm.ppf(0.99999), proc_vals.shape[0]), loc=np.mean(proc_vals))
    title = "Histogram for Residual Normal Gamma with $\\mu, \\mu_{W}, \\sigma_{w}, \\nu, \\gamma = " + str(
        mu) + " ," + str(mu_W) + " ," + str(var_W) + " ," + str(nu) + " ," + str(
        (t2 - t1) * gamma) + "$ at time $t = T_{horizon}$"
    plotter = HistogramPlotter(x1samples=proc_vals, pdf_values=pdf_vals, xlabel="X", ylabel="PDF",
                               plotlabel="Standard Normal Distribution", plottitle=title)

    plotter.plot_normal()
    plt.show()
    # plt.savefig("NTSCLTHistogram.pgf", bbox_inches='tight')

verify_CLT(subordinator_truncation=1e-2, nProcesses=100000)
