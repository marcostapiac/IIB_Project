import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.extend(['../Classes', '../Plotter'])
from ClassLevyJumpProcesses import AlphaStableSubordinator
from ClassNormalVarianceMeanProcesses import NormalVarianceMeanProcess
from ClassPlotter import HistogramPlotter
import numpy as np
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

def verify_CLT(t1=0.0, t2=1.0, num_obs=100, num_epochs=1000, subordinator_truncation=0.0, mu=0., mu_W=0.0,
               var_W=1.0, alpha=0.9, gamma=2.0, nProcesses=10000):
    sub = AlphaStableSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, alpha=alpha, gamma=gamma)
    process = NormalVarianceMeanProcess(mu=mu, mu_W=mu_W, var_W=var_W, subordinator=sub)
    proc_vals = []
    for _ in tqdm(range(nProcesses)):
        proc_vals.append(np.sum(process.generate_small_jumps()))

    proc_vals = (proc_vals - np.mean(proc_vals)) / np.std(proc_vals)
    pdf_vals = norm.pdf(np.linspace(norm.ppf(0.00001), norm.ppf(0.99999), proc_vals.shape[0]), loc=np.mean(proc_vals))
    title = "Histogram for Residual Normal Alpha Stable with $\\alpha, \\gamma = " + str(alpha) + " ," + str((t2-t1)*gamma) + " $"

    plotter = HistogramPlotter(x1samples=proc_vals, pdf_values=pdf_vals, xlabel="X", ylabel="PDF",
                               plotlabel="Standard Normal Distribution", plottitle=title)

    plotter.plot_normal()
    plt.show()
    # plt.savefig("NAlphaCLTHistogram.pgf", bbox_inches='tight')



verify_CLT(subordinator_truncation=1e-2,nProcesses=100000)
