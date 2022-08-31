import sys
sys.path.extend(['../Classes', '../Plotter'])
from ClassLevyJumpProcesses import AlphaStableSubordinator
from ClassNormalVarianceMeanProcesses import NormalVarianceMeanProcess
from ClassPlotter import HistogramPlotter
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib
import numpy as np
from tqdm import tqdm
from scipy.stats import norm

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
# Fixing animation params
fig, ax = plt.subplots()
frame_iterable = list(np.logspace(2, -4, 10))
t1 = 0.0
t2 = 1.0
num_obs = 200
num_epochs = 2000
mu = 0.
mu_W = 0.0
var_W = 1.0
alpha = 0.8
gamma = 1.0
nProcesses = 1000


def prepare_animation():
    def animate(frame_number):
        print(frame_number)
        ax.clear()
        subordinator_truncation = frame_iterable[frame_number]
        sub = AlphaStableSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation=subordinator_truncation, alpha=alpha,
                                gamma=gamma)
        process = NormalVarianceMeanProcess(mu=mu, mu_W=mu_W, var_W=var_W, subordinator=sub)
        proc_vals = []
        for _ in tqdm(range(nProcesses)):
            proc_vals.append(np.sum(process.generate_small_jumps()))
        # TODO: INVESTIGATE
        proc_vals = (proc_vals - np.mean(proc_vals)) / np.std(proc_vals)
        pdf_vals = norm.pdf(np.linspace(norm.ppf(0.00001), norm.ppf(0.99999), proc_vals.shape[0]),
                            loc=np.mean(proc_vals))
        title = "Histogram for Residual Normal Alpha Stable with $\\mu, \\mu_{W}, \\sigma_{w}, \\alpha, \\gamma = " + str(
            mu) + " ," + str(mu_W) + " ," + str(var_W) + " ," + str(alpha) + " ," + str(
            (t2 - t1) * gamma) + "$ at time $t = T_{horizon}$"
        plotter = HistogramPlotter(x1samples=proc_vals, pdf_values=pdf_vals, xlabel="X", ylabel="PDF",
                                   plotlabel="Standard Normal Distribution", plottitle=title, fig=fig, ax=ax)

        plotter.plot_normal()

    return animate


ani = animation.FuncAnimation(fig, prepare_animation(), frames=10, repeat=False, blit=False)

ani.save('normalGamma_animation_truncation.gif', fps=1)
plt.show()
