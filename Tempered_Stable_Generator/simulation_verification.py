from simulate_process_functions import *
from rvs_generator_functions import *
from plotting_functions import plot_qq, plot_histogram_TS
from matplotlib import pyplot as plt
from scipy.stats import ks_2samp

N_Jumps = 1000
N_epochs= 2000
mu = 0
kappa = 0.5
delta = 1
gamma = 1.26

TS_process_samples = TS_jumps(kappa, delta, gamma, N_Jumps, N_epochs, T_horizon=1)
TS_rvs = np.array([x for x in TS_rv_generator(kappa, delta, gamma, N_Jumps)])

fig, ax = plt.subplots(figsize=(14, 8))
plot_qq(TS_process_samples, TS_rvs, ax=ax)
plt.xlabel("Time")
plt.ylabel("Location")
plt.title("Tempered Stable process")

fig, ax2 = plt.subplots()
plot_histogram_TS(TS_process_samples, kappa, delta, gamma, np.linspace(min(TS_process_samples), max(TS_process_samples), TS_process_samples.shape[0]), ax)
pval = ks_2samp(TS_process_samples, TS_rvs).pvalue
ax.set_title(
    "TS Process Sample with $\kappa, \delta, \gamma, pval= " + str((kappa, delta, gamma, round(pval, 5))) + "$")
ax2.set_title(
    "TS Process Sample with $\kappa, \delta, \gamma, pval= " + str(
        (kappa, delta, gamma, round(pval, 5))) + "$")
plt.show()
