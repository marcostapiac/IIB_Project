from simulate_process_functions import *
from rvs_generator_functions import *
from plotting_functions import plot_qq, plot_histogram_GIG
from matplotlib import pyplot as plt
from scipy.stats import ks_2samp, kstest
from plotting_functions import plot_histogram_normal

N_Jumps = 100000
lambd = 0.1
delta = 2
gamma_param = 0.1

GIG_process_samples = GIG_jumps(delta=delta, gamma_param=gamma_param, lambd=lambd, N_Jumps=N_Jumps)
GIG_rvs = np.array([x[0] for x in generate_GIG_rvs(lambd, gamma_param ** 2, delta ** 2, N_Jumps)])

fig, ax = plt.subplots(figsize=(14, 8))
plot_qq(GIG_process_samples, GIG_rvs, ax=ax)
fig, ax2 = plt.subplots()
plot_histogram_GIG(GIG_process_samples, lambd, gamma_param, delta, ax2)
pval = ks_2samp(GIG_process_samples, GIG_rvs).pvalue
print(pval)
ax.set_title(
    "GIG Process Sample with $\delta, \gamma, \lambda, pval= " + str((delta, gamma_param, lambd, round(pval, 5))) + "$")
ax2.set_title(
    "GIG Process Sample with $\delta, \gamma, \lambda, pval = " + str(
        (delta, gamma_param, lambd, round(pval, 5))) + "$")
plt.show()
