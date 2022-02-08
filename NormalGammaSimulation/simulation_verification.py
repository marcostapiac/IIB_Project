from simulate_process_functions import *
from rvs_generator_functions import *
from plotting_functions import *
from scipy.stats import ks_2samp

N_Jumps = 50000
N_epochs = 1000
mu = 0
mu_W = 0
std_W = 1
nu=2
gamma=np.sqrt(2)

#process(N_Processes=3)
processes = process(mu=mu, nu=nu, gamma=gamma, mu_W=mu_W, std_W=std_W, T_horizon=1, N_Processes=N_Jumps, N_epochs=1000)
NG_jumps = []
for process in processes:
    NG_jumps.append(process[-1])
NG_jumps = np.array(NG_jumps)

NG_rvs = generate_RVs(mu, nu, gamma, mu_W, std_W, L=NG_jumps.shape[0])[0]
fig, ax = plt.subplots(figsize=(14,8))
plot_qq(NG_jumps, NG_rvs, ax=ax)
pval = ks_2samp(NG_jumps, NG_rvs).pvalue
ax.set_title("NG QQ Plot: pval $\\approx" + str(round(pval , 5)) + "$")

plt.xlabel("NG RVS")
plt.ylabel("NG Process at t=1")
print(pval)
plt.show()
