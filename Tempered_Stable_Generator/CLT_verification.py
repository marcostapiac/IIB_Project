from plotting_functions import *
from simulate_process_functions import *
from scipy.stats import kstest


# https://towardsdatascience.com/the-poisson-process-everything-you-need-to-know-322aa0ab9e9a
def CLT_NTS(mu, kappa, delta, gamma, mu_W, std_W, T_horizon, N_Processes, N_epochs, epoch_cutoff):
    error_process = generate_residual_process_at_T(mu, kappa, delta, gamma, mu_W, std_W, T_horizon, N_Processes,
                                                N_epochs, epoch_cutoff)
    fig, ax = plt.subplots()
    fig.figsize = (18, 18)
    plot_histogram_normal(error_process, ax=ax)
    pvalue = kstest(error_process, 'norm', [np.mean(error_process), np.std(error_process)]).pvalue
    ax.set_title("NTS Residuals where $\delta, \gamma, \kappa, \Gamma_i, pval= " + str(
        (delta, gamma, kappa, epoch_cutoff, round(pvalue, 5))) + "$")
    print(kstest(error_process, 'norm', [np.mean(error_process), np.std(error_process)]))
    plt.show()


CLT_NTS(mu=0, kappa=0.5, delta=1, gamma=1.26, mu_W=0, std_W=1, N_Processes=100000, T_horizon=1, N_epochs=1100,
        epoch_cutoff=1000)
