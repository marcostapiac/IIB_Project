import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from plotting_functions import *
from simulate_process_functions import generate_residual_process_at_T


def CLT2(mu=0, nu=0, gamma=0, mu_W=0, std_W=0,
         T_horizon=0, N_Processes=1, N_epochs=1000, truncation=1e-13):
    # Define random process variance
    # getcontext().prec = 17
    # a = Decimal((2 * nu / gamma ** 2) * (std_W ** 2 + 2 * mu_W ** 2 / gamma ** 2))
    # b = Decimal((2 * nu * mu_W ** 2) / gamma ** 2)
    # variance = Decimal(a - np.exp(-truncation * gamma ** 2 / 2) * (b * truncation + a))
    # epoch_cutoff = nu * special.exp1(float(truncation * gamma ** 2 / 2))
    epoch_cutoff = 2000
    error_process = generate_residual_process_at_T(mu, nu, gamma, mu_W, std_W, T_horizon, N_Processes, N_epochs,
                                                epoch_cutoff)

    pval = kstest(error_process, 'norm', [np.mean(error_process), np.std(error_process)]).pvalue
    fig2, ax2 = plt.subplots()
    fig2.figsize = (108, 22)
    plot_histogram_normal(error_process,
                          label="NG Residuals Histogram: $h(\Gamma_{i}) < $" + str(
                              truncation) + " and pval $\\approx" + str(
                              round(pval * 1e144, 3)) + "\\times$10e-144", ax=ax2)
    plt.savefig("Hist")
    fig4, ax4 = plt.subplots()
    fig4.figsize = (18, 22)
    plot_qq(error_process,
            np.random.normal(np.mean(error_process), np.std(error_process), size=error_process.shape[0]), ax=ax4,
            label="NG Residuals QQ Plot: $h(\Gamma_{i}) < $" + str(truncation) + " and pval $\\approx" + str(
                round(pval * 1e144, 3)) + "\\times$10e-144")
    print(pval)
    plt.show()


CLT2(mu=0, nu=2, gamma=np.sqrt(2), mu_W=0, std_W=1, T_horizon=1, N_Processes=10000, N_epochs=10000,
     truncation=1e-320)
