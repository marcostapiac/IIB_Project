from tqdm import tqdm
import numpy as np
from scipy.stats import kstest
from simulate_process_functions import hankel_squared
from plotting_functions import plot_histogram_normal
from matplotlib import pyplot as plt
from scipy.stats import gamma
from scipy.special import gammainc, gammaincc, gammaincinv
from scipy.special import gamma as gammaf


def residual_GIG_simple_Jumps(delta, lambd, T_horizon, N_epochs, epoch_cutoff):
    # GIG component for large truncation
    epochs = np.cumsum(np.random.exponential(scale=1 / 1, size=N_epochs)) / T_horizon
    epochs = np.array([epoch for epoch in epochs if epoch > epoch_cutoff])
    ts_C = (T_horizon * delta * gammaf(0.5)) / (np.sqrt(2) * np.pi)
    ts_alpha = 0.5
    x = (ts_alpha * epochs / ts_C) ** (-1 / ts_alpha)
    zs = np.sqrt(gamma.rvs(0.5, loc=0, scale=1 / (x / (2 * delta ** 2)),
                           size=len(x)))
    u = np.random.uniform(0., 1., len(x))
    prob_acc = 2 / (np.pi * zs * hankel_squared(np.abs(lambd), zs))
    GIG_component_jump_sizes = x[(u < prob_acc)]
    return GIG_component_jump_sizes


def generate_simple_residual_process_at_T(mu, mu_W, std_W, delta, gamma_param, lambd, N_Processes, T_horizon, N_epochs,
                                       epoch_cutoff):
    process_at_T = []
    for _ in tqdm(range(N_Processes)):
        jumps = residual_GIG_simple_Jumps(delta, lambd, T_horizon, N_epochs, epoch_cutoff)
        if lambd > 0:
            epochs = np.cumsum(np.random.exponential(scale=1 / 1, size=N_epochs)) / T_horizon
            epochs = np.array([epoch for epoch in epochs if epoch > epoch_cutoff])
            # GIG component for large truncation
            beta = gamma_param ** 2 / 2
            gamma_C = lambd
            x = np.exp(-epochs / gamma_C) / (beta)
            # Rejection sampling
            gamma_component_jumps = x
            jumps = np.append(jumps, gamma_component_jumps)
        sum_jumps = np.sum(jumps)
        square_root_jumps = np.sqrt(sum_jumps)
        error_val = np.random.normal(loc=mu_W * sum_jumps, scale=std_W * square_root_jumps)
        process_at_T.append(error_val)

    process_at_T = np.array(process_at_T)
    return process_at_T


def CLT_GHsimple(mu, mu_W, std_W, delta, gamma_param, lambd, N_Processes, T_horizon, N_epochs, epoch_cutoff):
    assert (np.absolute(lambd) >= 0.5)
    print(lambd)
    residual_process = generate_simple_residual_process_at_T(mu, mu_W, std_W, delta, gamma_param, lambd, N_Processes,
                                                       T_horizon, N_epochs, epoch_cutoff)
    fig, ax = plt.subplots()
    fig.figsize = (18, 18)
    plot_histogram_normal(residual_process, ax=ax)
    pvalue = kstest(residual_process, 'norm', [np.mean(residual_process), np.std(residual_process)]).pvalue
    ax.set_title("GIG Residuals where $\delta, \gamma, \lambda, \Gamma_i, pval= " + str(
        (delta, gamma_param, lambd, epoch_cutoff, round(pvalue, 5))) + "$")
    print(kstest(residual_process, 'norm', [np.mean(residual_process), np.std(residual_process)]))
    plt.show()


def error_GIG_harder_Jumps(delta, gamma_param, lambd, T_horizon, N_epochs, epoch_cutoff):
    a = np.pi * np.power(2.0, (1.0 - 2.0 * np.abs(lambd)))
    b = gammaf(np.abs(lambd)) ** 2
    c = 1 / (1 - 2 * np.abs(lambd))
    z1 = (a / b) ** c
    H0 = z1 * hankel_squared(np.abs(lambd), z1)
    # Generate gamma process
    epochs = np.cumsum(np.random.exponential(scale=1 / 1, size=N_epochs)) / T_horizon
    epochs = np.array([epoch for epoch in epochs if epoch > epoch_cutoff])
    beta = gamma_param ** 2 / 2
    gamma_C = z1 / (np.pi * np.pi * np.absolute(lambd) * H0)  # Shape parameter of process at t = 1
    # Approximation to gamma process for large jumps
    x = np.exp(-(epochs) / gamma_C) / beta
    jump_sizes = x
    """ Rejection sampling from Algorithm 6 (NOT needed because gammainc is small) """
    """jump_sizes_copy = np.exp(-epochs / gamma_C) / beta
    const1 = (z1 ** 2) * jump_sizes_copy / (2 * delta ** 2)
    GIG_prob_acc = np.absolute(lambd) * gammaf(np.abs(lambd)) * gammainc(np.abs(lambd), const1) / (
            const1 ** np.abs(lambd))
    # Really tiny jumps accepted: asymptotic version of gammainc at x->0:
    GIG_prob_acc[gammainc(np.abs(lambd), const1) == 0] = 1
    print(GIG_prob_acc)
    u = np.random.uniform(0., 1., size=jump_sizes.size)
    jump_sizes = jump_sizes[(u < GIG_prob_acc)]"""

    """ Sample from truncated Nakagami (for small enough jumps, all zs are 0! """
    """C1 = np.random.uniform(0., 1., size=jump_sizes.size)
    l = C1 * gammainc(np.absolute(lambd), (z1 ** 2 * jump_sizes) / (2 * delta ** 2))
    zs = np.sqrt(((2 * delta ** 2) / jump_sizes) * gammaincinv(np.absolute(lambd), l))
    zs[np.isnan(zs) == True] = 0 # Asymptotic expansion of density leads to z_i = 0 for 0/0"""
    """ Thinning for process N1 (believe asymptotic expansion for N1_prob_acc is 1 since zs are 0) """
    """u = np.random.uniform(0., 1., size=zs.size)
    N1_prob_acc = H0 / (hankel_squared(np.abs(lambd), zs) *
                        (zs ** (2 * np.abs(lambd))) / (z1 ** (2 * np.abs(lambd) - 1)))
    N1 = jump_sizes[(u < N1_prob_acc)]"""
    N1 = jump_sizes
    """N2"""
    """ Generate Tempered Stable Jump Size samples """
    alpha = 0.5
    beta = (gamma_param ** 2) / 2
    C = np.sqrt(2 * delta ** 2) * gammaf(0.5) / ((np.pi ** 2) * H0)
    print
    epochs = np.cumsum(np.random.exponential(scale=1 / 1, size=N_epochs)) / T_horizon
    epochs = np.array([epoch for epoch in epochs if epoch > epoch_cutoff])
    x = ((alpha * epochs) / C) ** (-1 / alpha)

    prob_acc = np.exp(-beta * x)
    u = np.random.uniform(0.0, 1.0, size=prob_acc.size)
    jump_sizes = x[(u < prob_acc)]

    """ Rejection sampling based on Algorithm 7 (GIG_probs very close to 1 when x is order e-08)"""
    GIG_prob_acc = gammaincc(0.5, (z1 ** 2) * jump_sizes / (2 * delta ** 2))
    u = np.random.uniform(0., 1., size=jump_sizes.size)
    jump_sizes = jump_sizes[(u < GIG_prob_acc)]

    # Simulate Truncated Square-Root Gamma Process:
    C2 = np.random.uniform(low=0.0, high=1.0, size=jump_sizes.size)
    zs = np.sqrt(
        ((2 * delta ** 2) / jump_sizes) * gammaincinv(0.5,
                                                      C2 * (gammaincc(0.5, (z1 ** 2) * jump_sizes / (2 * delta ** 2)))
                                                      + gammainc(0.5, (z1 ** 2) * jump_sizes / (2 * delta ** 2))))
    """Thinning for process N2"""
    u = np.random.uniform(0., 1., size=jump_sizes.size)
    N2_prob_acc = H0 / (zs * hankel_squared(np.abs(lambd), zs))
    N2 = jump_sizes[(u < N2_prob_acc)]
    jump_sizes = np.append(N1, N2)
    return jump_sizes


def generate_harder_residual_process_at_T(mu, mu_W, std_W, delta, gamma_param, lambd, N_Processes, T_horizon, N_epochs,
                                       epoch_cutoff):
    process_at_T = []
    for _ in tqdm(range(N_Processes)):
        jumps = error_GIG_harder_Jumps(delta, gamma_param, lambd, T_horizon, N_epochs, epoch_cutoff)
        if lambd > 0:
            epochs = np.cumsum(np.random.exponential(scale=1 / 1, size=N_epochs)) / T_horizon
            epochs = np.array([epoch for epoch in epochs if epoch > epoch_cutoff])
            beta = gamma_param ** 2 / 2
            gamma_C = lambd
            x = (1 / beta) * np.exp(-(epochs) / gamma_C)
            # Rejection sampling
            gamma_component_jumps = x
            jumps = np.append(jumps, gamma_component_jumps)
        sum_jumps = np.sum(jumps)
        square_root_jumps = np.sum(np.sqrt(jumps))
        error_val = np.random.normal(loc=mu + mu_W * sum_jumps, scale=std_W * square_root_jumps)
        process_at_T.append(error_val)

    process_at_T = np.array(process_at_T)
    return process_at_T


def CLT_GHharder(mu, mu_W, std_W, delta, gamma_param, lambd, N_Processes, T_horizon, N_epochs, epoch_cutoff):
    assert (np.absolute(lambd) < 0.5)
    print(lambd)
    residual_process = generate_harder_residual_process_at_T(mu, mu_W, std_W, delta, gamma_param, lambd, N_Processes,
                                                       T_horizon, N_epochs, epoch_cutoff)
    fig, ax = plt.subplots()
    fig.figsize = (18, 18)
    plot_histogram_normal(residual_process, ax=ax)
    print(np.mean(residual_process))
    pvalue = kstest(residual_process, 'norm', [np.mean(residual_process), np.std(residual_process)]).pvalue
    ax.set_title("GIG Residuals where $\delta, \gamma, \lambda, \Gamma_i, NEpochs, pval= " + str(
        (delta, gamma_param, lambd, epoch_cutoff, N_epochs, round(pvalue, 5))) + "$")
    print(kstest(residual_process, 'norm', [np.mean(residual_process), np.std(residual_process)]))
    plt.show()


def CLT(mu=0, mu_W=1, std_W=1, delta=2, gamma_param=-0.8, lambd=0.8, N_Processes=30000, T_horizon=1,
        N_epochs=3000, epoch_cutoff=1000):
    if np.abs(lambd) >= 0.5:
        CLT_GHsimple(mu, mu_W, std_W, delta, gamma_param, lambd, N_Processes, T_horizon, N_epochs, epoch_cutoff)
    else:
        CLT_GHharder(mu, mu_W, std_W, delta, gamma_param, lambd, N_Processes, T_horizon, N_epochs, epoch_cutoff)


CLT(delta=2, N_Processes=30000, lambd=-0.5, gamma_param=2)