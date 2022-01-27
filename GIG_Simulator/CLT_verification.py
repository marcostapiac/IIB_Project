from tqdm import tqdm
import numpy as np
from scipy.stats import kstest
from simulate_process_functions import hankel_squared
from plotting_functions import plot_histogram_normal
from matplotlib import pyplot as plt
from scipy.stats import gamma
from scipy.special import gammainc, gammaincc, gammaincinv
from scipy.special import gamma as gammaf


def error_GIG_simple_Jumps(delta, lambd, T_horizon, epochs):
    # GIG component for large truncation
    ts_C = (T_horizon * delta * gammaf(0.5)) / (np.sqrt(2) * np.pi)
    ts_alpha = 0.5
    x = (ts_alpha * epochs / ts_C) ** (-1 / ts_alpha)
    zs = np.sqrt(gamma.rvs(0.5, loc=0, scale=1 / (x / (2 * delta ** 2)),
                           size=len(x)))
    u = np.random.uniform(0., 1., len(x))
    prob_acc = 2 / (np.pi * zs * hankel_squared(np.abs(lambd), zs))
    GIG_component_jump_sizes = x[(u < prob_acc)]
    return GIG_component_jump_sizes


def generate_simple_error_process_at_T(mu, mu_W, std_W, delta, gamma_param, lambd, N_Processes, T_horizon, N_epochs,
                                       epoch_cutoff):
    process_at_T = []

    for _ in tqdm(range(N_Processes)):
        epochs = np.cumsum(np.random.exponential(scale=1 / 1, size=N_epochs)) / T_horizon
        epochs = np.array([epoch for epoch in epochs if epoch > epoch_cutoff])
        jumps = error_GIG_simple_Jumps(delta, lambd, T_horizon, epochs)
        if lambd > 0:
            # GIG component for large truncation
            beta = gamma_param ** 2 / 2
            gamma_C = max(0, lambd)
            x = 1 / (beta * (np.exp(epochs / gamma_C) - 1))
            prob_acc = (1 + beta * x) * np.exp(-beta * x)
            # Rejection sampling
            u = np.random.uniform(0., 1., size=x.size)
            gamma_component_jumps = x[(u < prob_acc)]
            jumps = np.append(jumps, gamma_component_jumps)
        sum_jumps = np.sum(jumps)
        square_root_jumps = np.sqrt(sum_jumps)
        error_val = np.random.normal(loc=mu_W * sum_jumps, scale=std_W * square_root_jumps)
        process_at_T.append(error_val)

    process_at_T = np.array(process_at_T)
    return process_at_T


def CLT_GHsimple(mu, mu_W, std_W, delta, gamma_param, lambd, N_Processes, T_horizon, N_epochs):
    assert (np.absolute(lambd) >= 0.5)
    epoch_cutoff = 2000
    error_process = generate_simple_error_process_at_T(mu, mu_W, std_W, delta, gamma_param, lambd, N_Processes,
                                                       T_horizon, N_epochs, epoch_cutoff)
    fig, ax = plt.subplots()
    fig.figsize = (18, 18)
    plot_histogram_normal(error_process, ax=ax)
    pvalue = kstest(error_process, 'norm', [np.mean(error_process), np.std(error_process)]).pvalue
    ax.set_title("GIG Residuals where $\delta, \gamma, \lambda, \Gamma_i, pval= " + str(
        (delta, gamma_param, lambd, epoch_cutoff, round(pvalue, 5))) + "$")
    print(kstest(error_process, 'norm', [np.mean(error_process), np.std(error_process)]))
    plt.show()


def error_GIG_harder_Jumps(delta, gamma_param, lambd, T_horizon, N_epochs, epoch_cutoff):
    a = np.pi * np.power(2.0, (1.0 - 2.0 * np.abs(lambd)))
    b = gammaf(np.abs(lambd)) ** 2
    c = 1 / (1 - 2 * np.abs(lambd))
    z1 = (a / b) ** c
    H0 = z1 * hankel_squared(np.abs(lambd), z1)
    """N1"""
    # Generate gamma process
    epochs = np.cumsum(np.random.exponential(scale=1 / 1, size=N_epochs)) / T_horizon
    epochs = np.array([epoch for epoch in epochs if epoch > epoch_cutoff])
    beta = gamma_param ** 2 / 2
    gamma_C = z1 / (np.pi * np.pi * np.absolute(lambd) * H0)  # Shape parameter of process at t = 1
    x = 1 / (beta * (np.exp(epochs / gamma_C) - 1))
    prob_acc = (1 + beta * x) * np.exp(-beta * x)
    # Rejection sampling
    u = np.random.uniform(0., 1., size=x.size)
    jump_sizes = x[(u < prob_acc)]
    """ Rejection sampling from Algorithm 6 """
    const1 = (z1 ** 2) * jump_sizes / (2 * delta ** 2)
    GIG_prob_acc = np.absolute(lambd) * gammaf(np.abs(lambd)) * gammainc(np.abs(lambd), const1) / (
            ((z1 ** 2) * jump_sizes / (2 * delta ** 2)) ** np.abs(lambd))
    u = np.random.uniform(0., 1., size=jump_sizes.size)
    jump_sizes = jump_sizes[(u < GIG_prob_acc)]

    """ Sample from truncated Nakagami """
    C1 = np.random.uniform(0., 1., size=jump_sizes.size)
    l = C1 * gammainc(np.absolute(lambd), (z1 ** 2 * jump_sizes) / (2 * delta ** 2))
    zs = np.sqrt(((2 * delta ** 2) / jump_sizes) * gammaincinv(np.absolute(lambd), l))

    """ Thinning for process N1 """
    u = np.random.uniform(0., 1., size=jump_sizes.size)
    N1_prob_acc = H0 / (hankel_squared(np.abs(lambd), zs) *
                        (zs ** (2 * np.abs(lambd))) / (z1 ** (2 * np.abs(lambd) - 1)))
    N1 = jump_sizes[(u < N1_prob_acc)]

    """N2"""
    """ Generate Tempered Stable Jump Size samples """
    alpha = 0.5
    beta = (gamma_param ** 2) / 2
    C = np.sqrt(2 * delta ** 2) * gammaf(0.5) / ((np.pi ** 2) * H0)
    epochs = np.cumsum(np.random.exponential(scale=1 / 1, size=N_epochs)) / T_horizon
    epochs = np.array([epoch for epoch in epochs if epoch > epoch_cutoff])
    x = ((alpha * epochs) / C) ** (-1 / alpha)
    prob_acc = np.exp(-beta * x)
    u = np.random.uniform(0.0, 1.0, size=prob_acc.size)
    jump_sizes = x[(u < prob_acc)]

    """ Rejection sampling based on Algorithm 7 """
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


def generate_harder_error_process_at_T(mu, mu_W, std_W, delta, gamma_param, lambd, N_Processes, T_horizon, N_epochs,
                                       epoch_cutoff):
    process_at_T = []
    for _ in tqdm(range(N_Processes)):
        jumps = error_GIG_harder_Jumps(delta, gamma_param, lambd, T_horizon, N_epochs, epoch_cutoff)
        if lambd > 0:
            epochs = np.cumsum(np.random.exponential(scale=1 / 1, size=N_epochs)) / T_horizon
            epochs = np.array([epoch for epoch in epochs if epoch > epoch_cutoff])
            beta = gamma_param ** 2 / 2
            gamma_C = max(0, lambd)
            x = 1 / (beta * (np.exp(epochs / gamma_C) - 1))
            prob_acc = (1 + beta * x) * np.exp(-beta * x)
            # Rejection sampling
            u = np.random.uniform(0., 1., size=x.size)
            gamma_component_jumps = x[(u < prob_acc)]
            jumps = np.append(jumps, gamma_component_jumps)
        sum_jumps = np.sum(jumps)
        square_root_jumps = np.sqrt(sum_jumps)
        error_val = np.random.normal(loc=mu_W * sum_jumps, scale=std_W * square_root_jumps)
        process_at_T.append(error_val)

    process_at_T = np.array(process_at_T)
    return process_at_T


def CLT_GHharder(mu, mu_W, std_W, delta, gamma_param, lambd, N_Processes, T_horizon, N_epochs):
    assert (np.absolute(lambd) < 0.5)
    epoch_cutoff = 3000
    error_process = generate_harder_error_process_at_T(mu, mu_W, std_W, delta, gamma_param, lambd, N_Processes, T_horizon, N_epochs, epoch_cutoff)
    fig, ax = plt.subplots()
    fig.figsize = (18, 18)
    plot_histogram_normal(error_process, ax=ax)
    pvalue = kstest(error_process, 'norm', [np.mean(error_process), np.std(error_process)]).pvalue
    ax.set_title("GIG Residuals where $\delta, \gamma, \lambda, \Gamma_i, pval= " + str(
        (delta, gamma_param, lambd, epoch_cutoff, round(pvalue, 5))) + "$")
    print(kstest(error_process, 'norm', [np.mean(error_process), np.std(error_process)]))
    plt.show()


def CLT(mu=0, mu_W=0, std_W=1, delta=2, gamma_param=-0.8, lambd=0.8, N_Processes=15000, T_horizon=1,
        N_epochs=10000):
    if np.abs(lambd) >= 0.5:
        CLT_GHsimple(mu, mu_W, std_W, delta, gamma_param, lambd, N_Processes, T_horizon, N_epochs)
    else:
        CLT_GHharder(mu, mu_W, std_W, delta, gamma_param, lambd, N_Processes, T_horizon, N_epochs)


CLT(delta=2, N_Processes=100000, lambd=-0.1)
