import numpy as np
from tqdm import tqdm
from scipy.special import exp1
from scipy.stats import truncexpon, expon
import matplotlib.pyplot as plt


def powerL(elements, index):
    return np.array([elements[i] ** index for i in range(len(elements))])


def gammaJumps(C, beta, epochs):
    """ Generates jumps from a Gamma process with parameters Ga(nu, gamma*gamma/2)"""
    x = beta ** -1 / (np.exp(epochs / C) - 1)
    prob_acc = (1 + beta * x) * np.exp(-beta * x)
    # Rejection sampling
    u = np.random.uniform(0., 1., size=x.size)
    jump_sizes = x[(u < prob_acc)]
    return jump_sizes


def gammaProcess(C, beta, epochs, time_ax, T):
    jump_sizes = gammaJumps(C, beta, epochs)
    jump_times = np.random.uniform(0., T, size=jump_sizes.size)
    gamma_process = np.array([np.cumsum(jump_sizes * (jump_times <= t))[-1] for t in time_ax])
    return gamma_process


def error_gammaJumps(C, beta, epochs, epoch_cutoff):
    # Jump sizes so large they are accepted with probability 1
    x = np.exp(-(epochs - epoch_cutoff) / C) / beta
    return x


def generate_residual_process_at_T(mu=0, nu=2, gamma=2, mu_W=0, std_W=1, T_horizon=1, N_Processes=1, N_epochs=1000,
                                   epoch_cutoff=1000):
    process_at_T = []

    for _ in tqdm(range(N_Processes)):
        epochs = np.cumsum(np.random.exponential(scale=1 / 1, size=N_epochs)) / T_horizon
        epochs = np.array([epoch for epoch in epochs if epoch > epoch_cutoff])
        gamma_Jumps = error_gammaJumps(nu, gamma ** 2 / 2, epochs, epoch_cutoff)
        sum_jumps = np.sum(gamma_Jumps)
        square_root_jumps = np.sum(np.sqrt(gamma_Jumps))  # SUM OF ROOTS
        residual_val = np.random.normal(loc=mu + mu_W * sum_jumps, scale=std_W * square_root_jumps)
        process_at_T.append(residual_val)

    process_at_T = np.array(process_at_T)
    return process_at_T


def process(mu=0, nu=2, gamma=2, mu_W=0, std_W=1, T_horizon=1, N_Processes=10000, N_epochs=1000):
    # Inputs are the  NG(mu, beta, nu, gamma2/2) parameters, conditionally Gaussian parameters (for U_i's),
    # and simulation parameters.
    processes = []
    time_ax = np.linspace(0, T_horizon, 100)
    for _ in tqdm(range(N_Processes)):
        epochs = np.cumsum(expon.rvs(scale=1 / 1, size=N_epochs)) / T_horizon
        gamma_jump_sizes = gammaJumps(nu, gamma, epochs)
        jump_times = np.random.uniform(0.0, T_horizon, size=gamma_jump_sizes.shape[0])
        U_is = np.random.normal(mu_W, std_W, size=gamma_jump_sizes.shape[0])
        deterministic_mu = mu * time_ax
        brownian_drift = mu_W * gamma_jump_sizes
        brownian_motion = np.array(
            [np.sqrt(gamma_jump_sizes[i]) * (U_is[i]) for i in range(gamma_jump_sizes.shape[0])])
        ng_jump_sizes = brownian_drift + brownian_motion
        ng_process = deterministic_mu + np.array(
            [np.sum((ng_jump_sizes * (jump_times <= t))) for t in
             time_ax])
        processes.append(ng_process)

    fig, ax = plt.subplots(figsize=(14, 8))
    for p in processes:
        ax.plot(time_ax, p)  # Plot different NG processes
    plt.grid()
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("Normal Gamma Process Samples for $\mu, \mu_{W}, \sigma_{W}, nu, \gamma =" + str(
        (mu, mu_W, std_W, nu, gamma)) + "$")
    plt.show()
    return processes


def truncated_process(mu=0, nu=2, gamma=2, mu_W=0, std_W=1, T_horizon=1, N_Processes=1, N_epochs=1000, truncation=1e-6):
    # Inputs are the  NG(mu, beta, nu, gamma2/2) parameters, conditionally Gaussian parameters (for U_i's),
    # and simulation parameters.
    processes = []
    epoch_cutoff = nu * exp1(truncation * gamma ** 2 / 2)
    time_ax = np.linspace(0, T_horizon, 100)
    for _ in tqdm(range(N_Processes)):
        epochs = np.cumsum(truncexpon.rvs(b=epoch_cutoff, scale=1 / 1, size=N_epochs)) / T_horizon
        gamma_jump_sizes = gammaJumps(nu, gamma ** 2 / 2, epochs)
        jump_times = np.random.uniform(0.0, T_horizon, size=gamma_jump_sizes.shape[0])
        U_is = np.random.normal(mu_W, std_W, size=gamma_jump_sizes.shape[0])
        deterministic_mu = mu * time_ax
        brownian_drift = mu_W * gamma_jump_sizes
        brownian_motion = np.array(
            [np.sqrt(gamma_jump_sizes[i]) * (U_is[i]) for i in range(gamma_jump_sizes.shape[0])])
        ng_jump_sizes = brownian_drift + brownian_motion
        ng_process = deterministic_mu + np.array(
            [np.sum((ng_jump_sizes * (jump_times <= t))) for t in
             time_ax])
        processes.append(ng_process)
    fig, ax = plt.subplots()
    for p1 in processes:
        ax.plot(time_ax, p1)  # Plot different NG processes
    plt.show()
