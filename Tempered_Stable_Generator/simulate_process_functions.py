import numpy as np
from scipy.special import gamma as gammaf
from tqdm import tqdm


def TS_jump(alpha, beta, C, N_epochs, T_horizon):
    epochs = np.cumsum(
        np.random.exponential(1, N_epochs)) / T_horizon  # Arrival times according to unit rate Poisson Process
    x = ((alpha * epochs) / C) ** (-1 / alpha)
    prob_acc = np.exp(-beta * x)
    u = np.random.uniform(0.0, 1.0, size=prob_acc.size)
    jump_sizes = x[(u < prob_acc)]
    return jump_sizes


def TS_jumps(kappa=0.5, delta=1, gamma=1.26, N_Processes=1000, N_epochs=2000, T_horizon=1):
    alpha = kappa
    beta = 0.5 * gamma * (1 / kappa)
    C = delta * (2 ** alpha) * alpha * 1 / gammaf(1 - alpha)
    jumps = np.zeros(N_Processes)
    for i in tqdm(range(N_Processes)):
        jumps[i] = (np.sum(TS_jump(alpha, beta, C, N_epochs, T_horizon)))
    return jumps


def TS_residual_jumps(alpha, beta, C, epochs):
    x = ((alpha * epochs) / C) ** (-1 / alpha)
    prob_acc = np.exp(-beta * x)
    u = np.random.uniform(0.0, 1.0, size=prob_acc.size)
    jump_sizes = x[(u < prob_acc)]
    return jump_sizes


def process(kappa=0.5, delta=1, gamma=1.26, N_Processes=1000, N_epochs=2000, T_horizon=1):
    # TS becomes Gamma for kappa = 0
    # TS becomes Levy positive stable for gamma = 0
    # Turn parameterization into Yaman's version
    alpha = kappa
    beta = 0.5 * gamma * (1 / kappa)
    C = delta * (2 ** alpha) * alpha * 1 / gammaf(1 - alpha)
    time_ax = np.linspace(0.0, T_horizon, num=N_epochs)
    processes = []
    for _ in tqdm(range(N_Processes)):
        jump_sizes = TS_jumps(alpha, beta, C, N_epochs, T_horizon)
        jump_times = np.random.uniform(0.0, T_horizon, size=jump_sizes.size)
        tempered_process = np.array([np.sum(jump_sizes * (jump_times <= t)) for t in time_ax])
        processes.append(tempered_process)

    process_at_t = []
    for i in range(N_Processes):
        # fig, ax = plt.subplots()
        # plt.plot(time_ax, processes[i])
        process_at_t.append(processes[i][-1])
    return process_at_t


def generate_residual_process_at_T(mu=0, kappa=0.5, delta=1, gamma=1.26, mu_W=0, std_W=1, T_horizon=1, N_Processes=1,
                                   N_epochs=1000,
                                   epoch_cutoff=1000):
    process_at_T = []
    alpha = kappa
    beta = 0.5 * gamma * (1 / kappa)
    C = delta * (2 ** alpha) * alpha * 1 / gammaf(1 - alpha)
    for _ in tqdm(range(N_Processes)):
        epochs = np.cumsum(np.random.exponential(scale=1 / 1, size=N_epochs)) / T_horizon
        epochs = np.array([epoch for epoch in epochs if epoch > epoch_cutoff])
        ts_jumps = TS_residual_jumps(alpha, beta, C, epochs)
        sum_jumps = np.sum(ts_jumps)
        square_root_jumps = np.sum(np.sqrt(ts_jumps))
        error_val = np.random.normal(loc=mu + mu_W * sum_jumps, scale=std_W * square_root_jumps)
        process_at_T.append(error_val)

    error_process_at_T = np.array(process_at_T)
    return error_process_at_T
