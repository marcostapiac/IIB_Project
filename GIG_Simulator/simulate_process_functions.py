import numpy as np
from tqdm import tqdm
from scipy.stats import gamma
from scipy.special import hankel1, hankel2, gammainc, gammaincc, gammaincinv
from scipy.special import gamma as gammaf


def unnorm_gammaincc(lam, z):
    return gammaf(lam) * gammaincc(lam, z)


def unnorm_gammainc(lam, z):
    return gammaf(lam) * gammainc(lam, z)


def hankel_squared(lam, z):
    return np.real(hankel1(lam, z) * hankel2(lam, z))


def generate_tempered_stable_jumps(alpha, beta, delta, N_epochs, T_horizon):
    epochs = np.cumsum(
        np.random.exponential(1, N_epochs)) / T_horizon
    C = T_horizon * delta * gammaf(0.5) / (np.sqrt(2) * np.pi)
    x = ((alpha * epochs) / C) ** (-1 / alpha)
    prob_acc = np.exp(-beta * x)
    u = np.random.uniform(0.0, 1.0, size=prob_acc.size)
    jump_sizes = x[(u < prob_acc)]

    return jump_sizes


def generate_gamma_jumps(C, beta, N_epochs, T_horizon):
    # Arrival times according to unit rate Poisson Process
    epochs = np.cumsum(
        np.random.exponential(1, N_epochs)) / T_horizon
    x = 1 / (beta * (np.exp(epochs / C) - 1))
    prob_acc = (1 + beta * x) * np.exp(-beta * x)
    # Rejection sampling
    u = np.random.uniform(0., 1., size=x.size)
    jump_sizes = x[(u < prob_acc)]
    return jump_sizes


def GIG_gamma_component(gamma_param, lambd, N_epochs, T_horizon):
    gamma_beta = gamma_param ** 2 / 2
    gamma_C = max(0, lambd)
    return generate_gamma_jumps(gamma_C, gamma_beta, N_epochs, T_horizon)


def GIG_simple_jumps(N_epochs, delta, gamma_param, lambd, T_horizon):
    x = generate_tempered_stable_jumps(0.5, (gamma_param ** 2) / 2, delta, N_epochs, T_horizon)
    zs = np.sqrt(gamma.rvs(0.5, loc=0, scale=1 / (x / (2 * delta ** 2)),
                           size=len(x)))
    u = np.random.uniform(0., 1., len(x))
    prob_acc = 2 / (np.pi * zs * hankel_squared(np.abs(lambd), zs))
    jump_sizes = x[(u < prob_acc)]
    return jump_sizes


def GIG_simple_process(delta=2, gamma_param=2, lambd=0.8, N_Processes=10000, T_horizon=1, N_epochs=1000):
    assert (np.absolute(lambd) >= 0.5)
    processes = []
    time_ax = np.linspace(0., T_horizon, N_epochs)
    for _ in tqdm(range(N_Processes)):
        GIG_jumps = GIG_simple_jumps(N_epochs, delta, gamma_param, lambd, T_horizon)
        if lambd > 0:
            gamma_jumps = GIG_gamma_component(gamma_param, lambd, N_epochs, T_horizon)
            jump_sizes = np.append(GIG_jumps, gamma_jumps)
        else:
            jump_sizes = GIG_jumps
        jump_times = np.random.uniform(0., T_horizon, size=jump_sizes.size)
        GIG = np.array([np.sum(jump_sizes * (jump_times <= t)) for t in time_ax])
        processes.append(GIG)
    return processes


def GIG_harder_jumps(delta, gamma_param, lambd, N_epochs, T_horizon):
    a = np.pi * np.power(2.0, (1.0 - 2.0 * np.abs(lambd)))
    b = gammaf(np.abs(lambd)) ** 2
    c = 1 / (1 - 2 * np.abs(lambd))
    z1 = (a / b) ** c
    H0 = z1 * hankel_squared(np.abs(lambd), z1)
    N1 = generate_N1(z1, H0, lambd, delta, gamma_param, N_epochs, T_horizon)
    N2 = generate_N2(z1, H0, lambd, delta, gamma_param, N_epochs, T_horizon)
    jump_sizes = np.append(N1, N2)
    return jump_sizes


def generate_N1(z1, H0, lambd, delta, gamma_param, N_epochs, T_horizon):
    # Generate gamma process
    beta = 0.5 * gamma_param ** 2
    C = z1 / (np.pi * np.pi * np.absolute(lambd) * H0)  # Shape parameter of process at t = 1
    jump_sizes = generate_gamma_jumps(C, beta, N_epochs, T_horizon)

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
    jump_sizes = jump_sizes[(u < N1_prob_acc)]
    return jump_sizes


def generate_N2(z1, H0, lambd, delta, gamma_param, N_epochs, T_horizon):
    """Generate point process N2 """

    """ Generate Tempered Stable Jump Size samples """
    alpha = 0.5
    beta = (gamma_param ** 2) / 2
    C = np.sqrt(2 * delta ** 2) * gammaf(0.5) / ((np.pi ** 2) * H0)
    epochs = np.cumsum(
        np.random.exponential(1, N_epochs)) / T_horizon
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
    jump_sizes = jump_sizes[(u < N2_prob_acc)]
    return jump_sizes


def GIG_harder_process(delta, gamma_param, lambd, N_Processes, T_horizon, N_epochs):
    assert (np.absolute(lambd) < 0.5)
    processes = []
    time_ax = np.linspace(0., T_horizon, num=N_epochs)
    for _ in tqdm(range(N_Processes)):
        GIG_jumps = GIG_harder_jumps(delta, gamma_param, lambd, N_epochs)
        if lambd > 0:
            gamma_jumps = GIG_gamma_component(gamma_param, lambd, N_epochs, T_horizon=T_horizon)
            jump_sizes = np.append(GIG_jumps, gamma_jumps)
        else:
            jump_sizes = GIG_jumps
        jump_times = np.random.uniform(0., T_horizon, size=jump_sizes.size)
        GIG = np.array([np.sum(jump_sizes * (jump_times <= t)) for t in time_ax])
        processes.append(GIG)
    return processes


def GIG_process(delta=2, gamma_param=0.2, lambd=-0.1, N_Processes=10000, T_horizon=1, N_epochs=1000):
    if np.abs(lambd) >= 0.5:
        processes = GIG_simple_process(delta, gamma_param, lambd, N_Processes, T_horizon, N_epochs)
    else:
        processes = GIG_harder_process(delta, gamma_param, lambd, N_Processes, T_horizon, N_epochs)
    return processes


def GIG_jumps(delta=2, gamma_param=0.2, lambd=-0.1, N_Jumps=10000, T_horizon=1, N_epochs=1000):
    jumps = np.array([])
    if np.abs(lambd) >= 0.5:
        for _ in tqdm(range(N_Jumps)):
            p1 = GIG_simple_jumps(N_epochs, delta, gamma_param, lambd, T_horizon)
            if lambd > 0:
                p2 = GIG_gamma_component(gamma_param, lambd, N_epochs, T_horizon)
                jumps = np.append(jumps, np.append(p1, p2).sum())
            else:
                jumps = np.append(jumps, p1.sum())
    else:
        for _ in tqdm(range(N_Jumps)):
            p1 = GIG_harder_jumps(delta, gamma_param, lambd, N_epochs, T_horizon)
            if lambd > 0:
                p2 = GIG_gamma_component(gamma_param, lambd, N_epochs, T_horizon)
                jumps = np.append(jumps, np.append(p1, p2).sum())
            else:
                jumps = np.append(jumps, p1.sum())
    return jumps

