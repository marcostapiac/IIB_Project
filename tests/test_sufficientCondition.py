import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
import scipy.special as special
from old_GIGJumps import GIG_harder_jumps
from scipy.special import gamma as gammaf


def normalAlpha():
    alpha = 0.8  # Determines Levy heavy-tailed-ness
    delta = 1
    C = delta * (2 ** alpha) * alpha * 1 / special.gamma(1 - alpha)
    T_horizon = 1
    N_epochs = 10000
    # mu_W = 0
    # std_W = 1
    vals = []
    subordinator_truncations = np.linspace(1e-7, 1e-5, 10000)  # Truncation on Alpha-Stable Jumps
    xaxis = []
    for subordinator_truncation in subordinator_truncations:
        epochs = np.cumsum(expon.rvs(scale=1 / 1, size=N_epochs)) / T_horizon
        jump_sizes = ((alpha * epochs) / C) ** (-1 / alpha)
        jump_sizes = jump_sizes[jump_sizes <= subordinator_truncation]
        sum = np.sum(jump_sizes)
        mean = C / (1 - alpha) * subordinator_truncation ** (1 - alpha)
        print(sum, mean)
        vals.append(sum / mean)
        xaxis.append(subordinator_truncation)
    plt.plot(xaxis, vals)
    # plt.yscale("log")
    # plt.xscale("log")
    plt.xlabel("Normal Alpha-Stable Subordinator Jump Truncation, $c$, $X_{i} \leq c$")
    plt.ylabel("Value of $S_{c}/E[S_{c}]$")
    plt.show()


def normalGamma():
    nu = C = 2
    gamma = 2
    beta = gamma ** 2 / 2
    T_horizon = 1
    N_epochs = 10000
    mu_W = 0
    std_W = 1
    vals = []
    subordinator_truncations = np.linspace(1e-207, 1e-206, 10000)  # Truncation on Gamma Jumps
    xaxis = []
    for subordinator_truncation in subordinator_truncations:
        epochs = np.cumsum(expon.rvs(scale=1 / 1, size=N_epochs)) / T_horizon
        x = np.exp(-epochs / C) / beta
        prob_acc = 1
        # Rejection sampling
        u = np.random.uniform(0., 1., size=x.size)
        jump_sizes = x[(u < prob_acc)]
        jump_sizes = jump_sizes[jump_sizes <= subordinator_truncation]
        sum = np.sum(jump_sizes)
        sigma_c2 = nu * subordinator_truncation
        vals.append(sum / sigma_c2)
        xaxis.append(subordinator_truncation)
    plt.plot(xaxis, vals)
    # plt.yscale("log")
    # plt.xscale("log")
    plt.xlabel("Normal Gamma Subordinator Jump Truncation, $c$, $X_{i} \leq c$")
    plt.ylabel("Value of $S_{c}/E[S_{c}]$")
    plt.show()


def normalTemperedStable():
    mu = 0
    kappa = 0.9
    delta = 1
    gamma = 1.26
    alpha = kappa
    beta = 0.5 * gamma * (1 / kappa)
    C = delta * (2 ** alpha) * alpha * 1 / gammaf(1 - alpha)
    T_horizon = 1
    N_epochs = 100000
    mu_W = 0
    std_W = 1
    vals = []
    subordinator_truncations = np.linspace(1e-5, 1e-7, 10000)  # Truncation on TS Jumps
    xaxis = []
    for subordinator_truncation in subordinator_truncations:
        epochs = np.cumsum(expon.rvs(scale=1 / 1, size=N_epochs)) / T_horizon
        x = ((alpha * epochs) / C) ** (-1 / alpha)
        prob_acc = np.exp(-beta * x)
        u = np.random.uniform(0.0, 1.0, size=prob_acc.size)
        jump_sizes = x[(u < prob_acc)]
        jump_sizes = jump_sizes[jump_sizes <= subordinator_truncation]
        print(jump_sizes)
        sum = np.sum(jump_sizes)
        sigma_c2 = subordinator_truncation ** (1 - alpha)
        vals.append(sum / sigma_c2)
        xaxis.append(subordinator_truncation)
    plt.figure(figsize=(14, 8))
    plt.plot(subordinator_truncations, vals)
    # plt.yscale("log")
    # plt.xscale("log")
    plt.xlabel("Normal Tempered Stable Subordinator Jumps, $c$, $X_{i} \leq c$")
    plt.ylabel("$S_{c}/E[S_{c}]$")
    plt.show()

def generalisedHyperbolic():
    delta = 2
    gamma_param = 0.2
    lambd = 0.1
    T_horizon = 1
    N_epochs = 1000
    mu_W = 0
    std_W = 1
    vals = []
    subordinator_truncations = np.logspace(-10, -14, 10000)  # Truncation on GIG Jumps
    for subordinator_truncation in subordinator_truncations:
        jump_sizes = GIG_harder_jumps(delta, gamma_param, lambd, T_horizon, N_epochs, subordinator_truncation=None)
        jump_sizes = jump_sizes[jump_sizes <= subordinator_truncation]
        print(jump_sizes)
        sum = np.sum(jump_sizes)
        mean = lambd * (lambd >= 0) * subordinator_truncation + delta * np.sqrt(2 / np.pi) * np.sqrt(
            subordinator_truncation)
        vals.append(sum / mean)
    print(max(vals))
    plt.figure(figsize=(14, 8))
    plt.plot(subordinator_truncations, vals)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Generalised Hyperbolic Subordinator Jump Truncation, $c$, $X_{i} \leq c$")
    plt.ylabel("Upper bound on value of $S_{c}/E[S_{c}]$")
    plt.show()


generalisedHyperbolic()
