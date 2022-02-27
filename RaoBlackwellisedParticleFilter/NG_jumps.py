import numpy as np
from scipy.stats import expon


def gammaJumps(C, beta, epochs):
    """ Generates jumps from a Gamma process with parameters Ga(nu, gamma*gamma/2)"""
    x = beta ** -1 / (np.exp(epochs / C) - 1)
    prob_acc = (1 + beta * x) * np.exp(-beta * x)
    # Rejection sampling
    u = np.random.uniform(0., 1., size=x.size)
    jump_sizes = x[(u < prob_acc)]
    return jump_sizes


def normalGammaJumps(mu=0, nu=2, gamma=2, mu_W=0, std_W=1, T_horizon=1, N_epochs=1000):
    # Inputs are the  NG(mu, beta, nu, gamma2/2) parameters, conditionally Gaussian parameters (for U_i's),
    # and simulation parameters.
    epochs = np.cumsum(expon.rvs(scale=1 / 1, size=N_epochs)) / T_horizon
    gamma_jump_sizes = gammaJumps(nu, gamma, epochs)  # h(Gamma_i)
    return mu + mu_W * gamma_jump_sizes + std_W ** 2 * np.sqrt(gamma_jump_sizes) * np.random.normal(loc=0, scale=1,
                                                                                                    size=
                                                                                                    gamma_jump_sizes.shape[
                                                                                                        0])
