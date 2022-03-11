import numpy as np


def gammaJumps(C, beta, epochs):
    """ Generates jumps from a Gamma process with parameters Ga(nu, gamma*gamma/2)"""
    x = beta ** -1 / (np.exp(epochs / C) - 1)
    prob_acc = (1 + beta * x) * np.exp(-beta * x)
    # Rejection sampling
    u = np.random.uniform(0., 1., size=x.size)
    jump_sizes = x*(u < prob_acc)
    return jump_sizes