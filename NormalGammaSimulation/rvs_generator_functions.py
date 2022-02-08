import numpy as np


def generate_RVs(mu, nu, gamma, mu_W, std_W, L=10000):
    tau = np.random.gamma(nu, scale=1 / gamma, size=L)
    x = np.random.normal(mu + mu_W * tau, np.sqrt(tau) * std_W, size=L)
    return x, tau
