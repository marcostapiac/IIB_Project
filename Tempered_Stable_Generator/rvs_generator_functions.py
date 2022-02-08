import numpy as np

def CMS_Method(alpha, beta, C, L, location=0):
    U = np.random.uniform(-np.pi * 0.5, np.pi * 0.5, L)
    W = np.random.exponential(1, L)
    zeta = -beta * np.tan(np.pi * alpha * 0.5)
    xi = np.arctan(-zeta) / alpha
    X = ((1 + zeta ** 2) ** (1 / (2 * alpha))) * (
            (np.sin(alpha * (U + np.full_like(U, xi, dtype=np.double)))) / (np.cos(U)) ** (1 / alpha)) * (
                (np.cos(U - alpha * (U + np.full_like(U, xi, dtype=np.double)))) / W) ** ((1 - alpha) / alpha)

    return C * X + location


def TS_rv_generator(kappa, delta, gamma, L, location=0):
    """Generate Stable Random Variables from S(alpha, beta, gamma, delta) = S(alpha, 1, 0, 1) = S(alpha, C=1)"""
    # CHANGE SCALES FOR LOG stable_rvs_scaling = delta * (2 ** kappa) * kappa * 1 / gamma(1 - kappa)
    stable_rvs = np.array(CMS_Method(kappa, 1, 1, L, location=location))
    U = np.random.uniform(0., 1., size=stable_rvs.size)
    prob_acc = np.exp(-0.5 * gamma * (1 / kappa) * stable_rvs)
    accepted_rvs = stable_rvs[(U < prob_acc)]
    return accepted_rvs
