import numpy as np


def psi(x, alpha, lambd):
    return -alpha * (np.cosh(x) - 1) - lambd * (np.exp(x) - x - 1)


def dpsi(x, alpha, lambd):
    return -alpha * np.sinh(x) - lambd * (np.exp(x) - 1)


def g(x, sd, td, f1, f2):
    a = 0
    b = 0
    c = 0
    if (x >= -sd) and (x <= td):
        a = 1
    elif (x > td):
        b = f1
    elif (x < -sd):
        c = f2
    return a + b + c


def generate_GIG_rvs(P, a, b, SampleSize):
    """ Code is translated from MATLAB Code from:
    Jan Patrick Hartkopf (2022).
    gigrnd (https://www.mathworks.com/matlabcentral/fileexchange/78805-gigrnd),
    MATLAB Central File Exchange. """
    """Setup - - we sample from the two parameter version of the GIG(alpha, omega)"""
    lambd = P
    omega = np.sqrt(a * b)
    swap = False
    if lambd < 0:
        lambd = lambd * -1
        swap = True
    alpha = np.sqrt(omega ** 2 + lambd ** 2) - lambd
    x = -psi(1, alpha, lambd)  # TODO CHECK
    if (x >= 0.5) and (x <= 2):
        t = 1
    elif x > 2:
        t = np.sqrt(2 / (alpha + lambd))
    elif x < 0.5:
        t = np.log(4 / (alpha + 2 * lambd))

    x = -psi(-1, alpha, lambd)  # TODO CHECK
    if (x >= 0.5) and (x <= 2):
        s = 1
    elif x > 2:
        s = np.sqrt(4 / (alpha * np.cosh(1) + lambd))
    elif x < 0.5:
        s = min(1 / lambd, np.log(1 + 1 / alpha + np.sqrt(1 / alpha ** 2 + 2 / alpha)))

    eta = -psi(t, alpha, lambd)
    zeta = -dpsi(t, alpha, lambd)
    theta = -psi(-s, alpha, lambd)
    xi = dpsi(-s, alpha, lambd)
    p = 1 / xi
    r = 1 / zeta
    td = t - r * eta
    sd = s - p * theta
    q = td + sd

    X = [0 for i in range(SampleSize)]
    for i in range(SampleSize):
        done = False
        while not done:
            U = np.random.uniform(0., 1., size=1)
            V = np.random.uniform(0., 1., size=1)
            W = np.random.uniform(0., 1., size=1)
            if U < (q / (p + q + r)):
                X[i] = -sd + q * V
            elif U < ((q + r) / (p + q + r)):
                X[i] = td - r * np.log(V)
            else:
                X[i] = -sd + p * np.log(V)
            f1 = np.exp(-eta - zeta * (X[i] - t))
            f2 = np.exp(-theta + xi * (X[i] + s))
            if (W * g(X[i], sd, td, f1, f2)) <= np.exp(psi(X[i], alpha, lambd)):
                done = True
    X = np.exp(X) * (lambd / omega + np.sqrt(1 + (lambd / omega) ** 2))
    if swap:
        X = 1 / X
    X = X / np.sqrt(a / b)
    return X
