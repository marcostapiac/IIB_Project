import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def visualise(observations):
    time_ax = np.linspace(0, len(observations), len(observations))
    plt.plot(time_ax, observations, '-')
    plt.xlabel("Time")
    plt.ylabel("Exchange rate")
    plt.title("EUR-USD Exchange Rate")
    plt.show()


def clean(view):
    mat = scipy.io.loadmat('dataEurUS.mat')
    tls = np.array([v[0] for v in mat['last_traded'][0][0][0]])  # Time of trade (units of days)
    signs = np.array([v[0] for v in mat['last_traded'][0][0][1]])
    zls = np.array([v[0] for v in mat['last_traded'][0][0][2]])  # Exchange rate
    if view: visualise(zls)
    return tls, signs, zls


def obtainData(view=False):
    tls, signs, zls = clean(view)
    return zls
