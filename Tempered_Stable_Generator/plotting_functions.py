import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma as gammaf
from scipy.stats import norm
import numbers
from tqdm import tqdm


def generate_TS_dist(alpha, beta, C, x):
    """
    scaling = C*special.gamma(1-alpha)*2**(-alpha)*alpha**(-1)
    pdf = np.exp(scaling*(2*beta)**alpha)*np.exp(-beta*x)*stats.levy_stable.pdf(x, alpha, 0, 0, scaling) # alpha, beta, location, scale"""
    N = len(x) - 1
    T = 1e6
    ks = np.linspace(0, T, num=N + 1)
    charf = [0 for _ in ks]
    i = 0
    for k in ks:
        charf[i] = np.exp(C * gammaf(-alpha) * (complex(beta, -k) ** alpha - beta ** alpha))
        i += 1
    w = [0.5]
    w.extend([1 for _ in range(1, N)])
    w.extend([0.5])
    delta = T / (np.pi * N)
    pdf = [0 for _ in ks]
    for i in tqdm(range(len(x))):
        sum = 0
        for j in range(len(ks)):
            num = complex(np.cos(-ks[j] * x[i]), np.sin(-ks[j] * x[i])) * charf[j]
            sum += w[j] * num.real
        pdf[i] = delta * sum
    return pdf


def plot_histogram_normal(histogram_sequence1, ax, label=None):
    numbins = 200
    binvals, bins, _ = ax.hist(histogram_sequence1, numbins, density=True, label="Process at t=1")
    xvals = np.linspace(norm.ppf(0.00001, scale=float(np.std(histogram_sequence1))),
                        norm.ppf(0.99999, scale=float(np.std(histogram_sequence1))), histogram_sequence1.shape[0])
    pdf = norm.pdf(xvals, scale=float(np.std(histogram_sequence1)))
    ax.plot(xvals, pdf,
            label="Standard Normal Distribution")
    ax.set_xlabel("X")
    ax.set_ylabel("PDF")
    ax.legend()
    ax.set_title(label)
    plt.grid()


def plot_qq(x1, y1, x2=None, quantiles=None, interpolation='nearest', ax=None, rug=False,
            rug_length=0.05, rug_kwargs=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel("Theoretical RVs")
    ax.set_ylabel("Generalised Inverse Gaussian Process at t = 1")
    if quantiles is None:
        quantiles = min(len(x1), len(y1))

    # Compute quantiles of the two samples
    if isinstance(quantiles, numbers.Integral):
        quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
    else:
        quantiles = np.atleast_1d(np.sort(quantiles))
    x_quantiles1 = np.quantile(x1, quantiles, interpolation=interpolation)
    y_quantiles1 = np.quantile(y1, quantiles, interpolation=interpolation)
    if x2 is not None:
        if quantiles is None:
            quantiles = min(len(x2), len(y1))

        # Compute quantiles of the two samples
        if isinstance(quantiles, numbers.Integral):
            quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
        else:
            quantiles = np.atleast_1d(np.sort(quantiles))
        x_quantiles2 = np.quantile(x2, quantiles, interpolation=interpolation)
        y_quantiles2 = np.quantile(y1, quantiles, interpolation=interpolation)

    # Draw the rug plots if requested
    if rug:
        # Default rug plot settings
        rug_x_params = dict(ymin=0, ymax=rug_length, c='gray', alpha=0.5)
        rug_y_params = dict(xmin=0, xmax=rug_length, c='gray', alpha=0.5)

        # Override default setting by any user-specified settings
        if rug_kwargs is not None:
            rug_x_params.update(rug_kwargs)
            rug_y_params.update(rug_kwargs)

        # Draw the rug plots
        for point in x1:
            ax.axvline(point, **rug_x_params)
        for point in y1:
            ax.axhline(point, **rug_y_params)
        if x2 is not None:
            for point in x2:
                ax.axvline(point, **rug_x_params)
            for point in y1:
                ax.axhline(point, **rug_y_params)

    # Draw the q-q plot and compare with y = x
    ax.scatter(x_quantiles1, y_quantiles1, c="black", label="Q-Q plot", **kwargs)
    if x2 is not None:
        ax.scatter(x_quantiles2, y_quantiles2, c='red', label="Q-Q plot", **kwargs)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    # now plot both limits against each other
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, label="45 degree line")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()


def plot_histogram_TS(TS_process, kappa, delta, gamma, x, ax):
    alpha = kappa
    beta = 0.5 * gamma * (1 / kappa)
    C = delta * (2 ** alpha) * alpha * 1 / gammaf(1 - alpha)
    """ Function to compare generated process with density at t = T """
    ax.set_xlabel("Jump Sizes")
    ax.set_ylabel("Probability Density")
    binvals, bins, _ = plt.hist(TS_process, 100, density=True, label="Histogram of Tempered Stable Process at t = T")
    pdf = generate_TS_dist(alpha, beta, C, x)
    plt.plot(np.linspace(min(binvals), max(binvals), 1000), pdf, label="PDF of Tempered Stable Distribution")
    ax.legend()
