import numpy as np
from matplotlib import pyplot as plt
from scipy.special import kv
from scipy.stats import norm
import numbers

def GIG_pdf(lambd, gamma, delta, x):
    return (gamma / delta) ** lambd * x ** (lambd - 1) * 0.5 * np.exp(
        -0.5 * (gamma ** 2 * x + delta ** 2 / x)) * kv(lambd, delta * gamma) ** (-1)


def plot_histogram_GIG(process, lambd, gamma, delta, ax):
    bins = np.arange(1e-5, 10000, 100)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    binvals, _, _ = plt.hist(np.clip(process, bins[0], bins[-1]), logbins, density=True, label="Histogram of GIG Process at t = 1")
    x = np.logspace(-3, np.log10(bins[-1]), num=10000)
    pdf = GIG_pdf(lambd, gamma, delta, x)
    ax.plot(x, pdf, label="PDF of GIG Distribution")
    ax.set_xscale('log')
    ax.set_xlabel("Jump Sizes")
    ax.set_ylabel("Probability Density")
    ax.legend()


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
