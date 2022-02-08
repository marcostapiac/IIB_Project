import numpy as np
import matplotlib.pyplot as plt
import numbers
from scipy.stats import gamma as gammaf
from scipy.stats import norm, rv_continuous
from scipy.special import gamma as gammaff
from scipy.special import kv


class NGDistribution(rv_continuous):
    def _pdf(self, x):
        mu, beta, nu, gamma, t = 0, 0, 4, 2, 1
        constant = (gamma ** (2 * t * nu) * (np.sqrt(beta ** 2 + gamma ** 2)) ** (1 - 2 * nu * t)) / (
                np.sqrt(2 * np.pi) * gammaff(t * nu) * (2 ** ((t * nu) - 1)))
        bessel_component = (gamma ** 2 / 2) * np.abs(x - mu)
        u1 = np.exp(beta * (x - mu))
        u2 = bessel_component ** (nu - 0.5)
        u3 = kv(
            nu - 0.5, bessel_component)
        return constant * u1 * u2 * u3


def gamma_pdf(linspace, alpha, beta):
    """ tau must be a linspace mumpy array with support on positive half-line"""
    return gammaf.pdf(linspace, alpha,
                           scale=1 / beta)


def conditionalGaussian_pdf(linspace, mu, std):
    """ linspace must be a linspace numpy array with support in entire real line """
    return norm.pdf(linspace, mu, std)


# X is gamma_process, y is RVs from gamma process
def plot_qq(x1, y1, x2=None, quantiles=None, interpolation='nearest', ax=None, rug=False,
            rug_length=0.05, rug_kwargs=None, label="Normal Gamma QQ Plot", **kwargs):
    # Get current axes if none are provided
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel("Normal RVs")
    ax.set_ylabel("NG Process Residual RVs at t = 1")
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
    ax.set_title(label)
    ax.legend()


def plot_histogram_normal(histogram_sequence1, label, ax):
    numbins = 200
    binvals, bins, _ = ax.hist(histogram_sequence1, numbins, density=True, label="Process at t=1")
    xvals = np.linspace(norm.ppf(0.00001, scale=float(np.std(histogram_sequence1))),
                        norm.ppf(0.99999, scale=float(np.std(histogram_sequence1))), histogram_sequence1.shape[0])
    ax.plot(xvals,
            norm.pdf(xvals, loc=float(np.mean(histogram_sequence1)), scale=float(np.std(histogram_sequence1))),
            label="Standard Normal Distribution")
    ax.set_xlabel("X")
    ax.set_ylabel("PDF")
    ax.legend()
    ax.set_title(label)
    plt.grid()


def plot_histogram_gamma(histogram_sequence1, loc, scale, ax):
    numbins = 200
    binvals, bins, _ = ax.hist(histogram_sequence1, numbins, density=True, label="Process at t=1")
    xvals = np.linspace(gammaf.ppf(0.000001, a=loc, scale=scale),
                        gammaf.ppf(0.9999, a=loc, scale=scale), histogram_sequence1.shape[0])
    pdf = gammaf.pdf(xvals, a=loc, scale=scale)
    ax.plot(xvals, pdf, label="Gamma Distribution")
    ax.set_xlabel("X")
    ax.set_ylabel("PDF")
    ax.legend()
    plt.grid()
