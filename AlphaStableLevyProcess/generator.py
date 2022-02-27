import numpy as np
import numbers
from scipy import special, stats
import matplotlib.pyplot as plt
from tqdm import tqdm


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


def CMS_Method(alpha, beta, C, L, location=0):
    U = np.random.uniform(-np.pi * 0.5, np.pi * 0.5, L)
    W = np.random.exponential(1, L)
    zeta = -beta * np.tan(np.pi * alpha * 0.5)
    xi = np.arctan(-zeta) / alpha
    X = ((1 + zeta ** 2) ** (1 / (2 * alpha))) * (
            (np.sin(alpha * (U + np.full_like(U, xi, dtype=np.double)))) / (np.cos(U)) ** (1 / alpha)) * (
                (np.cos(U - alpha * (U + np.full_like(U, xi, dtype=np.double)))) / W) ** ((1 - alpha) / alpha)

    return C * X + location


def process_jumps(alpha=0.8, delta=1, T_horizon=1, N_Processes=50000, N_epochs=1000):
    C = delta * (2 ** alpha) * alpha * 1 / special.gamma(1 - alpha)
    process_at_t = []
    for _ in tqdm(range(N_Processes)):
        epochs = np.cumsum(
            np.random.exponential(1, N_epochs)) / T_horizon  # Arrival times according to unit rate Poisson Process
        jump_sizes = ((alpha * epochs) / C) ** (-1 / alpha)
        process_at_t.append(np.sum(jump_sizes))
    fig, ax = plt.subplots()
    positive_stable_rvs = CMS_Method(alpha, 1, 1, L=N_Processes,
                                     location=0)  # C*CMS_Method(alpha, beta=1, c=delta, L=N_Processes, location=0)
    print(positive_stable_rvs)
    plot_qq(np.array(process_at_t), positive_stable_rvs, ax=ax)
    plt.xlabel("Time")
    plt.ylabel("Location")
    plt.title("Positive Alpha-Stable process")
    plt.show()


def process(alpha=0.8, delta=1, T_horizon=1, N_Processes=1000, N_epochs=1000):
    C = 1  # delta * (2 ** alpha) * alpha * 1 / special.gamma(1 - alpha)
    time_ax = np.linspace(0.0, T_horizon, num=N_epochs)
    processes = []
    for _ in tqdm(range(N_Processes)):
        epochs = np.cumsum(
            np.random.exponential(1, N_epochs)) / T_horizon  # Arrival times according to unit rate Poisson Process
        jump_sizes = ((alpha * epochs) / C) ** (-1 / alpha)
        jump_times = np.random.uniform(0.0, T_horizon, size=jump_sizes.size)
        positive_stable_process = np.array([np.sum(jump_sizes * (jump_times <= t)) for t in time_ax])
        processes.append(positive_stable_process)

    process_at_t = []
    fig, ax = plt.subplots()
    for p in processes:
        # plt.plot(time_ax, p)
        process_at_t.append(p[-1])
    positive_stable_rvs = stats.levy_stable.rvs(alpha, beta=1, loc=0, scale=1, size=len(
        process_at_t))  # C*CMS_Method(alpha, beta=1, c=delta, L=N_Processes, location=0)
    plot_qq(np.array(process_at_t), positive_stable_rvs, ax=ax)
    plt.xlabel("Time")
    plt.ylabel("Location")
    plt.title("Positive Alpha-Stable process")
    plt.show()


process_jumps(alpha=0.5)
