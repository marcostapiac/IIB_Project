import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, special
import numbers
from tqdm import tqdm


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
    """Code from matlab function """
    """Setup - - we sample from the two parameter version of the GIG(alpha, omega)"""
    lambd = P
    omega = np.sqrt(a * b)
    print(omega)
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


def GIG_pdf(lambd, gamma, delta, x):
    return (gamma / delta) ** lambd * x ** (lambd - 1) * 0.5 * np.exp(
        -0.5 * (gamma ** 2 * x + delta ** 2 / x)) * special.kv(lambd, delta * gamma) ** (-1)

def plot_histogram_GIG(process, lambd, gamma, delta, ax):
    """ Function to compare generated process with density at t = T """
    ax.set_xscale('log')
    ax.set_xlabel("Jump Sizes")
    ax.set_ylabel("Probability Density")
    bins = np.arange(1e-5, 10000, 100)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    binvals, _, _ = plt.hist(process, logbins, density=True, label="Histogram of GIG Process at t = 1")
    x = np.logspace(-3, np.log10(bins[-1]), num=10000)
    pdf = GIG_pdf(lambd, gamma, delta, x)
    plt.plot(x, pdf, label="PDF of GIG Distribution")
    ax.legend()


def plot_histogram_norm(histogram_sequence1, ax, label=None):
    numbins = 200
    binvals, bins, _ = ax.hist(histogram_sequence1, numbins, density=True, label="Process at t=1")
    xvals = np.linspace(stats.norm.ppf(0.00001, scale=float(np.std(histogram_sequence1))),
                        stats.norm.ppf(0.99999, scale=float(np.std(histogram_sequence1))), histogram_sequence1.shape[0])
    pdf = stats.norm.pdf(xvals, scale=float(np.std(histogram_sequence1)))
    ax.plot(xvals, stats.norm.pdf(xvals, scale=float(np.std(histogram_sequence1))),
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


def generate_tempered_stable_jumps(alpha, beta, delta, N_epochs, T_horizon=1):
    epochs = np.cumsum(
        np.random.exponential(1, N_epochs)) / T_horizon
    C = T_horizon * delta * special.gamma(0.5) / (np.sqrt(2) * np.pi)
    x = ((alpha * epochs) / C) ** (-1 / alpha)
    prob_acc = np.exp(-beta * x)
    u = np.random.uniform(0.0, 1.0, size=prob_acc.size)
    jump_sizes = x[(u < prob_acc)]

    return jump_sizes


def generate_gamma_jumps(C, beta, N_epochs, T_horizon=1):
    epochs = np.cumsum(
        np.random.exponential(1, N_epochs)) / T_horizon  # Arrival times according to unit rate Poisson Process
    x = 1 / (beta * (np.exp(epochs / C) - 1))
    prob_acc = (1 + beta * x) * np.exp(-beta * x)
    # Rejection sampling
    u = np.random.uniform(0., 1., size=x.size)
    jump_sizes = x[(u < prob_acc)]
    return jump_sizes


def GIG_gamma_component(gamma_param, lambd, N_epochs, T_horizon=1):
    gamma_beta = gamma_param ** 2 / 2
    gamma_C = max(0, lambd)
    assert (gamma_C > 0)
    return generate_gamma_jumps(gamma_C, gamma_beta, N_epochs, T_horizon)


def GIG_simple_jumps(N_epochs, delta=2, gamma_param=0.1, lambd=-0.8, T_horizon=1):
    x = generate_tempered_stable_jumps(0.5, (gamma_param ** 2) / 2, delta, N_epochs, T_horizon)
    zs = np.sqrt(stats.gamma.rvs(0.5, loc=0, scale=1 / (x / (2 * delta ** 2)),
                                 size=len(x)))
    u = np.random.uniform(0., 1., len(x))
    prob_acc = 2 / (np.pi * zs * np.absolute(special.hankel1(np.abs(lambd), zs)) ** 2)
    jump_sizes = x[(u < prob_acc)]
    return jump_sizes


def GIG_simple_process(delta=2, gamma_param=2, lambd=0.8, N_Processes=10000, T_horizon=1, N_epochs=1000):
    assert (np.absolute(lambd) >= 0.5)
    processes = []
    time_ax = np.linspace(0., T_horizon, N_epochs)
    fig, ax = plt.subplots()
    for _ in tqdm(range(N_Processes)):
        GIG_jumps = GIG_simple_jumps(N_epochs, delta, gamma_param, lambd, T_horizon)
        if lambd > 0:
            gamma_jumps = GIG_gamma_component(gamma_param, lambd, N_epochs, T_horizon)
            L = min(GIG_jumps.size, gamma_jumps.size)
            jump_sizes = GIG_jumps[:L] + gamma_jumps[:L]
        else:
            jump_sizes = GIG_jumps
        jump_times = np.random.uniform(0., T_horizon, size=jump_sizes.size)
        GIG = np.array([np.cumsum(jump_sizes * (jump_times <= t))[-1] for t in time_ax])
        processes.append(GIG)

    process_at_t = []
    for i in range(N_Processes):
        # ax.plot(time_ax, processes[i])
        process_at_t.append(processes[i][-1])
    """ Generate GIG rvs to create QQ plot or histogram"""
    GIG_rvs = generate_GIG_rvs(lambd, gamma_param ** 2, delta ** 2, len(process_at_t))
    GIG_rvs = [x[0] for x in GIG_rvs]

    plot_qq(process_at_t, np.array(GIG_rvs), ax=ax)
    fig, ax2 = plt.subplots()
    plot_histogram_GIG(process_at_t, lambd, gamma_param, delta, ax2)
    # ax.set_xlabel("Time")
    # ax.set_ylabel("Sample Path")
    pval = (stats.ks_2samp(process_at_t, GIG_rvs)).pvalue
    ax.set_title(
        "GIG Process Sample with $\delta, \gamma, \lambda, pval= " + str((delta, gamma_param, lambd, pval)) + "$")
    ax2.set_title(
        "GIG Process Sample with $\delta, \gamma, \lambda, pval = " + str((delta, gamma_param, lambd, pval)) + "$")
    plt.show()


def error_GIG_simple_Jumps(delta, gamma_param, lambd, T_horizon, epochs, epoch_cutoff):
    # GIG component for large truncation
    beta = gamma_param ** 2 / 2
    gamma_C = max(0, lambd)
    x = 1 / (beta * (np.exp(epochs / gamma_C) - 1))
    prob_acc = (1 + beta * x) * np.exp(-beta * x)
    # Rejection sampling
    u = np.random.uniform(0., 1., size=x.size)
    gamma_component_jumps = x[(u < prob_acc)]
    # GIG component for large truncation
    ts_C = (T_horizon * delta * special.gamma(0.5)) / (np.sqrt(2) * np.pi)
    ts_alpha = 0.5
    x = (ts_C / (ts_alpha * (epochs))) ** (1 / ts_alpha)
    zs = np.sqrt(stats.gamma.rvs(0.5, loc=0, scale=1 / (x / (2 * delta ** 2)),
                                 size=len(x)))
    u = np.random.uniform(0., 1., len(x))
    prob_acc = 2 / (np.pi * zs * np.absolute(special.hankel1(np.abs(lambd), zs)) ** 2)
    GIG_component_jump_sizes = x[(u < prob_acc)]
    return np.append(GIG_component_jump_sizes, gamma_component_jumps)


def generate_simple_error_process_at_T(mu=0, mu_W=0, std_W=1, delta=2, gamma_param=-0.8, lambd=-0.8, T_horizon=1,
                                       N_Processes=1, N_epochs=1000, epoch_cutoff=1000):
    process_at_T = []

    for _ in tqdm(range(N_Processes)):
        epochs = np.cumsum(np.random.exponential(scale=1 / 1, size=N_epochs)) / T_horizon
        epochs = np.array([epoch for epoch in epochs if epoch > epoch_cutoff])
        jumps = error_GIG_simple_Jumps(delta, gamma_param, lambd, T_horizon, epochs, epoch_cutoff)
        jumps = jumps / max(jumps)
        sum_jumps = np.sum(jumps)
        square_root_jumps = np.sqrt(sum_jumps)
        error_val = np.random.normal(loc=mu_W * sum_jumps, scale=std_W * square_root_jumps)
        process_at_T.append(error_val)

    process_at_T = np.array(process_at_T)
    return process_at_T


def CLT2_GHsimple(mu=0, mu_W=0, std_W=1, delta=2, gamma_param=2, lambd=0.8, N_Processes=15000, T_horizon=1,
                  N_epochs=10000, truncation=1e-2):
    assert (np.absolute(lambd) >= 0.5)
    epoch_cutoff = 2000
    error_process = generate_simple_error_process_at_T(mu, mu_W, std_W, delta, gamma_param, lambd, T_horizon,
                                                       N_Processes, N_epochs, epoch_cutoff)
    print(error_process)
    fig2, ax4 = plt.subplots()
    fig2.figsize = (18, 18)
    plot_histogram_norm(error_process, ax=ax4)
    ax4.set_title("GIG Residuals with parameters $\delta, \gamma, \lambda, \Gamma_i= " + str(
        (delta, gamma_param, lambd, epoch_cutoff)) + "$")

    ksstat = stats.kstest(error_process, 'norm', [np.mean(error_process), np.std(error_process)]).pvalue
    print(stats.kstest(error_process, 'norm', [np.mean(error_process), np.std(error_process)]))
    """
    fig2, ax2 = plt.subplots()
    fig2.figsize = (18, 18)
    plot_histogram(error_process,
                   label="GH Residuals Histogram: $h(\Gamma_{i}) < $" + str(truncation) + " and pval $\\approx" + str(
                       round(ksstat * 1e144, 3)) + "\\times$10e-144", ax=ax2)
    plt.savefig("Hist")
    fig4, ax4 = plt.subplots()
    fig2.figsize = (18, 18)
    plot_qq(error_process,
            np.random.normal(np.mean(error_process), np.std(error_process), size=error_process.shape[0]), ax=ax4,
            label="GB Residuals QQ Plot: $h(\Gamma_{i}) < $" + str(truncation) + " and pval $\\approx" + str(
                round(ksstat * 1e144, 3)) + "\\times$10e-144")
    plt.savefig("QQ")
    """
    plt.show()


def GIG_harder_jumps(delta=2, gamma_param=2.0, lambd=-0.1, N_epochs=50000):
    a = np.pi * np.power(2.0, (1.0 - 2.0 * np.abs(lambd)))
    b = special.gamma(np.abs(lambd)) ** 2
    c = 1 / (1 - 2 * np.abs(lambd))
    z1 = (a / b) ** c
    H0 = z1 * np.absolute(special.hankel1(np.abs(lambd), z1)) ** 2
    N1 = generate_N1(z1, H0, lambd, delta, gamma_param, N_epochs)
    N2 = generate_N2(z1, H0, lambd, delta, gamma_param, N_epochs)
    jump_sizes = np.append(N1, N2)
    return jump_sizes


def generate_N1(z1, H0, lambd, delta, gamma_param, N_epochs):
    # Generate gamma process
    beta = 0.5 * gamma_param ** 2
    C = z1 / (np.pi * np.pi * np.absolute(lambd) * H0)  # Shape parameter of process at t = 1
    jump_sizes = generate_gamma_jumps(C, beta, N_epochs=N_epochs)

    """ Rejection sampling from Algorithm 6 """
    const1 = (z1 ** 2) * jump_sizes / (2 * delta ** 2)
    GIG_prob_acc = np.absolute(lambd) * special.gamma(np.abs(lambd)) * special.gammainc(np.abs(lambd), const1) / (
            ((z1 ** 2) * jump_sizes / (2 * delta ** 2)) ** np.abs(lambd))
    u = np.random.uniform(0., 1., size=jump_sizes.size)
    jump_sizes = jump_sizes[(u < GIG_prob_acc)]

    """ Sample from truncated Nakagami """
    C1 = np.random.uniform(0., 1., size=jump_sizes.size)
    l = C1 * special.gammainc(np.absolute(lambd), (z1 ** 2 * jump_sizes) / (2 * delta ** 2))
    zs = np.sqrt(((2 * delta ** 2) / jump_sizes) * special.gammaincinv(np.absolute(lambd), l))

    """ Thinning for process N1 """
    u = np.random.uniform(0., 1., size=jump_sizes.size)
    N1_prob_acc = H0 / ((np.abs(special.hankel1(np.abs(lambd), zs)) ** 2) *
                        (zs ** (2 * np.abs(lambd))) / (z1 ** (2 * np.abs(lambd) - 1)))
    jump_sizes = jump_sizes[(u < N1_prob_acc)]
    return jump_sizes


def generate_N2(z1, H0, lambd, delta, gamma_param, N_epochs, T_horizon=1):
    """Generate point process N2 """

    """ Generate Tempered Stable Jump Size samples """
    N = 100
    alpha = 0.5
    beta = (gamma_param ** 2) / 2
    C = np.sqrt(2 * delta ** 2) * special.gamma(0.5) / ((np.pi ** 2) * H0)
    epochs = np.cumsum(
        np.random.exponential(1, N_epochs)) / T_horizon
    x = ((alpha * epochs) / C) ** (-1 / alpha)
    prob_acc = np.exp(-beta * x)
    u = np.random.uniform(0.0, 1.0, size=prob_acc.size)
    jump_sizes = x[(u < prob_acc)]

    """ Rejection sampling based on Algorithm 7 """
    GIG_prob_acc = special.gammaincc(0.5, (z1 ** 2) * jump_sizes / (2 * delta ** 2))
    u = np.random.uniform(0., 1., size=jump_sizes.size)
    jump_sizes = jump_sizes[(u < GIG_prob_acc)]

    """ Sample from truncated Nakagami """
    C2 = np.random.uniform(0., 1., size=jump_sizes.size)
    const1 = (special.gammaincc(0.5, (z1 ** 2) * jump_sizes / (2 * delta ** 2))) + special.gammainc(0.5,
                                                                                                    (
                                                                                                                z1 ** 2) * jump_sizes / (
                                                                                                            2 * delta ** 2))
    zs = np.sqrt(((2 * delta ** 2) / jump_sizes) * special.gammaincinv(0.5, C2 * const1))
    """Thinning for process N2"""
    u = np.random.uniform(0., 1., size=jump_sizes.size)
    N2_prob_acc = H0 / (zs * (np.abs(special.hankel1(np.abs(lambd), zs)) ** 2))
    jump_sizes = jump_sizes[(u < N2_prob_acc)]
    return jump_sizes


def GIG_harder_process(delta=2, gamma_param=0.2, lambd=-0.1, N_Processes=10000, T_horizon=1, N_epochs=1000):
    assert (np.absolute(lambd) < 0.5)
    processes = []
    time_ax = np.linspace(0., T_horizon, num=N_epochs)
    for _ in tqdm(range(N_Processes)):
        GIG_jumps = GIG_harder_jumps(delta, gamma_param, lambd, N_epochs)
        if lambd > 0:
            gamma_jumps = GIG_gamma_component(gamma_param, lambd, N_epochs, T_horizon=T_horizon)
            L = min(GIG_jumps.size, gamma_jumps.size)
            jump_sizes = np.append(GIG_jumps, gamma_jumps)
        else:
            jump_sizes = GIG_jumps
        jump_times = np.random.uniform(0., T_horizon, size=jump_sizes.size)
        GIG = np.array([np.cumsum(jump_sizes * (jump_times <= t))[-1] for t in time_ax])
        processes.append(GIG)

    fig, ax = plt.subplots()
    process_at_t = []
    for i in range(N_Processes):
        # ax.plot(time_ax, processes[i])
        process_at_t.append(processes[i][-1])

    """ Generate GIG rvs to create QQ plot or histogram"""
    GIG_rvs = generate_GIG_rvs(lambd, gamma_param ** 2, delta ** 2, len(process_at_t))
    GIG_rvs = [x[0] for x in GIG_rvs]

    plot_qq(process_at_t, np.array(GIG_rvs), ax=ax)
    fig, ax2 = plt.subplots()
    plot_histogram_GIG(process_at_t, lambd, gamma_param, delta, ax2)
    pval = (stats.ks_2samp(process_at_t, GIG_rvs)).pvalue
    ax.set_title(
        "GIG Process Sample with $\delta, \gamma, \lambda, pval= " + str((delta, gamma_param, lambd, round(pval,5))) + "$")
    ax2.set_title(
        "GIG Process Sample with $\delta, \gamma, \lambda, pval = " + str((delta, gamma_param, lambd, round(pval,5))) + "$")
    plt.show()


def error_GIG_harder_Jumps(delta, gamma_param, lambd, T_horizon, epochs, epoch_cutoff):
    # GIG component for large truncation
    beta = gamma_param ** 2 / 2
    gamma_C = max(0, lambd)
    gamma_component_jumps = np.exp(-(epochs) / gamma_C) / beta
    print(gamma_component_jumps)
    print("gammabaove")
    # GIG component for large truncation
    ts_C = T_horizon * delta * special.gamma(0.5) / (np.sqrt(2) * np.pi)
    ts_alpha = 0.5
    x = ((ts_alpha * (epochs)) / ts_C) ** (-1 / ts_alpha)
    zs = np.sqrt(stats.gamma.rvs(0.5, loc=0, scale=1 / (x / (2 * delta ** 2)),
                                 size=len(x)))
    u = np.random.uniform(0., 1., len(x))
    prob_acc = 2 / (np.pi * zs * np.absolute(special.hankel1(np.abs(lambd), zs)) ** 2)
    GIG_component_jump_sizes = x[(u < prob_acc)]
    print(GIG_component_jump_sizes)
    L = min(GIG_component_jump_sizes.size, gamma_component_jumps.size)
    print("gigabove")
    return gamma_component_jumps[:L] + GIG_component_jump_sizes[:L]


def generate_harder_error_process_at_T(mu=0, mu_W=0, std_W=1, delta=2, gamma_param=-0.8, lambd=-0.8, T_horizon=1,
                                       N_Processes=1, N_epochs=1000, epoch_cutoff=1000):
    process_at_T = []
    for _ in tqdm(range(N_Processes)):
        epochs = np.cumsum(np.random.exponential(scale=1 / 1, size=N_epochs)) / T_horizon
        epochs = np.array([epoch for epoch in epochs if epoch > epoch_cutoff])
        jumps = error_GIG_harder_Jumps(delta, gamma_param, lambd, T_horizon, epochs, epoch_cutoff)
        jumps = jumps / max(jumps)
        print(jumps)
        sum_jumps = np.sum(jumps)
        square_root_jumps = np.sqrt(sum_jumps)
        error_val = np.random.normal(loc=mu_W * sum_jumps, scale=std_W * square_root_jumps)
        process_at_T.append(error_val)

    process_at_T = np.array(process_at_T)
    return process_at_T


def CLT2_GHharder(mu=0, mu_W=0, std_W=1, delta=2, gamma_param=-0.8, lambd=0.8, N_Processes=15000, T_horizon=1,
                  N_epochs=10000, truncation=1e-2):
    assert (np.absolute(lambd) >= 0.5)
    epoch_cutoff = 3000
    print(epoch_cutoff)
    error_process = generate_simple_error_process_at_T(mu, mu_W, std_W, delta, gamma_param, lambd, T_horizon,
                                                       N_Processes, N_epochs, epoch_cutoff)
    print(error_process)
    fig2, ax4 = plt.subplots()
    fig2.figsize = (18, 18)
    plot_histogram_norm(error_process, ax=ax4)

    ksstat = stats.kstest(error_process, 'norm', [np.mean(error_process), np.std(error_process)]).pvalue
    print(stats.kstest(error_process, 'norm', [np.mean(error_process), np.std(error_process)]))
    """
    fig2, ax2 = plt.subplots()
    fig2.figsize = (18, 18)
    plot_histogram(error_process,
                   label="GH Residuals Histogram: $h(\Gamma_{i}) < $" + str(truncation) + " and pval $\\approx" + str(
                       round(ksstat * 1e144, 3)) + "\\times$10e-144", ax=ax2)
    plt.savefig("Hist")
    fig4, ax4 = plt.subplots()
    fig2.figsize = (18, 18)
    plot_qq(error_process,
            np.random.normal(np.mean(error_process), np.std(error_process), size=error_process.shape[0]), ax=ax4,
            label="GB Residuals QQ Plot: $h(\Gamma_{i}) < $" + str(truncation) + " and pval $\\approx" + str(
                round(ksstat * 1e144, 3)) + "\\times$10e-144")
    plt.savefig("QQ")
    """
    plt.show()


GIG_harder_process(N_epochs=1000, N_Processes=10000, lambd=-0.1, delta=2, gamma_param=0.1)
