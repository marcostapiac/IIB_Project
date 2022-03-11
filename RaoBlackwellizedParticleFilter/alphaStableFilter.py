import numpy as np
from tqdm import tqdm
from scipy.special import gamma
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter
from scipy.stats import norm
from data_processing import obtainData


def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


def LatentSampling(t1, t2, truncation, alpha, P, Theta):
    """ Aim to efficiently compute observation and latent state marginals and posteriors """
    time = t2
    delta_t = t2 - t1
    # Generate jump times from Gamma Process
    epochs = np.cumsum(np.random.exponential(scale=1.0 / delta_t, size=1000))
    Vs = np.random.uniform(t1, t2, size=epochs.shape[0])  # delta_t or T?
    epochs = epochs * (epochs <= delta_t * truncation)
    Vs = Vs[epochs > 0]
    epochs = epochs[epochs > 0]
    """ P=2 Langevin-specific matrices for precise implementation"""
    M1 = np.array([1 / Theta, 1]).reshape((P, 1))
    M2 = np.array([-1 / Theta, 0]).reshape((P, 1))
    mean_sum_1 = np.sum((epochs ** (-1 / alpha)) * (np.exp(Theta * (time - Vs))))
    mean_sum_2 = np.sum((epochs ** (-1 / alpha)))
    mean = mean_sum_1 * M1 + mean_sum_2 * M2
    M1 = np.array([[1 / (Theta ** 2), 1 / Theta], [1 / Theta, 1]]).reshape((P, P))
    M2 = np.array([[-2 / (Theta ** 2), -1 / Theta], [-1 / Theta, 0]]).reshape((P, P))
    M3 = np.array([[1 / (Theta ** 2), 0], [0, 0]]).reshape((P, P))
    assert (M1[0, 0] == 1 / Theta ** 2)
    assert (M2[1, 0] == -1 / Theta)
    assert (M3[1, 1] == 0)
    cov_sum_1 = np.sum((epochs ** (-2 / alpha)) * np.exp(2 * Theta * (time - Vs)))
    cov_sum_2 = np.sum((epochs ** (-2 / alpha)) * np.exp(Theta * (time - Vs)))
    cov_sum_3 = np.sum((epochs ** (-2 / alpha)))
    cov = cov_sum_1 * M1 + cov_sum_2 * M2 + cov_sum_3 * M3
    mc = (1 ** (1 / alpha)) * mean  # (delta_t ** (1 / alpha) * mean)
    Sc = (1 ** (2 / alpha)) * cov
    Sc += 1e-12 * np.eye(cov.shape[0])
    return mc, Sc


def CheckResampling():
    """ Simplest check is to always resample the particles """
    return True


def ResampleSystematic(a_updates, C_updates, E_Ns, log_weights):
    N_p = len(a_updates)
    new_as = [np.zeros(a_updates[0].shape)] * N_p
    new_Cs = [np.zeros(a_updates[0].shape)] * N_p
    new_ENs = [np.zeros(a_updates[0].shape)] * N_p
    u = np.zeros((N_p, 1))
    c = np.cumsum(np.exp(log_weights))
    c[-1] = 1.0
    i = 0
    u[0] = np.random.rand() / N_p

    for j in range(N_p):

        u[j] = u[0] + j / N_p

        while u[j] > c[i]:
            i = i + 1

        new_as[j] = a_updates[i]
        new_Cs[j] = C_updates[i]
        new_ENs[j] = E_Ns[i]

    log_weights = np.array([-np.log(N_p)] * N_p)
    return new_as, new_Cs, new_ENs, log_weights


def ResampleStratified(a_updates, C_updates, log_weights):
    """From filterpy.monte_carlo.stratified_resample"""
    N_p = len(log_weights)
    # make N subdivisions, and chose a random position within each one
    positions = (np.random.random(N_p) + range(N_p)) / N_p
    indexes = np.zeros(N_p, 'i')
    cumulative_sum = np.cumsum(np.exp(log_weights))
    i, j = 0, 0
    while i < N_p:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    new_as = [a_updates[i] for i in indexes]
    new_Cs = [C_updates[i] for i in indexes]
    log_weights = np.array([- np.log(N_p) for _ in indexes])
    return new_as, new_Cs, log_weights


def Resample(a_updates, C_updates, E_Ns, log_weights):
    N_p = len(log_weights)
    # make N subdivisions, and chose a random position within each one
    cumulative_sum = np.cumsum(np.exp(log_weights))
    assert (not np.isnan(cumulative_sum).all())
    # cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
    indexes = np.searchsorted(cumulative_sum, np.random.random(len(log_weights)))
    new_as = [a_updates[i] for i in indexes]
    new_Cs = [C_updates[i] for i in indexes]
    new_Ens = [E_Ns[i] for i in indexes]
    log_weights = np.array([-np.log(N_p) for _ in indexes])
    return new_as, new_Cs, new_Ens, log_weights


def getStateCovarianceMatrix(Theta, t1, t2, alpha, truncation, Sc, var_W):
    assert (t2 - t1 > 0)
    delta_t = t2 - t1
    const = (alpha / (2 - alpha)) * truncation ** (1 - (2 / alpha))
    M1 = np.array(
        [[1 / (2 * (Theta ** 3)), 1 / (2 * (Theta ** 2))], [1 / (2 * (Theta ** 2)), 1 / (2 * Theta)]])
    M2 = np.array(
        [[-2 / (Theta ** 3), -1 / (Theta ** 2)], [-1 / (Theta ** 2), 0]])
    M3 = np.array([[1 / (Theta ** 2), 0], [0, 0]])
    Sigma_c = (np.exp(2 * Theta * delta_t) - np.exp(2 * Theta * 0)) * M1 + (
            np.exp(Theta * delta_t) - np.exp(Theta * 0)) * M2 + delta_t * M3
    assert (Sigma_c.shape == (2, 2))
    Sigma_c = const * Sigma_c
    return var_W * np.array(Sc + Sigma_c)  # Partial Gaussian Approximation


def getStateMatrix(alpha, Theta, mc, P, truncation, delta_t):
    assert (delta_t > 0)
    Yc = ((alpha > 1) * (alpha / (alpha - 1)) * (truncation ** (1 - (1 / alpha)))) * (
            (np.exp(Theta * delta_t) - 1) * np.array([1 / (Theta ** 2), 1 / Theta]).reshape(
        (2, 1)) + delta_t * np.array([-1 / Theta, 0]).reshape((2, 1)))  # Specific to Langevin model
    expA = np.exp(Theta * delta_t) * np.array([[0, 1 / Theta], [0, 1]]) + np.array([[1, -1 / Theta], [0, 0]])
    ss_A = np.vstack(
        [np.hstack([expA, mc - Yc]),
         np.hstack([np.zeros(shape=(1, P)), np.array(1).reshape(-1, 1)])])
    assert (ss_A.shape == (P + 1, P + 1))
    return ss_A


def getParticleWeightUpdate(alpha_W, beta_W, time_index, F_N, E_N1, E_N, M):
    weight_increment = -M / 2 * np.log(2 * np.pi) - 0.5 * np.log(np.abs(F_N)) - (alpha_W + time_index / 2) * np.log(
        beta_W + E_N / 2)
    weight_increment += (alpha_W + (time_index - 1) / 2) * np.log(beta_W + E_N1 / 2)
    if time_index < 300:
        weight_increment += np.log(gamma(time_index / 2 + alpha_W)) - np.log(gamma((time_index - 1) / 2 + alpha_W))
    else:
        weight_increment += 0.5 * np.log((time_index - 1) / 2 + alpha_W)
    return weight_increment


def getParticleWeightPredDecomp(F_N, var_W, diff_E, M=1):
    return -(M / 2) * np.log(2 * np.pi) - 0.5 * np.log(var_W * np.abs(F_N)) - (diff_E / (2 * var_W))


def RB_Filter(time_index, a_updates, C_updates, a_predicts, C_predicts, log_weights, var_W, N_p, truncation,
              t1, t2, alpha, Theta, alpha_W, beta_W, E_Ns, H, Cv, B, observation_n, position_estimates,
              velocity_estimates, skewness_estimates, position_stds, velocity_stds, skewness_stds, Mcs=None, Scs=None):
    """ Single Rao-Blackwellized Filter iteration """
    P = B.shape[0] - 1
    M = 1  # Dimension of
    if CheckResampling():
        a_updates, C_updates, E_Ns, log_weights = ResampleSystematic(a_updates, C_updates, E_Ns, log_weights)
    delta_t = t2 - t1
    """ Nonlinear Filter"""
    for i in range(N_p):
        if Mcs:
            mc, Sc = Mcs[time_index], Scs[time_index]
        else:
            mc, Sc = LatentSampling(t1, t2, truncation, alpha, P, Theta)
        """ Given proposed mc, Sc, we can now define Kalman Filter matrices """
        ss_A = getStateMatrix(alpha, Theta, mc, P, truncation, delta_t)
        Ce = getStateCovarianceMatrix(Theta, t1, t2, alpha, truncation, Sc, 1)
        """ Kalman Filter"""
        E_N1 = E_Ns[i]
        a_predict, C_predict, a_update, C_update, F_N, E_N = KalmanFilter(ss_A, np.zeros(shape=(1, 1)), B, Ce,
                                                                          Cv, H,
                                                                          a_updates[i],
                                                                          C_updates[i], observation_n, E_N1=E_N1)
        """ Incremental Weight Update via Likelihood """
        log_weight_increment = getParticleWeightUpdate(alpha_W, beta_W, time_index, F_N, E_N1, E_N, M)
        # log_weight_increment = getParticleWeightPredDecomp(F_N, var_W, E_N)
        """ Sufficient statistics update keeps track of non-linear particles """
        a_updates[i] = a_update
        C_updates[i] = C_update
        a_predicts[i] = a_predict
        C_predicts[i] = C_predict
        E_Ns[i] = E_N
        log_weights[i] += log_weight_increment
    log_Wn = logsumexp(log_weights)
    log_weights -= log_Wn
    for i in range(N_p):
        position_estimates[time_index] += np.exp(log_weights[i]) * a_updates[i][0, 0]  # Estimated position mean
        velocity_estimates[time_index] += np.exp(log_weights[i]) * a_updates[i][1, 0]  # Estimated velocity mean
        skewness_estimates[time_index] += np.exp(log_weights[i]) * a_updates[i][2, 0]  # Estimated skewness mean
        position_stds[time_index] += np.exp(log_weights[i]) * (var_W * C_updates[i][0, 0] + (a_updates[i][0, 0] ** 2))
        velocity_stds[time_index] += np.exp(log_weights[i]) * (var_W * C_updates[i][1, 1] + (a_updates[i][1, 0] ** 2))
        skewness_stds[time_index] += np.exp(log_weights[i]) * (var_W * C_updates[i][2, 2] + (a_updates[i][2, 0] ** 2))
    position_stds[time_index] -= (position_estimates[time_index] ** 2)
    velocity_stds[time_index] -= (velocity_estimates[time_index] ** 2)
    skewness_stds[time_index] -= (skewness_estimates[time_index] ** 2)
    if -1e-1 < position_stds[time_index] < -1e-323:
        position_stds[time_index] = 0
    else:
        position_stds[time_index] = np.sqrt(position_stds[time_index])
    position_stds[time_index] = np.sqrt(position_stds[time_index])
    velocity_stds[time_index] = np.sqrt(velocity_stds[time_index])
    skewness_stds[time_index] = np.sqrt(skewness_stds[time_index])
    return a_updates, C_updates, a_predicts, C_predicts, N_p, E_Ns, log_weights, position_estimates, velocity_estimates, skewness_estimates, position_stds, velocity_stds, skewness_stds


def Langevin_2D(simulate_data=True):
    """ Case 3 (partial Gaussian approximation)"""
    """ Latent Space Model:  dx(t) = Ax(t)dt + HdW(t) """
    """ State is [x(t), xdot(t)], observations are always 1D """
    P = 2
    N_p = 200  # Number of particles initialised by ourselves AND constant if resampling
    """ Langevin Model parameters"""
    Theta, var_W, k_V, k_W, alpha_W, beta_W, Cv, mu_mu_W = getLangevinParameters()
    alpha, truncation = getAlphaStableParameters()
    H, B = getStateSpaceMatrices(P)
    if simulate_data:
        """ Simulate data """
        T_horizon = 50
        a_update = np.vstack([np.zeros(shape=(P, 1)), mu_mu_W]).reshape((P + 1, 1))  # a_0_0 (is shape of a_update P+1)
        C_update = np.vstack([np.hstack([np.zeros(shape=(P, P)), np.zeros(shape=(P, 1))]),
                              np.hstack([np.zeros(shape=(1, P)),
                                         np.array(1 * k_W).reshape(-1, 1)])])  # C_0_0 (does it need the var_W YES)?
        time_ax = np.linspace(0, T_horizon, 500)  # Generate time-axis during which we are tracking
        States = [a_update + np.random.multivariate_normal(mean=np.zeros(P + 1), cov=C_update)]
        Observations = [0]
        Mcs = [0]
        Scs = [0]
        for i in tqdm(range(1, len(time_ax))):
            t1 = time_ax[i - 1]
            t2 = time_ax[i]
            delta_t = t2 - t1
            mc, Sc = LatentSampling(t1, t2, truncation, alpha, P, Theta)
            ss_A = getStateMatrix(alpha, Theta, mc, P, truncation, delta_t)
            Ce = getStateCovarianceMatrix(Theta, t1, t2, alpha, truncation, Sc, 1)
            Mcs.append(mc)
            Scs.append(Sc)
            state = ss_A @ States[i - 1] + B @ np.random.multivariate_normal(mean=np.zeros(P), cov=var_W * Ce).reshape(
                (P, 1))
            obs = H @ state + np.random.normal(loc=0, scale=np.sqrt(var_W * Cv))
            States.append(state)
            Observations.append(obs[0, 0])
            position_states = []
            velocity_states = []
            skewness_states = []
            for s in States:
                position_states.append(s[0, 0])
                velocity_states.append(s[1, 0])
                skewness_states.append(s[2, 0])
    else:
        Observations = obtainData()
        time_ax = np.linspace(0, len(Observations), len(Observations))
        a_update = np.vstack([0, 0, mu_mu_W]).reshape((P + 1, 1))
        C_update = k_W * np.eye(P + 1)

    """ Initialise RBP Filter """
    L = len(Observations)
    a_predicts, C_predicts, a_updates, C_updates, position_estimates, velocity_estimates, skewness_estimates, \
    position_stds, velocity_stds, skewness_stds, log_weights, E_Ns = initialiseRBPFilter(H, Cv, N_p, L, a_update,
                                                                                         C_update)
    """ Run RBP Filter """
    for i in tqdm(range(1, L)):
        a_updates, C_updates, a_predicts, C_predicts, N_p, E_Ns, log_weights, position_estimates, velocity_estimates, \
        skewness_estimates, position_stds, velocity_stds, skewness_stds = RB_Filter(i, a_updates, C_updates, a_predicts,
                                                                                    C_predicts,
                                                                                    log_weights, var_W, N_p,
                                                                                    truncation, time_ax[i - 1],
                                                                                    time_ax[i],
                                                                                    alpha, Theta, alpha_W, beta_W, E_Ns,
                                                                                    H, Cv, B,
                                                                                    Observations[i], position_estimates,
                                                                                    velocity_estimates,
                                                                                    skewness_estimates, position_stds,
                                                                                    velocity_stds, skewness_stds)

    """ Plot results """
    if simulate_data:
        plotFilterResults(time_ax, position_states, position_estimates, position_stds, velocity_states,
                          velocity_estimates,
                          velocity_stds, skewness_states,
                          skewness_estimates, skewness_stds, Theta, N_p, k_V, alpha, var_W, k_W)
    else:
        plotFilterResults(time_ax, Observations, position_estimates, position_stds, np.zeros(L),
                          velocity_estimates,
                          velocity_stds, np.zeros(L),
                          skewness_estimates, skewness_stds, Theta, N_p, k_V, alpha, var_W, k_W)


def getStateSpaceMatrices(P):
    """ State space model matrices """
    H = np.array(np.append(1, np.zeros(shape=(1, P)))).reshape((1, P + 1))  # Size (1, P+1)
    B = np.vstack([np.identity(P), np.zeros(shape=(1, P))]).reshape((P + 1, P))  # Only present in cases 1 and 3
    return H, B


def getLangevinParameters():
    Theta = -1  # Mean reversion term -> small theta means jump lasts longer
    var_W = 1 ** 2  # Variance of U_is (or W_is) in generalised shot noise method.
    k_V = 1e1  # Variance scaling for observation noise (if matrix, need to change observation parameters)
    k_W = 1  # Variance scaling for mean (when included as latent param)
    alpha_W, beta_W = 1, 1  # Variance prior parameters
    Cv = 1 * k_V
    mu_mu_W = 0  # np.mean(observations)  # Mean prior mean
    return Theta, var_W, k_V, k_W, alpha_W, beta_W, Cv, mu_mu_W


def getAlphaStableParameters():
    alpha = 0.8  # Determines Levy heavy-tailed-ness
    truncation = 0.5 ** (-alpha)  # Epoch size cutoff
    return alpha, truncation


def initialiseRBPFilter(H, Cv, N_p, L, a_update, C_update):
    E_Ns = [((H @ a_update)[0, 0] ** 2) / ((H @ C_update @ H.T)[0, 0] + Cv)] * N_p
    log_weights = np.array([0.0] * N_p)  # Store the log weights
    log_Wn = logsumexp(log_weights)
    log_weights -= log_Wn
    a_updates = [a_update] * N_p
    C_updates = [C_update] * N_p
    a_predicts = [0] * N_p
    C_predicts = [0] * N_p
    position_estimates, velocity_estimates, skewness_estimates = np.zeros(L), np.zeros(L), np.zeros(L)
    position_stds, velocity_stds, skewness_stds = np.zeros(L), np.zeros(L), np.zeros(L)
    return a_predicts, C_predicts, a_updates, C_updates, position_estimates, velocity_estimates, skewness_estimates, position_stds, velocity_stds, skewness_stds, log_weights, E_Ns


def plotFilterResults(time_ax, position_states, position_estimates, position_stds, velocity_states, velocity_estimates,
                      velocity_stds, skewness_states,
                      skewness_estimates, skewness_stds, Theta, N_p, k_V, alpha, var_W, k_W, std_width = 1.96):
    fig, ax = plt.subplots(3, sharex=True, sharey=False, figsize=(14,9.5))
    ax[0].set_title("Alpha-Stable Particle Filtering, $\\theta, N_{particles}, k_{V}, \\alpha, \sigma_{W}^{2}, k_{W} = " + str(
        Theta) + ", " + str(N_p) + ", " + str(k_V) + ", " + str(alpha) + ", " + str(var_W) + ", " + str(k_W) + "$")
    ax[0].plot(time_ax, position_states, linestyle="dashed", color='black', label="True Signal")
    ax[0].plot(time_ax, position_estimates, linestyle="dotted", color='orange', label="Estimated Signal")
    ax[0].fill_between(time_ax, np.array(position_estimates) - std_width * np.array(position_stds),
                       np.array(position_estimates) + std_width * np.array(position_stds), label="$\pm 2$ standard deviations")
    ax[0].set_ylabel("Position, $x(t)$")
    ax[0].legend()
    ax[1].plot(time_ax, velocity_states, linestyle="dashed", color='black')
    ax[1].plot(time_ax, velocity_estimates, linestyle="dotted", color='orange')
    ax[1].fill_between(time_ax, np.array(velocity_estimates) - std_width * np.array(velocity_stds),
                       np.array(velocity_estimates) + std_width * np.array(velocity_stds))
    ax[1].set_ylabel("Velocity, $\dot{x}(t)$")
    ax[2].plot(time_ax, skewness_states, linestyle="dashed", color='black')
    ax[2].plot(time_ax, skewness_estimates, linestyle="dotted", color='orange')
    ax[2].fill_between(time_ax, np.array(skewness_estimates) - std_width * np.array(skewness_stds),
                       np.array(skewness_estimates) + std_width * np.array(skewness_stds))
    ax[2].set_xlabel("Time, $t$")
    ax[2].set_ylabel("Process Skewness, $\mu_{W}$")
    plt.show()


Langevin_2D(simulate_data=True)
