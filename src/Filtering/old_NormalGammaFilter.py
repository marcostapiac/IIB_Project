import numpy as np
from tqdm import tqdm
from scipy.special import gamma as gammaf
import matplotlib.pyplot as plt
from ClassKalmanFilter import KalmanFilter
from NG_jumps import gammaJumps
from data_processing import obtainData


# def getNormalGammaCovarianceMatrix(Theta, t1, t2, nu, gamma, subordinator_truncation, Sc):
#    delta_t = t2 - t1
#    const = (2 * nu / (gamma ** 2)) * (1 - np.exp(-(gamma ** 2 / 2) * subordinator_truncation))
#    M1 = np.array(
#        [[1 / (2 * (Theta ** 3)), 1 / (2 * (Theta ** 2))], [1 / (2 * (Theta ** 2)), 1 / (2 * Theta)]])
#    M2 = np.array(
#        [[-2 / (Theta ** 3), -1 / (Theta ** 2)], [-1 / (Theta ** 2), 0]])
#    M3 = np.array([[1 / (Theta ** 2), 0], [0, 0]])
#    Sigma_c = (np.exp(2 * Theta * delta_t) - np.exp(2 * Theta * 0)) * M1 + (
#            np.exp(Theta * delta_t) - np.exp(Theta * 0)) * M2 + delta_t * M33
#    assert (Sigma_c.shape == (2, 2))
#    Sigma_c = const * Sigma_c
#    return np.array(Sc + Sigma_c + 1e-320 * np.eye(Sigma_c.shape[0]))  # Partial Gaussian Approximation

def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


def CheckResampling():
    """ Simplest check is to always resample the particles """
    return True


def ResampleSystematic(kfs, N_p, log_weights):
    u = np.zeros((N_p, 1))
    c = np.cumsum(np.exp(log_weights))
    c[-1] = 1.0
    i = 0
    u[0] = np.random.rand() / N_p
    new_kfs = [0] * N_p
    for j in range(N_p):

        u[j] = u[0] + j / N_p

        while u[j] > c[i]:
            i = i + 1

        new_kfs[j] = kfs[i]

    log_weights = np.array([-np.log(N_p)] * N_p)
    return new_kfs, log_weights


def Resample(a_updates, C_updates, log_weights):
    """OUT OF DATE WITH NEW KALMAN FILTER OOPROGRAMMING"""
    log_Wn = np.log(np.sum(np.exp(log_weights)))
    log_weights -= log_Wn
    N_p = len(log_weights)
    # make N subdivisions, and chose a random position within each one
    cumulative_sum = np.cumsum(np.exp(log_weights))
    cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
    indexes = np.searchsorted(cumulative_sum, np.random.random(len(log_weights)))
    new_as = [a_updates[i] for i in indexes]
    new_Cs = [C_updates[i] for i in indexes]
    log_weights = np.array([-np.log(N_p) for _ in indexes])
    return new_as, new_Cs, log_weights


def sort_jumps(jumps, times):
    """
    Sort process jumps into chronological order
    """
    # indices of sorted times
    idx = np.argsort(times)

    # store times and jump sizes sorted in this order
    times = np.take(times, idx)
    sizes = np.take(jumps, idx)
    return sizes, times


def normalGammaLatentSampling(t1, t2, nu, gamma, subordinator_truncation, P, Theta, var_W=1.0):
    """ Aim to efficiently compute observation and latent state marginals and posteriors """
    time = t2
    delta_t = t2 - t1
    # Generate jump times from Gamma Process
    epochs = np.cumsum(np.random.exponential(scale=1.0 / delta_t, size=2000))  # scale = 1/delta_t
    gJumps = gammaJumps(nu, (gamma ** 2) / 2, epochs)
    gJumps = gJumps[gJumps >= subordinator_truncation]
    Vs = np.random.uniform(t1, t2, size=gJumps.shape[0])
    Vs = Vs[gJumps >= subordinator_truncation]
    gJumps, Vs = sort_jumps(gJumps, Vs)
    """ P=2 Langevin-specific matrices for precise implementation"""
    M1 = np.array([1. / Theta, 1]).reshape((P, 1))
    M2 = np.array([-1. / Theta, 0]).reshape((P, 1))
    mean_sum_1 = np.sum(gJumps * (np.exp(Theta * (time - Vs))))
    mean_sum_2 = np.sum(gJumps)
    mean = mean_sum_1 * M1 + mean_sum_2 * M2
    M1 = np.array([[1. / (Theta ** 2), 1. / Theta], [1. / Theta, 1]]).reshape((P, P))
    M2 = np.array([[-2. / (Theta ** 2), -1. / Theta], [-1. / Theta, 0]]).reshape((P, P))
    M3 = np.array([[1. / (Theta ** 2), 0], [0, 0]]).reshape((P, P))
    assert (M1[0, 0] == 1 / Theta ** 2)
    assert (M2[1, 0] == -1 / Theta)
    assert (M3[1, 1] == 0)
    cov_sum_1 = np.sum(gJumps * np.exp(2 * Theta * (time - Vs)))
    cov_sum_2 = np.sum(gJumps * np.exp(Theta * (time - Vs)))
    cov_sum_3 = mean_sum_2
    cov = cov_sum_1 * M1 + cov_sum_2 * M2 + cov_sum_3 * M3
    mc = mean
    Sc = var_W * cov
    return mc, Sc


def getNormalGammaStateMatrix(Theta, nu, gamma, mc, P, subordinator_truncation, delta_t):
    # Commenting out Yc since we do not consider compensation
    Yc = (2 * nu / (gamma ** 2)) * np.exp(-(gamma ** 2 / 2) * subordinator_truncation) * (
            (np.exp(Theta * delta_t) - 1) * np.array([1 / Theta ** 2, 1 / Theta]).reshape(
        (2, 1)) + delta_t * np.array([-1 / Theta, 0]).reshape((2, 1)))
    M1 = np.array([[0., 1. / Theta], [0., 1.]])
    M2 = np.array([[1., -1. / Theta], [0., 0.]])
    assert (M1[0, 0] == 0. and M1[0, 1] == 1. / Theta and M1[1, 0] == 0. and M1[1, 1] == 1. and M1.shape == (2, 2))
    expA = np.exp(Theta * delta_t) * M1 + M2
    ss_A = np.vstack(
        [np.hstack([expA, mc - 0]),
         np.hstack([np.zeros(shape=(1, P)), np.array(1.).reshape(-1, 1)])])
    assert (ss_A.shape == (P + 1, P + 1))
    return ss_A


def getParticleWeightUpdate(alpha_W, beta_W, time_index, F_N, E_N1, E_N, M):
    weight_increment = -M / 2 * np.log(2 * np.pi) - 0.5 * np.log(np.abs(F_N)) - (alpha_W + time_index / 2) * np.log(
        beta_W + E_N / 2)
    weight_increment += (alpha_W + (time_index - 1) / 2) * np.log(beta_W + E_N1 / 2)
    if time_index < 300:
        weight_increment += np.log(gammaf(time_index / 2 + alpha_W)) - np.log(gammaf((time_index - 1) / 2 + alpha_W))
    else:
        weight_increment += 0.5 * np.log((time_index - 1) / 2 + alpha_W)
    return weight_increment


def getParticleWeightPredDecomp(F_N, var_W, diff_E, M=1):
    return -(M / 2) * np.log(2 * np.pi) - 0.5 * np.log(var_W * np.abs(F_N)) - (diff_E / (2 * var_W))


def RB_Filter(time_index, a_predicts, C_predicts, log_weights, var_W, N_p, nu, gamma,
              subordinator_truncation,
              t1, t2, Theta, alpha_W, beta_W, kfs, P, observation_n, position_estimates,
              velocity_estimates, skewness_estimates, position_stds, velocity_stds, skewness_stds, Mcs=None, Scs=None):
    """ Single Rao-Blackwellized Filter iteration """
    M = 1
    if CheckResampling():
        kfs, log_weights = ResampleSystematic(kfs, N_p, log_weights)
    delta_t = t2 - t1
    """ Nonlinear Filter"""
    for i in range(N_p):
        if Mcs:
            mc, Sc = Mcs[time_index], Scs[time_index]
        else:
            mc, Sc = normalGammaLatentSampling(t1, t2, nu, gamma, subordinator_truncation, P, Theta)
        """ Given proposed mc, Sc, we can now define Kalman Filter matrices """
        ss_A = getNormalGammaStateMatrix(Theta, nu, gamma, mc, P, subordinator_truncation, delta_t)
        Ce = np.array(Sc)  # getNormalGammaCovarianceMatrix(Theta, t1, t2, nu, gamma, subordinator_truncation, Sc)
        """ Kalman Filter"""
        E_N1 = kfs[i].get_sum_E()
        kfs[i].set_state_predict_matrix(ss_A)
        kfs[i].set_covariance_predict_matrix(Ce)
        kfs[i].predict()
        a_predicts[i] = kfs[i].get_state_mean()
        C_predicts[i] = kfs[i].get_covariance_matrix()
        kfs[i].update(observation_n)
        """ Incremental Weight Update via Likelihood """
        log_weight_increment = getParticleWeightUpdate(alpha_W, beta_W, time_index, kfs[i].get_F_N(), E_N1,
                                                       kfs[i].get_sum_E(), M)
        # log_weight_increment = getParticleWeightPredDecomp(F_N, var_W, E_N)
        """ Sufficient statistics update keeps track of non-linear particles """

        log_weights[i] += log_weight_increment
    log_Wn = logsumexp(log_weights)
    log_weights -= log_Wn
    for i in range(N_p):
        position_estimates[time_index] += np.exp(log_weights[i]) * a_predicts[i][0, 0]  # Estimated position mean
        velocity_estimates[time_index] += np.exp(log_weights[i]) * a_predicts[i][1, 0]  # Estimated velocity mean
        skewness_estimates[time_index] += np.exp(log_weights[i]) * a_predicts[i][2, 0]  # Estimated skewness mean
        position_stds[time_index] += np.exp(log_weights[i]) * (var_W * C_predicts[i][0, 0] + (a_predicts[i][0, 0] ** 2))
        velocity_stds[time_index] += np.exp(log_weights[i]) * (var_W * C_predicts[i][1, 1] + (a_predicts[i][1, 0] ** 2))
        skewness_stds[time_index] += np.exp(log_weights[i]) * (var_W * C_predicts[i][2, 2] + (a_predicts[i][2, 0] ** 2))
    position_stds[time_index] -= (position_estimates[time_index] ** 2)
    velocity_stds[time_index] -= (velocity_estimates[time_index] ** 2)
    skewness_stds[time_index] -= (skewness_estimates[time_index] ** 2)
    if -1e-1 < position_stds[time_index] < -1e-323:
        position_stds[time_index] = 0
    else:
        position_stds[time_index] = np.sqrt(position_stds[time_index])
    velocity_stds[time_index] = np.sqrt(velocity_stds[time_index])
    skewness_stds[time_index] = np.sqrt(skewness_stds[time_index])
    return a_predicts, C_predicts, N_p, log_weights, position_estimates, velocity_estimates, skewness_estimates, position_stds, velocity_stds, skewness_stds


def Langevin_2D(simulate_data=True):
    """ Case 3 (partial Gaussian approximation)"""
    """ Latent Space Model:  dx(t) = Ax(t)dt + HdW(t) """
    """ State is [x(t), xdot(t)], observations are always 1D """
    P = 2
    T_horizon = 1.0
    N_p = 200  # Number of particles initialised by ourselves AND constant if resampling
    Theta, var_W, k_V, k_W, alpha_W, beta_W, Cv, mu_mu_W = langevinParameters()  # P-dim Langevin model
    nu, gamma, subordinator_truncation = gammaParameters()  # Subordinate process parameters

    """ State space model matrices """
    H, B = getStateSpaceMatrices(P)
    a_update = np.vstack([np.zeros(shape=(P, 1)), mu_mu_W]).reshape((P + 1, 1))  # a_0_0 (is shape of a_update P+1)
    C_update = np.vstack([np.hstack([np.zeros(shape=(P, P)), np.zeros(shape=(P, 1))]), np.hstack(
        [np.zeros(shape=(1, P)), np.array(1 * k_W).reshape(-1, 1)])])
    Mcs = [np.zeros(shape=(P, 1))]
    Scs = [C_update]
    if simulate_data:
        """ Simulate data """
        time_ax = np.linspace(0., T_horizon, 100)  # Generate time-axis during which we are tracking
        L = np.linalg.cholesky(var_W * (C_update + 1e-100 * np.eye(C_update.shape[0])))
        u = np.random.normal(0, 1.0, (P + 1)).reshape((P + 1, 1))
        normRV = a_update + L @ u
        States = [normRV]
        Observations = [0]
        for i in tqdm(range(1, len(time_ax))):
            t1 = time_ax[i - 1]
            t2 = time_ax[i]
            delta_t = t2 - t1
            mc, Sc = normalGammaLatentSampling(t1, t2, nu, gamma, subordinator_truncation, P, Theta, var_W)
            Mcs.append(mc)
            Scs.append(Sc)
            ss_A = getNormalGammaStateMatrix(Theta, nu, gamma, mc, P, subordinator_truncation, delta_t)
            Ce = var_W * Sc  # getNormalGammaCovarianceMatrix(Theta, t1, t2, nu, gamma, subordinator_truncation, Sc)
            # mean = np.array((2 * nu / (gamma ** 2)) * (1 - np.exp(-(gamma ** 2 / 2)) * subordinator_truncation) * (
            #        (np.exp(Theta * delta_t) - 1) * np.array([1 / Theta ** 2, 1 / Theta]).reshape(
            #    (2, 1)) + delta_t * np.array([-1 / Theta, 0]).reshape((2, 1)))).reshape((P, 1))
            # Cholesky for hidden state
            try:
                L = np.linalg.cholesky(Ce)
                u = np.random.randn(P)
                normRV = L @ u
            except:
                normRV = np.zeros(P)
            state = ss_A @ States[i - 1] + B @ (normRV.reshape((P,1)))
            # Normal random generation for observation
            v = np.sqrt(var_W * Cv) * np.random.randn()
            obs = H @ state + v
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


    """ Initialise RBP Filter """
    L = len(Observations)
    a_predicts, C_predicts, position_estimates, velocity_estimates, skewness_estimates, \
    position_stds, velocity_stds, skewness_stds, log_weights, kfs = initialiseRBPFilter(H, B, P, Cv, N_p, L, a_update,
                                                                                        C_update)
    """ Run RBP Filter """
    for i in tqdm(range(1, len(time_ax))):
        a_predicts, C_predicts, N_p, log_weights, position_estimates, velocity_estimates, \
        skewness_estimates, position_stds, velocity_stds, skewness_stds = \
            RB_Filter(i, a_predicts, C_predicts, log_weights, var_W, N_p, nu, gamma,
                      subordinator_truncation, time_ax[i - 1], time_ax[i], Theta, alpha_W, beta_W, kfs, P,
                      Observations[i],
                      position_estimates, velocity_estimates, skewness_estimates, position_stds, velocity_stds,
                      skewness_stds)

    """ Plot results """
    if simulate_data:
        plotFilterResults(time_ax, position_states, position_estimates, position_stds, velocity_states,
                          velocity_estimates,
                          velocity_stds, skewness_states,
                          skewness_estimates, skewness_stds, Theta, N_p, k_V, nu, gamma, var_W, k_W)
    else:
        plotFilterResults(time_ax, Observations, position_estimates, position_stds, np.zeros(L),
                          velocity_estimates,
                          velocity_stds, np.zeros(L),
                          skewness_estimates, skewness_stds, Theta, N_p, k_V, nu, gamma, var_W, k_W)


def getStateSpaceMatrices(P):
    """ State space model matrices """
    H = np.array(np.append(1, np.zeros(shape=(1, P)))).reshape((1, P + 1))  # Size (1, P+1)
    B = np.vstack([np.identity(P), np.zeros(shape=(1, P))]).reshape((P + 1, P))  # Only present in cases 1 and 3
    return H, B


def langevinParameters():
    """ Function to decide langevin model parameters """
    Theta = -1.  # Mean reversion term -> small theta means jump lasts longer
    var_W = 1 ** 2  # Variance of U_is (or W_is) in generalised shot noise method.
    k_V = 1e-1  # Variance scaling for observation noise (if matrix, need to change observation parameters)
    k_W = 1e-1  # Variance scaling for mean (when included as latent param)
    alpha_W, beta_W = 1., 1.  # Variance prior parameters
    Cv = 1 * k_V
    mu_mu_W = 10  # np.mean(observations)  # Mean prior mean
    return Theta, var_W, k_V, k_W, alpha_W, beta_W, Cv, mu_mu_W


def initialiseRBPFilter(H, B, P, Cv, N_p, L, a_initial, C_initial):
    E_N = ((H @ a_initial)[0, 0] ** 2) / ((H @ C_initial @ H.T)[0, 0] + Cv)
    log_weights = np.array([0.0] * N_p)  # Store the log weights
    log_Wn = logsumexp(log_weights)
    log_weights -= log_Wn  # Normalise weights
    a_predicts = [0] * N_p
    C_predicts = [0] * N_p
    position_estimates, velocity_estimates, skewness_estimates = np.zeros(L), np.zeros(L), np.zeros(L)
    position_stds, velocity_stds, skewness_stds = np.zeros(L), np.zeros(L), np.zeros(L)
    kfs = [KalmanFilter(a_initial, C_initial, 0, np.eye(P + 1), np.zeros(shape=(1, 1)), B, np.eye(P), H, Cv, E_N) for _
           in range(N_p)]
    return a_predicts, C_predicts, position_estimates, velocity_estimates, skewness_estimates, position_stds, velocity_stds, skewness_stds, log_weights, kfs


def gammaParameters():
    """ Function to choose the parameters of the gamma process during simulation """
    nu = 1
    gamma = np.sqrt(2)
    subordinator_truncation = 1e-300  # Truncation on Gamma Jumps
    return nu, gamma, subordinator_truncation


def plotFilterResults(time_ax, position_states, position_estimates, position_stds, velocity_states, velocity_estimates,
                      velocity_stds, skewness_states,
                      skewness_estimates, skewness_stds, Theta, N_p, k_V, nu, gamma, var_W, k_W, std_width=1.96):
    fig, ax = plt.subplots(3, sharex=True, sharey=False, figsize=(14, 9.5))
    ax[0].set_title("Normal Gamma Particle Filtering, $\\theta, N_{particles}, k_{V}, \\nu, \gamma, \sigma_{W}^{2}, "
                    "k_{W} = " + str(Theta) + ", " + str(N_p) + ", " + str(k_V) + ", " + str(nu) + ", " + str(
        round(gamma, 4)) + ", " + str(var_W) + ", " + str(k_W) + "$")
    ax[0].plot(time_ax, position_states, linestyle="dashed", color='black', label="True Signal")
    ax[0].plot(time_ax, position_estimates, linestyle="dotted", color='orange', label="Estimated Signal")
    ax[0].fill_between(time_ax, np.array(position_estimates) - std_width * np.array(position_stds),
                       np.array(position_estimates) + std_width * np.array(position_stds),
                       label="$\pm 2$ standard deviations")
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
