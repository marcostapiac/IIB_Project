import numpy as np
import scipy.special
from tqdm import tqdm
from scipy.special import gamma as gammaf
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter
from NG_jumps import gammaJumps


def normalGammaLatentSampling(t1, t2, nu, gamma, truncation, P, h, A, Theta):
    """ Aim to efficiently compute observation and latent state marginals and posteriors """
    time = t2
    delta_t = t2 - t1
    # Generate jump times from Gamma Process
    epochs = np.cumsum(np.random.exponential(scale=1.0, size=100000)) / delta_t
    epochs = epochs[epochs < delta_t * truncation]
    Vs = np.random.uniform(t1, t2, size=epochs.shape[0])  # delta_t or T?
    gJumps = gammaJumps(nu, gamma ** 2 / 2, epochs)
    L = len(gJumps)
    if L == 0:
        return np.zeros(shape=(P, 1)), np.zeros(shape=(P, P))
    """ P=2 Langevin-specific matrices for precise implementation"""
    CV1 = np.array([1 / Theta, 1]).reshape((P, 1))
    CV2 = np.array([-1 / Theta, 0]).reshape((P, 1))
    assert ((np.exp(Theta * (time - Vs[0])) * CV1 + CV2).all() == (np.exp(A * (time - Vs[0])) @ h).all())
    mean = gJumps[0] * (np.exp(Theta * (time - Vs[0])) * CV1 + CV2)
    M1 = np.array([[1 / (Theta ** 2), 1 / Theta], [1 / Theta, 1]]).reshape((P, P))
    M2 = np.array([[-2 / (Theta ** 2), -1 / Theta], [-1 / Theta, 0]]).reshape((P, P))
    M3 = np.array([[1 / (Theta ** 2), 0], [0, 0]]).reshape((P, P))
    assert (M1[0, 0] == 1 / Theta ** 2)
    assert (M2[1, 0] == -1 / Theta)
    assert (M3[1, 1] == 0)
    assert ((np.exp(2 * Theta * (time - Vs[0])) * M1 + np.exp(Theta * (time - Vs[
        0])) * M2 + M3).all() == (np.exp(A * (time - Vs[0])) @ h @ h.T @ np.exp(A * (time - Vs[0])).T).all())
    cov = gJumps[0] * (np.exp(2 * Theta * (time - Vs[0])) * M1 + np.exp(Theta * (time - Vs[
        0])) * M2 + M3)
    for i in range(1, L):
        ei = gJumps[i]
        mean += ei * (np.exp(Theta * (time - Vs[i])) * CV1 + CV2)
        cov += ei * (np.exp(2 * Theta * (time - Vs[i])) * M1 + np.exp(
            Theta * (time - Vs[i])) * M2 + M3)  # expi @ h @ h.T @ expi.T
    mc = mean  # (delta_t ** (1 / alpha) * mean) @ h
    Sc = cov
    return mc, Sc


def CheckResampling():
    """ Simplest check is to always resample the particles """
    return True


def ResampleStratified(a_updates, C_updates, log_weights):
    """From filterpy.monte_carlo.stratified_resample"""
    log_Wn = np.log(np.sum(np.exp(log_weights)))
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
    log_weights = np.array([log_Wn - np.log(N_p) for _ in indexes])
    return new_as, new_Cs, log_weights


def Resample(a_updates, C_updates, log_weights):
    log_Wn = np.log(np.sum(np.exp(log_weights)))
    log_weights -= log_Wn
    N_p = len(log_weights)
    # make N subdivisions, and chose a random position within each one
    cumulative_sum = np.cumsum(np.exp(log_weights))
    assert (not np.isnan(cumulative_sum).all())
    cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
    indexes = np.searchsorted(cumulative_sum, np.random.random(len(log_weights)))
    new_as = [a_updates[i] for i in indexes]
    new_Cs = [C_updates[i] for i in indexes]
    log_weights = np.array([- np.log(N_p) for _ in indexes])
    return new_as, new_Cs, log_weights


def getNormalGammaCovarianceMatrix(Theta, t1, t2, nu, gamma, subordinator_truncation, Sc):
    const = 2 * nu / (gamma ** 2) * (1 - np.exp(-gamma ** 2 / 2 * subordinator_truncation))
    Sigma_c = (np.exp(2 * Theta * t2) - np.exp(2 * Theta * t1)) * np.array(
        [[1 / (2 * Theta ** 3), 1 / (2 * Theta ** 2)], [1 / (2 * Theta ** 2), 1 / (2 * Theta)]])
    assert (Sigma_c.shape == (2, 2))  # (P,P)
    Sigma_c += (np.exp(Theta * t2) - np.exp(Theta * t1)) * np.array(
        [[-2 / (Theta ** 3), -1 / (Theta ** 2)], [-1 / (Theta ** 2), 0]])
    Sigma_c += (t2 - t1) * np.array([[1 / Theta ** 2, 0], [0, 0]])
    Sigma_c *= const
    assert (np.linalg.eigvals(Sigma_c).all() > 0)
    return np.array(Sc + Sigma_c)  # Partial Gaussian Approximation


def getNormalGammaStateMatrix(Theta, nu, gamma, mc, P, subordinator_truncation, delta_t):
    Yc = 2 * nu / (gamma ** 2) * np.exp(-gamma ** 2 / 2 * subordinator_truncation) * (
            (np.exp(Theta * delta_t) - 1) * np.array([1 / Theta ** 2, 1 / Theta]).reshape(
        (2, 1)) + np.array([-1 / Theta, 0]).reshape((2, 1)))
    expA = np.exp(Theta * delta_t) * np.array([[0, 1 / Theta], [0, 1]]) + np.array([[1, -1 / Theta], [0, 0]])
    ss_A = np.vstack(
        [np.hstack([expA, mc - Yc]),
         np.hstack([np.zeros(shape=(1, P)), np.array(1).reshape(1, 1)])])
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


def RB_Filter(time_index, a_updates, C_updates, Mcs, Scs, log_weights, N_p, nu, gamma, truncation,
              subordinator_truncation,
              t1, t2, h, Theta, alpha_W, beta_W, sde_A, E_Ns, H, Cv, B, observation_n, position_estimates,
              velocity_estimates, mean_estimates):
    """ Single Rao-Blackwellized Filter iteration """
    P = B.shape[0] - 1
    M = 1  # Dimension of observations
    delta_t = t2 - t1
    """ Nonlinear Filter"""
    for i in range(N_p):
        mc, Sc = normalGammaLatentSampling(t1, t2, nu, gamma, truncation, P, h, sde_A, Theta)
        Mcs[i] = mc
        Scs[i] = Sc
        """ Given proposed mc, Sc, we can now define Kalman Filter matrices """
        ss_A = getNormalGammaStateMatrix(Theta, nu, gamma, mc, P, subordinator_truncation, delta_t)
        Ce = getNormalGammaCovarianceMatrix(Theta, t1, t2, nu, gamma, subordinator_truncation, Sc)
        """ Kalman Filter"""
        E_N1 = E_Ns[i]
        _, _, a_update, C_update, F_N, E_N = KalmanFilter(ss_A, np.zeros(shape=(1, 1)), B, Ce, Cv, H, a_updates[i],
                                                          C_updates[i], observation_n, E_N1=E_N1)
        """ Sufficient statistics update keeps track of non-linear particles """
        a_updates[i] = a_update
        C_updates[i] = C_update
        E_Ns[i] = E_N
        """ Incremental Weight Update via Likelihood """
        weight_increment = getParticleWeightUpdate(alpha_W, beta_W, time_index, F_N, E_N1, E_N, M)
        log_weights[i] += weight_increment
    Wn = (np.sum(np.exp(log_weights)))
    for i in range(N_p):
        position_estimates[time_index] += np.exp(log_weights[i]) * a_updates[i][0, 0]  # Estimated position mean
        velocity_estimates[time_index] += np.exp(log_weights[i]) * a_updates[i][1, 0]  # Estimated position mean
        mean_estimates[time_index] += np.exp(log_weights[i]) * a_updates[i][2, 0]  # Estimated position mean
    position_estimates[time_index] = position_estimates[time_index] / Wn
    velocity_estimates[time_index] = velocity_estimates[time_index] / Wn
    mean_estimates[time_index] = mean_estimates[time_index] / Wn
    return a_updates, C_updates, Mcs, Scs, N_p, E_Ns, log_weights, position_estimates, velocity_estimates, mean_estimates


def initialiseRBPFilter(a_update, C_update, mu_mu_W, N_p, L):
    log_weights = np.array([-np.log(N_p)] * N_p)  # Store the log weights
    E_Ns = np.zeros(N_p)
    a_updates = [a_update] * N_p
    C_updates = [C_update] * N_p
    Mcs = [0] * N_p
    Scs = [0] * N_p
    position_estimates = [0] * L
    velocity_estimates = [0] * L
    mean_estimates = [mu_mu_W] * L
    assert (a_updates.shape == (N_p, 3))  # P+1 = 3
    assert (C_updates.shape == (N_p, 3, 3))
    return log_weights, E_Ns, a_updates, C_updates, Mcs, Scs, position_estimates, velocity_estimates, mean_estimates


def langevinParameters():
    Theta = -0.1  # Friction constant / speed reversion term
    var_W = 4 ** 2  # Variance of U_is (or W_is) in generalised shot noise method.
    k_V = 1  # Variance scaling for observation noise (if matrix, need to change observation parameters)
    k_W = 0.01  # Variance scaling for mean (when included as latent param)
    alpha_W, beta_W = 1, 1  # Variance prior parameters
    Cv = k_V
    mu_mu_W = 0  # Mean prior mean
    return Theta, var_W, k_V, k_W, alpha_W, beta_W, Cv, mu_mu_W


def gammaParameters():
    nu = 2
    gamma = 2
    subordinator_truncation = 0.05  # Truncation on Gamma Jumps
    truncation = nu * scipy.special.exp1(gamma ** 2 / 2 * subordinator_truncation)  # Epoch size cutoff
    return nu, gamma, subordinator_truncation, truncation


def plotFilterResults(time_ax, position_states, position_estimates, velocity_states, velocity_estimates, mean_states,
                      mean_estimates, particles, log_weights):
    fig, ax = plt.subplots()
    ax.plot(time_ax, position_states, linestyle="dashed", color='black', label="Real State")
    ax.plot(time_ax, position_estimates, linestyle="dotted", color='orange', label="Estimated State")
    ax.set_title("Rao-Blackwellised Particle Filtering")
    ax.set_xlabel("Time, $t$")
    ax.set_ylabel("Position, $x(t)$")
    plt.legend()
    fig, ax = plt.subplots()
    ax.plot(time_ax, velocity_states, linestyle="dashed", color='black', label="Real State")
    ax.plot(time_ax, velocity_estimates, linestyle="dotted", color='orange', label="Estimated State")
    ax.set_title("Rao-Blackwellised Particle Filtering")
    ax.set_xlabel("Time, $t$")
    ax.set_ylabel("Velocity, $\dot{x}(t)$")
    plt.legend()
    fig, ax = plt.subplots()
    ax.plot(time_ax, mean_states, linestyle="dashed", color='black', label="Real State")
    ax.plot(time_ax, mean_estimates, linestyle="dotted", color='orange', label="Estimated State")
    ax.set_title("Rao-Blackwellised Particle Filtering")
    ax.set_xlabel("Time, $t$")
    ax.set_ylabel("Process Mean, $\mu_{W}$")
    plt.legend()
    """ Plot particle-weight distribution """
    fig, ax = plt.subplots()
    ax.scatter(particles, np.exp(log_weights), s=8, color='black', label="Real State")
    ax.set_title("Rao-Blackwellised Particle Filtering")
    ax.set_xlabel("Kalman Mean")
    ax.set_ylabel("Particle Weight")
    plt.legend()
    plt.show()


def Langevin_2D():
    """ Case 3 (partial Gaussian approximation)"""
    """ Latent Space Model:  dx(t) = Ax(t)dt + HdW(t) """
    """ State is [x(t), xdot(t)], observations are always 1D """
    P = 2
    Theta, var_W, k_V, k_W, alpha_W, beta_W, Cv, mu_mu_W = langevinParameters()  # P-dim Langevin model
    nu, gamma, subordinator_truncation, truncation = gammaParameters()  # Subordinate process parameters
    """ Langevin Model matrices """
    sde_A = np.array([[0, 1], [0, Theta]]).reshape((P, P))  # Langevin model for p=2
    h = np.array([0, 1]).reshape((P, 1))
    """ State space model matrices """
    H = np.array(np.append(1, np.zeros(shape=(1, P)))).reshape((1, P + 1))  # Size (1, P+1)
    B = np.vstack([np.identity(P), np.zeros(shape=(1, P))])  # Only present in cases 1 and 3
    """ Initialise filter """
    a_update = np.vstack([np.zeros(shape=(P, 1)), mu_mu_W]).reshape((P + 1, 1))  # a_0_0 (is shape of a_update P+1)
    C_update = np.vstack([np.hstack([np.zeros(shape=(P, P)), np.zeros(shape=(P, 1))]),
                          np.hstack([np.zeros(shape=(1, P)),
                                     np.array(var_W * k_W).reshape(1, 1)])])  # C_0_0 (does it need the var_W YES)?
    C_update += + 1e-20 * np.eye(C_update.shape[0])  # Make positive definite
    T_horizon = 1000
    N_p = 1000  # Number of particles initialised by ourselves AND constant if resampling
    time_ax = np.linspace(0, T_horizon, 1000)  # Generate time-axis during which we are tracking
    States = [a_update + np.matmul(np.linalg.cholesky(C_update), np.random.normal(size=P + 1)).reshape((P + 1, 1))]
    Observations = [0]
    for i in tqdm(range(1, len(time_ax))):
        t1 = time_ax[i - 1]
        t2 = time_ax[i]
        delta_t = t2 - t1
        mc, Sc = normalGammaLatentSampling(t1, t2, nu, gamma, truncation, P, h, sde_A, Theta)
        ss_A = getNormalGammaStateMatrix(Theta, nu, gamma, mc, P, subordinator_truncation, delta_t)
        Ce = getNormalGammaCovarianceMatrix(Theta, t1, t2, nu, gamma, subordinator_truncation, Sc)
        state = ss_A @ States[i - 1] + B @ np.matmul(np.linalg.cholesky(var_W * Ce), np.random.normal(size=P)).reshape(
            (P, 1))
        obs = H @ state + np.random.normal(loc=0, scale=var_W * Cv)
        States.append(state)
        Observations.append(obs[0, 0])

    position_states = []
    velocity_states = []
    mean_states = []
    for s in States:
        position_states.append(s[0, 0])
        velocity_states.append(s[1, 0])
        mean_states.append(s[2, 0])

    """ Initialise RBP Filter """
    log_weights, E_Ns, a_updates, C_updates, Mcs, Scs, position_estimates, velocity_estimates, mean_estimates \
        = initialiseRBPFilter(a_update, C_update, mu_mu_W, N_p, L=len(time_ax))
    """ Run RBP Filter """
    for i in tqdm(range(1, len(time_ax))):
        if CheckResampling():
            a_updates, C_updates, log_weights = Resample(a_updates, C_updates, log_weights)
        a_updates, C_updates, Mcs, Scs, N_p, E_Ns, log_weights, position_estimates, velocity_estimates, \
        mean_estimates = RB_Filter(
            i, a_updates, C_updates,
            Mcs, Scs, log_weights,
            N_p, nu, gamma, truncation, subordinator_truncation,
            time_ax[i - 1],
            time_ax[i], h,
            Theta, alpha_W, beta_W,
            sde_A, E_Ns, H, Cv, B,
            Observations[i],
            position_estimates, velocity_estimates, mean_estimates)

    """ Obtain state particles"""
    particles = []
    for a in a_updates:
        particles.append(a[0, 0])
    Wn = np.sum(np.exp(log_weights))
    log_weights -= np.log(Wn)
    """ Plot results """
    plotFilterResults(time_ax, position_states, position_estimates, velocity_states, velocity_estimates, mean_states,
                      mean_estimates, particles, log_weights)


Langevin_2D()
