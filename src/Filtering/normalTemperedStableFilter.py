import sys
sys.path.insert(1, '/Users/marcos/PycharmProjects/IIB_Project_Code/LevyJumpProcessesClass')
import numpy as np
from tqdm import tqdm
from ClassKalmanFilter import KalmanFilter
from data_processing import obtainData
from ClassLevyJumpProcesses import TemperedStableSubordinator
from ClassLangevinModel import LangevinModel
from ClassRaoBlackwellisedParticleFilter import RaoBlackwellisedFilter


def run_filter(kappa, gamma, delta, simulate_data=True):
    P = 2
    M = 1
    Theta = -1.0
    var_W = np.sqrt(1 ** 2) ** 2
    k_V = 1e-5
    k_W = 1e-5
    mu_mu_W = 1.0
    T_horizon = 10
    nEpochs = 2000
    subordinator_truncation = 1e-3
    L = 100
    Np = 500
    lModel = LangevinModel(P, M, Theta, var_W, k_V, k_W, mu_mu_W)
    if simulate_data:
        time_ax = np.cumsum(np.random.exponential(scale=1.0 / T_horizon,
                                                  size=L + 1))  # np.linspace(0.0, T_horizon, L) # Generate time-axis during which we are tracking
        a_update = np.vstack([np.zeros(shape=(P, 1)), lModel.get_mean_mean()]).reshape(
            (P + 1, 1))
        L = np.linalg.cholesky(
            (lModel.get_observation_variance() * lModel.get_state_covariance_matrix() + 1e-100 * np.eye(P + 1)))
        u = np.random.randn(P + 1)
        normRV = a_update + L @ (u.reshape((P + 1, 1)))
        Observations = [0]
        state = normRV
        position_states = [normRV[0, 0]]
        velocity_states = [normRV[1, 0]]
        skewness_states = [normRV[2, 0]]
        for i in tqdm(range(1, len(time_ax))):
            t1, t2 = time_ax[i - 1], time_ax[i]
            tsSubordinator = TemperedStableSubordinator(t1, t2, nEpochs, nEpochs, subordinator_truncation, kappa, delta, gamma)
            mc, Sc = lModel.filter_sampling(tsSubordinator)
            lModel.update_state_covariance_matrix(tsSubordinator.get_covariance_constant(), t2 - t1, Sc)
            state = lModel.simulate_state(state, t2 - t1, mc)
            obs = lModel.simulate_observation(state)
            Observations.append(obs[0, 0])
            position_states.append(state[0, 0])
            velocity_states.append(state[1, 0])
            skewness_states.append(state[2, 0])
    else:
        Observations, time_ax = obtainData()

    """ Start filtering given the data """
    numObs = len(Observations)
    alpha_W = 1.0
    beta_W = 1.0
    lModel = LangevinModel(P, M, Theta, var_W, k_V, k_W, mu_mu_W)
    a_initial = np.vstack([np.zeros(shape=(P, 1)), lModel.get_mean_mean()]).reshape((P + 1, 1))
    kf = KalmanFilter(a_initial, lModel.get_state_covariance_matrix(), 0, np.eye(P + 1), np.zeros(shape=(1, 1)),
                      lModel.get_state_B(), np.eye(P),
                      lModel.get_observation_H(), lModel.get_observation_covariance_matrix())
    rbf = RaoBlackwellisedFilter(Np, numObs, kf, alpha_W, beta_W)
    a_predicts = [0] * Np
    C_predicts = [0] * Np
    for i in tqdm(range(1, numObs)):
        tsSubordinator = TemperedStableSubordinator(time_ax[i-1], time_ax[i], nEpochs, nEpochs, subordinator_truncation, kappa, delta, gamma)
        a_predicts, C_predicts = rbf.filter(i, time_ax[i]-time_ax[i-1], tsSubordinator, Observations[i], lModel,
                                            a_predicts, C_predicts)

    plot_title = "Normal Tempered Stable Particle Filtering, $\\theta, N_{particles}, k_{V}, \kappa, \delta, \gamma, " \
                 "\sigma_{W}^{2}, k_{W} = " + str(Theta) + ", " + str(Np) + ", " + str(k_V) + ", " + str(kappa) + ", " + str(
            delta) + ", " + str(gamma) + ", " + str(var_W) + ", " + str(k_W) + "$"
    rbf.plotFilterResults(time_ax, position_states, rbf.get_position_estimates(), rbf.get_position_stds(),
                          velocity_states,
                          rbf.get_velocity_estimates(), rbf.get_velocity_stds(), skewness_states,
                          rbf.get_skewness_estimates(),
                          rbf.get_skewness_stds(), plot_title, std_width=1.96)


run_filter(0.5, 1.0, 1.26)
