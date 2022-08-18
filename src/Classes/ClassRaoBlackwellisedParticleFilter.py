import numpy as np
from scipy.special import gamma as gammaf
import copy
from ClassLangevinModel import LangevinModel
from ClassLevyJumpProcesses import AlphaSubordinator
from ClassPlotter import TimeSeriesPlotter
import matplotlib.pyplot as plt


class RaoBlackwellisedFilter:

    def __logsumexp(self):
        c = self.log_weights.max()
        logWn = c + np.log(np.sum(np.exp(self.log_weights - c)))
        self.log_weights -= logWn

    def __init__(self, number_of_particles, L, kf, alpha_W, beta_W):
        self.N_p = number_of_particles
        self.log_weights = np.array([0.0] * self.N_p)  # Store the log weights
        self.__logsumexp()  # normalise initialised weights
        self.position_estimates, self.velocity_estimates, self.skewness_estimates = np.zeros(L), np.zeros(L), np.zeros(
            L)
        self.position_stds, self.velocity_stds, self.skewness_stds = np.zeros(L), np.zeros(L), np.zeros(L)
        # One Kalman Filter object per particle (since each particle will have different latent variables)
        self.kfs = [copy.deepcopy(kf) for _ in range(self.N_p)]
        """ Conjugate prior values for variance (only present in case 3) """
        self.__alpha_W = alpha_W
        self.__beta_W = beta_W

    def get_alpha_W(self):
        return self.__alpha_W

    def get_beta_W(self):
        return self.__beta_W

    def get_position_estimates(self):
        return self.position_estimates

    def get_velocity_estimates(self):
        return self.velocity_estimates

    def get_skewness_estimates(self):
        return self.skewness_estimates

    def get_position_stds(self):
        return self.position_stds

    def get_velocity_stds(self):
        return self.velocity_stds

    def get_skewness_stds(self):
        return self.skewness_stds

    def CheckResampling(self):
        """ Simplest check is to always resample the particles """
        return True

    def resample(self, option):
        kfs = self.kfs
        log_weights = self.log_weights
        N_p = len(log_weights)
        if option == 0:
            # make N subdivisions, and chose a random position within each one
            cumulative_sum = np.cumsum(np.exp(log_weights))
            cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
            indexes = np.searchsorted(cumulative_sum, np.random.random(len(log_weights)))
            new_kfs = [copy.deepcopy(kfs[i]) for i in indexes]
            log_weights = np.array([-np.log(N_p) for _ in indexes])
            self.kfs = new_kfs
            self.log_weights = log_weights
        elif option == 1:
            """ From filterpy.monte_carlo.stratified_resample """
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
            new_kfs = [copy.deepcopy(kfs[i]) for i in indexes]
            log_weights = np.array([- np.log(N_p) for _ in indexes])
            self.kfs = new_kfs
            self.log_weights = log_weights
        elif option == 2:
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

                new_kfs[j] = copy.deepcopy(kfs[i])

            log_weights = np.array([-np.log(N_p)] * N_p)
            self.kfs = new_kfs
            self.log_weights = log_weights
        else:
            try:
                assert (0 <= option <= 2)
            except AssertionError:
                print(
                    "Please choose one of the following resampling algorithms: " "0: Multinomial Resampling " "1: Stratified Resampling " "2: Systematic Resampling")

    def update_particle_weight(self, particle_index, alpha_W, beta_W, obs_index, F_N, E_N1, E_N, M):
        """ Weight update specific to Levy State Space Model """
        weight_increment = -M / 2 * np.log(2 * np.pi) - 0.5 * np.log(np.abs(F_N)) - (alpha_W + obs_index / 2) * np.log(
            beta_W + E_N / 2)
        weight_increment += (alpha_W + (obs_index - 1) / 2) * np.log(beta_W + E_N1 / 2)
        if obs_index < 300:  # Heuristic to avoid overflow
            weight_increment += np.log(gammaf(obs_index / 2 + alpha_W)) - np.log(
                gammaf((obs_index - 1) / 2 + alpha_W))
        else:
            weight_increment += 0.5 * np.log((obs_index - 1) / 2 + alpha_W)
        self.log_weights[particle_index] += weight_increment[0, 0]

    def __get_state_estimates(self, obs_index, var_W, a_predicts, C_predicts):
        for i in range(self.N_p):
            self.position_estimates[obs_index] += np.exp(self.log_weights[i]) * a_predicts[i][0, 0]
            self.velocity_estimates[obs_index] += np.exp(self.log_weights[i]) * a_predicts[i][1, 0]
            self.skewness_estimates[obs_index] += np.exp(self.log_weights[i]) * a_predicts[i][2, 0]
            self.position_stds[obs_index] += np.exp(self.log_weights[i]) * (
                    var_W * C_predicts[i][0, 0] + (a_predicts[i][0, 0] ** 2))
            self.velocity_stds[obs_index] += np.exp(self.log_weights[i]) * (
                    var_W * C_predicts[i][1, 1] + (a_predicts[i][1, 0] ** 2))
            self.skewness_stds[obs_index] += np.exp(self.log_weights[i]) * (
                    var_W * C_predicts[i][2, 2] + (a_predicts[i][2, 0] ** 2))
        self.position_stds[obs_index] -= (self.position_estimates[obs_index] ** 2)
        self.velocity_stds[obs_index] -= (self.velocity_estimates[obs_index] ** 2)
        self.skewness_stds[obs_index] -= (self.skewness_estimates[obs_index] ** 2)
        self.position_stds[obs_index] = np.sqrt(self.position_stds[obs_index])
        self.velocity_stds[obs_index] = np.sqrt(self.velocity_stds[obs_index])
        self.skewness_stds[obs_index] = np.sqrt(self.skewness_stds[obs_index])

    def __kalman_step(self, i, ss_A, Ce, observation_n, a_predicts, C_predicts):
        self.kfs[i].set_state_predict_matrix(ss_A)
        self.kfs[i].set_covariance_predict_matrix(Ce)
        a_predicts[i] = self.kfs[i].get_state_mean()
        C_predicts[i] = self.kfs[i].get_covariance_matrix()
        self.kfs[i].predict()
        self.kfs[i].update(observation_n)
        return a_predicts, C_predicts

    def filter(self, time_index, delta_t, subordinator, observation_n, lModel: LangevinModel, a_predicts, C_predicts, Mcs=None, Scs=None):
        if self.CheckResampling():
            self.resample(2)
        """ Nonlinear Filter"""
        for i in range(self.N_p):
            if Mcs:
                mc, Sc = Mcs[time_index], Scs[time_index]
            else:
                mc, Sc = lModel.filter_sampling(subordinator)
            """ Given proposed mc, Sc, we can now define Kalman Filter matrices """
            ss_A = lModel.get_state_A(delta_t, mc) # Do we subtract Yc in latent variables???
            lModel.update_state_covariance_matrix(subordinator.get_covariance_constant(), delta_t, Sc)
            Ce = lModel.get_state_covariance_matrix()
            """ Kalman Filter"""
            E_N1 = self.kfs[i].get_sum_E()
            a_predicts, C_predicts = self.__kalman_step(i, ss_A, Ce, observation_n, a_predicts, C_predicts)
            """ Incremental Weight Update via Likelihood """
            self.update_particle_weight(i, self.__alpha_W, self.__beta_W, time_index, self.kfs[i].get_F_N(), E_N1,
                                        self.kfs[i].get_sum_E(), 1)
        self.__logsumexp()
        self.__get_state_estimates(time_index, lModel.get_observation_variance(), a_predicts, C_predicts)
        return a_predicts, C_predicts

    def plotFilterResults(self, time_ax, position_states, position_estimates, position_stds, velocity_states,
                          velocity_estimates,
                          velocity_stds, skewness_states,
                          skewness_estimates, skewness_stds, title, std_width=1.96):
        fig, ax = plt.subplots(3, sharex=True, sharey=False, figsize=(14, 9.5))
        plotter = TimeSeriesPlotter(time_ax, position_states, plotlabel="True Signal", ax=ax[0])
        plotter.plot()
        plotter = TimeSeriesPlotter(time_ax, position_estimates, plottitle=title, ylabel="Position, $x(t)$", plotlabel="Estimated Signal", ax=ax[0])
        plotter.plot()
        ax[0].fill_between(time_ax, np.array(position_estimates) - std_width * np.array(position_stds),
                           np.array(position_estimates) + std_width * np.array(position_stds),
                           label="$\pm 2$ standard deviations")

        plotter = TimeSeriesPlotter(time_ax, velocity_states, plotlabel="True Signal", ax=ax[1])
        plotter.plot()
        plotter = TimeSeriesPlotter(time_ax, velocity_estimates, ylabel="Velocity, $\dot{x}(t)$",
                                    plotlabel="Estimated Signal", ax=ax[1])
        plotter.plot()
        ax[1].fill_between(time_ax, np.array(velocity_estimates) - std_width * np.array(velocity_stds),
                           np.array(velocity_estimates) + std_width * np.array(velocity_stds))

        plotter = TimeSeriesPlotter(time_ax, skewness_states,
                                    plotlabel="True Signal", ax=ax[2])
        plotter.plot()
        plotter = TimeSeriesPlotter(time_ax, skewness_estimates, xlabel="Time, $t$",
                                    ylabel="Process Skewness, $\mu_{W}$", plotlabel="Estimated Signal", ax=ax[2])
        plotter.plot()
        ax[2].fill_between(time_ax, np.array(skewness_estimates) - std_width * np.array(skewness_stds),
                           np.array(skewness_estimates) + std_width * np.array(skewness_stds))
        plt.show()
