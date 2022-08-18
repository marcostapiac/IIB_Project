import numpy as np


class LangevinModel:
    def __init__(self, state_dimension, observation_dimension, theta, var_W, k_V, k_W, mu_mu_W):
        self.__P = state_dimension
        self.__M = observation_dimension
        self.__theta = theta
        self.__observation_variance = var_W
        self.__obs_var_scaling = k_V
        self.__mean_var_scaling = k_W
        self.__mean_mean = mu_mu_W
        """ Matrices used in latent sampling - change depending on model used """
        self.__M1 = np.array([1 / theta, 1]).reshape((self.__P, 1))
        self.__M2 = np.array([-1 / theta, 0]).reshape((self.__P, 1))
        self.__C1 = np.array([[1 / (theta ** 2), 1 / theta], [1 / theta, 1]]).reshape((self.__P, self.__P))
        self.__C2 = np.array([[-2 / (theta ** 2), -1 / theta], [-1 / theta, 0]]).reshape((self.__P, self.__P))
        self.__C3 = np.array([[1 / (theta ** 2), 0], [0, 0]]).reshape((self.__P, self.__P))
        """ Matrices to define state matrix A """
        self.__state_M1 = np.array([[0., 1. / self.__theta], [0., 1.]])
        self.__state_M2 = np.array([[1., -1. / self.__theta], [0., 0.]])
        """ More matrices to define state space model """
        self.__B = np.vstack([np.identity(state_dimension), np.zeros(shape=(1, state_dimension))]).reshape(
            (state_dimension + 1, state_dimension))  # Only present in cases 1 and 3
        self.__H = np.array(np.append(1, np.zeros(shape=(1, state_dimension)))).reshape(
            (1, state_dimension + 1))  # Size (1, P+1)
        """ Covariance matrices for state space model """
        self.__Ce = var_W * np.eye(self.__P+1) #np.vstack(
            #[np.hstack([np.zeros(shape=(state_dimension, state_dimension)), np.zeros(shape=(state_dimension, 1))]),
            # np.hstack(
            #     [np.zeros(shape=(1, state_dimension)), np.array(1 * k_W).reshape(-1, 1)])])  # initialised state
        # noise matrix (do we include var_W???)
        self.__Cv = var_W * k_V  # observation noise matrix (do we include var_W here or just 1.0 ??????)

    def get_state_dimension(self):
        return self.__P

    def get_theta(self):
        return self.__theta

    def get_observation_variance(self):
        return self.__observation_variance

    def get_obs_var_scaling(self):
        return self.__obs_var_scaling

    def get_mean_var_scaling(self):
        return self.__mean_var_scaling

    def get_mean_mean(self):
        return self.__mean_mean

    def get_M1(self):
        return self.__M1

    def get_M2(self):
        return self.__M2

    def get_C1(self):
        return self.__C1

    def get_C2(self):
        return self.__C2

    def get_C3(self):
        return self.__C3

    def get_state_B(self):
        return self.__B

    def get_observation_H(self):
        return self.__H

    def get_state_A(self, delta_t, lvars):
        return self.__complete_A(delta_t, lvars)

    def __complete_A(self, delta_t, latent_variables):
        M1 = self.__state_M1
        M2 = self.__state_M2
        expA = np.exp(self.__theta * delta_t) * M1 + M2
        return np.vstack(
            [np.hstack([expA, latent_variables]),
             np.hstack([np.zeros(shape=(1, self.__P)), np.array(1).reshape(-1, 1)])])

    def get_observation_covariance_matrix(self):
        return self.__Cv

    def get_state_covariance_matrix(self):
        return self.__Ce

    def update_state_covariance_matrix(self, const, delta_t, Sc):
        # Truncation is on the subordinator jumps, not epochs -> need to use expression relating both
        if const != 0.0:
            Theta = self.__theta
            C1 = self.__C1 / (2 * Theta)
            C2 = self.__C2 / Theta
            C3 = self.__C3
            Sigma_c = (np.exp(2 * Theta * delta_t) - 1.0) * C1 + (np.exp(Theta * delta_t) - 1.0) * C2 + delta_t * C3
            Sigma_c = const * Sigma_c
            self.__Ce = np.array(Sc + Sigma_c)  # Partial Gaussian Approximation
        else:
            self.__Ce = np.array(Sc)

    def filter_sampling(self, Subordinator):
        subordinator_truncation = Subordinator.get_truncation()
        Theta = self.__theta
        time = Subordinator.get_maxT()
        # Generate jump times from Subordinator Process
        Subordinator.generate_jumps(Subordinator.generate_epochs())
        jumps = Subordinator.get_jump_sizes()
        jumps = jumps * (jumps >= subordinator_truncation)
        Vs = Subordinator.generate_jump_times(jumps.shape[0])
        Vs = Vs[jumps > 0]
        jumps = jumps[jumps > 0]
        M1 = self.__M1
        M2 = self.__M2
        mean_sum_1 = np.sum(jumps * (np.exp(Theta * (time - Vs))))
        mean_sum_2 = np.sum(jumps)
        mean = mean_sum_1 * M1 + mean_sum_2 * M2
        C1 = self.__C1
        C2 = self.__C2
        C3 = self.__C3
        if Subordinator.__class__.__name__ == "AlphaSubordinator":  # Only Normal STD-Mean Process
            cov_sum_1 = np.sum(jumps ** 2 * np.exp(2 * Theta * (time - Vs)))
            cov_sum_2 = np.sum(jumps ** 2 * np.exp(Theta * (time - Vs)))
            cov_sum_3 = np.sum(jumps ** 2)
        else:  # All other processes are Normal Variance-Mean
            cov_sum_1 = np.sum(jumps * np.exp(2 * Theta * (time - Vs)))
            cov_sum_2 = mean_sum_1
            cov_sum_3 = mean_sum_2
        cov = cov_sum_1 * C1 + cov_sum_2 * C2 + cov_sum_3 * C3
        return mean, cov

    def simulate_state(self, state, delta_t, lvars):
        P = self.__P
        try:
            L = np.linalg.cholesky(self.__Ce * self.__observation_variance)
            u = np.random.normal(0.0, 1.0, P).reshape((P, 1))
            normRV = L @ u
        except np.linalg.LinAlgError:
            normRV = np.zeros(P)
        return self.get_state_A(delta_t, lvars) @ state + self.__B @ (normRV.reshape((P, 1)))

    def simulate_observation(self, state):
        return self.__H @ state + np.sqrt(self.__Cv) * np.random.randn()
