import numpy as np


class KalmanFilter:
    def __init__(self, initial_mean, initial_covariance, initial_control, state_predict_matrix, control_predict_matrix,
                 noise_predict_matrix, covariance_predict_matrix, state_update_matrix, covariance_update_matrix):
        """ x_k+1 = Ax_k + Uu_k + Be_k, e_k ~ N(0, Ce), Ce = Q;
            y_k+1 = Hx_k+1 + v_k, v_k ~ N(0, Cv), Cv = R """
        self.__state_mean = initial_mean
        self.__covariance_matrix = initial_covariance
        self.__control = initial_control
        self.__A = state_predict_matrix
        self.__U = control_predict_matrix
        self.__B = noise_predict_matrix
        self.__Q = covariance_predict_matrix
        self.__H = state_update_matrix
        self.__R = covariance_update_matrix
        self.__sum_E = ((self.__H @ self.__state_mean)[0, 0] ** 2) / (
                (self.__H @ self.__covariance_matrix @ self.__H.T)[
                    0, 0] + self.__R)  # Term used in Rao-Blackwellised filter
        self.__F_N = 0  # Term used in Rao-Blackwellised filter

    def set_state_predict_matrix(self, new_A):
        """ Required for use of single KF object during RBFilter"""
        self.__A = new_A

    def set_covariance_predict_matrix(self, new_Q):
        """ Required for use of single KF object during RBFilter"""
        self.__Q = new_Q

    def set_covariance_update_matrix(self, new_R):
        """ Required for use of single KF object during RBFilter"""
        self.__R = new_R

    def set_state_mean(self, mean):
        self.__state_mean = mean

    def set_sum_E(self, E):
        self.__sum_E = E

    def set_F_N(self, F):
        self.__F_N = F

    def set_covariance_matrix(self, cov):
        self.__covariance_matrix = cov

    def get_state_mean(self):
        return self.__state_mean

    def get_covariance_matrix(self):
        return self.__covariance_matrix

    def get_sum_E(self):
        return self.__sum_E

    def get_F_N(self):
        return self.__F_N

    def predict(self):
        """ We assume state_mean is multidimensional and control is 1-d; if not, change matrix syntax accordingly
         * if scalar; @ if matrix """
        self.__state_mean = self.__A @ self.__state_mean  # + self.__U * self.__control
        self.__covariance_matrix = self.__A @ self.__covariance_matrix @ self.__A.T + self.__B @ self.__Q @ self.__B.T

    def update(self, observation):
        """ We assume observation is 1-dimensional """
        self.__F_N = self.__H @ self.__covariance_matrix @ self.__H.T + self.__R
        inv_F_N = np.linalg.inv(self.__F_N)
        kalman_gain = np.array(self.__covariance_matrix @ self.__H.T) @ inv_F_N
        obs_mean = self.__H @ self.__state_mean
        self.__state_mean = self.__state_mean + kalman_gain @ (observation - obs_mean)
        self.__covariance_matrix = np.array(
            np.eye(self.__H.shape[1]) - np.array(kalman_gain @ self.__H)) @ self.__covariance_matrix
        self.__sum_E = self.__sum_E + ((observation - obs_mean) ** 2) @ inv_F_N
