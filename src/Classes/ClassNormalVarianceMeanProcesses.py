import numpy as np
from ClassLevyJumpProcesses import LevyJumpProcess


class NormalVarianceMeanProcess:
    def __init__(self, mu, mu_W, var_W, subordinator: LevyJumpProcess):
        self.__drift = mu
        self.__mean = mu_W
        self.__var_W = var_W
        self.__subordinator = subordinator

    def get_drift(self):
        return self.__drift

    def get_mean(self):
        return self.__mean

    def get_var_W(self):
        return self.__var_W

    def generate_jumps(self, time):
        self.__subordinator.generate_jumps(self.__subordinator.generate_epochs())
        sub_jumps = self.__subordinator.get_jump_sizes()
        return np.random.normal(self.__drift + self.__mean * sub_jumps, np.sqrt(self.__var_W*sub_jumps), size=sub_jumps.shape[0])


    def marginal_samples(self, numSamples, tHorizon):
        subSamples = self.__subordinator.marginal_samples(numSamples, tHorizon)
        return self.__drift*tHorizon + self.__mean * subSamples + np.sqrt(
            self.__var_W * subSamples) * np.random.randn(numSamples)

    def generate_path(self):
        time_ax = np.linspace(self.__subordinator.get_minT(), self.__subordinator.get_maxT(),
                              self.__subordinator.get_num_obs())
        self.__subordinator.generate_jumps(self.__subordinator.generate_epochs())
        sub_jumps = self.__subordinator.get_jump_sizes()
        jump_times = self.__subordinator.generate_jump_times(sub_jumps.shape[0])
        jumps = self.__mean * sub_jumps + np.sqrt(
            self.__var_W * sub_jumps) * np.random.randn(sub_jumps.shape[0])
        return np.array([np.sum((self.__drift * t + jumps) * (jump_times <= t)) for t in time_ax])
