import numpy as np
from scipy.stats import levy_stable
from scipy.special import gamma as gammaf
from scipy.special import gammaincc as uppergammaf
from scipy.stats import gamma as gammaDist
from scipy.special import hankel1, hankel2, gammainc, gammaincc, gammaincinv
from tqdm import tqdm

class LevyJumpProcess:
    """ Class to simulate Lévy processes based on rejection sampling """

    def __init__(self, minT, maxT, num_samples, num_epochs, subordinator_truncation):
        """ Each process is defined by its jumps, jump times, time scale, and epochs """
        self.__epoch_number = num_epochs
        self.__tMin = minT
        self.__tMax = maxT
        self.__num_obs = num_samples
        self.__jump_times = None
        self.__jumps = None
        self.__truncation = subordinator_truncation
        self.__covariance_constant = 0.0
        try:
            assert (maxT - minT > 0.0)
        except AssertionError:
            print("Ensure process time scale is strictly positive")

    def get_minT(self):
        return self.__tMin

    def get_maxT(self):
        return self.__tMax

    def get_num_obs(self):
        return self.__num_obs

    def get_jump_times(self):
        return self.__jump_times

    def get_jump_sizes(self):
        return self.__jumps

    def get_truncation(self):
        return self.__truncation

    def get_epoch_number(self):
        return self.__epoch_number

    def get_covariance_constant(self):
        return self.__covariance_constant

    def set_jump_sizes(self, jumps):
        self.__jumps = jumps

    def generate_epochs(self):
        delta_t = self.__tMax - self.__tMin
        return np.cumsum(np.random.exponential(scale=1.0 / delta_t, size=self.__epoch_number))

    def generate_jump_times(self, num_times):
        self.__jump_times = np.random.uniform(self.__tMin, self.__tMax, size=num_times)
        return self.__jump_times

    def generate_path(self, truncation=0.0):
        time_ax = np.linspace(self.__tMin, self.__tMax, self.__num_obs)
        epochs = self.generate_epochs()
        epochs = epochs * (epochs >= truncation)
        self.generate_jumps(epochs)
        jumps = self.__jumps
        self.generate_jump_times(jumps.shape[0])
        jump_times = self.__jump_times
        return np.array([np.sum(jumps * (jump_times <= t)) for t in time_ax])

    def generate_jumps(self, epochs):
        return None

    def marginal_samples(self, numSamples, tHorizon):
        return None

    @staticmethod
    def rejection_sampling(probs, jumps):
        us = np.random.uniform(0., 1., size=probs.size)
        xs = np.where(probs > us, jumps, 0.)
        return xs[xs > 0.]


class AlphaStableSubordinator(LevyJumpProcess):

    def __init__(self, minT, maxT, num_samples, num_epochs, subordinator_truncation, alpha, gamma):
        """ delta parameter in Barndorff-Nielsen for positive stable is gamma parameter here"""
        super().__init__(minT, maxT, num_samples, num_epochs, subordinator_truncation)
        self.__alpha = alpha
        self.__beta = 1.0
        self.__gamma = gamma
        self.__delta = 0.0
        self.__C = (2 / np.pi) * gammaf(alpha) * np.sin(np.pi * alpha / 2) * gamma * alpha
        self.__covariance_constant = (alpha / (2. - alpha)) * ((self.__C / alpha) ** (1. - (2. / alpha))) * (
                subordinator_truncation ** (2. - alpha))

    def get_alpha(self):
        return self.__alpha

    def get_C(self):
        return self.__C

    def generate_jumps(self, epochs):
        super().set_jump_sizes(((self.__alpha * epochs) / self.__C) ** (-1 / self.__alpha))

    def marginal_samples(self, numSamples, tHorizon):
        """ Scale parameter is related to gamma via gamma = scale**alpha (Levy Random Fields, Wolpert) """
        return levy_stable.rvs(self.__alpha, beta=self.__beta, loc=self.__delta,
                               scale=(tHorizon * self.__gamma) ** (1 / self.__alpha),
                               size=numSamples)  # CMS_Method(alpha, 1, gamma, L=N_Processes, location=delta)


class GammaSubordinator(LevyJumpProcess):
    def __init__(self, minT, maxT, num_samples, num_epochs, subordinator_truncation, nu=1.0, gamma=np.sqrt(5)):
        super().__init__(minT, maxT, num_samples, num_epochs, subordinator_truncation)
        self.__nu = nu
        self.__gamma = gamma

    def get_nu(self):
        return self.__nu

    def get_gamma(self):
        return self.__gamma

    def generate_jumps(self, epochs):
        beta = self.__gamma ** 2 / 2
        x = 1. / (beta * (np.exp(epochs / self.__nu) - 1))
        prob_acc = (1 + beta * x) * np.exp(-beta * x)
        # Rejection sampling
        super().set_jump_sizes(self.rejection_sampling(prob_acc, x))

    def marginal_samples(self, numSamples, tHorizon):
        return gammaDist.rvs(a=tHorizon * self.__nu, loc=0, scale=2.0 / (self.__gamma ** 2), size=numSamples)


class TemperedStableSubordinator(LevyJumpProcess):
    def __init__(self, minT, maxT, num_samples, num_epochs, subordinator_truncation, kappa=0.5, delta=1, gamma=1.26):
        super().__init__(minT, maxT, num_samples, num_epochs, subordinator_truncation)
        self.__kappa = kappa
        self.__delta = delta
        self.__gamma = gamma
        self.__covariance_constant = 4 * delta * kappa * (gamma ** ((2 - kappa) / kappa)) * (
                gammaf(2 - kappa) / gammaf(1 - kappa)) * uppergammaf(
            2 - kappa,
            0.5 * subordinator_truncation * gamma ** (
                    1 / kappa))

    def get_kappa(self):
        return self.__kappa

    def get_delta(self):
        return self.__delta

    def get_gamma(self):
        return self.__gamma

    def get_covariance_constant(self):
        return self.__covariance_constant

    def generate_jumps(self, epochs):
        beta = self.__gamma ** (1 / self.__kappa) / 2.0
        C = self.__delta * (2 ** self.__kappa) * self.__kappa * (1 / gammaf(1 - self.__kappa))
        x = ((self.__kappa * epochs) / C) ** (-1 / self.__kappa)
        prob_acc = np.exp(-beta * x)
        super().set_jump_sizes(self.rejection_sampling(prob_acc, x))

    def marginal_samples(self, numSamples, tHorizon):
        x = levy_stable.rvs(self.__kappa, beta=1.0, loc=0.0, scale=(tHorizon * self.__delta) ** (1 / self.__kappa),
                            size=numSamples)
        beta = 0.5 * self.__gamma ** (1 / self.__kappa)
        prob_acc = np.exp(-beta * x)
        return self.rejection_sampling(prob_acc, x)


class GIGSubordinator(LevyJumpProcess):
    def __init__(self, minT, maxT, num_samples, num_epochs, subordinator_truncation, delta=2.0, gamma=0.2, lambd=-0.1):
        super().__init__(minT, maxT, num_samples, num_epochs, subordinator_truncation)
        self.__delta = delta
        self.__lambd = lambd
        self.__gamma = gamma
        self.__covariance_constant = 0.0  # TODO

    def get_lambd(self):
        return self.__lambd

    def get_delta(self):
        return self.__delta

    def get_gamma(self):
        return self.__gamma

    def get_covariance_constant(self):
        return self.__covariance_constant
    @staticmethod
    def __psi(x, alpha, lambd):
        return -alpha * (np.cosh(x) - 1) - lambd * (np.exp(x) - x - 1)
    @staticmethod
    def __dpsi(x, alpha, lambd):
        return -alpha * np.sinh(x) - lambd * (np.exp(x) - 1)
    @staticmethod
    def __g(x, sd, td, f1, f2):
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
    @staticmethod
    def __unnorm_gammaincc(lam, z):
        return gammaf(lam) * gammaincc(lam, z)

    @staticmethod
    def __unnorm_gammainc(lam, z):
        return gammaf(lam) * gammainc(lam, z)

    @staticmethod
    def __hankel_squared(lam, z):
        return np.real(hankel1(lam, z) * hankel2(lam, z))

    def __generate_gamma_jumps(self, C, beta):
        epochs = self.generate_epochs()
        x = 1 / (beta * (np.exp(epochs / C) - 1))
        prob_acc = (1 + beta * x) * np.exp(-beta * x)
        return self.rejection_sampling(prob_acc, x)

    def __generate_tempered_stable_jumps(self, alpha, beta, delta):
        epochs = self.generate_epochs()
        C = (self.get_maxT() - self.get_minT()) * delta * gammaf(0.5) / (np.sqrt(2) * np.pi)
        x = ((alpha * epochs) / C) ** (-1 / alpha)
        prob_acc = np.exp(-beta * x)
        return self.rejection_sampling(prob_acc, x)

    def __GIG_gamma_component(self):
        return self.__generate_gamma_jumps(max(0.0, self.__lambd), self.__gamma ** 2 / 2)

    def __generate_N1(self, z1, H0, lambd, delta, gamma_param):
        # Generate gamma process
        beta = 0.5 * gamma_param ** 2
        C = z1 / (np.pi * np.pi * np.absolute(lambd) * H0)  # Shape parameter of process at t = 1
        jump_sizes = self.__generate_gamma_jumps(C, beta,)

        """ Rejection sampling from Algorithm 6 """
        const1 = (z1 ** 2) * jump_sizes / (2 * delta ** 2)
        GIG_prob_acc = np.absolute(lambd) * gammaf(np.abs(lambd)) * gammainc(np.abs(lambd), const1) / (
                ((z1 ** 2) * jump_sizes / (2 * delta ** 2)) ** np.abs(lambd))
        u = np.random.uniform(0., 1., size=jump_sizes.size)
        jump_sizes = jump_sizes[(u < GIG_prob_acc)]

        """ Sample from truncated Nakagami """
        C1 = np.random.uniform(0., 1., size=jump_sizes.size)
        l = C1 * gammainc(np.absolute(lambd), (z1 ** 2 * jump_sizes) / (2 * delta ** 2))
        zs = np.sqrt(((2 * delta ** 2) / jump_sizes) * gammaincinv(np.absolute(lambd), l))

        """ Thinning for process N1 """
        u = np.random.uniform(0., 1., size=jump_sizes.size)
        N1_prob_acc = H0 / (self.__hankel_squared(np.abs(lambd), zs) *
                            (zs ** (2 * np.abs(lambd))) / (z1 ** (2 * np.abs(lambd) - 1)))
        jump_sizes = jump_sizes[(u < N1_prob_acc)]
        return jump_sizes

    def __generate_N2(self, z1, H0, lambd, delta, gamma_param, N_epochs, T_horizon):
        """Generate point process N2 """

        """ Generate Tempered Stable Jump Size samples """
        alpha = 0.5
        beta = (gamma_param ** 2) / 2
        C = np.sqrt(2 * delta ** 2) * gammaf(0.5) / ((np.pi ** 2) * H0)
        epochs = np.cumsum(
            np.random.exponential(1, N_epochs)) / T_horizon
        x = ((alpha * epochs) / C) ** (-1 / alpha)
        prob_acc = np.exp(-beta * x)
        u = np.random.uniform(0.0, 1.0, size=prob_acc.size)
        jump_sizes = x[(u < prob_acc)]

        """ Rejection sampling based on Algorithm 7 """
        GIG_prob_acc = gammaincc(0.5, (z1 ** 2) * jump_sizes / (2 * delta ** 2))
        u = np.random.uniform(0., 1., size=jump_sizes.size)
        jump_sizes = jump_sizes[(u < GIG_prob_acc)]

        # Simulate Truncated Square-Root Gamma Process:
        C2 = np.random.uniform(low=0.0, high=1.0, size=jump_sizes.size)
        zs = np.sqrt(
            ((2 * delta ** 2) / jump_sizes) * gammaincinv(0.5,
                                                          C2 * (
                                                              gammaincc(0.5, (z1 ** 2) * jump_sizes / (2 * delta ** 2)))
                                                          + gammainc(0.5, (z1 ** 2) * jump_sizes / (2 * delta ** 2))))

        """Thinning for process N2"""
        u = np.random.uniform(0., 1., size=jump_sizes.size)
        N2_prob_acc = H0 / (zs * self.__hankel_squared(np.abs(lambd), zs))
        jump_sizes = jump_sizes[(u < N2_prob_acc)]
        return jump_sizes

    def __GIG_simple_jumps(self):
        delta = self.__delta
        x = self.__generate_tempered_stable_jumps(0.5, (self.__gamma ** 2) / 2, delta)
        zs = np.sqrt(gammaDist.rvs(0.5, loc=0, scale=1 / (x / (2 * delta ** 2)), size=len(x)))
        u = np.random.uniform(0., 1., len(x))
        prob_acc = 2 / (np.pi * zs * self.__hankel_squared(np.abs(self.__lambd), zs))
        jump_sizes = x[(u < prob_acc)]
        return jump_sizes

    def __GIG_harder_jumps(self):
        delta = self.__delta
        gamma_param = self.__gamma
        lambd = self.__lambd
        N_epochs = self.get_epoch_number()
        T_horizon = self.get_maxT() - self.get_minT()
        a = np.pi * np.power(2.0, (1.0 - 2.0 * np.abs(lambd)))
        b = gammaf(np.abs(lambd)) ** 2
        c = 1 / (1 - 2 * np.abs(lambd))
        z1 = (a / b) ** c
        H0 = z1 * self.__hankel_squared(np.abs(self.__lambd), z1)
        N1 = self.__generate_N1(z1, H0, lambd, delta, gamma_param)
        N2 = self.__generate_N2(z1, H0, lambd, delta, gamma_param, N_epochs, T_horizon)
        jump_sizes = np.append(N1, N2)
        return jump_sizes

    def generate_jumps(self, epochs=None):
        """ Function does NOT use epochs """
        if np.abs(self.__lambd) >= 0.5:
            jumps = self.__GIG_simple_jumps()
        else:
            jumps = self.__GIG_harder_jumps()
        if self.__lambd > 0:
            p2 = self.__GIG_gamma_component()
            jumps = np.append(jumps, p2)
        super().set_jump_sizes(jumps)

    def marginal_samples(self, numSamples, tHorizon):
        """ Code is translated from MATLAB Code from:
            Jan Patrick Hartkopf (2022).
            gigrnd (https://www.mathworks.com/matlabcentral/fileexchange/78805-gigrnd),
            MATLAB Central File Exchange.
            Setup - - we sample from the two parameter version of the GIG(alpha, omega) where:
            P, a, b = lambd, gamma_param ** 2, delta ** 2,
        """

        """ Which parameter is scaled by TIME ??? """

        a = self.__gamma**2
        b = self.__delta ** 2
        lambd = self.__lambd
        omega = np.sqrt(a * b)
        swap = False
        if lambd < 0:
            lambd = lambd * -1
            swap = True
        alpha = np.sqrt(omega ** 2 + lambd ** 2) - lambd
        x = -self.__psi(1, alpha, lambd)  # TODO CHECK
        if (x >= 0.5) and (x <= 2):
            t = 1
        elif x > 2:
            t = np.sqrt(2 / (alpha + lambd))
        elif x < 0.5:
            t = np.log(4 / (alpha + 2 * lambd))

        x = -self.__psi(-1, alpha, lambd)  # TODO CHECK
        if (x >= 0.5) and (x <= 2):
            s = 1
        elif x > 2:
            s = np.sqrt(4 / (alpha * np.cosh(1) + lambd))
        elif x < 0.5:
            s = min(1 / lambd, np.log(1 + 1 / alpha + np.sqrt(1 / alpha ** 2 + 2 / alpha)))

        eta = -self.__psi(t, alpha, lambd)
        zeta = -self.__dpsi(t, alpha, lambd)
        theta = -self.__psi(-s, alpha, lambd)
        xi = self.__dpsi(-s, alpha, lambd)
        p = 1 / xi
        r = 1 / zeta
        td = t - r * eta
        sd = s - p * theta
        q = td + sd

        X = [0 for _ in range(numSamples)]
        for i in range(numSamples):
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
                if (W * self.__g(X[i], sd, td, f1, f2)) <= np.exp(self.__psi(X[i], alpha, lambd)):
                    done = True
        X = np.exp(X) * (lambd / omega + np.sqrt(1 + (lambd / omega) ** 2))
        if swap:
            X = 1 / X
        X = X / np.sqrt(a / b)
        X = X.reshape((1, X.shape[0]))
        return X[0]
