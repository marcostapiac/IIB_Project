import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def CheckResampling():
    """ Simplest check is to always resample the particles """
    return True


def log_normal_evaluation(y, x):
    # Y_k = exp(X_k/2)*V_k, V_k ~ N(0,1)
    return -0.5 * np.log(2 * np.pi) - 0.5 * x - ((y ** 2) / (2 * np.exp(x)))


def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


def resample(particles, log_weights):
    N_p = np.shape(log_weights)
    # make N subdivisions, and chose a random position within each one
    cumulative_sum = np.cumsum(np.exp(log_weights))
    cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
    indexes = np.searchsorted(cumulative_sum, np.random.random(np.shape(log_weights)))
    new_particles = [particles[i] for i in indexes]
    log_weights = np.array([-np.log(N_p) for _ in indexes])
    return new_particles, log_weights


def extend(particles):
    # X_k = 0.95X_k-1 + W_k, W_k ~ N(0,1)
    N = np.shape(particles)[0]
    return 0.95 * particles + np.random.randn(N, 1)


def update_weights(particles, observation, log_weights):
    N = np.shape(log_weights)[0]
    for i in range(len(log_weights)):
        log_weights[i] = - np.log(N) + log_normal_evaluation(observation, particles[i, -1])
    log_weights -= logsumexp(log_weights)
    return log_weights


def plot_true_hidden(time, x):
    plt.plot(time, x, label="True Hidden State", )
    plt.title("Particle Filter for Exponential Volatility Model")
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.legend()
    plt.show()


def simulate_data(T=1000):
    X = np.array([0])  # How else to initialise?
    Y = np.array([np.exp(X[0] / 2) * np.random.randn()])
    for t in range(1, T):
        # X_k = 0.95X_k-1 + W_k, W_k ~ N(0,1)
        # Y_k = exp(X_(k-1)/2)V_k, V_k ~ N(0,1)
        X = np.append(X, 0.95 * X[t - 1] + np.random.randn())
        Y = np.append(Y, np.exp(X[t] / 2) * np.random.randn())
    return X, Y


def particle_filter():
    """ Test for particle filter using the exponential volatility model """
    T = 1000
    X, Y = simulate_data(T)
    # Number of particles
    N = 10
    states = np.array([np.random.normal(0, np.sqrt(2), size=N)])
    states = states.reshape((N, 1))
    mean_states = np.zeros(T)
    log_weights = np.zeros(N)
    for n in range(N):
        log_weights[n] = log_normal_evaluation(Y[0], states[0])
    mean_states[0] = np.exp(log_weights).T @ states[:, -1]
    for t in tqdm(range(1, T)):
        if CheckResampling():
            resample(states, log_weights)
        prev_particles = states[:, -1].reshape((N, 1))
        states = np.concatenate((states, extend(prev_particles)), axis=1)
        log_weights = update_weights(states, Y[t], log_weights)
        mean_states[t] = np.exp(log_weights).T @ states[:, -1]

    time = np.linspace(0, 1000, num=T)
    plt.plot(time, mean_states, label="Estimated State")
    plot_true_hidden(time, X)


particle_filter()
