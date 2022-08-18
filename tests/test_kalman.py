import numpy as np
from ClassKalmanFilter import KalmanFilter
import matplotlib.pyplot as plt

def KalmanFilterNonObject(A, B, Q, R, H, a_prev, C_prev, observation, E_N1):
    state_mean = A @ a_prev
    covariance_matrix = A @ C_prev @ A.T + B @ Q @ B.T
    F_N = H @ covariance_matrix @ H.T + R
    inv_F_N = np.linalg.inv(F_N)
    kalman_gain = np.array(covariance_matrix @ H.T) @ inv_F_N
    obs_mean = H @ state_mean
    a_update = state_mean + kalman_gain @ (observation - obs_mean)
    C_update = np.array(
        np.eye(H.shape[1]) - np.array(kalman_gain @ H)) @ covariance_matrix
    sum_E = E_N1 + ((observation - obs_mean) ** 2) @ inv_F_N
    return state_mean, covariance_matrix, a_update, C_update, F_N, sum_E


# https://machinelearningspace.com/object-tracking-python/
def test_KF():
    """ Tracking:
     x(k) = Ax(k-1) + Uu(k-1) + w(k), w(k) = N(0, Q)
     y(k) = Hx(k) + v(k), v(k) = N(0, R)"""
    dt = 0.1
    time_ax = np.arange(0, 100, dt)
    L = len(time_ax)
    real_state = 0.1 * ((time_ax ** 2) - time_ax)
    # Define a model
    A = np.array([[1, dt], [0, 1]]).reshape((2, 2))
    U = np.array([0.5 * dt ** 2, dt]).reshape((2, 1))
    H = np.matrix([[1, 0]])
    R = 1.2 ** 2
    std_acc = 0.25  # we assume that the standard deviation of the acceleration is 0.25 (m/s^2)
    Q = np.array([[0.25 * dt ** 4, 0.5 * dt ** 3], [0.5 * dt ** 3, dt ** 2]]).reshape((2, 2)) * (std_acc ** 2)
    mean_update = np.array([0, 0]).reshape((2, 1))
    cov_update = np.eye(Q.shape[0])
    u = 2.0  # Initial (and constant) control
    measurements = [0]
    predictions_marcos = [0]
    kf = KalmanFilter(mean_update, cov_update, u, A, U, np.eye(Q.shape[0]), Q, H, R)

    for i in tqdm(range(1, L)):
        kf.predict()
        predictions_marcos.append(kf.get_state_mean()[0][0])
        observation = H * real_state[i] + np.random.normal(loc=0., scale=50.)
        measurements.append(observation)
        kf.update(observation.item(0))

    fig, ax = plt.subplots()
    ax.plot(time_ax, predictions_marcos, color="blue", label="Predicted State")
    # plt.plot(time_ax, observations, color="black", label="Noisy Observations")
    ax.plot(time_ax, real_state, color="red", label="True State")
    plt.legend()
    plt.title("Linear Gaussian SS Model Kalman Filter Marcos")
    plt.xlabel("Time /s")
    plt.ylabel("Position /m")
    plt.show()


def test_LGSSM_KF():
    """
    x(k) = Ax(k - 1) + Uu(k - 1) + w(k), w(k) = N(0, Q)
    y(k) = Hx(k) + v(k), v(k) = N(0, R)
    """
    A = np.eye(2)
    U = np.zeros(shape=(2, 1))
    B = np.eye(2)
    u = 0
    Q = 1 ** 2 * np.eye(2)
    H = np.array([1, 0]).reshape((1, 2))
    R = 1 ** 2
    time_ax = np.arange(0, 100, 1)
    """ Initialise filter """
    x = np.random.multivariate_normal(mean=np.zeros(Q.shape[0]), cov=Q).reshape((Q.shape[0], 1))
    mean_update = np.zeros(shape=(2, 1))
    cov_update = np.eye(2)
    position_preds = [mean_update[0]]
    observations = [0]
    states = [x]
    true_position = [x[0]]
    kf = KalmanFilter(mean_update, cov_update, u, A, U, B, Q, H, R)
    """ Run Kalman Filter"""
    for i in range(1, len(time_ax)):
        x = A @ states[i - 1] + np.random.multivariate_normal(mean=np.zeros(Q.shape[0]), cov=Q).reshape((Q.shape[0], 1))
        y = H @ x + np.random.normal(loc=0, scale=R)
        states.append(x)
        observations.append(y)
        true_position.append(x[0])

        kf.predict()
        position_preds.append(kf.get_state_mean()[0][0])
        kf.update(y)

    plt.plot(time_ax, position_preds, label="Kalman Estimate", linestyle='dashed', color='r')
    plt.plot(time_ax, true_position, label="State", color='b')
    # plt.plot(time_ax, observations, label="Observations", color='black')
    plt.title("Kalman Filter for LGSSM")
    plt.legend()
    plt.xlabel("Time /s")
    plt.ylabel("Position /m")
    plt.show()
