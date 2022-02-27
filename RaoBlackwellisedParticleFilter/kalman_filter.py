import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def KalmanFilter(A, U, B, Ce, Cv, H, prev_mean, prev_cov, observation, prev_control=np.zeros(shape=(1, 1)), E_N1=0):
    """SS model:
    x(k) = Ax(k-1) + Uu(k-1)+ Bw(k), w(k) = N(0, Ce)
    y(k) = Hx(k) + v(k), v(k) = N(0, Cv)
    """
    a_predict = A @ prev_mean + U @ prev_control  # a_t_t-1
    C_predict = A @ prev_cov @ A.T + B @ Ce @ B.T  # C_t_t-1
    F_N = np.array(H @ C_predict @ H.T)[0, 0] + Cv  # Needs to be changed if M != 1
    K = (C_predict @ H.T) / F_N
    obs_hat = np.array(H @ a_predict)[0, 0]  # Needs to be edited if M != 1
    a_update = a_predict + K * (observation - obs_hat)
    I = np.eye(H.shape[1])
    C_update = (I - (K * H)) @ C_predict  # Change syntax if M != 1
    E_N = E_N1 + (observation - obs_hat) ** 2 / F_N  # Change syntax if M != 1
    return a_predict, C_predict, a_update, C_update, F_N, E_N


# https://machinelearningspace.com/object-tracking-python/
def test_KF():
    """ Tracking:
     x(k) = Ax(k-1) + Bu(k-1) + w(k), w(k) = N(0, Q)
     y(k) = Hx(k) + v(k), v(k) = N(0, R)"""
    dt = 0.1
    time_ax = np.arange(0, 100, dt)
    L = len(time_ax)
    # Define a model track
    A = np.array([[1, dt], [0, 1]]).reshape((2, 2))
    B = np.array([0.5 * dt ** 2, dt]).reshape((2, 1))
    H = np.array([1., 0.]).reshape((1, 2))
    std_acc = 0.25  # we assume that the standard deviation of the acceleration is 0.25 (m/s^2)
    Q = np.array([[0.25 * dt ** 4, 0.5 * dt ** 3], [0.5 * dt ** 3, dt ** 2]]).reshape((2, 2)) * std_acc ** 2
    R = 5.0 ** 2
    real_state = 0.1 * ((time_ax ** 2) - time_ax)
    observations = real_state + np.random.normal(loc=0, scale=50, size=L)
    mean_update = np.array([0, 0]).reshape((2, 1))
    cov_update = np.eye(Q.shape[0])
    u = np.array([2.0]).reshape((1, 1))
    assert (A[0, 0] == 1)
    assert (A[0, 1] == dt)
    assert (A[1, 0] == 0)
    assert (A[1, 1] == 1)
    assert (B[0, 0] == 0.5 * dt ** 2)
    assert (B[1, 0] == dt)
    assert (H[0, 0] == 1)
    assert (H[0, 1] == 0.)
    assert (Q[0, 0] == 0.25 * dt ** 4 * std_acc ** 2)
    assert (Q[0, 1] == 0.5 * dt ** 3 * std_acc ** 2)
    assert (Q[1, 0] == 0.5 * dt ** 3 * std_acc ** 2)
    assert (Q[1, 1] == dt ** 2 * std_acc ** 2)
    assert (np.array(np.eye(Q.shape[0]).T @ Q @ np.eye(Q.shape[0])).all() == Q.all())
    predictions_marcos = []
    err1 = 0
    err2 = 0

    for i in tqdm(range(L)):
        mean_pred, cov_pred, mean_update, cov_update, _, _ = KalmanFilter(A, B, np.eye(Q.shape[0]), Q, R, H,
                                                                          mean_update, cov_update, observations[i], u)

        predictions_marcos.append(mean_pred[0][0])
        err1 += np.abs(real_state[i] - mean_pred[0][0]) ** 2
        err2 += np.abs(real_state[i] - mean_update[0][0]) ** 2

    fig, ax = plt.subplots()
    ax.plot(time_ax, predictions_marcos, color="blue", label="Predicted State")
    # plt.plot(time_ax, observations, color="black", label="Noisy Observations")
    ax.plot(time_ax, real_state, color="red", label="True State")
    plt.legend()
    plt.title("Linear Gaussian SS Model Kalman Filter Marcos")
    plt.xlabel("Time /s")
    plt.ylabel("Position /m")

    plt.legend()
    plt.title("Linear Gaussian SS Model Kalman Filter Github")
    plt.xlabel("Time /s")
    plt.ylabel("Position /m")
    plt.show()


def test_LGSSM_KF():
    """
    x(k) = Ax(k - 1) + Bu(k - 1) + w(k), w(k) = N(0, Q)
    y(k) = Hx(k) + v(k), v(k) = N(0, R)
    """
    A = np.array([[1, 0], [0, 1]]).reshape((2, 2))
    B = np.zeros(shape=(2, 2))
    u = np.zeros(shape=(2, 1))
    Q = 2 ** 2 * np.eye(2)
    H = np.array([1, 0]).reshape((1, 2))
    R = 2**2
    time_ax = np.arange(0, 10, 0.1)
    """ Initialise filter """
    x = np.random.multivariate_normal(mean=[0,0], cov=Q)
    mean_update = np.zeros(shape=(2,1))
    cov_update = np.eye(2)
    preds = []
    observations = []
    """ Run Kalman Filter"""
    for i in range(len(time_ax)):
        y = H @ x + np.random.normal(loc=0, scale=R)
        observations.append(y)
        mean_pred, cov_pred, mean_update, cov_update, _, _ = KalmanFilter(A, B, np.eye(Q.shape[0]), Q, R, H,
                                                                          mean_update, cov_update, y, u)

        preds.append(mean_pred[0])
    plt.plot(time_ax, preds, label="Kalman Estimate", linestyle='dashed', color='r')
    plt.plot(time_ax, observations, label="Observations", color='b')
    plt.title("Kalman Filter for LGSSM")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.show()