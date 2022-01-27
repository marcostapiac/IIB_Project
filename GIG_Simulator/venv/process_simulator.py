from simulate_process_functions import *
from rvs_generator_functions import *

lambd = -0.1
gamma_param = 2
delta = 2
T_horizon = 1.0
N_Processes= 10
N_epochs = 1000

processes = GIG_process(delta=delta, gamma_param=gamma_param, lambd=lambd, N_Processes=N_Processes, T_horizon=T_horizon, N_epochs=N_epochs)
fig, ax = plt.subplots()
time_ax = np.linspace(0., T_horizon, num=N_epochs)
for process in processes:
    ax.plot(time_ax, process)
ax.grid(True)
ax.set_title("Sample Paths from GIG Process")
ax.set_xlabel("Time")
ax.set_ylabel("Position")
plt.show()
