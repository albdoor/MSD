import numpy as np
import matplotlib.pyplot as plt

def load_joint_data(npy_filename, n, time_int):
    # data = np.load(npy_filename, allow_pickle=True)
    # data = data[0] if isinstance(data[0], list) else data
    data = np.loadtxt(npy_filename, delimiter=',')
    data = data.T
    q = np.zeros((time_int, n))
    qd = np.zeros((time_int, n))
    qdd = np.zeros((time_int, n))

    for i in range(n):
        q[:, i] = data[i]

    for i in range(n, 2 * n):
        qd[:, i - n] = data[i]

    for i in range(2 * n, 3 * n):
        qdd[:, i - 2 * n] = data[i]

    return q, qd, qdd

n = 5

time = np.linspace(0, 10, 1000)


print(len(time)) # Why is it equal to 1000 and in other cycles it gives 1001 elements? Because they start from 0? Maybe

q_csv, qd_csv, qdd_csv = load_joint_data('./data/robot_dynamics_output.csv', n, len(time) + 1)
print("Shape of q:", np.shape(q_csv))
print("Shape of qd:", np.shape(qd_csv))
print("Shape of qdd:", np.shape(qdd_csv))
