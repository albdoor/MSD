import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
import os

threshold = 1e-13


def load_joint_data(npy_filename, n, time_int, filetype):
    if filetype == 'csv':
        data = np.loadtxt(npy_filename, delimiter=',', skiprows=1)
        data = data.T
        # print(np.shape(data))
        q = np.zeros((time_int, n))
        # print(np.shape(q))
        qd = np.zeros((time_int, n))
        qdd = np.zeros((time_int, n))
    elif filetype == 'npy':
        data = np.load(npy_filename, allow_pickle=True)
        data = data[0] if isinstance(data[0], list) else data
        q = np.vstack((data[0]))
        qd = np.vstack((data[1])) 
        qdd = np.vstack((data[2]))
        return q, qd, qdd

    for i in range(n):
        q[:, i] = data[i]

    for i in range(n, 2 * n):
        print(i)
        qd[:, i - n] = data[i]

    for i in range(2 * n, 3 * n):
        qdd[:, i - 2 * n] = data[i]

    return q, qd, qdd


def formulaText(q, qd, qdd, links, g):
    theta, alpha, r, m, I, j_type, b = links[0]
    tau2 = 1/3 * m * r * qdd[0] + 1/3 * m * r**2 * qdd[1] + 0.5 * r**2 * np.cos(q[1]) * qdd[0] + 0.5 * m * g * r * np.cos(q[0] + q[1]) + 0.5 * m * r**2 * np.sin(q[1]) * qd[0]**2
    tau1 = 1/3 * m * r**2 * qdd[0] + 4/3 * m * r**2 * qdd[0] + 1/3 * m * r**2 * qdd[1] + m * np.cos(q[1]) * r**2 * qdd[0] 
    tau1 += 0.5 * m * r**2 * np.cos(q[1]) * qdd[1] - m * np.sin(q[1]) * r * qd[0] * qd[1] - 0.5 * m * np.sin(q[1]) * r * qd[1]**2 
    tau1 +=  0.5 * m * g * r * np.cos(q[0]) + 0.5 * m * g * r * np.cos(q[0] + q[1]) + m * g * r * np.cos(q[0])
    return np.array([tau1, tau2]) # 2 * np.sin(0.5 * t), 1.5 * np.cos(1.5 * t)


def tau_input(t): # initial input used
    return np.array([0,0]) # 2 * np.sin(0.5 * t), 1.5 * np.cos(1.5 * t)



def plot_graphs(n, data1, data2):
    plt.figure(figsize=(10, 5))
    for i in range(n):
        plt.subplot(1, n, i+1)    
        plt.plot(time, data1[:, i], label=f'Torque {i+1} Input')
        plt.plot(time, data2[:, i], '--r', label=f'Torque {i+1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.title('Joint Torques Over Time')
    plt.show()    








time = np.linspace(0, 10, 1000)  # Time steps from 0 to 10 seconds
torques = []

torquesLE = []

n = 2

q_csv, qd_csv, qdd_csv = load_joint_data('./data/LEForw_2link.csv', n, len(time), 'csv') # './providedForward/rl_multilink_simulation.csv'

# './providedForwardMod/rl_multilink_simulation2.csv' , './data/planarDoublePend.csv'

print("Shape of q:", np.shape(q_csv))
print("Shape of qd:", np.shape(qd_csv))
print("Shape of qdd:", np.shape(qdd_csv))


print("Type of q:", type(q_csv))
print("Type of qd:", type(qd_csv))
print("Type of qdd:", type(qdd_csv))


g = 9.81
gravity = np.array([[0, -g, 0, 0]])



links = [
    (0, 0, 1.0, 1.0, np.diag([0.0, 1/12 * 1, 1/12 * 1]), 1, 0.),  # Link 1
    (0, 0, 1.0, 1.0, np.diag([0.0, 1/12 * 1, 1/12 * 1]), 1, 0.)   # Link 2
]



# links = [
#     (0, 0, 1.0, 1.0, np.diag([1, 1, 1]), 1, 0.),  # Link 1
#     (0, 0, 1.0, 1.0, np.diag([1, 1, 1]), 1, 0.)      # Link 2
# ]


#range(2): 

for t_idx in range(len(time)): #len(time)
    q = q_csv[t_idx]   # Joint positions from CSV
    qd = qd_csv[t_idx] # Joint velocities from CSV
    qdd = qdd_csv[t_idx] # Joint accelerations from CSV
    print('********************************************************* Start of the Program *********************************************************')
    torque = formulaText(q, qd, qdd, links, g)  # Compute torques using RNEA
    torques.append(torque)
    torque2 = tau_input(time[t_idx])
    torquesLE.append(torque2)

torques = np.array(torques)
torquesLE = np.array(torquesLE)

torques[np.abs(torques) < threshold] = 0.0
torquesLE[np.abs(torquesLE) < threshold] = 0.0



plot_graphs(n, torquesLE, torques)
