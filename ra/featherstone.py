import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
print(os.getcwd())  # Print the current working directory
print(os.listdir()) # List all files in the current directory
threshold = 1e-13


# =========================
# Spatial Algebra Utilities
# =========================

def tau_input(t):
    return np.array([0.5 * np.sin(3 * t), 0.88 * np.cos(2 * t), 0, 0, 0, 0, 0, 0, 0, 0, 0]) # , 0.075 * np.cos(2 * t) 0.88 * np.cos(2 * t), 0.88 * np.cos(2 * t)
# 0, 0 * np.sin(1.5 * t), 0 * np.cos(2 * t)


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
        qd[:, i - n] = data[i]

    for i in range(2 * n, 3 * n):
        qdd[:, i - 2 * n] = data[i]

    return q, qd, qdd


# -----------------------------
# Spatial algebra utilities
# -----------------------------

def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def crm(v):
    w = v[:3] # check the dimensions
    vlin = v[3:]
    return np.block([
        [skew(w), np.zeros((3,3))],
        [skew(vlin), skew(w)]
    ])

def crf(v):
    return -crm(v).T

# -----------------------------
# Spatial inertia
# -----------------------------

def spatial_inertia(m, Ic, c):
    C = skew(c)
    return np.block([
        [Ic + m * C @ C.T, m * C],
        [m * C.T, m * np.eye(3)]
    ])



# # np.block([
#         [Ic + m * C @ C.T, m * C],
#         [m * C.T, m * np.eye(3)]
#     ])
# -----------------------------
# Joint transform (planar)
# -----------------------------

def Xrotz(theta, r):
    c = np.cos(theta)
    s = np.sin(theta)

    R = np.array([
        [c,-s,0],
        [s, c,0],
        [0, 0,1]
    ])

    X = np.zeros((6,6))
    X[:3,:3] = R
    X[3:,3:] = R
    X[3:,:3] = skew(np.array([r,0,0])) @ R
    return X

# -----------------------------
# Featherstone inverse dynamics
# -----------------------------

def featherstone_id(q, qd, qdd, links, gravity):
    n = len(links)

    S = np.array([0,0,1,0,0,0])  # revolute z-axis

    v = [np.zeros(6) for _ in range(n)]
    a = [np.zeros(6) for _ in range(n)]
    f = [np.zeros(6) for _ in range(n)]
    Xup = [None]*n
    I = [None]*n

    # Base acceleration (gravity)
    a0 = np.array([0,0,0, -gravity[0], -gravity[1], -gravity[2]])

    # Forward pass
    for i in range(n):
        _, _, l, m, Ic_com, _, _ = links[i]
        c = np.array([-l/2, 0, 0])

        I[i] = spatial_inertia(m, Ic_com, c)
        Xup[i] = Xrotz(q[i], l)
        # print('Featherstone I:', I[i])

        vJ = S * qd[i]

        if i == 0:
            v[i] = vJ
            a[i] = Xup[i] @ a0 + S * qdd[i] + crm(v[i]) @ vJ
        else:
            v[i] = Xup[i] @ v[i-1] + vJ
            a[i] = Xup[i] @ a[i-1] + S * qdd[i] + crm(v[i]) @ vJ

        f[i] = I[i] @ a[i] + crf(v[i]) @ (I[i] @ v[i])

    # Backward pass
    tau = np.zeros(n)
    for i in reversed(range(n)):
        tau[i] = S @ f[i]
        if i > 0:
            f[i-1] += Xup[i].T @ f[i]

    
    # print("Featherstone V:", v)
    # print("Featherstone a:", a)

    return tau

# -----------------------------
# Link data generator
# -----------------------------

def link_data(n, l=1.0, m=1.0):
    links = []
    for _ in range(n):
        # if _ == 2:
        #     l = 0.001
        #     m = 0.001
        #     Ic = np.diag([0.0, (1/12)*m*l*l, (1/12)*m*l*l])
        #     links.append((0,0,l,m,Ic,1,0.0))
        # else:
            Ic = np.diag([0.0, (1/12)*m*l*l, (1/12)*m*l*l])
            links.append((0,0,l,m,Ic,1,0.0))

    return links





def plot_graphs(n, data1, data2):
    plt.figure(figsize=(10, 5))
    for i in range(n):
        plt.subplot(1, n, i+1)    
        plt.plot(time_step, data1[:, i], label=f'Torque {i} Input')
        plt.plot(time_step, data2[:, i], '--r', label=f'Torque {i}')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.title('Joint Torques Over Time')
    plt.show()    


if __name__ == "__main__":

    # Define the manipulator links: (theta, alpha, length, mass, inertia tensor, joint type: 0 - translational, 1 - rotational, damping coeff.)
    # n = 2
    n = 2

    # links = [
    #     (0, 0, 1.0, 1.0, np.diag([0.0, 1/12 * 1, 1/12 * 1]), 1, 0.),  # Link 1
    #     (0, 0, 1.0, 1.0, np.diag([0.0, 1/12 * 1, 1/12 * 1]), 1, 0.)   # Link 2
    # ]


    links = link_data(n)
    # links = [
    #     (0, 0, 1.0, 1.0, np.diag([1, 1, 1]), 1, 0.),  # Link 1
    #     (0, 0, 1.0, 1.0, np.diag([1, 1, 1]), 1, 0.),
    #     (0, 0, 1.0, 1.0, np.diag([1, 1, 1]), 1, 0.)      # Link 2
    # ]

    time_step = np.linspace(0, 10, 1000)  # Time steps from 0 to 10 seconds
    torques = []

    torquesLE = []


    out_dir = './ra'

    trj_data = f"{out_dir}/trajectory_data_gen{n}.csv"

    q_csv, qd_csv, qdd_csv = load_joint_data(trj_data, n, len(time_step), 'csv') # './providedForward/rl_multilink_simulation.csv' './data/LEForw.csv'
    # './providedForwardMod/rl_multilink_simulation2.csv'

    print("Shape of q:", np.shape(q_csv))
    print("Shape of qd:", np.shape(qd_csv))
    print("Shape of qdd:", np.shape(qdd_csv))

    print("Type of q:", type(q_csv))
    print("Type of qd:", type(qd_csv))
    print("Type of qdd:", type(qdd_csv))

    # def random_q(t):
    #     return np.sin(t) + 0.5 * np.cos(0.5 * t)

    # def random_qd(t):
    #     return np.cos(t) - 0.25 * np.sin(0.5 * t)

    # def random_qdd(t):
    #     return -np.sin(t) - 0.125 * np.cos(0.5 * t)

    g = 9.81
    gravity = np.array([0, g, 0])

    t_total_start = time.perf_counter()


    def ftst(time_step, q_csv, qd_csv, qdd_csv, links, gravity):
        for t_idx in range(len(time_step)): #len(time_step)
            q = q_csv[t_idx]   # Joint positions from CSV
            qd = qd_csv[t_idx] # Joint velocities from CSV
            qdd = qdd_csv[t_idx] # Joint accelerations from CSV
            print(f'Performing FTST for timestep {t_idx}')
            torque = featherstone_id(q, qd, qdd, links, gravity)  # Compute torques using RNEA
            torques.append(torque)
            torque2 = tau_input(time_step[t_idx])
            torquesLE.append(torque2)
            
        return torques, torquesLE


    torques, torquesLE = ftst(time_step, q_csv, qd_csv, qdd_csv, links, gravity)

    print(torques)

    t_total_end = time.perf_counter()
    elapsed = t_total_end - t_total_start
    print(f"Total runtime: {elapsed:.4f} s")
    print(f"Average per timestep: {elapsed/len(time_step):.6f} s")



    torques = np.array(torques)
    torquesLE = np.array(torquesLE)

    print(torques)

    # torques[np.abs(torques) < threshold] = 0.0
    # torquesLE[np.abs(torquesLE) < threshold] = 0.0

        # Create a dictionary with the data
    cols = torques.shape[1] if (hasattr(torques, "ndim") and torques.ndim > 1) else 1
    data = {f't{i+1}': (torques[:, i] if cols > 1 else torques[:]) for i in range(cols)}

    df = pd.DataFrame(data)

    torque_data = f"{out_dir}/torquesFst{n}.csv"


    df.to_csv(torque_data, index=False)



    # Plot the torques

    plot_graphs(n, torquesLE, torques)



# def of parameters: moment of inertia, mass, center of mass, 


# find from Featherstone textbook 2-link or 3-link simulations, try to replicate it. Compare with the obtained results. 



# check the notations for the link parameters in the Featherstone.

# n = 3, m1=m2 = 1, m3 = 0.001, same for Ic. or just make Ic3 really small.