import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
print(os.getcwd())  # Print the current working directory
print(os.listdir()) # List all files in the current directory
threshold = 1e-13

def tau_input(t):
    return np.array([0.5 * np.sin(3 * t), 0.88 * np.cos(2 * t)]) # , 0.075 * np.cos(2 * t) 0.88 * np.cos(2 * t), 0.88 * np.cos(2 * t)
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

def rotation_matrix_to_base(q, links, link_idx):
    cumul_angle = 0
    res = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        ])

    if (link_idx == -1):
        return res

    if (link_idx == 0):
        theta, alpha, r, m, I, j_type, b = links[link_idx]
        res = np.array([
            [np.cos(q[link_idx]), -np.sin(q[link_idx]), 0],
            [np.sin(q[link_idx]), np.cos(q[link_idx]), 0],
            [0, 0, 1]
        ])
        res[np.abs(res) < threshold] = 0.0
        return res
    if (link_idx >= 1):
        for j in range(link_idx + 1):
            theta, alpha, r, m, I, j_type, b = links[j]
            # print(j)
            if j_type == 1:
                # print(j)
                temp = np.array([
                    [np.cos(q[link_idx]), -np.sin(q[link_idx]), 0],
                    [np.sin(q[link_idx]), np.cos(q[link_idx]), 0],
                    [0, 0, 1]
                ])
                res = res @ temp
        # res[np.abs(res) < threshold] = 0.0
        return res





def rotation_matrix(i, j, q, links):
    iden_matrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
        ])

    if (i == j) or j == -1:
        return iden_matrix
    
    
    if (i == -1):
        if i == j:
            return iden_matrix
        else :
            return rotation_matrix_to_base(q, links, j)
    
    if ((j - i) == 1): 
        theta, alpha, r, m, I, j_type, b = links[j]
        res = np.array([
            [np.cos(q[j]), -np.sin(q[j]), 0],
            [np.sin(q[j]), np.cos(q[j]), 0],
            [0, 0, 1]
        ])
        # res[np.abs(res) < threshold] = 0.0
        return res 
    elif (abs(j)-abs(i)) == -1:#check
            theta, alpha, r, m, I, j_type, b = links[j]
            res = np.array([
            [np.cos(q[j]), -np.sin(q[j]), 0],
            [np.sin(q[j]), np.cos(q[j]), 0],
            [0, 0, 1]
            ]).T
            # res[np.abs(res) < threshold] = 0.0
            return res  
    elif (j-i > 1): #check
        res = np.eye(4)
        # print(j)
        # print(i)
        for k in range(i, j):
            theta, alpha, r, m, I, j_type, b = links[k]
            temp = np.array([
            [np.cos(q[j]), -np.sin(q[j]), 0],
            [np.sin(q[j]), np.cos(q[j]), 0],
            [0, 0, 1]
            ])
            res = res @ temp
            # print('----------------------------- Coordinate Transform -----------------------------------------------')
            # print(k)
            # print(res)    
        # res[np.abs(res) < threshold] = 0.0
        return res


# ------------------------------ New Functions ---------------------------------------

def skew_symmetric(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]]) 

def vector_from_skew(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def composite_mass(links, i):
    total_mass = 0
    for j in range(0, i):
        theta, alpha, r, m, I, j_type, b = links[j]
        total_mass += m
    return total_mass


def transform_inertia_tensor(I_local, R_ki, q, links, i, j):
    """
    Transform inertia tensor from the link's local frame (i) 
    to another frame (k).
    
    Parameters
    ----------
    I_local : (3,3) array_like
        Inertia tensor expressed in the link's own frame (i).
    R_ki : (3,3) array_like
        Rotation matrix from frame i to frame k.
        That is, it transforms a vector from frame i into frame k.
        
    Returns
    -------
    I_global : (3,3) ndarray
        Inertia tensor expressed in frame k.
    """
    theta, alpha, r, m, I_local, j_type, b = links[i]
    R_ki = rotation_matrix(i, j, q, links)
    I_global = R_ki @ I_local @ R_ki.T
    return I_global

def unit_inertia(links, i):
    theta, alpha, r, m, I_local, j_type, b = links[i]
    I_unit = I_local / m
    return I_unit

def r_skew(i, j, links):
    if i == j:
        theta, alpha, r, m, I, j_type, b = links[i]
        return skew_symmetric(np.array([[r], [0], [0]]).reshape(3, ))
    elif (i - j == 1):
        theta, alpha, r, m, I, j_type, b = links[i]
        res = (rotation_matrix(i-1, i, q, links).T) @ r_skew(j, j, links)
        return skew_symmetric(res.reshape(3, ))    
    

def omega_plus(i, q, qd, links, omega):
    R = rotation_matrix(i, i + 1, q, links)
    res = R.T @ (R @ omega - np.array([[0], [0], [qd[i + 1]]]))
    return res.reshape((3, ))


def p_skew(i, j, links):
    if i == j:
        theta, alpha, r, m, I, j_type, b = links[i]
        return skew_symmetric(np.array([[r/2], [0], [0]]).reshape(3, ))
    elif (i - j == 1):
        theta, alpha, r, m, I, j_type, b = links[i]
        res = (rotation_matrix(i-1, i, q, links).T) @ p_skew(j, j, links)
        return skew_symmetric(res.reshape(3, ))

# Must work on the doc: insert equations from textbooks
# ------------------------------ New Functions ---------------------------------------

def rnea(q, qd, qdd, links, gravity):
    n = len(links)  # Number of links
    omega = np.zeros((n, 3))  # Angular velocity
    omegad = np.zeros((n, 3))  # Angular acceleration
    Omega = np.zeros((n, 3))
    pdd = np.zeros((n, 3))  # Linear acceleration
    mu = np.zeros((n, 3))
    R = []  # Rotation matrices
    Rbase = []
    comp_mass = []
    u_skew = []
    k = []
    k_comp = []
    for i in range(n):
        theta, alpha, r, m, I, j_type, b = links[i]
        rskew = r_skew(i, i, links)
        pskew = p_skew(i+1, i, links)
        E_inertia = unit_inertia(links, i)
        next_comp_mass = composite_mass(links, i+1)
        comp_mass[i] = next_comp_mass + m
        u_skew[i] = m * rskew + next_comp_mass * pskew
        k[i] = I - m * rskew * rskew - next_comp_mass * pskew * pskew
        k_comp[i] = k[i] - 0.5 * np.trace(k[i]) * E_inertia

    for i in range(-1, n - 2):
        theta, alpha, r, m, I, j_type, b = links[i]
        print('======================= Main Loop ==================================')
        print(b)
        R.append(rotation_matrix(i, i + 1, q, links))
        print(R[i])
        # print(R[i])
        Rbase.append(rotation_matrix_to_base(q, links, i))

        if i == -1:
            if (j_type):
                omega[i+1] =  (np.array([[0], [0], [qd[i]]])).reshape((3, ))
                omegaplus = skew_symmetric(omega_plus(i, q, qd, links, omega[i+1]))
                omegad[i+1] = (omegaplus @ np.array([[0], [0], [qd[i]]]) + np.array([[0], [0], [qdd[i]]])).reshape((3, ))
                Omega[i+1] = skew_symmetric(omegad[i+1]) + skew_symmetric(omega[i+1]) @ skew_symmetric(omega[i+1])
                pdd[i+1] = R[i].T @ (gravity)
                tempmu = -(Omega[i+1] @ k_comp[i+1]) + (Omega[i+1] @ k_comp[i+1]).T
                mu[i+1] = vector_from_skew(tempmu)
                print('--------------------- i == 0 Case ---------------------------')
                print(omega[i])
                print(omegad[i])
        else:
            if (j_type):
                pvec = vector_from_skew(p_skew(i+1, i, links))
                omega[i+1] =  (R[i].T @ omega[i] + np.array([[0], [0], [qd[i]]])).reshape((3, ))
                omegaplus = skew_symmetric(omega_plus(i, q, qd, links, omega[i+1]))
                omegad[i+1] = (R[i].T @ omegad[i] + omegaplus @ np.array([[0], [0], [qd[i]]]) + np.array([[0], [0], [qdd[i]]])).reshape((3, ))
                Omega[i+1] = skew_symmetric(omegad[i+1]) + skew_symmetric(omega[i+1]) @ skew_symmetric(omega[i+1])
                pdd[i+1] = R[i].T @ (Omega[i] @ pvec + pdd[i])
                tempmu = -(Omega[i+1] @ k_comp[i+1]) + (Omega[i+1] @ k_comp[i+1]).T
                mu[i+1] = vector_from_skew(tempmu)
    # Backward recursion: force & torque propagation
    udd = np.zeros((n, 3))  # Force
    n_torque = [np.zeros(3) for _ in range(n + 1)]  # To    rque on each link
    tau = np.zeros(n)  # Joint torques
    
    for i in reversed(range(n)):
        theta, alpha, r, m, I, j_type, b = links[i]
        print(i)
        if i == n-1:
            uvec = vector_from_skew(u_skew[i])
            udd[i] = Omega[i] @ uvec
            print('------------------ Passing Through i = n - 1 -------------------------------------')
            n_torque[i] = mu[i] + u_skew[i] @ pdd[i]                  #(R[i+1] @ udd[i]) 
            print(n_torque[i])
        else: 
            uvec = vector_from_skew(u_skew[i])
            udd[i] = Omega[i] @ uvec + R[i] @ udd[i + 1]
            n_torque[i] = mu[i] + u_skew[i] @ pdd[i] + skew_symmetric(pvec) @ ((R[i+1] @ udd[i])) + R[i] @ n_torque[i+1]                  #(R[i+1] @ udd[i]) 
            print('----------------- Passing Through i < n - 1 -------------------------------')
        
        if j_type == 1:
            print('--------------------- Passing Through ---------------------------------------')
            tau[i] = np.dot(n_torque[i].reshape(3,), np.array([0, 0, 1]).T) + b * qd[i]
            print(tau[i])
        else:  # Prismatic
            tau[i] = np.dot(f[i], R[i].T @ np.array([0, 0, 1])) + b * qd[i]    
    
    return tau




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


# Define the manipulator links: (theta, alpha, length, mass, inertia tensor, joint type: 0 - translational, 1 - rotational, damping coeff.)
n = 2
# n = 3

links = [
    (0, 0, 1.0, 1.0, np.diag([0.0, 1/12 * 1, 1/12 * 1]), 1, 0.),  # Link 1
    (0, 0, 1.0, 1.0, np.diag([0.0, 1/12 * 1, 1/12 * 1]), 1, 0.)   # Link 2
]


# links = [
#     (0, 0, 1.0, 1.0, np.diag([0.0, 1/12 * 1, 1/12 * 1]), 1, 0.),  # Link 1
#     (0, 0, 1.0, 1.0, np.diag([0.0, 1/12 * 1, 1/12 * 1]), 1, 0.),   # Link 2
#     (0, 0, 1.0, 1.0, np.diag([0.0, 1/12 * 1, 1/12 * 1]), 1, 0.)   # Link 2
# ]

# links = [
#     (0, 0, 1.0, 1.0, np.diag([1, 1, 1]), 1, 0.),  # Link 1
#     (0, 0, 1.0, 1.0, np.diag([1, 1, 1]), 1, 0.),
#     (0, 0, 1.0, 1.0, np.diag([1, 1, 1]), 1, 0.)      # Link 2
# ]

time = np.linspace(0, 10, 1000)  # Time steps from 0 to 10 seconds
torques = []

torquesLE = []

q_csv, qd_csv, qdd_csv = load_joint_data('./data/LEForw_2link.csv', n, len(time), 'csv') # './providedForward/rl_multilink_simulation.csv' './data/LEForw.csv'
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

for t_idx in range(len(time)): #len(time)
    q = q_csv[t_idx]   # Joint positions from CSV
    qd = qd_csv[t_idx] # Joint velocities from CSV
    qdd = qdd_csv[t_idx] # Joint accelerations from CSV
    torque = rnea(q, qd, qdd, links, gravity)  # Compute torques using RNEA
    torques.append(torque)
    torque2 = tau_input(time[t_idx])
    torquesLE.append(torque2)

torques = np.array(torques)
torquesLE = np.array(torquesLE)

# torques[np.abs(torques) < threshold] = 0.0
# torquesLE[np.abs(torquesLE) < threshold] = 0.0


# Plot the torques

plot_graphs(n, torquesLE, torques)


# Add options for the joints
# verification simulation for double pendulum
# The torque is external
# To input torque as some sine function


# 12.03.2025
# consider single-link pendulum
# external torque again
# 



# 14.04.25
# try making m2 = 0, or make m1 = 1000m2



# 5 link system with varying joint types, same mass, length, I
# recursive lagrangian 


# 15.08.25
# check the indices
# send the code