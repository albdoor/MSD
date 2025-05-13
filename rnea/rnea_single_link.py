import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
print(os.getcwd())  # Print the current working directory
print(os.listdir()) # List all files in the current directory

def tau_input(t):
    return np.array([2 * np.sin(0.5 * t)])

def load_joint_data(npy_filename):
    """
    Loads joint position (q), velocity (qd), and acceleration (qdd) from a CSV file.

    """
    data = np.load(npy_filename, allow_pickle=True)
    data = data[0] if isinstance(data[0], list) else data
    q = np.vstack((data[0]))
    qd = np.vstack((data[1])) 
    qdd = np.vstack((data[2])) 
    return q, qd, qdd

def rotation_matrix_to_base(q, links, link_idx):
    
    # Accumulate rotations up to the specified link
    cumul_angle = 0
    for i in range(link_idx):
        theta, alpha, r, m, I, j_type, b = links[i]
        if j_type == '1':
            theta = q[i]
            cumul_angle += theta
    
    R_direct = np.array([
        [np.cos(cumul_angle), -np.sin(cumul_angle), 0],
        [np.sin(cumul_angle), np.cos(cumul_angle), 0],
        [0, 0, 1]
    ])
    
    # Return the direct calculation (more efficient)
    return R_direct


def manual_cross_product(b, c):
    return np.array([
        b[1]*c[2] - b[2]*c[1],
        b[2]*c[0] - b[0]*c[2],
        b[0]*c[1] - b[1]*c[0]
    ])

def rotation_matrix(theta, alpha=0):
    return np.array([
        [np.cos(theta), -np.cos(alpha) * np.sin(theta), np.sin(alpha) * np.sin(theta)],
        [np.sin(theta), np.cos(alpha) * np.cos(theta), -np.sin(alpha) * np.cos(theta)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])

def s_vector(theta, l):
    return np.array([[-l/2], [0], [0]]) 

    # if len(theta) == 1:
    #     return np.array([[-l/2*np.cos(theta[0])], 
    #                     [-l/2*np.sin(theta[0])], 
    #                     [0]])
    # else: 
    #     theta_fin = 0
    #     for i in range(len(theta)):
    #         theta_fin += theta[i]
    #     return np.array([[-l/2*np.cos(theta_fin)], 
    #                     [-l/2*np.sin(theta_fin)], 
    #                     [0]])

 

def p_star_vector(theta, l):
    return np.array([[l], [0], [0]]) 

    # if len(theta) == 1:
    # else:   
    #     theta_fin = 0
        
    #     for i in range(len(theta)):
    #         theta_fin += theta[i]
    #     return np.array([[l*np.cos(theta_fin)], 
    #                     [l*np.sin(theta_fin)], 
    #                     [0]])


def rotation_matrix_n(n, theta, alpha = 0):
    rotation_final = np.eye(3)
    if n > 1:
        for i in range(n):
            rotation_final = rotation_final @ rotation_matrix(theta[i], alpha[i])
    return rotation_final

def rnea(q, qd, qdd, links, gravity):
    n = len(links)  # Number of links
    tau = np.zeros(n)  # Joint torques
    
    # Forward recursion: velocity & acceleration propagation
    omega = np.zeros((n, 3))  # Angular velocity
    omegad = np.zeros((n, 3))  # Angular acceleration
    v = np.zeros((n, 3))  # Linear velocity
    vd = np.zeros((n, 3))  # Linear acceleration
    a_c = np.zeros((n, 3))
    R = []  # Rotation matrices
    Rbase = []
    p_star = []
    s_bar = []
    for i in range(n):
        theta, alpha, r, m, I, j_type, b = links[i]
        R.append(rotation_matrix(float(q[i]), alpha))
        # print(R[i])
        Rbase.append(rotation_matrix_to_base(q, links, len(links)))
        p_star.append(p_star_vector(q, r))
        s_bar.append(s_vector(q, r))
        if i == 0:
            if (j_type):
                omega[i] =  R[i].T @ np.array([0, 0, qd[i]])
                omegad[i] = R[i].T @ np.array([0, 0, qdd[i]])
                vd[i] = np.cross(omegad[i], p_star[i].T) + np.cross(omega[i], np.cross(omega[i], p_star[i].T)) + R[i].T @ gravity
            else:   
                omega[i] = np.zeros(3)
                omegad[i] = np.zeros(3)
                vd[i] = gravity + R[i] @ np.array([qdd[i], 0, 0])  # Motion along joint axis
                
            # print(vd[i])
        else:
            if (j_type):
                omega[i] = R[i - 1].T @ (R[i - 1] @ omega[i - 1] + np.array([0, 0, qd[i]]))
                omegad[i] = R[i - 1].T @ (Rbase[i-1] @ omegad[i - 1] + np.dot(Rbase[i-1] @ omega[i], np.array([0, 0, qd[i]])) + np.array([0, 0, qdd[i]]))
                # vd[i] = R[i - 1] @ vd[i - 1] + np.dot(rotation_matrix_n(i, [q[i-1], q[i]], [0, 0]) @ omegad[i], [r, 0, 0])
                # vd[i] += np.dot(rotation_matrix_n(i, [q[i-1], q[i]], [0, 0]) @ omega[i], np.dot(omega[i], [r, 0, 0]))
                vd[i] = omegad[i] @ (p_star[i]) + np.dot(omega[i], omega[i] @ (p_star[i])) + R[i-1].T @ (Rbase[i-1] @ vd[i-1])
                # rework the vd above___________________---

            else:
                omega[i] = R[i - 1] @ omega[i - 1]  # No angular motion
                omegad[i] = R[i - 1] @ omegad[i - 1]
                vd[i] = R[i - 1] @ vd[i - 1] + R[i] @ np.array([qdd[i], 0, 0])
        # a_c[i] = rotation_matrix_to_base([q[i-1], q[i]], links, i) @ vd[i] + np.dot(rotation_matrix_to_base([q[i-1], q[i]], links, i) @ omegad[i], [r/2, 0, 0]) + np.dot(rotation_matrix_to_base([q[i-1], q[i]], links, i) @ omega[i], np.dot(omega[i], [r/2, 0, 0]))
        # print(omegad[i].T @ np.dot(Rbase[i], s_bar[i]).T)

        a_c[i] = np.cross(omegad[i],  s_bar[i].T) + np.cross(omega[i],  np.cross(omega[i],  s_bar[i].T)) + vd[i]

    # Backward recursion: force & torque propagation
    F = np.zeros((n, 3))  # Force
    N = np.zeros((n, 3))  # Torque
    f = [np.zeros(3) for _ in range(n + 1)]  # Force on each link
    n_torque = [np.zeros(3) for _ in range(n + 1)]  # Torque on each link
    tau = np.zeros(n)  # Joint torques

    for i in reversed(range(n)):
        theta, alpha, r, m, I, j_type, b = links[i]
        if i == n-1:
            F[i] = m * a_c[i]
            N[i] = np.dot(Rbase[i].T @ I @ Rbase[i], omegad[i]) + np.cross(omega[i], np.dot(Rbase[i].T @ I @ Rbase[i], omega[i])) #Why is the second term giving zero
            f[i] = F[i]
            n_torque[i] = np.cross(p_star[i].T + s_bar[i].T, F[i]) + N[i]
        else: 
            # Compute force: F = m * a
            # Compute torque: N = I * ω + ω × (I * ω)
            # Propagate force: f_i = R_{i+1} * f_{i+1} + F_i
            F[i] = m * a_c[i]
            N[i] = np.dot(Rbase[i].T @ I @ Rbase[i], omegad[i]) + np.dot(omega[i], np.dot(Rbase[i].T @ I @ Rbase[i], omega[i]))

            # Propagate torque: n_i = R_{i+1} * n_{i+1} + (p_i × f_i) + N_i
            f[i] = np.dot(R[i+1], Rbase[i+1] @ f[i+1]) + F[i]

            n_torque[i] = R[i+1] @(Rbase[i+1] @ n_torque[i+1] + np.cross(Rbase[i+1] @ p_star, Rbase[i+1] @ f[i]) ) + np.cross(Rbase[i+1] @ p_star + Rbase[i+1] @ s_bar, F[i]) + N[i]

            # Compute joint torque
        if j_type == 1:
            tau[i] = np.dot(n_torque[i].flatten(), R[i].T @ np.array([0, 0, 1]).T) + b * qd[i]
        else:  # Prismatic
            tau[i] = np.dot(f[i], R[i - 1].T @ np.array([0, 0, 1])) + b * qd[i]

    
    
    return tau

# Define the manipulator links: (theta, alpha, length, mass, inertia tensor, joint type: 0 - translational, 1 - rotational, damping coeff.)
links = [
    (0, 0, 1.0, 1.0, np.diag([0, 1/12 * 1, 1/12 * 1]), 1, 0)  # Link 1
]


time = np.linspace(0, 10, 1000)  # Time steps from 0 to 10 seconds

torques = []

torquesLE = []

q_csv, qd_csv, qdd_csv = load_joint_data('./data/simplePendLE.npy')
print("Shape of q:", np.shape(q_csv))
print("Shape of qd:", np.shape(qd_csv))
print("Shape of qdd:", np.shape(qdd_csv))

# print("Type of q:", type(q_csv))
# print("Type of qd:", type(qd_csv))
# print("Type of qdd:", type(qdd_csv))
# def random_q(t):
#     return np.sin(t) + 0.5 * np.cos(0.5 * t)

# def random_qd(t):
#     return np.cos(t) - 0.25 * np.sin(0.5 * t)

# def random_qdd(t):
#     return -np.sin(t) - 0.125 * np.cos(0.5 * t)

g = 9.81
gravity = np.array([0, g, 0]) ################# The error was here!!!!!!!!!!!!!!!!!!!!!!

for t_idx in range(len(time)):
    q = q_csv[t_idx]   # Joint positions from CSV
    qd = qd_csv[t_idx] # Joint velocities from CSV
    qdd = qdd_csv[t_idx] # Joint accelerations from CSV
    torque = rnea(q, qd, qdd, links, gravity)  # Compute torques using RNEA
    torques.append(torque)
    torque2 = tau_input(time[t_idx])
    torquesLE.append(torque2)

torques = np.array(torques)

torquesLE = np.array(torquesLE)
print(torques[0])
print(torquesLE[0])

# Plot the torques
plt.figure(figsize=(7, 5))

# plt.subplot(1, 1, 1)
plt.plot(time, torques[:, 0], 'b' , label='Torque 1')
plt.plot(time, torquesLE[:, 0], 'r--', label='Torque 1 Input')
# plt.plot(time, qdd_csv[:, 0], label='qdd input')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.title('Joint Torques Over Time')

plt.show()

# plt.subplot(1, 2, 2)
# plt.plot(time, torques[:, 1], label='Torque 2')
# plt.plot(time, torquesLE[:, 1], label='Torque 2 Input')
# plt.xlabel('Time (s)')
# plt.ylabel('Torque (Nm)')
# plt.legend()
# plt.title('Joint Torques Over Time')
# plt.show()


# Add options for the joints
# verification simulation for double pendulum
# The torque is external
# To input torque as some sine function


# 12.03.2025
# consider single-link pendulum
# external torque again
# 


# try doing teo-link with one link fixed
# fix torque as 0 for the first link



# forward simulation in matlab
# compare theta, thetadot
# write RNEA in matlab



# double-link system
# do forward simulation again for double link
# theta1, theta1dot, 