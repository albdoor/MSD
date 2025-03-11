import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
print(os.getcwd())  # Print the current working directory
print(os.listdir()) # List all files in the current directory


def tau_input(t):
    return np.array([2 * np.sin(0.5 * t), 1.5 * np.cos(0.5 * t)])

def load_joint_data(npy_filename):
    """
    Loads joint position (q), velocity (qd), and acceleration (qdd) from a CSV file.

    """
    data = np.load(npy_filename, allow_pickle=True)
    data = data[0] if isinstance(data[0], list) else data
    q = np.vstack((data[0], data[1])).T
    qd = np.vstack((data[2], data[3])).T 
    qdd = np.vstack((data[4], data[5])).T 
    return q, qd, qdd



def rotation_matrix(theta, alpha):
    return np.array([
        [np.cos(theta), -np.cos(alpha) * np.sin(theta), np.sin(alpha) * np.sin(theta)],
        [np.sin(theta), np.cos(alpha) * np.cos(theta), -np.sin(alpha) * np.cos(theta)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])

def rotation_matrix_n(n, theta, alpha):
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
    for i in range(n):
        theta, alpha, r, m, I, j_type, b = links[i]
        R.append(rotation_matrix(float(q[i]), alpha))
        if i == 0:
            if (j_type):
                omega[i] = np.array([0, 0, qd[i]])
                omegad[i] = np.array([0, 0, qdd[i]])
                vd[i] = gravity
            else: 
                omega[i] = np.zeros(3)
                omegad[i] = np.zeros(3)
                vd[i] = gravity + R[i] @ np.array([qdd[i], 0, 0])  # Motion along joint axis
                
            # print(vd[i])
        else:
            if (j_type):
                omega[i] = R[i - 1] @ omega[i - 1] + np.array([0, 0, qd[i]])
                omegad[i] = R[i - 1] @ (R[i] @ omegad[i - 1] + np.dot(omega[i], np.array([0, 0, qd[i]])) + np.array([0, 0, qdd[i]]))
                vd[i] = R[i - 1] @ vd[i - 1] + np.dot(rotation_matrix_n(i, [q[i-1], q[i]], [0, 0]) @ omegad[i], [r, 0, 0])
                vd[i] += np.dot(rotation_matrix_n(i, [q[i-1], q[i]], [0, 0]) @ omega[i], np.dot(omega[i], [r, 0, 0]))
               
            else:
                omega[i] = R[i - 1] @ omega[i - 1]  # No angular motion
                omegad[i] = R[i - 1] @ omegad[i - 1]
                vd[i] = R[i - 1] @ vd[i - 1] + R[i] @ np.array([qdd[i], 0, 0])
        a_c[i] = rotation_matrix_n(i, [q[i-1], q[i]], [0, 0]) @ vd[i] + np.dot(rotation_matrix_n(i, [q[i-1], q[i]], [0, 0]) @ omegad[i], [r/2, 0, 0]) + np.dot(rotation_matrix_n(i, [q[i-1], q[i]], [0, 0]) @ omega[i], np.dot(omega[i], [r/2, 0, 0]))
    # Backward recursion: force & torque propagation
    F = np.zeros((n, 3))  # Force
    N = np.zeros((n, 3))  # Torque

    for i in reversed(range(n)):
        theta, alpha, r, m, I, j_type, b = links[i]
        F[i] = m * a_c[i]
        N[i] = I @ omegad[i] + np.dot(omega[i], I @ omega[i])
        if i < n - 1:
            F[i] += R[i].T @ F[i + 1]
            N[i] += R[i].T @ N[i + 1] + np.dot([r, 0, 0], R[i].T @ F[i + 1])
    
        if j_type == 1:  # Rotational joint: torque
            tau[i] = N[i][2] + b * qd[i]
        else:  # Translational joint: force
            tau[i] = F[i][0] + b * qd[i] # Force along the x-axis of motion
    
    
    
    
    return tau

# Define the manipulator links: (theta, alpha, length, mass, inertia tensor, joint type: 0 - translational, 1 - rotational, damping coeff.)
links = [
    (0, 0, 1.0, 5.0, np.diag([0.1, 0.1, 0.2]), 1, 0.),  # Link 1
    (0, 0, 1.0, 1.0, np.diag([0.1, 0.1, 0.2]), 1, 0.)   # Link 2
]

time = np.linspace(0, 10, 1000)  # Time steps from 0 to 10 seconds
torques = []

torquesLE = []

q_csv, qd_csv, qdd_csv = load_joint_data('./data/doublelinkLE.npy')
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
gravity = np.array([0, -g, 0])

for t_idx in range(len(time)):
    q = q_csv[t_idx]   # Joint positions from CSV
    qd = qd_csv[t_idx] # Joint velocities from CSV
    qdd = qdd_csv[t_idx] # Joint accelerations from CSV
    torque = rnea(q, qd, qdd, links, gravity)  # Compute torques using RNEA
    torques.append(torque)
    torque2 = tau_input(t_idx)
    torquesLE.append(torque2)

torques = np.array(torques)
torquesLE = np.array(torquesLE)


# Plot the torques
plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.plot(time, torques[:, 0], label='Torque 1')
plt.plot(time, torquesLE[:, 0], label='Torque 1 input')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.title('Joint Torques Over Time')



plt.subplot(1, 2, 2)
plt.plot(time, torques[:, 1], label='Torque 2')
plt.plot(time, torquesLE[:, 1], label='Torque 2 input')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.title('Joint Torques Over Time')
plt.show()


# Add options for the joints
# verification simulation for double pendulum
# The torque is external
# To input torque as some sine function