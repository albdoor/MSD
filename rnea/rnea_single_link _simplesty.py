import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
print(os.getcwd())  # Print the current working directory
print(os.listdir()) # List all files in the current directory
g = 9.81
def tau_input(t):
    return np.array([2 * np.sin(0.5 * t), 1.5 * np.cos(0.5 * t)])

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
    # Start with identity matrix
    
    # Accumulate rotations up to the specified link
    cumul_angle = 0
    for i in range(link_idx + 1):
        theta, alpha, r, m, I, j_type, b = links[i]
        if j_type == '1':
            # Only revolute joints contribute to rotation
            theta = q[i]
            cumul_angle += theta
    
    # Alternative implementation using the cumulative angle directly
    # This is more efficient but equivalent to the matrix multiplications above
    R_direct = np.array([
        [np.cos(cumul_angle), -np.sin(cumul_angle), 0],
        [np.sin(cumul_angle), np.cos(cumul_angle), 0],
        [0, 0, 1]
    ])
    
    # Return the direct calculation (more efficient)
    return R_direct



def rotation_matrix(theta, alpha):
    return np.array([
        [np.cos(theta), -np.cos(alpha) * np.sin(theta), np.sin(alpha) * np.sin(theta)],
        [np.sin(theta), np.cos(alpha) * np.cos(theta), -np.sin(alpha) * np.cos(theta)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])

def s_vector(theta, l):
    if len(theta) == 1:
        return np.array([[-l/2*np.cos(theta[0])], 
                        [-l/2*np.sin(theta[0])], 
                        [0]])
    else: 
        theta_fin = 0
        for i in range(len(theta)):
            theta_fin += theta[i]
        return np.array([[-l/2*np.cos(theta_fin)], 
                        [-l/2*np.sin(theta_fin)], 
                        [0]])

def p_star_vector(theta, l):
    if len(theta) == 1:
        return np.array([[l*np.cos(theta[0])], [l*np.sin(theta[0])], [0]]) 
    else:   
        theta_fin = 0
        
        for i in range(len(theta)):
            theta_fin += theta[i]
        return np.array([[l*np.cos(theta_fin)], 
                        [l*np.sin(theta_fin)], 
                        [0]])


def rotation_matrix_n(n, theta, alpha):
    rotation_final = np.eye(3)
    if n > 1:
        for i in range(n):
            rotation_final = rotation_final @ rotation_matrix(theta[i], alpha[i])
    return rotation_final
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def rnea(q, qd, qdd, links, gravity):
    theta, r, m, I, j_type, b = links[0]
    n = len(links)  # Number of links
    tau = np.zeros(n)  # Joint torques
    tau = 1/3 * m * r**2 * qdd + 0.5 * m *g * r * np.cos(q)
    return tau

# Define the manipulator links: (theta, alpha, length, mass, inertia tensor, joint type: 0 - translational, 1 - rotational, damping coeff.)
links = [
    (0, 1.0, 1.0, np.diag([0.0, 0.0, 0.1]), 1, 0.)  # Link 1
]
print(links[0])
time = np.linspace(0, 10, 1000)  # Time steps from 0 to 10 seconds
torques = []

torquesLE = []

q_csv, qd_csv, qdd_csv = load_joint_data('./data/simplePendLE.npy')
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
    torque2 = tau_input(time[t_idx])
    torquesLE.append(torque2)

torques = np.array(torques)
print(np.shape(torques))
torquesLE = np.array(torquesLE)


# Plot the torques
plt.figure(figsize=(12, 12))
print(np.shape(qdd_csv))
print(np.shape(torques))
# plt.subplot(1, 2, 1)
plt.plot(time, torques[:, 0], label='Torque 1')
plt.plot(time, torquesLE[:, 0], label='Torque 1 Input')
plt.plot(time, qdd_csv[:, 0], label='qdd input')
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


# 28.03.2025
# double-check the parameters for LE and NE
# improve the matlab and LE jupiter 
# compare the code with the textbook and code the symbolic notation for the code


# extract values for each term from this implementation
# extract the inertial term from the matrix implementation
# go term by term
