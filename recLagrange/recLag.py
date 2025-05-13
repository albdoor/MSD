import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
print(os.getcwd())  # Print the current working directory
print(os.listdir()) # List all files in the current directory


def tau_input(t):
    return np.array([2 * np.sin(0.5 * t), 1.5 * np.cos(1.5 * t)])


def load_joint_data(npy_filename):
    """
    Loads joint position (q), velocity (qd), and acceleration (qdd) from a CSV file.

    """
    # data = np.load(npy_filename, allow_pickle=True)
    # data = data[0] if isinstance(data[0], list) else data
    data = np.loadtxt(npy_filename, delimiter=',')
    data = data.T
    q = np.vstack((data[0], data[1])).T
    qd = np.vstack((data[2], data[3])).T 
    qdd = np.vstack((data[4], data[5])).T 
    return q, qd, qdd



def coord_transform(theta, l):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, l * np.cos(theta)],
        [np.sin(theta), np.cos(theta), 0, l * np.sin(theta)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def coord_transform_to_base(q, links, link_idx):
    cumul_angle = 0
    for i in range(link_idx):
        theta, alpha, r, m, I, j_type, b = links[i]
        if j_type == '1':
            the = q[i]
            cumul_angle += the
    # R_direct = np.array([
    #     [np.cos(cumul_angle), -np.sin(cumul_angle), 0, l * (cos())],
    #     [np.sin(cumul_angle), np.cos(cumul_angle), 0],
    #     [0, 0, 1]
    # ])
    # Return the direct calculation (more efficient)
    l = 1
    R_direct = np.array([
        [np.cos(cumul_angle), -np.sin(cumul_angle), 0, l * (np.cos(cumul_angle) + np.cos(q[0]))],
        [np.sin(cumul_angle), np.cos(cumul_angle), 0, l * (np.sin(cumul_angle) + np.sin(q[0]))],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return R_direct


def Q_mat():
    return np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

# def find_U_matrix(i, j, k):
#     if((i >= k) & (k >= j)):
#         coord_transform_to_base(q[j-1]) @ Q_mat() @ coord_transform(qk-1) 

# def calculate_dmatrices(U, J, d_index, k_index):
#     d = np.zeros
#     for j in range(max(k_index, d_index)):
#         d += np.trace()





def recLag(q, qd, qdd, links, gravity):
    n = len(links)  # number of links
    tau = np.zeros(n)  # joint torques
    A_mat = []  # coordinate transformation matrices
    A_mat_base = []
    Q = Q_mat()
    U = [] # potential energy matrices
    J = [] # inertia matrices
    D = [] # D matrices
    h = []
    m = []
    Uh = []
    c = []
    l = []
    for i in range(n):
        theta, alpha, r, mass, I, j_type, b = links[i]
        m.append(mass)
        l.append(r)
        J.append(I)
        A_mat.append(coord_transform(float(q[i]), r))
        # print(R[i])
        A_mat_base.append(coord_transform_to_base(q, links, i))
        if (i == 0):
            U.append((Q @ A_mat[i]).reshape(4, 4))
            print(U)
        else:
            U.append((Q @ A_mat_base[i]).reshape(4, 4))
            U.append((A_mat[i-1] @ Q @ A_mat[i]).reshape(4, 4))
            print('Printing U---------------------------')
            print(U)
            D.append(np.trace(U[i-1] @ J[i-1] @ U[i-1].T) + np.trace(U[i] @ J[i] @ U[i].T))
            D.append(np.trace(U[i+1] @ J[i] @ U[i].T))
            D.append(np.trace(U[i+1] @ J[i] @ U[i].T))
            D.append(np.trace(U[i+1] @ J[i] @ U[i+1].T))
            print('Printing D---------------------------')
            print(D)
            
            h.append(np.trace((Q @ Q @ A_mat[i-1]) @ J[i-1] @ U[i-1].T + (Q @ Q @ A_mat_base[i]) @ J[i] @ U[i].T) * qd[i-1]**2 + 
                     np.trace((Q @ A_mat[i-1] @ Q @ A_mat[i]) @ J[i] @ U[i].T) * qd[i-1] * qd[i]  + 
                     np.trace((A_mat[i-1] @ Q @ A_mat[i-1].T @ Q @ A_mat_base[i]) @ J[i] @ U[i].T) * qd[i-1] * qd[i] + 
                     np.trace((A_mat[i-1] @ Q @ Q @ A_mat[i]) @ J[i] @ U[i].T) * qd[i]**2)
            h.append(np.trace(Q @ Q @ A_mat_base[i]) * qd[i-1]**2 + 
                     np.trace((Q @ A_mat[i-1] @ Q @ A_mat[i]) @ J[i] @ U[i+1].T) * qd[i-1] * qd[i] + 
                     np.trace((A_mat[i-1] @ Q @ A_mat[i-1].T @ Q @ A_mat_base[i]) @ J[i] @ U[i+1].T) * qd[i-1] * qd[i] + #qd2 removed in the textbook: To be checked
                     np.trace((A_mat[i-1] @ Q @ Q @ A_mat[i]) @ J[i] @ U[i+1].T) * qd[i]**2)
            print(np.shape(m[i-1] * gravity @ U[i-1]))
            print(np.shape(np.array([[-l[i-1]/2], [0], [0], [1]])))
            
            c.append(-(m[i-1] * gravity @ U[i-1] @ np.array([[-l[i-1]/2], [0], [0], [1]]) + m[i] * gravity @ U[i] @ np.array([[-l[i]/2], [0], [0], [1]])))
            c.append(-m[i] * gravity @ U[i+1] @ np.array([[-l[i-1]/2], [0], [0], [1]]))
            print('Printing C---------------------------')
            
            print(np.array(c).reshape(2,1))
            print(-m[i] * gravity @ U[i+1] @ np.array([[-l[i-1]/2], [0], [0], [1]]))

    tau = np.array(D).reshape((2, 2)) @ np.array([[qdd[0]], [qdd[1]]]) + np.array(h).reshape(2,1) + np.array(c).reshape(2,1)# Joint torques
    print("PRINTING TAU::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    print(tau)
    
    return tau











m1 = 1
m2 = 1
l1 = 1
l2 = 1

# Define the manipulator links: (theta, alpha, length, mass, inertia tensor, joint type: 0 - translational, 1 - rotational, damping coeff.)
links = [
    (0, 0, l1, m1, np.array([
        [1/3 * m1 * l1**2, 0, 0, -0.5 * m1 * l1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-0.5 * m1 * l1, 0, 0, m1],
    ]), 1, 0.),  # Link 1
    (0, 0, l2, m2, np.array([
        [1/3 * m2 * l2**2, 0, 0, -0.5 * m2 * l2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-0.5 * m2 * l2, 0, 0, m2],
    ]), 1, 0.)   # Link 2
]

time = np.linspace(0, 10, 1000)  # Time steps from 0 to 10 seconds
torques = []

torquesLE = []

q_csv, qd_csv, qdd_csv = load_joint_data('./data/planarDoublePend.csv')
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
gravity = np.array([[0, -g, 0, 0]])

for t_idx in range(len(time)):
    q = q_csv[t_idx]   # Joint positions from CSV
    qd = qd_csv[t_idx] # Joint velocities from CSV
    qdd = qdd_csv[t_idx] # Joint accelerations from CSV
    torque = recLag(q, qd, qdd, links, gravity)  # Compute torques using RNEA
    torques.append(torque)
    torque2 = tau_input(time[t_idx])
    torquesLE.append(torque2)

torques = np.array(torques)
torquesLE = np.array(torquesLE)


# Plot the torques


plt.figure(figsize=(15, 5))


# plt.subplot(1, 3, 1)
# plt.plot(time, torques[:, 0], '--b', label='Torque 1')
# plt.plot(time, torquesLE[:, 0], label='Torque 1 Input')
# plt.plot(time, torques[:, 1], '--g', label='Torque 2')
# plt.plot(time, torquesLE[:, 1], label='Torque 2 Input')
# plt.xlabel('Time (s)')
# plt.ylabel('Torque (Nm)')
# plt.legend()
# plt.title('Joint Torques Over Time')

plt.subplot(1, 2, 1)
plt.plot(time, torques[:, 0], label='Torque 1')
plt.plot(time, torquesLE[:, 0], '--r', label='Torque 1 Input')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.title('Joint Torques Over Time')


plt.subplot(1, 2, 2)
plt.plot(time, torques[:, 1], label='Torque 2')
plt.plot(time, torquesLE[:, 1], '--r', label='Torque 2 Input')
# plt.plot(time, q_csv[:, 1], label='q2')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.title('Joint Torques Over Time')
plt.show()

