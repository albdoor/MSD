import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
print(os.getcwd())  # Print the current working directory
print(os.listdir()) # List all files in the current directory
threshold = 1e-13

def tau_input(t):
    return np.array([0.5 * np.sin(3 * t), 0.88 * np.cos(2 * t), 0, 0]) # , 0.075 * np.cos(2 * t) 0.88 * np.cos(2 * t), 0.88 * np.cos(2 * t)
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


def s_vector(q, l):
       return np.array([[-l/2], [0], [0]]) 

def p_star_vector(q, l, i, j):
    if (i == j):
        return np.array([[l], [0], [0]])
    if (i - j == 1):
        res = (rotation_matrix(i-1, i, q, links).T) @ p_star_vector(q, l, j, j)
        return res 


def rnea(q, qd, qdd, links, gravity):
    n = len(links)  # Number of links
    omega = np.zeros((n, 3))  # Angular velocity
    omegad = np.zeros((n, 3))  # Angular acceleration
    vd = np.zeros((n, 3))  # Linear acceleration
    a_c = np.zeros((n, 3))
    R = []  # Rotation matrices
    Rbase = []
    p_star = []
    s_bar = []
    for i in range(n):
        theta, alpha, r, m, I, j_type, b = links[i]
        print('======================= Main Loop ==================================')
        print(b)
        R.append(rotation_matrix(i - 1, i, q, links))
        print(R[i])
        # print(R[i])
        Rbase.append(rotation_matrix_to_base(q, links, i))
        print(Rbase[i])
        p_star.append(p_star_vector(q, r, i, i))
        print(p_star[i])
        s_bar.append(s_vector(q[i], r))
        print(s_bar[i])
        if i == 0:
            if (j_type):
                omega[i] =  (R[i].T @ np.array([[0], [0], [qd[i]]])).reshape((3, ))
                omegad[i] = (R[i].T @ np.array([[0], [0], [qdd[i]]])).reshape((3, ))
                vd[i] = (np.cross(omegad[i], (p_star[i]).T) + np.cross(omega[i], np.cross(omega[i], p_star[i].T)) + R[i].T @ gravity).reshape((3, ))
                print('--------------------- i == 0 Case ---------------------------')
                print(omega[i])
                print(omegad[i])
                print(vd[i])

            else:   
                omega[i] = np.zeros(3)
                omegad[i] = np.zeros(3)
                vd[i] = gravity + R[i] @ np.array([[qdd[i]], [0], [0]])  # Motion along joint axis
                    
            # print(vd[i])
        else:
            if (j_type):
                print('-------------------------- i > 1 Case ------------------------------')
                omega[i] = (R[i].T @ (omega[i - 1] + np.array([0, 0, qd[i]]))).reshape((3, )) #check
                omegad[i] = (R[i].T @ (omegad[i - 1] + np.cross(omega[i-1], np.array([0, 0, qd[i]])) + np.array([0, 0, qdd[i]])).reshape((3, ))) #check
                vd[i] = (R[i].T @ vd[i - 1] + np.cross(omegad[i], (p_star[i]).T) + np.cross(omega[i], np.cross(omega[i], (p_star[i]).T))).reshape((3, )) #check
                print(omega[i])
                print(omegad[i])
                print(vd[i])
            else:
                omega[i] = R[i - 1] @ omega[i - 1]  # No angular motion
                omegad[i] = R[i - 1] @ omegad[i - 1]
                vd[i] = R[i - 1] @ vd[i - 1] + R[i] @ np.array([[qdd[i]], [0], [0]])
        a_c[i] = np.cross(omegad[i],  (s_bar[i]).T) + np.cross(omega[i],  np.cross(omega[i],  (s_bar[i]).T)) + vd[i]
        print(a_c[i])
    # Backward recursion: force & torque propagation
    F = np.zeros((n, 3))  # Force
    N = np.zeros((n, 3))  # Torque
    f = [np.zeros(3) for _ in range(n + 1)]  # Force on each link
    n_torque = [np.zeros(3) for _ in range(n + 1)]  # To    rque on each link
    tau = np.zeros(n)  # Joint torques
    
    for i in reversed(range(n)):
        theta, alpha, r, m, I, j_type, b = links[i]
        print(i)
        if i == n-1:
            F[i] = m * a_c[i]
            N[i] = np.dot(Rbase[i].T @ I @ Rbase[i], omegad[i]) + np.cross(omega[i], np.dot(Rbase[i].T @ I @ Rbase[i], omega[i]))
            f[i] = F[i]
            print('------------------ Passing Through i = n - 1 -------------------------------------')
            n_torque[i] = np.cross((p_star[i]).T + (s_bar[i]).T, F[i]) + N[i]
            print(f[i])
            print(n_torque[i])
        else: 
            print('----------------- Passing Through i < n - 1 -------------------------------')
            F[i] = m * a_c[i]
            N[i] = np.dot(Rbase[i].T @ I @ Rbase[i], omegad[i]) + np.cross(omega[i], np.dot(Rbase[i].T @ I @ Rbase[i], omega[i]))
            f[i] = R[i+1] @ f[i+1] + F[i]
            n_torque[i] = (R[i+1] @ (n_torque[i+1] + np.cross((p_star_vector(q, r, i+1, i)).T, f[i+1])).T + (np.cross((p_star[i]).T + (s_bar[i]).T, F[i]) + N[i]).T).reshape(3, )

        if j_type == 1:
            print('--------------------- Passing Through ---------------------------------------')
            tau[i] = np.dot(n_torque[i].reshape(3,), R[i].T @ np.array([0, 0, 1]).T) + b * qd[i]
            print(tau[i])
        else:  # Prismatic
            tau[i] = np.dot(f[i], R[i].T @ np.array([0, 0, 1])) + b * qd[i]    
    
    return tau




def plot_graphs(n, data1, data2):
    plt.figure(figsize=(10, 5))
    for i in range(n):
        plt.subplot(1, n, i+1)    
        plt.plot(time, data1[:, i], label=f'Torque {i} Input')
        plt.plot(time, data2[:, i], '--r', label=f'Torque {i}')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.title('Joint Torques Over Time')
    plt.show()    


# Define the manipulator links: (theta, alpha, length, mass, inertia tensor, joint type: 0 - translational, 1 - rotational, damping coeff.)
# n = 2
n = 4

# links = [
#     (0, 0, 1.0, 1.0, np.diag([0.0, 1/12 * 1, 1/12 * 1]), 1, 0.),  # Link 1
#     (0, 0, 1.0, 1.0, np.diag([0.0, 1/12 * 1, 1/12 * 1]), 1, 0.)   # Link 2
# ]


links = [
    (0, 0, 1.0, 1.0, np.diag([0.0, 1/12 * 1, 1/12 * 1]), 1, 0.),  # Link 1
    (0, 0, 1.0, 1.0, np.diag([0.0, 1/12 * 1, 1/12 * 1]), 1, 0.),   # Link 2
    (0, 0, 1.0, 1.0, np.diag([0.0, 1/12 * 1, 1/12 * 1]), 1, 0.),   # Link 2
    (0, 0, 1.0, 1.0, np.diag([0.0, 1/12 * 1, 1/12 * 1]), 1, 0.)   # Link 2
    
]

# links = [
#     (0, 0, 1.0, 1.0, np.diag([1, 1, 1]), 1, 0.),  # Link 1
#     (0, 0, 1.0, 1.0, np.diag([1, 1, 1]), 1, 0.),
#     (0, 0, 1.0, 1.0, np.diag([1, 1, 1]), 1, 0.)      # Link 2
# ]

time = np.linspace(0, 10, 1000)  # Time steps from 0 to 10 seconds
torques = []

torquesLE = []

q_csv, qd_csv, qdd_csv = load_joint_data('./trajectory_data_gen.csv', n, len(time), 'csv') # './providedForward/rl_multilink_simulation.csv' './data/LEForw.csv'
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

    # Create a dictionary with the data
cols = torques.shape[1] if (hasattr(torques, "ndim") and torques.ndim > 1) else 1
data = {f't{i+1}': (torques[:, i] if cols > 1 else torques[:]) for i in range(cols)}

df = pd.DataFrame(data)
df.to_csv('torquesNE.csv', index=False)



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