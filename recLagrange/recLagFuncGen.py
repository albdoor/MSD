import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
print(os.getcwd())  # Print the current working directory
print(os.listdir()) # List all files in the current directory


def Uij(i, j, q, links):  # provides the integration matrix    TO BE CHECKED
    if (j > i):
        return 0
    else:
        theta, alpha, r, m, I, j_type, b = links[i]
        if (i == j) and (j == 0):
            u_deriv =  Q_mat() @ coord_transform_new(j-1, i, q, links)
            # print(u_deriv)
            return u_deriv
        # if (j == 0) and (i == 1):
        #     u_deriv =  Q_mat() @ coord_transform_to_base(q, links, i)
        #     return u_deriv
        u_deriv = coord_transform_to_base(q, links, j-1) @ Q_mat() @ coord_transform_new(j-1, i, q, links)
        # print(u_deriv)
        return u_deriv


def Uijk(i, j, k, q, links):  # provides the double integration matrix   TO BE CHECKED
    if ((i < j) or (i < k)):
        return 0
    elif ((i >= k) and (k >= j)):
        theta, alpha, r, m, I, j_type, b = links[i] 
        uijk_deriv = coord_transform_to_base(q, links, j-1) @ Q_mat() @ coord_transform_new(j-1, k-1, q, links) @ Q_mat() @ coord_transform_new(k-1, i, q, links)
        return uijk_deriv
    elif ((i >= j) and (j >= k)):
        theta, alpha, r, m, I, j_type, b = links[i]     
        uijk_deriv = coord_transform_to_base(q, links, k-1) @ Q_mat() @ coord_transform_new(k-1, j-1, q, links) @ Q_mat() @ coord_transform_new(j-1, i, q, links)
        return uijk_deriv


def d_func(i, k, links, q):
    sum = 0
    j = max(i, k)
 
    for j in range(j, len(links)):
        theta, alpha, r, m, I, j_type, b = links[j]

        inertia_tensor = I
        ujk = Uij(j, k, q, links)

        uji = Uij(j, i, q, links)
  
        sum += np.trace(ujk @ inertia_tensor @ uji.T) 
    return sum

def hikm_func(i, k, m, links, q):
    sum = 0
    j = max(i, k, m)
    for j in range(j, len(links)):
        theta, alpha, r, mass, I, j_type, b = links[j]
        inertia_tensor = I
        ujkm = Uijk(j, k, m, q, links)
        uji = Uij(j, i, q, links)
        sum += np.trace(ujkm @ inertia_tensor @ uji.T)
    return sum

def h_func(i, links, q, qd):
    m = 0
    k = 0
    res = 0
    for k in range(len(links)):
        for m in range(len(links)):
           res += hikm_func(i, k, m, links, q) * qd[k] * qd[m] 
    
    return res

def c_func(i, links, g, q):
    j = i
    res = 0
    for j in range(i, len(links)):
        theta, alpha, r, m, I, j_type, b = links[j]
        uji = Uij(j, i, q, links)
        rmid = np.array([[-r/2], [0], [0], [1]])
        res += (-m * g @ uji @ rmid)
    return res    



def tau_input(t): # initial input used
    return np.array([2 * np.sin(0.5 * t), 2 * np.sin(0.5 * t)])


def load_joint_data(npy_filename, n, time_int, filetype):
    # data = np.load(npy_filename, allow_pickle=True)
    # data = data[0] if isinstance(data[0], list) else data
    # data = np.loadtxt(npy_filename, delimiter=',')
    # data = data.T
    # q = np.vstack((data[0], data[1])).T
    # qd = np.vstack((data[2], data[3])).T 
    # qdd = np.vstack((data[4], data[5])).T 
    # return q, qd, qdd
    if filetype == 'csv':
        data = np.loadtxt(npy_filename, delimiter=',', skiprows=1)
        data = data.T
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


def coord_transform_new(i, j, q, links):
    '''
    if (abs(j - i) > 1):
        id_matr = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ])
        theta, alpha, r, m, I, j_type, b = links[j]
        for k in range(i, j):
            id_matr = id_matr @ coord_transform_new(k, k + 1, q, links)
        return id_matr
    '''
    if ((j - i) == 1): 
        theta, alpha, r, m, I, j_type, b = links[j]
        return np.array([
            [np.cos(q[j]), -np.sin(q[j]), 0, r * np.cos(q[j])],
            [np.sin(q[j]), np.cos(q[j]), 0, r * np.sin(q[j])],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    elif j-i == -1:
            theta, alpha, r, m, I, j_type, b = links[j]
            return np.array([
            [np.cos(q[j]), -np.sin(q[j]), 0, r * np.cos(q[j])],
            [np.sin(q[j]), np.cos(q[j]), 0, r * np.sin(q[j])],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]).T

    if (i == j):
        return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ])

    if (i == -1):
        return coord_transform_to_base(q, links, j)  

# j-i == 1 review
# review the above function

def coord_transform(theta, l):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, l * np.cos(theta)],
        [np.sin(theta), np.cos(theta), 0, l * np.sin(theta)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def coord_transform_to_base(q, links, link_idx):
    cumul_angle = 0
    res = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ])

    if (link_idx == -1):
        return res
        
        
    for j in range(link_idx + 1):
        theta, alpha, r, m, I, j_type, b = links[j]
        # print(j)
        if j_type == 1:
            # print(j)
            res = res @ coord_transform_new(j - 1, j, q, links)
    return res

            # the = q[i]
            # cumul_angle += the
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






def recLag(q, qd, qdd, links, gravity):
    n = len(links)  # number of links
    tau = np.zeros(n)  # joint torques
    D = [] # D matrices
    h = []
    c = []

    i = 0
    for i in range(n):
        h.append(h_func(i, links, q, qd))
        c.append(c_func(i, links, gravity, q))
        for k in range(n):
            D.append(d_func(i, k, links, q))
    
    D = np.array(D).reshape((n, n))
    
    h = np.array(h).reshape((n, 1))
    
    c = np.array(c).reshape((n, 1))

    tau = D @ np.array(qdd).reshape((n, 1)) + h + c

    return tau





m1 = 1
m2 = 1
l1 = 1
l2 = 1

# Define the manipulator links: (theta, alpha, length, mass, inertia tensor, joint type: 0 - translational, 1 - rotational, damping coeff.)
'''
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
    ]), 1, 0.)
]
'''
'''
links = [
    (0, 0, 1, 1, np.array([
        [1/3 * m1 * l1**2, 0, 0, -0.5 * m1 * l1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-0.5 * m1 * l1, 0, 0, m1],
    ]), 1, 0.),  # Link 1
    (0, 0, 1, 1, np.array([
        [1/3 * m2 * l2**2, 0, 0, -0.5 * m2 * l2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-0.5 * m2 * l2, 0, 0, m2],
    ]), 1, 0.),   # Link 2
    (0, 0, 1, 1, np.array([
        [1/3 * m1 * l1**2, 0, 0, -0.5 * m1 * l1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-0.5 * m1 * l1, 0, 0, m1],
    ]), 1, 0.),
    (0, 0, 1, 1, np.array([
        [1/3 * m1 * l1**2, 0, 0, -0.5 * m1 * l1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-0.5 * m1 * l1, 0, 0, m1],
    ]), 1, 0.),
    (0, 0, 1, 1, np.array([
        [1/3 * m1 * l1**2, 0, 0, -0.5 * m1 * l1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-0.5 * m1 * l1, 0, 0, m1],
    ]), 1, 0.)
]
'''
links = [(0, 0, l1, m1, np.array([
        [1/3 * m1 * l1**2, 0, 0, -0.5 * m1 * l1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-0.5 * m1 * l1, 0, 0, m1],
    ]), 1, 0.)]

time = np.linspace(0, 10, 1000)  # Time steps from 0 to 10 seconds
torques = []

torquesLE = []

n = 1

q_csv, qd_csv, qdd_csv = load_joint_data('./data/singlelinkLE.npy', n, len(time), 'npy')
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

#range(2): 

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
plt.show()

'''
plt.subplot(1, 2, 2)
plt.plot(time, torques[:, 1], label='Torque 2')
plt.plot(time, torquesLE[:, 1], '--r', label='Torque 2 Input')
# plt.plot(time, q_csv[:, 1], label='q2')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.title('Joint Torques Over Time')
'''





# 5-link planar pendulum for both NE and LE
# non-zero input (torque) only for the first link
# zero for others


# 27.06.25
# 1 link system for NE and LE 
# And then 2 link system 


# two-link system
# reduce the mass of 2nd link gradually
# 1/2 of the 1sy mass, 1/4, and so forth
# check the iteration of the link code
# check the index notation
# try debugging line-by-line
# search in the internet 