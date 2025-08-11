import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
import os
print(os.getcwd())  # Print the current working directory
print(os.listdir()) # List all files in the current directory
threshold = 1e-13

def Uij(i, j, q, links):  # provides the integration matrix    TO BE CHECKED
    if (j > i):
        return 0
    else:
        theta, alpha, r, m, I, j_type, b = links[i]
        if (i == j) and (j == 0):
            u_deriv =  Q_mat() @ coord_transform_new(j-1, i, q, links)
            # print("-------------------- Passing through Uij ----------------------")
            # print(coord_transform_new(j-1, i, q, links))
            # print(q[i])
            # print(u_deriv)
            return u_deriv
        # if (j == 0) and (i == 1):
        #     u_deriv =  Q_mat() @ coord_transform_to_base(q, links, i)
        #     return u_deriv
        u_deriv = coord_transform_to_base(q, links, j-1) @ Q_mat() @ coord_transform_new(j-1, i, q, links)
        # print('----------------------- Passing through Uij function ----------------------------')
        # print(coord_transform_to_base(q, links, j-1))
        # print(coord_transform_new(j-1, i, q, links))
        # print(u_deriv)
        return u_deriv


def Uijk(i, j, k, q, links):  # provides the double integration matrix   TO BE CHECKED
    if ((i < j) or (i < k)):
        return 0
    elif ((i >= k) and (k >= j)):
        theta, alpha, r, m, I, j_type, b = links[i] 
        # print('--------------------------- Passing through Uijk ----------------------------------')
        # print(np.shape(coord_transform_to_base(q, links, j-1)))
        # print(np.shape(coord_transform_new(j-1, k-1, q, links)))
        # print(k-1)
        # print(i)
        # print(np.shape(coord_transform_new(k-1, i, q, links)))
        # print(f' ----------------------------- U {i} {j} {k} --------------------------')
        uijk_deriv = coord_transform_to_base(q, links, j-1) @ Q_mat() @ coord_transform_new(j-1, k-1, q, links) @ Q_mat() @ coord_transform_new(k-1, i, q, links)
        # print(uijk_deriv)
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
        # print("-------------- Printing ujk and uji --------------------------------")
        # print(f' ----------------------------- U {j} {k} --------------------------')
        # print(ujk)
        # print(ujk)
        # print(uji)
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
    return np.array([0, 0.1 * np.sin(1.5 * t), 0.075 * np.cos(2 * t)]) # 2 * np.sin(0.5 * t), 1.5 * np.cos(1.5 * t)


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
    iden_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ])

    if (i == j) or j == -1:
        return iden_matrix
    
    
    if (i == -1):
        if i == j:
            return iden_matrix
        else :
            return coord_transform_to_base(q, links, j)
    
    if ((j - i) == 1): 
        theta, alpha, r, m, I, j_type, b = links[j]
        res = np.array([
            [np.cos(q[j]), -np.sin(q[j]), 0, r * np.cos(q[j])],
            [np.sin(q[j]), np.cos(q[j]), 0, r * np.sin(q[j])],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        # res[np.abs(res) < threshold] = 0.0
        return res 
    elif (abs(j)-abs(i)) == -1:#check
            theta, alpha, r, m, I, j_type, b = links[j]
            res = np.array([
            [np.cos(q[j]), -np.sin(q[j]), 0, r * np.cos(q[j])],
            [np.sin(q[j]), np.cos(q[j]), 0, r * np.sin(q[j])],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
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
                [np.cos(q[j]), -np.sin(q[j]), 0, r * np.cos(q[j])],
                [np.sin(q[j]), np.cos(q[j]), 0, r * np.sin(q[j])],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            res = res @ temp
            # print('----------------------------- Coordinate Transform -----------------------------------------------')
            # print(k)
            # print(res)
        # res[np.abs(res) < threshold] = 0.0
        return res



    


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

    if (link_idx == 0):
        theta, alpha, r, m, I, j_type, b = links[link_idx]
        res = np.array([
            [np.cos(q[link_idx]), -np.sin(q[link_idx]), 0, r * np.cos(q[link_idx])],
            [np.sin(q[link_idx]), np.cos(q[link_idx]), 0, r * np.sin(q[link_idx])],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        # res[np.abs(res) < threshold] = 0.0
        return res
    if (link_idx >= 1):
        for j in range(link_idx + 1):
            theta, alpha, r, m, I, j_type, b = links[j]
            # print(j)
            if j_type == 1:
                # print(j)
                temp = np.array([
                [np.cos(q[j]), -np.sin(q[j]), 0, r * np.cos(q[j])],
                [np.sin(q[j]), np.cos(q[j]), 0, r * np.sin(q[j])],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
                res = res @ temp
        # res[np.abs(res) < threshold] = 0.0
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
    # R_direct[np.abs(R_direct) < threshold] = 0.0
    return R_direct


def Q_mat():
    return np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

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






def recLag(q, qd, links, gravity):
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
    
    print('************************************* Printing D, h, c *********************************************************')
    D = np.array(D).reshape((n, n))
    print(D)
    h = np.array(h).reshape((n, 1))
    print(h)
    c = np.array(c).reshape((n, 1))
    print(c)
    # tau = D @ np.array(qdd).reshape((n, 1)) +  h + c

    return D, h ,c


def dynamics(t, y):
    theta = y[:n]
    thetadot = y[n:]
    
    # Dmat = D(theta)
    # hvec = h(theta, thetadot)
    # cvec = c(theta)
    tauvec = np.array(tau_input(t)).reshape((n, 1))

    Dmat, hvec, cvec = recLag(theta, thetadot, links, gravity)

    theta_ddot = np.linalg.solve(Dmat, tauvec - hvec - cvec).flatten()

    return np.concatenate([thetadot, theta_ddot])



# 11.07.25
# try turning off variables D, h, c individually
# figure out which matrix generates oscillations
# run the 2, 3, 5 links system for LE, RNEA, based on the obtained data
# 





m1 = 1
m2 = 1
l1 = 1
l2 = 1

n = 3

g = 9.81
gravity = np.array([[0, -g, 0, 0]])

# Define the manipulator links: (theta, alpha, length, mass, inertia tensor, joint type: 0 - translational, 1 - rotational, damping coeff.)

links = [
    (0, 0, l1, m1, np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]), 1, 0.),  # Link 1
    (0, 0, l2, m2, np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]), 1, 0.),
    (0, 0, l2, m2, np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]), 1, 0.)
]

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
    ]), 1, 0.)   # Link 2
]


'''

'''
links = [
    (0, 0, 1, 1, np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]), 1, 0.),  # Link 1
    (0, 0, 1, 1, np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]), 1, 0.),   # Link 2
    (0, 0, 1, 1, np.array([
        [m1 * l1**2, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, m1],
    ]), 1, 0.)
]
'''
'''
links = [(0, 0, l1, m1, np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]), 1, 0.)]
'''
time = np.linspace(0, 10, 1001)  # Time steps from 0 to 10 seconds

t_span = (0, 10)
n_steps = 1000
t_eval = np.linspace(*t_span, n_steps)

# Initial conditions: [theta1, theta2, theta1_dot, theta2_dot]
y0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

sol = solve_ivp(dynamics, t_span, y0, t_eval=t_eval, method='RK45')


thetaddot = np.zeros((n, len(sol.t)))

for i in range(len(sol.t)):
    y_i = sol.y[:, i]
    theta_i = y_i[0:n]
    thetadot_i = y_i[n:n+n]
    tauvec_i = np.array(tau_input(sol.t[i])).reshape((n, 1))
    Dmat_i, hvec_i, cvec_i = recLag(theta_i, thetadot_i, links, gravity)
    thetaddot[:, i] = np.linalg.solve(Dmat_i, tauvec_i - hvec_i - cvec_i).flatten()


print(np.shape(sol.y[0]))
print(np.shape(thetaddot))

plt.plot(sol.t, thetaddot[0, :], label='θ₁')
plt.plot(sol.t, thetaddot[1, :], label='θ₂')
plt.plot(sol.t, thetaddot[2, :], label='θ3')
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.title('Forward Dynamics Simulation')
plt.legend()
plt.grid()
plt.show()


data = np.vstack([
    sol.y[0], sol.y[1], sol.y[2],       # theta1, theta2
    sol.y[3], sol.y[4], sol.y[5],       # thetadot1, thetadot2
    thetaddot[0], thetaddot[1], thetaddot[2]   # thetaddot1, thetaddot2
]).T  # Transpose to shape (n_points, 7)

# Create DataFrame
df = pd.DataFrame(data, columns=[
    "theta1", "theta2", "theta3",
    "thetadot1", "thetadot2", "thetadot3",
    "thetaddot1", "thetaddot2", "thetaddot3"
])

# Save to CSV
df.to_csv("./data/LEForw.csv", index=False)
# Plot the torques

# plt.figure(figsize=(15, 5))


# plt.subplot(1, 3, 1)
# plt.plot(time, torques[:, 0], '--b', label='Torque 1')
# plt.plot(time, torquesLE[:, 0], label='Torque 1 Input')
# plt.plot(time, torques[:, 1], '--g', label='Torque 2')
# plt.plot(time, torquesLE[:, 1], label='Torque 2 Input')
# plt.xlabel('Time (s)')
# plt.ylabel('Torque (Nm)')
# plt.legend()
# plt.title('Joint Torques Over Time')

# plt.subplot(1, 2, 1)
# plt.plot(time, torques[:, 0], label='Torque 1')
# plt.plot(time, torquesLE[:, 0], '--r', label='Torque 1 Input')
# plt.xlabel('Time (s)')
# plt.ylabel('Torque (Nm)')
# plt.legend()
# plt.title('Joint Torques Over Time')

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
# plt.show()





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




# 17.07.25
# RNEA modified for general cases: for n-number of links, and for revolute joints and prismatic joints
# Test RNEA and LE
# Use data from forward simulation and use for RNEA and LE: from n=1 ... 5
# 
# 
# 23.07.25
# for each sample time, calculate the matrices M, C, G - D, C, H
# extract values of these matrices
# separate code for D, C, H


# make sure that the absolute angles are used
# Dias' code may be with general frame
# make sure that it is with local frame 



# 29.07.25
# forward simu for 3-link 
# x = [ q qd] xd = [qd qdd] = [qd -M-1 (C*qd + D*qd + G - tau) ]
# Put coefficients for 
# gammai = {1, 0}
# General NE
# 