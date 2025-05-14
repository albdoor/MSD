import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def coord_transform_new(i, j, q, links):
    if (abs(i - j) == 1):
        theta, alpha, r, m, I, j_type, b = links[j]
        return np.array([
            [np.cos(q[j]), -np.sin(q[j]), 0, r * np.cos(q[j])], 
            [np.sin(q[j]), np.cos(q[j]), 0, r * np.sin(q[j])], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]])
    if (i == j):
        return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ])

    if (i == -1):
        return coord_transform_to_base(q, links, j)  

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



q = np.array([-14.2751, -0.2323])
qd = np.array([-0.50962526, -16.63187589])

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

def Q_mat():
    return np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])


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


def c_func(i, links, g, q):
    j = i
    res = 0
    print(j)
    for j in range(i, len(links)):
        print(j)
        theta, alpha, r, m, I, j_type, b = links[j]
        uji = Uij(j, i, q, links)
        print(uji)
        rmid = np.array([[-r/2], [0], [0], [1]])
        res += (-m * g @ uji @ rmid)
        print(res)
    return res  


def d_func(i, k, links, q):
    sum = 0
    j = max(i, k)
 
    for j in range(j, len(links)):
        theta, alpha, r, m, I, j_type, b = links[j]
        print(j)
        inertia_tensor = I
        print(inertia_tensor)
        ujk = Uij(j, k, q, links)
        print(ujk)

        uji = Uij(j, i, q, links)
        print(uji)
        sum += np.trace(ujk @ inertia_tensor @ uji.T) 
    return sum




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



gravity = np.array([[0, -9.81, 0, 0]])

# print(coord_transform_to_base(q, links, 1))
# print(coord_transform_new(-1, 0, q, links))

print(h_func(0, links, q, qd))

