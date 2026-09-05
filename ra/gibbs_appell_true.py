import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

# ============================================================
# Equation (14)
# x = ω × s = Ω s
# ============================================================

def tau_input(t):
    return np.array([0.5 * np.sin(3 * t), 0.88 * np.cos(2 * t), 0, 
                     0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0]) # , 0.075 * np.cos(2 * t) 0.88 * np.cos(2 * t), 0.88 * np.cos(2 * t)
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


def skew(v):

    return np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0]
    ])


# ============================================================
# Equation (2)
# Rotation matrix A_i,i-1
# ============================================================

def A_i_im1(theta, alpha = 0):

    c = np.cos(theta)
    s = np.sin(theta)

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ c, -s*ca,  s*sa],
        [ s,  c*ca, -c*sa],
        [ 0,     sa,    ca]
    ])


# ============================================================
# Equation (19)
# w_i vector
# ============================================================

def w_i(alpha = 0):

    return np.array([
        0,
        np.sin(alpha),
        np.cos(alpha)
    ])


# ============================================================
# Equation (20)
# X_i matrix
# ============================================================

def X_i(s_i, a_i, alpha_i = 0):

    ca = np.cos(alpha_i)
    sa = np.sin(alpha_i)

    return np.array([
        [0,             s_i*ca,      -s_i*sa],
        [-s_i*ca,       0,            a_i],
        [s_i*sa,       -a_i,          0]
    ])


# ============================================================
# Equation (24)
# Inertia tensor
# ============================================================

def inertia_rod(m, l):

    Izz = (1/12) * m * l**2

    return np.diag([
        0,
        Izz,
        Izz
    ])


# ============================================================
# Equation (23)
# Gibbs Function
# ============================================================

def gibbs_function(
    Omega,
    Omega_d,
    accel,
    links,
    n_vectors
):

    G = 0.0

    for i in range(len(links)):

        theta, alpha, r, m, J, j_type, b = links[i]

        Om = Omega[i]
        Omd = Omega_d[i]

        a_i = accel[i].reshape(3,1)

        n_i = n_vectors[i].reshape(3,1)

        # ----------------------------------------------------
        # Term 1
        # tr(Omega_dot J Omega_dot^T)
        # ----------------------------------------------------

        term1 = np.trace(
            Omd @ J @ Omd.T
        )

        # ----------------------------------------------------
        # Term 2
        # 2 Omega^2 J Omega_dot^T
        # ----------------------------------------------------

        term2 = 2 * np.trace(
            (Om @ Om) @ J @ Omd.T
        )

        # ----------------------------------------------------
        # Term 3
        # 2 Omega^2 n a^T
        # ----------------------------------------------------

        term3 = 2 * np.trace(
            (Om @ Om) @ n_i @ a_i.T
        )

        # ----------------------------------------------------
        # Term 4
        # 2 Omega_dot n a^T
        # ----------------------------------------------------

        term4 = 2 * np.trace(
            Omd @ n_i @ a_i.T
        )

        # ----------------------------------------------------
        # Term 5
        # m a a^T
        # ----------------------------------------------------

        term5 = m * np.trace(
            a_i @ a_i.T
        )

        G += 0.5 * (
            term1 +
            term2 +
            term3 +
            term4 +
            term5
        )

    return G


# ============================================================
# Main Recursive Gibbs-Appell Algorithm
# Equations (15)-(38)
# ============================================================
def T_matrix(omega, n):
        w1, w2, w3 = omega
        n1, n2, n3 = n

        T = np.array([
            [
                w2 * n2 + w3 * n3,
                -w1 * n2,
                -w1 * n3
            ],
            [
                -w2 * n1,
                w1 * n1 + w3 * n3,
                -w2 * n3
            ],
            [
                -w3 * n1,
                -w3 * n2,
                w1 * n1 + w2 * n2
            ]
        ])

        return T


def gibbs_appell_inverse_dynamics(
    q,
    qd,
    qdd,
    links,
    gravity
):

    N = len(q)

    # ========================================================
    # Storage
    # ========================================================

    omega = [np.zeros(3) for _ in range(N+1)]

    omegad = [np.zeros(3) for _ in range(N+1)]

    accel = [np.zeros(3) for _ in range(N+1)]

    Omega = [np.zeros((3,3)) for _ in range(N+1)]

    Omega_d = [np.zeros((3,3)) for _ in range(N+1)]

    A = [np.eye(3) for _ in range(N+2)]

    beta = [np.zeros(3) for _ in range(N+2)]

    lambd = [np.zeros(3) for _ in range(N+2)]

    tau = np.zeros(N)

    # ========================================================
    # Boundary condition
    # ========================================================

    accel[0] = -gravity

    # ========================================================
    # Link data
    # ========================================================


    J_all = []

    n_vectors = []

    for i in range(N):
        theta, alpha, l, m, J, j_type, b = links[i]


        alpha = 0.0

        theta = q[i]

        J = inertia_rod(m, l)

        n_i = np.array([
            l/2,
            0,
            0
        ])


        J_all.append(J)

        n_vectors.append(n_i)

    # ========================================================
    # FORWARD RECURSION
    # Equations (15)-(18)
    # ========================================================



    for i in range(1, N+1):

        idx = i - 1

        theta, alpha, a_i, m_i, J_i, joint_type, b = links[idx]

        # ----------------------------------------------------
        # Rotation matrix
        # ----------------------------------------------------

        A[i] = A_i_im1(q[idx], alpha)

        # ----------------------------------------------------
        # Equation (19)
        # ----------------------------------------------------

        w = w_i(alpha)

        # ----------------------------------------------------
        # Equation (15)
        # ω_i
        # ----------------------------------------------------

        omega[i] = (
            A[i].T @ omega[i-1]
            + w * qd[idx]
        )

        # ----------------------------------------------------
        # Omega matrix
        # ----------------------------------------------------

        Omega[i] = skew(omega[i])

        # ----------------------------------------------------
        # Equation (16)
        # ωdot_i
        # ----------------------------------------------------

        A_dot = (
            -Omega[i] @ A[i]
        )

        omegad[i] = (
            A[i].T @ omegad[i-1]
            + A_dot.T @ omega[i-1]
            + w * qdd[idx]
        )

        # ----------------------------------------------------
        # Omega_dot
        # ----------------------------------------------------

        Omega_d[i] = skew(omegad[i])

        # ----------------------------------------------------
        # Equation (20)
        # ----------------------------------------------------

        Xmat = X_i(
            qd[idx],
            -l/2,
            alpha
        )

        # ----------------------------------------------------
        # p_i-1,i vector
        # ----------------------------------------------------

        p = np.array([
            a_i,
            0,
            0
        ])

        # ----------------------------------------------------
        # Equation (17)
        # a_i
        # ----------------------------------------------------

        accel[i] = (
            A[i].T @ accel[i-1]
            + Xmat @ omegad[i]
            + (Omega[i] @ Omega[i]) @ p
        )

    # ========================================================
    # Equation (23)
    # Gibbs function
    # ========================================================

    G = gibbs_function(
        Omega[1:],
        Omega_d[1:],
        accel[1:],
        links,
        n_vectors
    )

    # ========================================================
    # BACKWARD RECURSION
    # Equations (35)-(38)
    # ========================================================

    beta[N+1] = np.zeros(3)

    lambd[N+1] = np.zeros(3)

    A[N+1] = np.eye(3)

    for i in reversed(range(1, N+1)):

        idx = i - 1

        theta, alpha, a_i, m_i, J_i, joint_type, b = links[idx]

        # ----------------------------------------------------
        # Equation (20)
        # ----------------------------------------------------

        Xmat = X_i(
            qd[idx],
            -l/2,
            alpha
        )

        # ----------------------------------------------------
        # G_i matrix
        # ----------------------------------------------------

        n_i = n_vectors[idx]

        G_i = m_i * skew(n_i)

        # ----------------------------------------------------
        # F_i matrix
        # ----------------------------------------------------

        F_i = (
            np.trace(J_i) * np.eye(3)
            - J_i
        )

        # ----------------------------------------------------
        # H_i matrix
        # ----------------------------------------------------

        H_i = (
            Omega[i] @ J_i
            - J_i @ Omega[i]
        )

        # ----------------------------------------------------
        # T_i matrix
        # ----------------------------------------------------

        T_i = T_matrix(omega[i], n_i)

        # ----------------------------------------------------
        # Equation (35)
        # ----------------------------------------------------

        beta[i] = (
            beta[i+1] @ A[i+1].T
            - omegad[i] @ G_i
            - omega[i] @ T_i
            - m_i * accel[i]
        )

        # ----------------------------------------------------
        # Equation (36)
        # ----------------------------------------------------

        lambd[i] = (
            lambd[i+1] @ A[i+1].T
            + beta[i] @ Xmat
            - omegad[i] @ F_i
            - omega[i] @ H_i
            - accel[i] @ G_i
        )

        # ----------------------------------------------------
        # Equation (37)
        # ----------------------------------------------------

        tau[idx] = (
            -lambd[i] @ w_i(alpha) + b*qd[idx]
        )

    return (
        tau
    )



def link_data(n):
    links = []
    for i in range(n):
        # if i == 2:
        #     links.append((0, 0, 1, 0.001, np.diag([0, 1/12 * 1, 1/12 * 1]), 1, 0.))
        # else:
            links.append((0, 0, 1.0, 1.0, np.diag([0, 1/12 * 1, 1/12 * 1]), 1, 50.0))
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


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    n = 3

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

    time_step = np.linspace(0, 5, 500)  # Time steps from 0 to 5 seconds
    torques = []

    torques2 = []

    out_dir = './ra/data5s'

    trj_data = f"{out_dir}/trajectory_data_gen{n}.csv"


    # lengths = [1.0, 1.0]

    # masses = [1.0, 1.0]

    gravity = np.array([
        0,
        9.81,        
        0
    ])

    q_csv, qd_csv, qdd_csv = load_joint_data(trj_data, n, len(time_step), 'csv') # './providedForward/rl_multilink_simulation.csv' './data/LEForw.csv'




    def gib(time_step, q_csv, qd_csv, qdd_csv, links, gravity):
        for t_idx in range(len(time_step)): #len(time)
            q = q_csv[t_idx]   # Joint positions from CSV
            qd = qd_csv[t_idx] # Joint velocities from CSV
            qdd = qdd_csv[t_idx] # Joint accelerations from CSV
            torque = gibbs_appell_inverse_dynamics(q, qd, qdd, links, gravity)  # Compute torques using RNEA
            torques.append(torque)
            torque2 = tau_input(time_step[t_idx])
            torques2.append(torque2)
            
        return torques, torques2

    torques, torques2 = gib(time_step, q_csv, qd_csv, qdd_csv, links, gravity)

    
    torques = np.array(torques)
    torques2 = np.array(torques2)

    print("Torques:")
    print(np.shape(torques))

         # Create a dictionary with the data
    cols = torques.shape[1] if (hasattr(torques, "ndim") and torques.ndim > 1) else 1
    data = {f't{i+1}': (torques[:, i] if cols > 1 else torques[:]) for i in range(cols)}

    df = pd.DataFrame(data)

    # torque_data = f"{out_dir}/torquesNE{n}.csv"
    torque_data = f"{out_dir}/torquesGibbs{n}.csv"



    df.to_csv(torque_data, index=False)



    # Plot the torques

    plot_graphs(n, torques2, torques)



# NE LE Hamiltonian Featherstone Methodology -> Simulations, Results Remove project picture, plan for the research, not cite eqs in methodology,
# change a bit graphs