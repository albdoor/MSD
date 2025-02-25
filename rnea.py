import numpy as np
import matplotlib.pyplot as plt

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
        theta, alpha, r, m, I = links[i]
        R.append(rotation_matrix(q[i], alpha))
        if i == 0:
            omega[i] = np.array([0, 0, qd[i]])
            omegad[i] = np.array([0, 0, qdd[i]])
            vd[i] = gravity
            # print(vd[i])
        else:
            omega[i] = R[i - 1] @ omega[i - 1] + np.array([0, 0, qd[i]])
            omegad[i] = R[i - 1] @ (R[i] @ omegad[i - 1] + np.dot(omega[i], np.array([0, 0, qd[i]])) + np.array([0, 0, qdd[i]]))
            vd[i] = R[i - 1] @ vd[i - 1] + np.dot(rotation_matrix_n(i, [q[i-1], q[i]], [0, 0]) @ omegad[i], [r, 0, 0])
            vd[i] += np.dot(rotation_matrix_n(i, [q[i-1], q[i]], [0, 0]) @ omega[i], np.dot(omega[i], [r, 0, 0]))
            a_c[i] = rotation_matrix_n(i, [q[i-1], q[i]], [0, 0]) @ vd[i] + np.dot(rotation_matrix_n(i, [q[i-1], q[i]], [0, 0]) @ omegad[i], [r/2, 0, 0]) 
            + np.dot(rotation_matrix_n(i, [q[i-1], q[i]], [0, 0]) @ omega[i], np.dot(omega[i], [r/2, 0, 0]))
    # Backward recursion: force & torque propagation
    F = np.zeros((n, 3))  # Force
    N = np.zeros((n, 3))  # Torque

    for i in reversed(range(n)):
        theta, alpha, r, m, I = links[i]
        F[i] = m * a_c[i]
        N[i] = I @ omegad[i] + np.dot(omega[i], I @ omega[i])
        if i < n - 1:
            F[i] += R[i].T @ F[i + 1]
            N[i] += R[i].T @ N[i + 1] + np.dot([r, 0, 0], R[i].T @ F[i + 1])
        tau[i] = N[i][2]  # Torque around z-axis
    
    
    
    return tau

# Define the manipulator links: (theta, alpha, length, mass, inertia tensor)
links = [
    (0, 0, 1.0, 5.0, np.diag([0.1, 0.1, 0.2])),  # Link 1
    (0, 0, 1.0, 1.0, np.diag([0.1, 0.1, 0.2]))   # Link 2
]

time = np.linspace(0, 10, 500)  # Time steps from 0 to 20 seconds
torques = []

def random_q(t):
    return np.sin(t) + 0.5 * np.cos(0.5 * t)

def random_qd(t):
    return np.cos(t) - 0.25 * np.sin(0.5 * t)

def random_qdd(t):
    return -np.sin(t) - 0.125 * np.cos(0.5 * t)

g = 9.81
gravity = np.array([0, -g, 0])

for t in time:
    q = np.array([random_q(t), random_q(t)])
    qd = np.array([random_qd(t), random_qd(t)])
    qdd = np.array([random_qdd(t), random_qdd(t)])
    torque = rnea(q, qd, qdd, links, gravity)
    torques.append(torque)

torques = np.array(torques)

# Plot the torques
plt.figure(figsize=(10, 5))
plt.plot(time, torques[:, 0], label='Torque 1')
plt.plot(time, torques[:, 1], label='Torque 2')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.title('Joint Torques Over Time')
plt.show()


# Add options for the joints
# verification simulation for double pendulum
# The torque is external
# To input torque as some sine function