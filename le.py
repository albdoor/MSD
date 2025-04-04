import numpy as np
import matplotlib.pyplot as plt

# Define system parameters
m = 1.0    # Mass of the link (kg)
l = 1.0    # Length of the link (m)
I = 0.1    # Moment of inertia (kg.m^2)
g = 9.81   # Gravity acceleration (m/s^2)

# Define simulation parameters
T = 10     # Total simulation time (s)
dt = 0.01  # Time step
time = np.arange(0, T, dt)  # Time vector

# Initialize state variables
q = np.zeros(len(time))    # Joint angle (rad)
qd = np.zeros(len(time))   # Joint velocity (rad/s)
qdd = np.zeros(len(time))  # Joint acceleration (rad/s²)

# Initial conditions
q[0] = 0 * np.pi/180 # 0 degrees initial angle
qd[0] = 0.0       # Initial angular velocity

# Define sinusoidal torque input
def torque_input(t):
    return 2 * np.sin(0.5 * t)  # Example sinusoidal torque

# Forward dynamics simulation using Euler integration
for i in range(len(time) - 1):
    tau = torque_input(time[i])  # Compute torque at time t
    qdd[i] = (tau - m * g * l * np.sin(q[i])) / I  # Compute angular acceleration
    qd[i + 1] = qd[i] + qdd[i] * dt  # Integrate velocity
    q[i + 1] = q[i] + qd[i] * dt  # Integrate position

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.plot(time, q, label="q (rad)")
plt.xlabel("Time (s)")
plt.ylabel("Joint Angle (rad)")
plt.legend()
plt.grid()

plt.subplot(1, 3, 2)
plt.plot(time, qd, label="qd (rad/s)", color="g")
plt.xlabel("Time (s)")
plt.ylabel("Joint Velocity (rad/s)")
plt.legend()
plt.grid()

plt.subplot(1, 3, 3)
plt.plot(time, qdd, label="qdd (rad/s²)", color="r")
plt.xlabel("Time (s)")
plt.ylabel("Joint Acceleration (rad/s²)")
plt.legend()
plt.grid()

plt.suptitle("Single-Link Manipulator: Forward Dynamics (Lagrange-Euler)")
plt.show()

np.save('./data/singlelinkLE', np.array([q, qd, qdd]))














































# import numpy as np
# import matplotlib.pyplot as plt

# # Define system parameters (link lengths, masses, inertia)
# m1, m2 = 1.0, 1.0  # Mass of links
# l1, l2 = 1.0, 1.0  # Length of links
# lc1, lc2 = l1 / 2, l2 / 2  # Center of mass positions
# I1, I2 = 0.1, 0.1  # Moments of inertia
# g = 9.81  # Gravity acceleration

# # Compute Mass Matrix M(q)
# def M_matrix(q):
#     th1, th2 = q
#     M11 = I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(th2))
#     M12 = I2 + m2 * (lc2**2 + l1 * lc2 * np.cos(th2))
#     M21 = M12
#     M22 = I2 + m2 * lc2**2
#     return np.array([[M11, M12], [M21, M22]])

# # Compute Coriolis and Centrifugal Matrix C(q, qd)
# def C_matrix(q, qd):
#     th1, th2 = q
#     th1d, th2d = qd
#     C11 = -m2 * l1 * lc2 * np.sin(th2) * th2d
#     C12 = -m2 * l1 * lc2 * np.sin(th2) * (th1d + th2d)
#     C21 = m2 * l1 * lc2 * np.sin(th2) * th1d
#     C22 = 0
#     return np.array([[C11, C12], [C21, C22]])

# # Compute Gravity Vector G(q)
# def G_vector(q):
#     th1, th2 = q
#     G1 = (m1 * g * lc1 + m2 * g * l1) * np.cos(th1) + m2 * g * lc2 * np.cos(th1 + th2)
#     G2 = m2 * g * lc2 * np.cos(th1 + th2)
#     return np.array([G1, G2])

# # Forward Dynamics: Compute qdd given q, qd, and tau
# def forward_dynamics(q, qd, tau):
#     M = M_matrix(q)
#     C = C_matrix(q, qd)
#     G = G_vector(q)
#     qdd = np.linalg.inv(M) @ (tau - C @ qd - G)
#     return qdd

# # Simulation settings
# time = np.linspace(0, 10, 1000)  # 1000 time steps over 10 seconds
# dt = time[1] - time[0]  # Time step
# q = np.zeros((len(time), 2))  # Joint angles
# qd = np.zeros((len(time), 2))  # Joint velocities
# qdd = np.zeros((len(time), 2))  # Joint accelerations

# # Initial conditions
# q[0] = np.array([np.pi / 4, np.pi / 6])  # Initial angles (45 and 30 degrees)
# qd[0] = np.array([0, 0])  # Initial velocities

# # Simulated joint torques (e.g., sinusoidal torque input)
# def tau_input(t):
#     return np.array([2 * np.sin(0.5 * t), 1.5 * np.cos(0.5 * t)])

# # Forward dynamics simulation
# for i in range(len(time) - 1):
#     tau = tau_input(time[i])  # Get input torques
#     qdd[i] = forward_dynamics(q[i], qd[i], tau)  # Compute accelerations
#     qd[i + 1] = qd[i] + qdd[i] * dt  # Integrate velocity
#     q[i + 1] = q[i] + qd[i] * dt  # Integrate position

# np.save('./data/doublelinkLE', np.array([q[:, 0], q[:, 1], qd[:, 0], qd[:, 1], qdd[:, 0], qdd[:, 1]]))

# # Plot results
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 3, 1)
# plt.plot(time, q[:, 0], label="Theta1 (rad)")
# plt.plot(time, q[:, 1], label="Theta2 (rad)")
# plt.xlabel("Time (s)")
# plt.ylabel("Joint Angles")
# plt.legend()
# plt.grid()

# plt.subplot(1, 3, 2)
# plt.plot(time, qd[:, 0], label="Theta1d (rad/s)")
# plt.plot(time, qd[:, 1], label="Theta2d (rad/s)")
# plt.xlabel("Time (s)")
# plt.ylabel("Joint Velocities")
# plt.legend()
# plt.grid()

# plt.subplot(1, 3, 3)
# plt.plot(time, qdd[:, 0], label="Theta1dd (rad/s²)")
# plt.plot(time, qdd[:, 1], label="Theta2dd (rad/s²)")
# plt.xlabel("Time (s)")
# plt.ylabel("Joint Accelerations")
# plt.legend()
# plt.grid()

# plt.suptitle("2-Link Manipulator: Forward Dynamics (Lagrange-Euler)")
# plt.show()
