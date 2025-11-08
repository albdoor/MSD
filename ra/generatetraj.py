import numpy as np
import pandas as pd

def qs(t):
    return np.array([0.5 * np.sin(3 * t), 0.88 * np.cos(2 * t), 1 * np.sin(t)]) # , 0.075 * np.cos(2 * t) 0.88 * np.cos(2 * t), 0.88 * np.cos(2 * t)


def qds(t):
    return np.array([0.5 * 3 * np.cos(3 * t), -0.88 * 2 * np.sin(2 * t), 1 * np.cos(t)]) # , 0.075 * np.cos(2 * t) 0.88 * np.cos(2 * t), 0.88 * np.cos(2 * t)


def qdds(t):
    return np.array([0.5 * 3 * -3 * np.sin(3 * t), -0.88 * 2 * 2 * np.cos(2 * t), -1 * np.sin(t)])

def generate_trajectory(t, n):
    q = qs(t)
    qd = qds(t)
    qdd = qdds(t)
    return q, qd, qdd



def graphs(q, qd, qdd, t):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, q[0, :], label='q1')
    plt.plot(t, q[1, :], label='q2')
    plt.plot(t, q[2, :], label='q3')
    plt.title('Position vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (rad)')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(t, qd[0, :], label='qd1')
    plt.plot(t, qd[1, :], label='qd2')
    plt.plot(t, qd[2, :], label='qd3')
    plt.title('Velocity vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(t, qdd[0, :], label='qdd1')
    plt.plot(t, qdd[1, :], label='qdd2')
    plt.plot(t, qdd[2, :], label='qdd3')
    plt.title('Acceleration vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (rad/s²)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Time parameters
    t_start = 0
    t_end = 10
    n_points = 1000
    t = np.linspace(t_start, t_end, n_points)

    # Generate trajectory
    q = np.zeros((3, n_points))
    qd = np.zeros((3, n_points))
    qdd = np.zeros((3, n_points))

    for i in range(n_points):
        q[:, i], qd[:, i], qdd[:, i] = generate_trajectory(t[i], 3)


    # Create a dictionary with the data
    data = {
        'q1': q[0, :],
        'q2': q[1, :],
        'q3': q[2, :],
        'qd1': qd[0, :],
        'qd2': qd[1, :],
        'qd3': qd[2, :],
        'qdd1': qdd[0, :],
        'qdd2': qdd[1, :],
        'qdd3': qdd[2, :]
    }

    # Create a pandas DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv('trajectory_data.csv', index=False)

    # Plot the results
    graphs(q, qd, qdd, t)