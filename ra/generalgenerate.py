import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def qs(t, n):
    q = np.zeros((n, len(t)))
    for i in range(n):
        q[i, :] = np.sin((i + 1) * t)  # Sinusoidal function for position
    return q

def qds(t, n):
    qd = np.zeros((n, len(t)))
    for i in range(n):
        qd[i, :] = (i + 1) * np.cos((i + 1) * t)  # Derivative of sinusoidal for velocity
    return qd

def qdds(t, n):
    qdd = np.zeros((n, len(t)))
    for i in range(n):
        qdd[i, :] = -(i + 1) ** 2 * np.sin((i + 1) * t)  # Second derivative for acceleration
    return qdd

def generate_trajectory(t, n):
    q = qs(t, n)
    qd = qds(t, n)
    qdd = qdds(t, n)
    return q, qd, qdd

def graphs(q, qd, qdd, t):
    n = q.shape[0]
    plt.figure(figsize=(12, 8))

    for i in range(n):
        plt.subplot(3, 1, 1)
        plt.plot(t, q[i, :], label=f'q{i+1}')
        
        plt.subplot(3, 1, 2)
        plt.plot(t, qd[i, :], label=f'qd{i+1}')
        
        plt.subplot(3, 1, 3)
        plt.plot(t, qdd[i, :], label=f'qdd{i+1}')

    plt.subplot(3, 1, 1)
    plt.title('Position vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (rad)')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.title('Velocity vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
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
    n_links = 4  # Number of links
    t = np.linspace(t_start, t_end, n_points)

    # Generate trajectory
    q, qd, qdd = generate_trajectory(t, n_links)

    # Create a dictionary with the data
# ...existing code...
    # Create a dictionary with the data in desired column order
    data = {}
    # First all positions
    for i in range(n_links):
        data[f'q{i+1}'] = q[i, :]
    # Then all velocities 
    for i in range(n_links):
        data[f'qd{i+1}'] = qd[i, :]
    # Finally all accelerations
    for i in range(n_links):
        data[f'qdd{i+1}'] = qdd[i, :]

    # Create a pandas DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv('trajectory_data_gen.csv', index=False)

    # Plot the results
    graphs(q, qd, qdd, t)