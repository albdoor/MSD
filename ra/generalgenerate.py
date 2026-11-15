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
    plots = (
        (q, r'$q$', 'Position vs Time', r'$q$ (rad)', None),
        (qd, r'$\dot{q}$', 'Velocity vs Time', r'$\dot{q}$ (rad/s)', None),
        (qdd, r'$\ddot{q}$', 'Acceleration vs Time', r'$\ddot{q}$ (rad/s²)', (-5, 5)),
    )

    for values, label_prefix, title, ylabel, ylim in plots:
        plt.figure(figsize=(10, 6))
        for i in range(n):
            plt.plot(t, values[i, :], label=f'{label_prefix}{i+1}')

        plt.title(title, fontsize=22, fontweight='bold')
        plt.xlabel('Time (s)', fontsize=18, fontweight='bold')
        plt.ylabel(ylabel, fontsize=18, fontweight='bold')
        if ylim is not None:
            plt.ylim(*ylim)
        plt.legend()
        plt.grid(True, which='major', linewidth=1.5, alpha=0.8)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.tight_layout()

    plt.show()
if __name__ == "__main__":
    # Time parameters
    t_start = 0
    t_end = 5
    n_points = t_end * 1000  # 100 points per second
    n_links = 2  # Number of links
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
    # df.to_csv(f"./ra/trajectory_data_gen{n_links}.csv", index=False)

    # Plot the results
    graphs(q, qd, qdd, t)