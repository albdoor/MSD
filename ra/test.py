import numpy as np
import matplotlib.pyplot as plt


from recLagFuncGen import recLag, load_joint_data, link_data as rlg_link_data, plot_graphs
from rneagen import rnea, link_data as rnea_link_data
from featherstone import featherstone_id, link_data as fst_link_data

# def plot_graphs_torque(n, torques_le, torques_ne, torques_fst, time):
#     lw = 2.5           # line width
#     fs = 16
#     for i in range(n):
#         plt.subplot(n, 1, i+1)
#         plt.plot(time, torques_le[f't{i+1}'], 'b-', label='Lagrange', linewidth=lw)
#         plt.plot(time, torques_ne[f't{i+1}'], 'r--', label='Newton-Euler', linewidth=lw)
#         plt.plot(time, torques_fst[f't{i+1}'], 'g-.', label='Featherstone', linewidth=lw)
#         plt.title(f'Joint {i+1} Torque Comparison', fontsize=fs)
#         plt.xlabel('Time (s)', fontsize=fs-1)
#         plt.ylabel('Torque (Nm)', fontsize=fs-1)
#         plt.legend(fontsize=fs-5)
#         plt.grid(True)

#     plt.tight_layout()


if __name__ == "__main__":

    n = 2 
    m1 = 1.0
    l1 = 1.0
    g = 9.81

    time_step = np.linspace(0, 10, 1000)  # Time steps from 0 to 10 seconds

    torques = []

    torquesLE = []


    out_dir = './ra'

    trj_data = f"{out_dir}/trajectory_data_gen{n}.csv"

    q_csv, qd_csv, qdd_csv = load_joint_data(trj_data, n, len(time_step), 'csv') # './providedForward/rl_multilink_simulation.csv' './data/LEForw.csv'
    # './providedForwardMod/rl_multilink_simulation2.csv'

    links_rnea = rnea_link_data(n)
    links_fst = fst_link_data(n)
    links_rlg = rlg_link_data(n)


    g_ftst = np.array([0, g, 0])
    g_rlg = np.array([[0, -g, 0, 0]])
    g_rnea = np.array([0, g, 0])
    torques_rlg = []
    torques_rnea = []
    torques_ftst = []


    for t_idx in range(5): #len(time)
        q = q_csv[t_idx]   # Joint positions from CSV
        qd = qd_csv[t_idx] # Joint velocities from CSV
        qdd = qdd_csv[t_idx] # Joint accelerations from CSV
        print('Performing FTST')
        torques_ftst.append(featherstone_id(q, qd, qdd, links_fst, g_ftst))  # Compute torques using FTST
        print('Performing RNEA')
        torques_rnea.append(rnea(q, qd, qdd, links_rnea, g_rnea))  # Compute torques using RNEA
        print('Performing RLG')
        torques_rlg.append(recLag(q, qd, qdd, links_rlg, g_rlg))  # Compute torques using RLG

    torques_ftst = np.array(torques_ftst)
    torques_rnea = np.array(torques_rnea)
    torques_rlg = np.array(torques_rlg)
    
    print('RNEA: ')
    print(torques_rnea)
    print('FTST: ')
    print(torques_ftst)
    print('RLG: ')
    print(torques_rlg)





