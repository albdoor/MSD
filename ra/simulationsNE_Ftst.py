import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from datetime import datetime




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
    execution_times = []  # Store execution times and link count
    
    for n in range(2, 21):
        m1 = 1.0
        l1 = 1.0
        g = 9.81
        t_end = 5
        time_step = np.linspace(0, t_end, t_end * 100)  # Time steps from 0 to 50 seconds

        torques = []

        torquesLE = []


        out_dir = './data5s'

        trj_data = f"{out_dir}/trajectory_data_gen{n}.csv"

        q_csv, qd_csv, qdd_csv = load_joint_data(trj_data, n, len(time_step), 'csv') # './providedForward/rl_multilink_simulation.csv' './data/LEForw.csv'
        # './providedForwardMod/rl_multilink_simulation2.csv'

        links_rnea = rnea_link_data(n)
        links_fst = fst_link_data(n)
        links_rlg = rlg_link_data(n)


        g_ftst = np.array([0, -g, 0])
        g_rlg = np.array([[0, -g, 0, 0]])
        g_rnea = np.array([0, g, 0])
        torques_rlg = []
        torques_rnea = []
        torques_ftst = []
        time_elapsed = []
        
        print('Performing FTST')
        t_total_start = time.perf_counter()
        
        for t_idx in range(len(time_step)): #len(time)
            q = q_csv[t_idx]   # Joint positions from CSV
            qd = qd_csv[t_idx] # Joint velocities from CSV
            qdd = qdd_csv[t_idx] # Joint accelerations from CSV
            torques_ftst.append(featherstone_id(q, qd, qdd, links_fst, g_ftst))  # Compute torques using FTST

        t_total_end = time.perf_counter()
        elapsed = t_total_end - t_total_start


        time_elapsed.append(elapsed)
        

        torques_ftst = np.array(torques_ftst)

        cols = torques_ftst.shape[1] if (hasattr(torques_ftst, "ndim") and torques_ftst.ndim > 1) else 1
        data = {f't{i+1}': (torques_ftst[:, i] if cols > 1 else torques_ftst[:]) for i in range(cols)}

        df = pd.DataFrame(data)

        torque_data = f"{out_dir}/torquesFst{n}.csv"
        # torque_data = f"{out_dir}/torquesFst{n}.csv"

        df.to_csv(torque_data, index=False)

        print('Performing RNEA')
        t_total_start = time.perf_counter()

        for t_idx in range(len(time_step)): #len(time)
            q = q_csv[t_idx]   # Joint positions from CSV
            qd = qd_csv[t_idx] # Joint velocities from CSV
            qdd = qdd_csv[t_idx] # Joint accelerations from CSV
            torques_rnea.append(rnea(q, qd, qdd, links_rnea, g_rnea))  # Compute torques using RNEA


        t_total_end = time.perf_counter()
        elapsed = t_total_end - t_total_start
        time_elapsed.append(elapsed)
        
        
        torques_rnea = np.array(torques_rnea)

        cols = torques_rnea.shape[1] if (hasattr(torques_rnea, "ndim") and torques_rnea.ndim > 1) else 1
        data = {f't{i+1}': (torques_rnea[:, i] if cols > 1 else torques_rnea[:]) for i in range(cols)}

        df = pd.DataFrame(data)

        torque_data = f"{out_dir}/torquesNE{n}.csv"
        # torque_data = f"{out_dir}/torquesFst{n}.csv"

        df.to_csv(torque_data, index=False)


        # print('Performing RLG')
        # now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # print(now)
        # t_total_start = time.perf_counter()
        # for t_idx in range(len(time_step)): #len(time)
        #     q = q_csv[t_idx]   # Joint positions from CSV
        #     qd = qd_csv[t_idx] # Joint velocities from CSV
        #     qdd = qdd_csv[t_idx] # Joint accelerations from CSV
        #     torques_rlg.append(recLag(q, qd, qdd, links_rlg, g_rlg))  # Compute torques using RLG
    
     
        # t_total_end = time.perf_counter()
        # elapsed = t_total_end - t_total_start
        # time_elapsed.append(elapsed)

        # torques_rlg = np.array(torques_rlg)

        
        # cols = torques_rlg.shape[1] if (hasattr(torques_rlg, "ndim") and torques_rlg.ndim > 1) else 1
        # data = {f't{i+1}': (torques_rlg[:, i] if cols > 1 else torques_rlg[:]) for i in range(cols)}

        # df = pd.DataFrame(data)

        # torque_data = f"{out_dir}/data10s/torquesLE{n}.csv"

        # df.to_csv(torque_data, index=False)



        print(f'Completed computations for n={n} joints.')
        print(f'Time taken for FTST: {time_elapsed[0]:.4f} seconds')
        print(f'Time taken for RNEA: {time_elapsed[1]:.4f} seconds')
        
        # Store execution time data
        execution_times.append({
            'num_links': n,
            'ftst_time': time_elapsed[0],
            'ne_time': time_elapsed[1]
        })
        # print(f'Time taken for RLG: {time_elapsed[0]:.4f} seconds')
    
    # Save execution times to CSV
    df_times = pd.DataFrame(execution_times)
    times_csv = f"{out_dir}/execution_times5s.csv"
    df_times.to_csv(times_csv, index=False)
    print(f'\nExecution times saved to {times_csv}')



        




