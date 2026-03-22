import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from datetime import datetime




from recLagFuncGen import recLag, load_joint_data, link_data as rlg_link_data, output_counters as rlg_output_counters, plot_graphs, reset_counters as rlg_reset_counters
from rneagen import rnea, link_data as rnea_link_data, output_counters as rnea_output_counters, reset_counters as rnea_reset_counters
from featherstone import featherstone_id, link_data as fst_link_data, output_counters as ftst_output_counters, reset_counters as ftst_reset_counters

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
    execution_counters = []  # Store execution times and link count
    
    for n in range(2, 21):
        m1 = 1.0
        l1 = 1.0
        g = 9.81
        t_end = 90
        time_step = np.linspace(0, t_end, t_end * 100)  # Time steps from 0 to 50 seconds

        torques = []

        torquesLE = []


        out_dir = './data90s'

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
        rlg_counters = []
        rnea_counters = []
        ftst_counters = []

        addCount_rlg, multCount_rlg, trigCount_rlg = 0, 0, 0
        addCount_rnea, multCount_rnea, trigCount_rnea = 0, 0, 0
        addCount_ftst, multCount_ftst, trigCount_ftst = 0,  0, 0
        
        print('Performing FTST')
        t_total_start = time.perf_counter()
        
        for t_idx in range(1): #len(time)
            q = q_csv[t_idx]   # Joint positions from CSV
            qd = qd_csv[t_idx] # Joint velocities from CSV
            qdd = qdd_csv[t_idx] # Joint accelerations from CSV
            torques_ftst.append(featherstone_id(q, qd, qdd, links_fst, g_ftst))  # Compute torques using FTST
            addCount_ftst, multCount_ftst, trigCount_ftst = ftst_output_counters()
            ftst_reset_counters()

        t_total_end = time.perf_counter()
        elapsed = t_total_end - t_total_start


        # time_elapsed.append(elapsed)
        

        torques_ftst = np.array(torques_ftst)


        print('Performing RNEA')
        t_total_start = time.perf_counter()

        for t_idx in range(1): #len(time)
            q = q_csv[t_idx]   # Joint positions from CSV
            qd = qd_csv[t_idx] # Joint velocities from CSV
            qdd = qdd_csv[t_idx] # Joint accelerations from CSV
            torques_rnea.append(rnea(q, qd, qdd, links_rnea, g_rnea))  # Compute torques using RNEA
            addCount_rnea, multCount_rnea, trigCount_rnea = rnea_output_counters()
            rnea_reset_counters()



        t_total_end = time.perf_counter()
        elapsed = t_total_end - t_total_start
        # time_elapsed.append(elapsed)
        
        
        torques_rnea = np.array(torques_rnea)

 

        print('Performing RLG')
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(now)
        t_total_start = time.perf_counter()
        for t_idx in range(1): #len(time)
            q = q_csv[t_idx]   # Joint positions from CSV
            qd = qd_csv[t_idx] # Joint velocities from CSV
            qdd = qdd_csv[t_idx] # Joint accelerations from CSV
            torques_rlg.append(recLag(q, qd, qdd, links_rlg, g_rlg))  # Compute torques using RLG
            addCount_rlg, multCount_rlg, trigCount_rlg = rlg_output_counters()
            rlg_reset_counters()
    
     
        t_total_end = time.perf_counter()
        elapsed = t_total_end - t_total_start
        # time_elapsed.append(elapsed)

        torques_rlg = np.array(torques_rlg)




        print(f'Completed computations for n={n} joints.')
        
        # Store execution time data
        execution_counters.append({
            'num_links': n,
            'rlg_AddCount': addCount_rlg,
            'rlg_MultCount': multCount_rlg,
            'rlg_TrigCount': trigCount_rlg,
            'rnea_AddCount': addCount_rnea,
            'rnea_MultCount': multCount_rnea,
            'rnea_TrigCount': trigCount_rnea,
            'ftst_AddCount': addCount_ftst,
            'ftst_MultCount': multCount_ftst,
            'ftst_TrigCount': trigCount_ftst
        })
        # print(f'Time taken for RLG: {time_elapsed[0]:.4f} seconds')
        df_times = pd.DataFrame(execution_counters)
        times_csv = f"{out_dir}/execution_counters90s.csv"
        df_times.to_csv(times_csv, index=False)
        print(f'\nExecution counters saved to {times_csv}')
    
    # Save execution times to CSV




        




