import csv

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

print(os.getcwd())

# Read the CSV files

def show_last_link(torques_le, torques_ne, torques_fst, torques_ham, torques_lie, time, n, zoom_range=None):
    series = [
        (torques_le[f't{n}'], 'b-', 'Lagrange'),
        (torques_ne[f't{n}'], 'r--', 'Newton-Euler'),
        (torques_fst[f't{n}'], 'g-.', 'Featherstone'),
        (torques_ham[f't{n}'], 'm:', 'Gibbs-Appell'),
        (torques_lie[f't{n}'], 'c:', 'Lie Group'),
    ]

    if zoom_range is not None:
        zoom_start, zoom_end = zoom_range
        mask = (time >= zoom_start) & (time <= zoom_end)
        if zoom_start >= zoom_end or not np.any(mask):
            raise ValueError('zoom_range must select at least one time sample.')
        fig, (ax, zoom_ax) = plt.subplots(
            2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        zoom_ax = None

    for values, style, label in series:
        ax.plot(time, values, style, label=label, linewidth=2.5)
        if zoom_ax is not None:
            zoom_ax.plot(time, values, style, linewidth=1.5)

    if zoom_ax is not None:
        zoom_values = np.concatenate([np.asarray(values)[mask] for values, _, _ in series])
        y_min, y_max = zoom_values.min(), zoom_values.max()
        y_padding = (y_max - y_min) * 0.05 or 0.01
        zoom_ax.set_xlim(zoom_start, zoom_end)
        zoom_ax.set_ylim(y_min - y_padding, y_max + y_padding)
        zoom_ax.set_title(f'Zoomed section: {zoom_start:g}-{zoom_end:g} s', fontsize=16, fontweight='bold')
        zoom_ax.set_xlabel('Time (s)', fontsize=16, fontweight='bold')
        zoom_ax.set_ylabel('Torque (Nm)', fontsize=16, fontweight='bold')
        zoom_ax.grid(True, alpha=0.6)
        ax.axvspan(zoom_start, zoom_end, color='gray', alpha=0.12)

    ax.set_title(f'Joint {n} Torque Comparison', fontsize=22, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Torque (Nm)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, frameon=True)
    ax.grid(True, which='major', linewidth=1.2, alpha=0.8)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    n = 20

    # out_dir = './ra'
    # out_dir = './ra/data5s'
    t_end = 10

    out_dir = f'./ra/data{t_end}s'


    csv_name_le= f"{out_dir}/torquesLE{n}.csv"
    csv_name_ne= f"{out_dir}/torquesNE{n}.csv"
    csv_name_fst= f"{out_dir}/torquesFst{n}.csv"
    csv_name_gib= f"{out_dir}/torquesGibbs{n}.csv"
    csv_name_lie = f"{out_dir}/torquesLie{n}.csv"

    # csv_name_fst= f"{out_dir}/torquesFst{3}.csv"



    torques_le = pd.read_csv(csv_name_le)
    torques_ne = pd.read_csv(csv_name_ne)
    torques_fst = pd.read_csv(csv_name_fst)
    torques_gib = pd.read_csv(csv_name_gib)
    torques_lie = pd.read_csv(csv_name_lie) 
    # Create time vector
    time = np.linspace(0, t_end, len(torques_le))

    # Get number of joints from column count
    n_joints = len(torques_le.columns)
    print(n_joints)

    lw = 2.5           # line width
    fs = 16 

    plt.rcParams.update({'xtick.labelsize': fs-2, 'ytick.labelsize': fs-2})


    # Create subplots for each joint
    plt.figure(figsize=(15, 2.5*n_joints))


    # Plot each joint
    for i in range(n_joints):
        plt.subplot(n_joints, 1, i+1)
        plt.plot(time, torques_le[f't{i+1}'], 'b-', label='Lagrange', linewidth=lw)
        plt.plot(time, torques_ne[f't{i+1}'], 'r--', label='Newton-Euler', linewidth=lw)
        plt.plot(time, torques_fst[f't{i+1}'], 'g-.', label='Featherstone', linewidth=lw)
        plt.plot(time, torques_gib[f't{i+1}'], 'm:', label='Gibbs-Appell', linewidth=lw)
        plt.plot(time, torques_lie[f't{i+1}'], 'c:', label='Lie Group', linewidth=lw)
        plt.title(f'Joint {i+1} Torque Comparison', fontsize=fs+4, fontweight='bold')
        plt.xlabel('Time (s)', fontsize=fs+4, fontweight='bold')
        plt.ylabel('Torque (Nm)', fontsize=fs+4, fontweight='bold')
        plt.legend(fontsize=fs-2, frameon=True)
        plt.tick_params(axis='both', which='major', labelsize=fs+4)
        plt.grid(True, which='major', linewidth=1.5, alpha=0.8)

    plt.tight_layout()

    # Calculate and print the RMS error for each joint
    def rms_error(actual, predicted):
        return np.sqrt(np.mean((actual - predicted) ** 2))

    # Calculate RMS errors for all joints
    rms_errors = []
    for i in range(n_joints):
        rms = rms_error(torques_le[f't{i+1}'], torques_ne[f't{i+1}'])
        rms_errors.append(rms)
        print(f"RMS Error for Joint {i+1}: {rms:.6f}")
        
    
    
    def calculate_errors(actual, predicted):
        """Calculate different error metrics between actual and predicted values."""
        # L1 norm (Manhattan distance)
        l1_norm = np.mean(np.abs(actual - predicted))
        
        # L2 norm (Euclidean distance)
        l2_norm = np.sqrt(np.mean((actual - predicted)**2))
        
        # L∞ norm (Maximum absolute error)
        linf_norm = np.max(np.abs(actual - predicted))
        
        # RMS error 
        rms = np.sqrt(np.mean((actual - predicted)**2))
        
        # Relative RMS error (normalized)
        rel_rms = rms / (np.sqrt(np.mean(actual**2)) + 1e-10)
        
        return {
            'L1': l1_norm,
            'L2': l2_norm,
            'Linf': linf_norm,
            'RMS': rms,
            'RelRMS': rel_rms
        }

    # Calculate errors for all joints
    for i in range(n_joints):
        errors = calculate_errors(torques_le[f't{i+1}'], torques_ne[f't{i+1}'])
        print(f"\nJoint {i+1} Error Metrics:")
        print(f"L1 Norm (Average Absolute Error): {errors['L1']:.6f}")
        print(f"L2 Norm (Euclidean): {errors['L2']:.6f}")
        print(f"L∞ Norm (Maximum Error): {errors['Linf']:.6f}")
        print(f"RMS Error: {errors['RMS']:.6f}")
        print(f"Relative RMS Error: {errors['RelRMS']:.6f}")
        
        
    plt.show()

    if n <= 4:
        for joint in range(1, n_joints + 1):
            show_last_link(torques_le, torques_ne, torques_fst, torques_gib, torques_lie, time, joint)
    else: 
        show_last_link(
            torques_le, torques_ne, torques_fst, torques_gib, torques_lie,
            time, n_joints, zoom_range=(6.0, 6.5)   
        )


    # read Hamiltonian
    # add 5 link and 6 link comparison


# bar charts comp time vs N links