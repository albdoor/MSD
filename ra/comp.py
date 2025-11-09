import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
print(os.getcwd())

# Read the CSV files
torques_le = pd.read_csv('./torquesLE.csv')
torques_ne = pd.read_csv('./torquesNE.csv')

# Create time vector
time = np.linspace(0, 10, len(torques_le))

# Get number of joints from column count
n_joints = len(torques_le.columns)
print(n_joints)

# Create subplots for each joint
plt.figure(figsize=(15, 3*n_joints))

# Plot each joint
for i in range(n_joints):
    plt.subplot(n_joints, 1, i+1)
    plt.plot(time, torques_le[f't{i+1}'], 'b-', label='Lagrange')
    plt.plot(time, torques_ne[f't{i+1}'], 'r--', label='Newton-Euler')
    plt.title(f'Joint {i+1} Torque Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.legend()
    plt.grid(True)

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
    