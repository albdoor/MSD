import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV files
torques_le = pd.read_csv('torquesLE.csv')
torques_ne = pd.read_csv('torquesNE.csv')

# Create time vector (assuming 1000 points from 0 to 10 seconds)
time = np.linspace(0, 10, len(torques_le))

# Create subplots for each joint
plt.figure(figsize=(15, 10))

# Plot Joint 1
plt.subplot(3, 1, 1)
plt.plot(time, torques_le['t1'], 'b-', label='Lagrange')
plt.plot(time, torques_ne['t1'], 'r--', label='Newton-Euler')
plt.title('Joint 1 Torque Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.grid(True)

# Plot Joint 2
plt.subplot(3, 1, 2)
plt.plot(time, torques_le['t2'], 'b-', label='Lagrange')
plt.plot(time, torques_ne['t2'], 'r--', label='Newton-Euler')
plt.title('Joint 2 Torque Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.grid(True)

# Plot Joint 3
plt.subplot(3, 1, 3)
plt.plot(time, torques_le['t3'], 'b-', label='Lagrange')
plt.plot(time, torques_ne['t3'], 'r--', label='Newton-Euler')
plt.title('Joint 3 Torque Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Calculate and print the RMS error for each joint
def rms_error(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

rms_1 = rms_error(torques_le['t1'], torques_ne['t1'])
rms_2 = rms_error(torques_le['t2'], torques_ne['t2'])
rms_3 = rms_error(torques_le['t3'], torques_ne['t3'])

print(f"RMS Error for Joint 1: {rms_1:.6f}")
print(f"RMS Error for Joint 2: {rms_2:.6f}")
print(f"RMS Error for Joint 3: {rms_3:.6f}")