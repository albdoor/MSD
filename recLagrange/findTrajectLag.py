import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import os
print(os.getcwd())  # Print the current working directory
print(os.listdir()) # List all files in the current directory


class RobotDynamicsSolver:
    def __init__(self, num_links, link_masses, link_lengths, gravity=9.81):
        """
        Initialize the robot dynamics solver.
        
        Parameters:
        -----------
        num_links : int
            Number of links in the robot
        link_masses : list or ndarray
            Masses of each link
        link_lengths : list or ndarray
            Lengths of each link
        gravity : float
            Gravitational acceleration constant
        """
        self.n = num_links
        self.m = np.array(link_masses)
        self.l = np.array(link_lengths)
        self.g = gravity
        
        # Verify inputs
        assert len(link_masses) == num_links, "Number of masses must equal num_links"
        assert len(link_lengths) == num_links, "Number of lengths must equal num_links"
    
    def compute_M(self, theta):
        """
        Compute the mass matrix M exactly according to equation (49):
        
        M_{i,j} = Σ(k=a to n) m_k [Σ(m=a to k-1) l_m^2 + Σ(m=b to a-1) l_m Σ(p=a to k) l_p cos(Σ(q=m+1 to p) θ_q) 
                  + 2 Σ(m=a to k-1) Σ(p=m+1 to k) l_m l_p cos(Σ(q=m+1 to p) θ_q)]
        
        where a = max(i,j) and b = min(i,j)
        """
        n = self.n
        m = self.m
        l = self.l
        
        M = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                a = max(i, j)  # indices are 0-based, so max(i,j) corresponds to max(i+1,j+1) in 1-based indexing
                b = min(i, j)
                
                M_ij = 0
                
                # Sum from k=a to n-1 (0-based indexing)
                for k in range(a, n):
                    mk = m[k]
                    
                    # First term: Σ(m=a to k-1) l_m^2
                    term1 = 0
                    for m_idx in range(a, k):
                        term1 += l[m_idx]**2
                    
                    # Second term: Σ(m=b to a-1) l_m Σ(p=a to k) l_p cos(Σ(q=m+1 to p) θ_q)
                    term2 = 0
                    for m_idx in range(b, a):
                        for p in range(a, k + 1):
                            # Compute Σ(q=m+1 to p) θ_q
                            if p >= m_idx + 1:
                                theta_sum = sum(theta[q] for q in range(m_idx + 1, p + 1))
                                term2 += l[m_idx] * l[p] * np.cos(theta_sum)
                    
                    # Third term: 2 Σ(m=a to k-1) Σ(p=m+1 to k) l_m l_p cos(Σ(q=m+1 to p) θ_q)
                    term3 = 0
                    for m_idx in range(a, k):
                        for p in range(m_idx + 1, k + 1):
                            # Compute Σ(q=m+1 to p) θ_q
                            theta_sum = sum(theta[q] for q in range(m_idx + 1, p + 1))
                            term3 += l[m_idx] * l[p] * np.cos(theta_sum)
                    
                    M_ij += mk * (term1 + term2 + 2 * term3)
                
                M[i, j] = M_ij
        
        return M
    

    def compute_C(self, theta, theta_dot):
        """
        Compute the Coriolis matrix C exactly according to equations (50), (51), and (52):
        
        Case 1 (i = j): Equation (50)
        C_{i,j} = -2 Σ(k=i+1 to n) m_k Σ(m=i to k-1) Σ(p=m+1 to k) l_p (Σ(q=m+1 to p) θ_q) sin(Σ(q=m+1 to p) θ_q)
        
        Case 2 (i < j): Equation (51)
        C_{i,j} = -Σ(k=j to n) m_k [Σ(m=i to j-1) Σ(p=j to k) l_p (Σ(q=m+1 to p) θ_q) sin(Σ(q=m+1 to p) θ_q) 
                  + 2 Σ(m=j to k-1) Σ(p=j+1 to k) l_p (Σ(q=m+1 to p) θ_q) sin(Σ(q=m+1 to p) θ_q)]
        
        Case 3 (i > j): Equation (52)
        C_{i,j} = Σ(k=i to n) m_k [Σ(m=j to i-1) Σ(p=i to k) (Σ(q=1 to m) θ_q) sin(Σ(q=m+1 to p) θ_q) 
                  - 2 Σ(m=i to k-1) Σ(p=i+1 to k) (Σ(q=m+1 to p) θ_q) sin(Σ(q=m+1 to p) θ_q)]
        """
        n = self.n
        m = self.m
        l = self.l
        
        C = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Case 1: i = j (Equation 50)
                    C_ij = 0
                    for k in range(i + 1, n):
                        inner_sum = 0
                        for m_idx in range(i, k):
                            for p in range(m_idx + 1, k + 1):
                                # Compute Σ(q=m+1 to p) θ_q
                                theta_sum = sum(theta[q] for q in range(m_idx + 1, p + 1))
                                # Note: the equation shows l_p multiplied by the theta sum, but this seems to be θ̇
                                inner_sum += l[p] * theta_dot[m_idx] * np.sin(theta_sum)
                        C_ij += -2 * m[k] * inner_sum
                    C[i, j] = C_ij
                    
                elif i < j:
                    # Case 2: i < j (Equation 51)
                    C_ij = 0
                    for k in range(j, n):
                        # First term
                        term1 = 0
                        for m_idx in range(i, j):
                            for p in range(j, k + 1):
                                theta_sum = sum(theta[q] for q in range(m_idx + 1, p + 1))
                                term1 += l[p] * theta_dot[m_idx] * np.sin(theta_sum)
                        
                        # Second term
                        term2 = 0
                        for m_idx in range(j, k):
                            for p in range(j + 1, k + 1):
                                theta_sum = sum(theta[q] for q in range(m_idx + 1, p + 1))
                                term2 += l[p] * theta_dot[m_idx] * np.sin(theta_sum)
                        
                        C_ij += -m[k] * (term1 + 2 * term2)
                    C[i, j] = C_ij
                    
                else:  # i > j
                    # Case 3: i > j (Equation 52)
                    C_ij = 0
                    for k in range(i, n):
                        # First term
                        term1 = 0
                        for m_idx in range(j, i):
                            for p in range(i, k + 1):
                                theta_sum1 = sum(theta[q] for q in range(0, m_idx + 1))  # Σ(q=1 to m) -> 0 to m in 0-based
                                theta_sum2 = sum(theta[q] for q in range(m_idx + 1, p + 1))
                                term1 += theta_sum1 * np.sin(theta_sum2)
                        
                        # Second term
                        term2 = 0
                        for m_idx in range(i, k):
                            for p in range(i + 1, k + 1):
                                theta_sum = sum(theta[q] for q in range(m_idx + 1, p + 1))
                                term2 += theta_dot[m_idx] * np.sin(theta_sum)
                        
                        C_ij += m[k] * (term1 - 2 * l[p] * term2)
                    C[i, j] = C_ij
        
        return C
    
    def compute_G(self, theta):
        """
        Compute the gravity vector G exactly according to equation (53):
        
        G(i) = Σ(k=1 to n) [g sin(Σ(q=1 to k) θ_q) (Σ(p=k to n) m_p l_k)]
        """
        n = self.n
        m = self.m
        l = self.l
        g = self.g
        
        G = np.zeros(n)
        
        for i in range(n):
            G_i = 0
            
            for k in range(n):  # k from 1 to n (0 to n-1 in 0-based)
                # Compute Σ(q=1 to k) θ_q -> sum from 0 to k in 0-based indexing
                theta_sum = sum(theta[q] for q in range(k + 1))
                
                # Compute Σ(p=k to n) m_p l_k -> sum from k to n-1 in 0-based indexing
                mass_length_sum = sum(m[p] for p in range(k, n)) * l[k]
                
                G_i += g * np.sin(theta_sum) * mass_length_sum
            
            G[i] = G_i
        
        return G
    
    def state_derivative(self, t, state):
        """
        Compute the derivative of the state vector.
        
        state = [theta_1, theta_2, ..., theta_n, theta_dot_1, theta_dot_2, ..., theta_dot_n]
        """
        n = self.n
        theta = state[:n]
        theta_dot = state[n:]
        
        # Compute matrices
        M = self.compute_M(theta)
        C = self.compute_C(theta, theta_dot)
        G = self.compute_G(theta)
        
        # Solve for theta_ddot: M * theta_ddot = -C * theta_dot - G
        theta_ddot = np.linalg.solve(M, -C.dot(theta_dot) - G)
        
        # Return the derivatives
        return np.concatenate([theta_dot, theta_ddot])
    
    def solve(self, initial_theta, initial_theta_dot, t_span, t_eval=None):
        """
        Solve the robot dynamics equation M𝜃̈ + C𝜃̇ + G = 0
        
        Parameters:
        -----------
        initial_theta : list or ndarray
            Initial joint angles
        initial_theta_dot : list or ndarray
            Initial joint velocities
        t_span : tuple
            (t_start, t_end) time span for integration
        t_eval : ndarray, optional
            Time points at which to evaluate the solution
        
        Returns:
        --------
        sol : OdeSolution
            Solution object from solve_ivp
        """
        initial_state = np.concatenate([initial_theta, initial_theta_dot])
        
        # Solve the ODE
        sol = solve_ivp(
            self.state_derivative,
            t_span,
            initial_state,
            method='RK45',
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9
        )
        
        return sol
    
    def plot_solution(self, sol):
        """Plot the solution of the robot dynamics"""
        n = self.n
        t = sol.t
        theta = sol.y[:n, :]
        theta_dot = sol.y[n:, :]
        
        # Create figure
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot joint angles
        for i in range(n):
            axs[0].plot(t, theta[i], label=f'θ_{i+1}')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Joint Angle (rad)')
        axs[0].set_title('Joint Angles')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot joint velocities
        for i in range(n):
            axs[1].plot(t, theta_dot[i], label=f'θ̇_{i+1}')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Joint Velocity (rad/s)')
        axs[1].set_title('Joint Velocities')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        return fig
    
    def compute_accelerations(self, sol):
        """
        Compute theta_ddot (accelerations) from the solution
        """
        n = self.n
        t = sol.t
        theta = sol.y[:n, :]
        theta_dot = sol.y[n:, :]
        print(f"Length of time array: {len(t)}")
        theta_ddot = np.zeros((n, len(t)))
        
        for i, time in enumerate(t):
            state = sol.y[:, i]
            derivatives = self.state_derivative(time, state)
            theta_ddot[:, i] = derivatives[n:]  # acceleration part
        
        return theta_ddot
    
    def save_to_csv(self, sol, filename='robot_dynamics.csv'):
        """
        Save the solution to a CSV file with columns:
        time, theta_1, theta_2, ..., theta_dot_1, theta_dot_2, ..., theta_ddot_1, theta_ddot_2, ...
        """
        import pandas as pd
        
        n = self.n
        t = sol.t
        theta = sol.y[:n, :]
        theta_dot = sol.y[n:, :]
        theta_ddot = self.compute_accelerations(sol)
        
        # Create column names
        columns = ['time']
        columns.extend([f'theta_{i+1}' for i in range(n)])
        columns.extend([f'theta_dot_{i+1}' for i in range(n)])
        columns.extend([f'theta_ddot_{i+1}' for i in range(n)])
        
        # Create data array
        data = np.column_stack([
            t,
            theta.T,
            theta_dot.T,
            theta_ddot.T
        ])
        
        # Create DataFrame and save
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        return df
    
    def save_to_numpy(self, sol, filename_prefix='robot_dynamics'):
        """
        Save the solution to numpy files (.npz format)
        Creates separate arrays for time, theta, theta_dot, and theta_ddot
        """
        n = self.n
        t = sol.t
        theta = sol.y[:n, :]
        theta_dot = sol.y[n:, :]
        theta_ddot = self.compute_accelerations(sol)
        
        # Save as compressed numpy file
        np.savez_compressed(
            f'{filename_prefix}.npz',
            time=t,
            theta=theta,
            theta_dot=theta_dot,
            theta_ddot=theta_ddot,
            num_links=n,
            link_masses=self.m,
            link_lengths=self.l,
            gravity=self.g
        )
        print(f"Data saved to {filename_prefix}.npz")
        
        # Also save individual arrays if needed
        np.save(f'{filename_prefix}_time.npy', t)
        np.save(f'{filename_prefix}_theta.npy', theta)
        np.save(f'{filename_prefix}_theta_dot.npy', theta_dot)
        np.save(f'{filename_prefix}_theta_ddot.npy', theta_ddot)
        print(f"Individual arrays saved as {filename_prefix}_*.npy")
    
    def load_from_numpy(self, filename_prefix='robot_dynamics'):
        """
        Load data from numpy files
        """
        data = np.load(f'{filename_prefix}.npz')
        return {
            'time': data['time'],
            'theta': data['theta'],
            'theta_dot': data['theta_dot'],
            'theta_ddot': data['theta_ddot'],
            'num_links': data['num_links'],
            'link_masses': data['link_masses'],
            'link_lengths': data['link_lengths'],
            'gravity': data['gravity']
        }

# Example usage:
if __name__ == "__main__":
    # Example parameters for a 2-link robot
    num_links = 5
    link_masses = [1.0, 1.0, 1.0, 1.0, 1.0]  # kg
    link_lengths = [1.0, 1.0, 1.0, 1.0, 1.0]  # m
    
    # Initialize solver
    solver = RobotDynamicsSolver(num_links, link_masses, link_lengths)
    
    # Initial conditions
    initial_theta = [np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # 45 degrees for each joint
    initial_theta_dot = [0.0, 0.0, 0.0, 0.0, 0.0]  # starting from rest
    
    # Time span
    t_span = (0, 10)  # 10 seconds
    t_eval = np.linspace(0, 10, 1000)  # 1000 evaluation points
    
    # Solve the system
    print("Solving robot dynamics...")
    sol = solver.solve(initial_theta, initial_theta_dot, t_span, t_eval)
    
    # Save data to files
    print("\nSaving data to files...")
    
    # Save to CSV
    df = solver.save_to_csv(sol, './data/robot_dynamics_output.csv')
    print(f"CSV contains {len(df)} time steps and {len(df.columns)} columns")
    
    # Save to numpy files
    solver.save_to_numpy(sol, './data/robot_dynamics_output')
    
    # Demonstrate loading from numpy
    print("\nLoading data back from numpy file...")
    loaded_data = solver.load_from_numpy('./data/robot_dynamics_output')
    print(f"Loaded data shape - theta: {loaded_data['theta'].shape}")
    print(f"Time range: {loaded_data['time'][0]:.3f} to {loaded_data['time'][-1]:.3f} seconds")
    
    # Display first few rows of data
    print("\nFirst 5 time steps:")
    print(df.head())
    
    # Plot the solution
    print("\nPlotting solution...")
    solver.plot_solution(sol)
    plt.show()
    
    print("\nSolution computed and saved successfully!")
    print(f"Final joint angles: {sol.y[:num_links, -1]}")
    print(f"Final joint velocities: {sol.y[num_links:, -1]}")
    
    # Show file contents summary
    print(f"\nFiles created:")
    print(f"- robot_dynamics_output.csv (CSV format)")
    print(f"- robot_dynamics_output.npz (compressed numpy)")
    print(f"- robot_dynamics_output_*.npy (individual arrays)")



# meeting 13.06.2025
    # eqs 16 17 18