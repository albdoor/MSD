"""
lie_group.py

Procedural (non-OOP) Lie-group inverse dynamics implementation for serial
manipulators.

This version is intentionally structured like a conventional procedural
Newton-Euler / Lagrange-Euler script:

    1. load_joint_data(...)
    2. link_data(n)
    3. loop through trajectory points
    4. calculate torque for each point
    5. save the torque matrix to CSV

No classes are used.

The formulation is the body-fixed Lie-group recursion:

    C_i = C_(i-1) B_i exp(X_i q_i)

    V_i = Ad(C_(i,i-1)) V_(i-1) + X_i qd_i

    Vdot_i =
        Ad(C_(i,i-1)) Vdot_(i-1)
        - qd_i ad(X_i) V_i
        + X_i qdd_i

    W_i =
        Ad(C_(i+1,i)).T W_(i+1)
        + M_i Vdot_i
        - ad(V_i).T M_i V_i
        + W_app_i

    tau_i = X_i.T W_i

The trajectory arrays use the same convention as the user's existing code:

    q.shape   = (number_of_time_points, number_of_links)
    qd.shape  = (number_of_time_points, number_of_links)
    qdd.shape = (number_of_time_points, number_of_links)

CSV format expected by load_joint_data():

    q1 ... qn qd1 ... qdn qdd1 ... qddn

one time point per row.

The implementation uses NumPy and SciPy.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm


# =============================================================================
# DATA LOADING
# =============================================================================

def load_joint_data(filename, n, time_int, filetype):
    """
    Load q, qd and qdd using the same basic interface as the user's
    existing implementations.

    Parameters
    ----------
    filename : str
        CSV or NPY filename.
    n : int
        Number of joints/links.
    time_int : int
        Number of trajectory points for CSV input.
    filetype : str
        'csv' or 'npy'.

    Returns
    -------
    q, qd, qdd : ndarray
        Each has shape (time_int, n) for CSV data.
    """

    filetype = filetype.lower()

    if filetype == 'csv':

        data = np.loadtxt(
            filename,
            delimiter=',',
            skiprows=1
        )

        data = np.atleast_2d(data)

        if data.shape[1] != 3 * n:
            raise ValueError(
                f"Expected {3*n} columns for {n} links, "
                f"but the CSV contains {data.shape[1]} columns."
            )

        if data.shape[0] != time_int:
            raise ValueError(
                f"Expected {time_int} trajectory points, "
                f"but the CSV contains {data.shape[0]} rows."
            )

        q = data[:, 0:n]
        qd = data[:, n:2*n]
        qdd = data[:, 2*n:3*n]

        return q, qd, qdd

    elif filetype == 'npy':

        data = np.load(filename, allow_pickle=True)

        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.tolist()

        q = np.asarray(data[0], dtype=float)
        qd = np.asarray(data[1], dtype=float)
        qdd = np.asarray(data[2], dtype=float)

        q = np.atleast_2d(q)
        qd = np.atleast_2d(qd)
        qdd = np.atleast_2d(qdd)

        if q.shape[1] != n:
            raise ValueError("q does not contain the requested number of joints.")

        return q, qd, qdd

    else:
        raise ValueError("filetype must be 'csv' or 'npy'.")


# =============================================================================
# BASIC LIE-GROUP OPERATIONS
# =============================================================================

def skew(v):
    """
    Return the 3x3 skew matrix of a 3-vector.

        skew(v) @ x = v x x
    """
    v = np.asarray(v, dtype=float).reshape(3)

    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0]
    ])


def ad_matrix(V):
    """
    Construct the 6x6 Lie-algebra ad matrix.

    V = [omega
         v]

    ad(V) =
        [ omega^      0 ]
        [ v^       omega^ ]

    so that ad(V) @ X represents the Lie bracket [V, X].
    """
    V = np.asarray(V, dtype=float).reshape(6)

    omega = V[0:3]
    v = V[3:6]

    return np.block([
        [skew(omega), np.zeros((3, 3))],
        [skew(v),     skew(omega)]
    ])


def screw_hat(X):
    """
    Convert a 6-vector screw X = [omega; v] into a 4x4 se(3) matrix.
    """
    X = np.asarray(X, dtype=float).reshape(6)

    Xi = np.zeros((4, 4))
    Xi[0:3, 0:3] = skew(X[0:3])
    Xi[0:3, 3] = X[3:6]

    return Xi


def exp_screw(X, q):
    """
    Compute exp(X^ q) using the SE(3) matrix exponential.
    """
    return expm(screw_hat(X) * q)


def make_transform(R, r):
    """
    Construct an SE(3) homogeneous transformation.
    """
    C = np.eye(4)
    C[0:3, 0:3] = np.asarray(R, dtype=float)
    C[0:3, 3] = np.asarray(r, dtype=float)

    return C


def inverse_transform(C):
    """
    Inverse of an SE(3) transform.
    """
    R = C[0:3, 0:3]
    r = C[0:3, 3]

    C_inv = np.eye(4)
    C_inv[0:3, 0:3] = R.T
    C_inv[0:3, 3] = -R.T @ r

    return C_inv


def adjoint(C):
    """
    Adjoint transformation for twists.

        Ad_C =
            [ R       0 ]
            [ r^ R    R ]

    This is the convention used in the body-fixed formulation.
    """
    R = C[0:3, 0:3]
    r = C[0:3, 3]

    return np.block([
        [R,                np.zeros((3, 3))],
        [skew(r) @ R,      R]
    ])


# =============================================================================
# LINK DATA
# =============================================================================

def link_data(n, lengths=1.0, masses=1.0, damping=0.0,
              com_fraction=0.5):
    """
    Generate the link data for an arbitrary n-link planar serial robot.

    This function deliberately returns ordinary Python/NumPy data instead
    of a Link class, matching the procedural style of the user's previous
    implementations.

    Each entry is:

        [B_i, X_i, mass_i, I_C_i, d_i, damping_i]

    where:

        B_i       = zero-position relative SE(3) transform
        X_i       = body-fixed joint screw
        mass_i    = link mass
        I_C_i     = inertia tensor about COM in body coordinates
        d_i       = body-frame vector from link origin to COM
        damping_i = viscous damping coefficient
    """

    if n < 1:
        raise ValueError("n must be at least 1.")

    def expand(value, name):
        if np.isscalar(value):
            return np.full(n, float(value))

        value = np.asarray(value, dtype=float).reshape(-1)

        if len(value) != n:
            raise ValueError(
                f"{name} must contain {n} values."
            )

        return value

    lengths = expand(lengths, "lengths")
    masses = expand(masses, "masses")
    damping = expand(damping, "damping")

    links = []

    for i in range(n):

        L = lengths[i]
        m = masses[i]

        # -------------------------------------------------------------
        # B_i:
        # At q_i = 0, frame i is translated by L along x.
        # -------------------------------------------------------------

        B_i = make_transform(
            np.eye(3),
            np.array([L, 0.0, 0.0])
        )

        # -------------------------------------------------------------
        # X_i:
        # Revolute joint about body z axis through the joint origin.
        #
        # X_i = [e_i
        #        x_i x e_i]
        #
        # Here x_i = 0, e_i = [0,0,1].
        # -------------------------------------------------------------

        X_i = np.array([
            0.0, 0.0, 1.0,
            0.0, 0.0, 0.0
        ])

        # -------------------------------------------------------------
        # COM location in body frame.
        # -------------------------------------------------------------

        d_i = np.array([
            com_fraction * L,
            0.0,
            0.0
        ])

        # -------------------------------------------------------------
        # Slender-rod inertia about COM.
        # -------------------------------------------------------------

        I_C_i = np.diag([
            m * L**2 / 12.0,
            m * L**2 / 12.0,
            m * L**2 / 12.0
        ])

        links.append([
            B_i,
            X_i,
            m,
            I_C_i,
            d_i,
            damping[i]
        ])

    return links


# =============================================================================
# BODY-FIXED LINK MASS MATRIX
# =============================================================================

def link_mass_matrix(link):
    """
    Construct the 6x6 body-fixed mass matrix.

    For

        V = [omega
             v]

    and COM offset d from the body-frame origin:

        M =
        [ Theta      m d^ ]
        [ -m d^      m I ]

    where

        Theta = I_C - m d^ d^
    """

    B, X, m, I_C, d, damping = link

    d_hat = skew(d)

    Theta = I_C - m * (d_hat @ d_hat)

    M = np.block([
        [Theta,              m * d_hat],
        [-m * d_hat,         m * np.eye(3)]
    ])

    return M


# =============================================================================
# FORWARD LIE-GROUP RECURSION
# =============================================================================

def forward_lie_group(q, qd, qdd, links):
    """
    Forward recursion.

    Returns
    -------
    C : (n,4,4)
        Absolute link configurations.
    Crel : (n,4,4)
        Relative transformations C_(i,i-1).
    V : (n,6)
        Body-fixed twists.
    Vdot : (n,6)
        Body-fixed accelerations.
    """

    n = len(links)

    q = np.asarray(q, dtype=float).reshape(n)
    qd = np.asarray(qd, dtype=float).reshape(n)
    qdd = np.asarray(qdd, dtype=float).reshape(n)

    C = np.zeros((n, 4, 4))
    Crel = np.zeros((n, 4, 4))

    V = np.zeros((n, 6))
    Vdot = np.zeros((n, 6))

    C_parent = np.eye(4)
    V_parent = np.zeros(6)
    Vdot_parent = np.zeros(6)

    for i in range(n):

        B_i = links[i][0]
        X_i = links[i][1]

        # -------------------------------------------------------------
        # C_i = C_(i-1) B_i exp(X_i q_i)
        # -------------------------------------------------------------

        motion = exp_screw(X_i, q[i])

        C[i] = C_parent @ B_i @ motion

        # -------------------------------------------------------------
        # C_(i,i-1) = C_i^(-1) C_(i-1)
        # -------------------------------------------------------------

        Crel[i] = inverse_transform(C[i]) @ C_parent

        Ad = adjoint(Crel[i])

        # -------------------------------------------------------------
        # V_i = Ad V_(i-1) + X_i qdot_i
        # -------------------------------------------------------------

        V[i] = (
            Ad @ V_parent
            + X_i * qd[i]
        )

        # -------------------------------------------------------------
        # Vdot_i =
        #       Ad Vdot_(i-1)
        #       - qdot_i ad(X_i) V_i
        #       + X_i qdd_i
        # -------------------------------------------------------------

        Vdot[i] = (
            Ad @ Vdot_parent
            - qd[i] * (ad_matrix(X_i) @ V[i])
            + X_i * qdd[i]
        )

        C_parent = C[i]
        V_parent = V[i]
        Vdot_parent = Vdot[i]

    return C, Crel, V, Vdot


# =============================================================================
# APPLIED GRAVITY WRENCHES
# =============================================================================

def gravity_wrenches(C, links, gravity):
    """
    Calculate body-frame gravity wrenches.

    gravity is specified in the inertial/base frame.

    For each link:

        f_g = m R.T g

        t_g = d x f_g

        W_g = [t_g
               f_g]
    """

    n = len(links)

    gravity = np.asarray(gravity, dtype=float).reshape(3)

    Wg = np.zeros((n, 6))

    for i in range(n):

        B, X, m, I_C, d, damping = links[i]

        R = C[i][0:3, 0:3]

        gravity_body = R.T @ gravity

        force = m * gravity_body
        moment = np.cross(d, force)

        Wg[i] = np.hstack([
            moment,
            force
        ])

    return Wg


# =============================================================================
# LIE-GROUP INVERSE DYNAMICS FOR ONE DATA POINT
# =============================================================================

def inverse_dynamics(
    q,
    qd,
    qdd,
    links,
    gravity=(0.0, -9.81, 0.0),
    include_damping=True,
    external_wrenches=None,
    return_intermediates=False
):
    """
    Calculate joint generalized forces/torques for one trajectory point.

    Parameters
    ----------
    q, qd, qdd:
        Arrays of length n.
    links:
        Output of link_data(n).
    gravity:
        Gravity vector in inertial coordinates.
    include_damping:
        Include b_i qdot_i.
    external_wrenches:
        Optional array with shape (n,6). Each wrench is expressed in the
        corresponding body frame and applied at that link-frame origin.
    return_intermediates:
        If True, return a dictionary containing all recursive quantities.

    Returns
    -------
    tau:
        n-vector of joint generalized forces.
    """

    n = len(links)

    q = np.asarray(q, dtype=float).reshape(n)
    qd = np.asarray(qd, dtype=float).reshape(n)
    qdd = np.asarray(qdd, dtype=float).reshape(n)

    C, Crel, V, Vdot = forward_lie_group(
        q, qd, qdd, links
    )

    # -------------------------------------------------------------
    # Applied wrenches
    # -------------------------------------------------------------

    W_app = gravity_wrenches(
        C,
        links,
        gravity
    )

    if external_wrenches is not None:

        external_wrenches = np.asarray(
            external_wrenches,
            dtype=float
        )

        if external_wrenches.shape != (n, 6):
            raise ValueError(
                f"external_wrenches must have shape {(n,6)}."
            )

        W_app += external_wrenches

    # -------------------------------------------------------------
    # Link mass matrices
    # -------------------------------------------------------------

    M = np.zeros((n, 6, 6))

    for i in range(n):
        M[i] = link_mass_matrix(links[i])

    # -------------------------------------------------------------
    # Backward wrench recursion
    # -------------------------------------------------------------

    W = np.zeros((n, 6))

    for i in reversed(range(n)):

        # ---------------------------------------------------------
        # Local inertial wrench:
        #
        # M_i Vdot_i
        # - ad(V_i).T M_i V_i
        # + W_app_i
        # ---------------------------------------------------------

        local_wrench = (
            M[i] @ Vdot[i]
            - ad_matrix(V[i]).T @ (M[i] @ V[i])
            + W_app[i]
        )

        # ---------------------------------------------------------
        # Add wrench transmitted from child.
        #
        # W_i =
        #   Ad(C_(i+1,i)).T W_(i+1)
        #   + local_wrench
        # ---------------------------------------------------------

        if i == n - 1:

            W[i] = local_wrench

        else:

            W[i] = (
                adjoint(Crel[i + 1]).T @ W[i + 1]
                + local_wrench
            )

    # -------------------------------------------------------------
    # Joint generalized forces
    #
    # Q_i = X_i.T W_i
    # -------------------------------------------------------------

    tau = np.zeros(n)

    for i in range(n):

        X_i = links[i][1]

        tau[i] = X_i @ W[i]

        if include_damping:
            damping_i = links[i][5]
            tau[i] += damping_i * qd[i]

    if return_intermediates:

        return {
            "tau": tau,
            "C": C,
            "Crel": Crel,
            "V": V,
            "Vdot": Vdot,
            "M": M,
            "W_app": W_app,
            "W": W
        }

    return tau


# =============================================================================
# TRAJECTORY CALCULATION
# =============================================================================

def generate_torque_data(
    q,
    qd,
    qdd,
    links,
    gravity=(0.0, -9.81, 0.0),
    include_damping=True
):
    """
    Calculate torque for every trajectory point.

    q, qd, qdd:
        shape = (N,n)

    Returns:
        tau:
        shape = (N,n)
    """

    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    qdd = np.asarray(qdd, dtype=float)

    if q.ndim != 2:
        raise ValueError("q must have shape (N,n).")

    if q.shape != qd.shape or q.shape != qdd.shape:
        raise ValueError(
            "q, qd and qdd must have identical shapes."
        )

    n = len(links)

    if q.shape[1] != n:
        raise ValueError(
            f"Trajectory contains {q.shape[1]} joints, "
            f"but link_data contains {n} links."
        )

    number_of_points = q.shape[0]

    tau = np.zeros((number_of_points, n))

    for k in range(number_of_points):

        tau[k] = inverse_dynamics(
            q[k],
            qd[k],
            qdd[k],
            links,
            gravity=gravity,
            include_damping=include_damping
        )

    return tau


# =============================================================================
# CSV OUTPUT
# =============================================================================

def save_torque_data(filename, tau):
    """
    Save torque data in the same simple DataFrame/CSV style as the user's
    previous implementations.
    """

    tau = np.asarray(tau, dtype=float)

    if tau.ndim != 2:
        raise ValueError("tau must have shape (N,n).")

    n = tau.shape[1]

    data = {}

    for i in range(n):
        data[f't{i+1}'] = tau[:, i]

    df = pd.DataFrame(data)

    df.to_csv(
        filename,
        index=False
    )


# =============================================================================
# OPTIONAL VALIDATION FUNCTIONS
# =============================================================================

def validate_forward_recursion(q, qd, qdd, links):
    """
    Return the forward Lie-group recursion quantities for inspection.

    This is useful when comparing the Lie-group code against the user's
    Newton-Euler implementation.
    """

    C, Crel, V, Vdot = forward_lie_group(
        q, qd, qdd, links
    )

    return {
        "C": C,
        "Crel": Crel,
        "V": V,
        "Vdot": Vdot
    }


def validate_mass_matrices(links):
    """
    Check symmetry and positive definiteness of all link mass matrices.
    """

    results = []

    for i in range(len(links)):

        M = link_mass_matrix(links[i])

        symmetry_error = np.linalg.norm(
            M - M.T
        )

        eigenvalues = np.linalg.eigvalsh(
            0.5 * (M + M.T)
        )

        results.append({
            "link": i + 1,
            "symmetry_error": symmetry_error,
            "minimum_eigenvalue": np.min(eigenvalues),
            "positive_definite": np.all(eigenvalues > 0.0)
        })

    return results


# =============================================================================
# EXAMPLE EXECUTION
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------
    # Number of links
    # -------------------------------------------------------------

    n = 5

    # -------------------------------------------------------------
    # Simulation settings
    # -------------------------------------------------------------

    time_int = 500

    time_step = np.linspace(
        0.0,
        5.0,
        time_int
    )

    # -------------------------------------------------------------
    # Link data
    #
    # Change n to 2, 3, 4, 5, ... without changing the algorithm.
    # -------------------------------------------------------------

    links = link_data(
        n=n,
        lengths=1.0,
        masses=1.0,
        damping=50.0
    )

    # -------------------------------------------------------------
    # Load trajectory
    #
    # Expected CSV columns:
    #
    # q1 ... qn qd1 ... qdn qdd1 ... qddn
    # -------------------------------------------------------------

    out_dir = './ra'

    trajectory_file = (
        f"{out_dir}/data5s/"
        f"trajectory_data_gen{n}.csv"
    )

    if os.path.exists(trajectory_file):

        q, qd, qdd = load_joint_data(
            trajectory_file,
            n,
            time_int,
            'csv'
        )

    else:

        # ---------------------------------------------------------
        # Example trajectory if the user's trajectory file is not
        # present.
        # ---------------------------------------------------------

        q = np.zeros((time_int, n))
        qd = np.zeros((time_int, n))
        qdd = np.zeros((time_int, n))

        amplitudes = np.linspace(
            0.30,
            0.10,
            n
        )

        frequencies = np.linspace(
            1.0,
            0.5,
            n
        )

        for i in range(n):

            A = amplitudes[i]
            w = frequencies[i]

            q[:, i] = A * np.sin(w * time_step)

            qd[:, i] = (
                A * w * np.cos(w * time_step)
            )

            qdd[:, i] = (
                -A * w**2
                * np.sin(w * time_step)
            )

    # -------------------------------------------------------------
    # Calculate torque for every trajectory point
    # -------------------------------------------------------------

    tau = generate_torque_data(
        q,
        qd,
        qdd,
        links,
        gravity=(0.0, 9.81, 0.0),
        include_damping=True
    )

    # -------------------------------------------------------------
    # Print basic information
    # -------------------------------------------------------------

    print("Lie-group inverse dynamics")
    print("--------------------------")
    print("Number of links:", n)
    print("Number of trajectory points:", len(q))
    print("q shape:", q.shape)
    print("qd shape:", qd.shape)
    print("qdd shape:", qdd.shape)
    print("tau shape:", tau.shape)

    print("\nFirst torque point:")
    print(tau[0])

    print("\nLast torque point:")
    print(tau[-1])

    # -------------------------------------------------------------
    # Validate link mass matrices
    # -------------------------------------------------------------

    print("\nMass matrix validation:")

    validation = validate_mass_matrices(links)

    for result in validation:

        print(
            f"Link {result['link']}: "
            f"symmetry error = "
            f"{result['symmetry_error']:.3e}, "
            f"minimum eigenvalue = "
            f"{result['minimum_eigenvalue']:.3e}, "
            f"positive definite = "
            f"{result['positive_definite']}"
        )

    # -------------------------------------------------------------
    # Save torque data
    # -------------------------------------------------------------

    torque_file = (
        f"{out_dir}/data5s/"
        f"torquesLie_Group{n}.csv"
    )

    torque_parent = Path(torque_file).parent
    torque_parent.mkdir(
        parents=True,
        exist_ok=True
    )

    save_torque_data(
        torque_file,
        tau
    )

    print("\nSaved torque data to:")
    print(torque_file)
