"""
gibbs_appell.py

General-purpose Gibbs-Appell inverse dynamics for serial rigid-link robots.

Basis:
    S. M. Mirtaheri and H. Zohoor,
    "Efficient formulation of the Gibbs–Appell equations for constrained
    multibody systems", Multibody System Dynamics, 2021.

The implementation uses the unconstrained specialization of the paper's
Eqs. (20)-(30) and Appendix B:

    U_i = [ m_i a_Gi
            I_i alpha_i + omega_i x (I_i omega_i) ]

    eta_i = d(v_tilde_Gi) / d(q_dot)
          = [ Jv_i
              Jw_i ]

    U* = sum_i eta_i.T @ U_i

For an unconstrained serial manipulator:
    U* = Q

For inverse dynamics with gravity and other external forces separated from
the actuator torques:

    tau = U* - Q_external

This file is intentionally self-contained. All classes and functions are
contained in this one file.

Input trajectory convention:
    q, qd, qdd:
        shape (N, n) for N trajectory samples and n joints

Output:
    tau:
        shape (N, n)

The implementation supports revolute ("R") and prismatic ("P") joints.

Important convention:
    DH parameters use the standard homogeneous transform

        A_i = Rot(z, theta_i) Trans(z, d_i) Trans(x, a_i) Rot(x, alpha_i)

    For revolute joints:
        theta_i = theta_offset + q_i
        d_i     = d

    For prismatic joints:
        theta_i = theta
        d_i     = d + q_i

    Link inertia is specified about the link COM and expressed in the link
    coordinate frame. The COM vector is also expressed in the link frame.

This is a dynamics implementation for a serial open-chain robot. The
constrained-system machinery of the paper (quasivelocities, constraint
matrix [a], sigma/omega partitioning, etc.) is not required for the
unconstrained inverse-dynamics problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float]]


# ============================================================================
# Basic linear algebra
# ============================================================================

def skew(v: ArrayLike) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix associated with a 3-vector."""
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def cross_matrix(v: ArrayLike) -> np.ndarray:
    """Alias for skew()."""
    return skew(v)


def check_vector(v: ArrayLike, name: str) -> np.ndarray:
    """Convert an input to a 3-vector and validate its shape."""
    a = np.asarray(v, dtype=float)
    if a.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {a.shape}")
    return a


# ============================================================================
# DH transformation
# ============================================================================

def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """
    Standard Denavit-Hartenberg homogeneous transformation.

    Maps coordinates from frame i to frame i-1.
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


# ============================================================================
# Link
# ============================================================================

@dataclass
class Link:
    """
    One rigid link/joint of a serial robot.

    Parameters
    ----------
    a:
        DH link length.
    alpha:
        DH link twist [rad].
    d:
        DH offset.
    theta:
        DH angle offset [rad].
    mass:
        Link mass [kg].
    com:
        COM position expressed in the link frame [m].
    inertia:
        3x3 inertia tensor about the COM, expressed in the link frame
        [kg*m^2].
    joint_type:
        "R" for revolute or "P" for prismatic.
    name:
        Optional link name.
    """

    a: float
    alpha: float
    d: float
    theta: float
    mass: float
    com: np.ndarray
    inertia: np.ndarray
    joint_type: str = "R"
    name: str = ""

    def __post_init__(self):
        self.com = check_vector(self.com, "com")

        self.inertia = np.asarray(self.inertia, dtype=float)
        if self.inertia.shape != (3, 3):
            raise ValueError(
                f"inertia must have shape (3,3), got {self.inertia.shape}"
            )

        self.joint_type = self.joint_type.upper()
        if self.joint_type not in ("R", "P"):
            raise ValueError("joint_type must be 'R' or 'P'.")

        if self.mass <= 0.0:
            raise ValueError("mass must be positive.")

        # Symmetrize tiny numerical asymmetries.
        self.inertia = 0.5 * (self.inertia + self.inertia.T)


# ============================================================================
# Robot
# ============================================================================

class Robot:
    """
    Serial open-chain robot for Gibbs-Appell inverse dynamics.

    The class follows the structure of the paper's formulation:

        1. Calculate COM velocity Jacobians eta_i.
        2. Calculate COM accelerations and angular accelerations.
        3. Build U_i from Newton-Euler momentum derivatives.
        4. Form U* = sum eta_i.T U_i.
        5. Subtract generalized external forces to obtain actuator torques.

    For an unconstrained system, this corresponds to the unconstrained
    specialization of Eq. (24)/(28)-(30) of the paper.
    """

    def __init__(
        self,
        links: Sequence[Link],
        gravity: ArrayLike = (0.0, 0.0, -9.81),
    ):
        if len(links) == 0:
            raise ValueError("Robot must contain at least one link.")

        self.links = list(links)
        self.n = len(self.links)
        self.gravity = check_vector(gravity, "gravity")

    # ---------------------------------------------------------------------
    # Joint variables
    # ---------------------------------------------------------------------

    def _joint_parameters(self, link: Link, q_i: float):
        """Return theta and d after inserting the joint coordinate."""
        if link.joint_type == "R":
            theta = link.theta + q_i
            d = link.d
        else:
            theta = link.theta
            d = link.d + q_i

        return theta, d

    def _joint_axis_and_origin(
        self,
        T_parent: np.ndarray,
    ):
        """
        Joint axis z_(i-1) and joint origin expressed in the base frame.
        """
        z = T_parent[:3, 2].copy()
        p = T_parent[:3, 3].copy()
        return z, p

    # ---------------------------------------------------------------------
    # Forward kinematics
    # ---------------------------------------------------------------------

    def forward_kinematics(
        self,
        q: ArrayLike,
    ):
        """
        Compute base-frame transforms for all link frames.

        Returns
        -------
        T:
            List of n homogeneous transformations T_0_i.
        joint_origins:
            List of joint origins p_0_(i-1).
        joint_axes:
            List of joint axes z_0_(i-1).
        """
        q = self._check_state(q, "q")

        T = []
        joint_origins = []
        joint_axes = []

        T_parent = np.eye(4)

        for i, link in enumerate(self.links):
            axis, origin = self._joint_axis_and_origin(T_parent)

            joint_axes.append(axis)
            joint_origins.append(origin)

            theta, d = self._joint_parameters(link, q[i])

            A = dh_transform(
                theta=theta,
                d=d,
                a=link.a,
                alpha=link.alpha,
            )

            T_parent = T_parent @ A
            T.append(T_parent.copy())

        return T, joint_origins, joint_axes

    # ---------------------------------------------------------------------
    # COM geometry
    # ---------------------------------------------------------------------

    def center_of_mass_positions(
        self,
        q: ArrayLike,
    ):
        """
        Return base-frame COM positions for all links.
        """
        T, _, _ = self.forward_kinematics(q)

        com_positions = []

        for i, link in enumerate(self.links):
            T_i = T[i]
            p = T_i[:3, :3] @ link.com + T_i[:3, 3]
            com_positions.append(p)

        return com_positions

    # ---------------------------------------------------------------------
    # Jacobians eta_i
    # ---------------------------------------------------------------------

    def link_jacobian(
        self,
        q: ArrayLike,
        link_index: int,
    ):
        """
        Calculate eta_i = [Jv; Jw] for the COM of link i.

        This is the implementation of the Appendix-B relationship

            eta_i = d(v_tilde_Gi) / d(q_dot)

        with

            v_tilde_Gi = [v_Gi, omega_i].

        Returns
        -------
        eta:
            6xn matrix.
        """
        q = self._check_state(q, "q")

        if not (0 <= link_index < self.n):
            raise IndexError("link_index out of range.")

        T, joint_origins, joint_axes = self.forward_kinematics(q)

        link = self.links[link_index]
        T_i = T[link_index]

        R_i = T_i[:3, :3]
        p_i = T_i[:3, 3]

        p_com = p_i + R_i @ link.com

        Jv = np.zeros((3, self.n))
        Jw = np.zeros((3, self.n))

        for j in range(link_index + 1):

            z = joint_axes[j]
            p_joint = joint_origins[j]

            if self.links[j].joint_type == "R":
                Jv[:, j] = np.cross(z, p_com - p_joint)
                Jw[:, j] = z

            else:
                Jv[:, j] = z
                Jw[:, j] = 0.0

        eta = np.vstack((Jv, Jw))

        return eta

    # ---------------------------------------------------------------------
    # All eta matrices
    # ---------------------------------------------------------------------

    def all_eta(
        self,
        q: ArrayLike,
    ):
        """Return eta_i for every link."""
        return [
            self.link_jacobian(q, i)
            for i in range(self.n)
        ]

    # ---------------------------------------------------------------------
    # Spatial velocity
    # ---------------------------------------------------------------------

    def link_velocity(
        self,
        q: ArrayLike,
        qd: ArrayLike,
        link_index: int,
    ):
        """Return [v_G, omega] for one link."""
        q = self._check_state(q, "q")
        qd = self._check_state(qd, "qd")

        eta = self.link_jacobian(q, link_index)
        return eta @ qd

    # ---------------------------------------------------------------------
    # Numerical Jacobian derivative
    # ---------------------------------------------------------------------

    def eta_derivative(
        self,
        q: ArrayLike,
        qd: ArrayLike,
        link_index: int,
        method: str = "finite_difference",
        eps: float = 1e-7,
    ):
        """
        Calculate d(eta_i)/dt.

        Since eta_i depends on q:

            eta_dot_i = sum_j d(eta_i)/dq_j * qd_j.

        A central finite difference is used here. This keeps the implementation
        general for arbitrary DH chains.

        For high-performance studies, this function can later be replaced
        by an analytical recursive Jacobian-derivative implementation.
        """
        q = self._check_state(q, "q")
        qd = self._check_state(qd, "qd")

        if method != "finite_difference":
            raise ValueError(
                "Only 'finite_difference' is currently supported."
            )

        eta_dot = np.zeros((6, self.n))

        for j in range(self.n):
            h = eps * max(1.0, abs(q[j]))

            qp = q.copy()
            qm = q.copy()

            qp[j] += h
            qm[j] -= h

            eta_p = self.link_jacobian(qp, link_index)
            eta_m = self.link_jacobian(qm, link_index)

            d_eta_dqj = (eta_p - eta_m) / (2.0 * h)

            eta_dot += d_eta_dqj * qd[j]

        return eta_dot

    # ---------------------------------------------------------------------
    # Kinematics for one link
    # ---------------------------------------------------------------------

    def link_kinematics(
        self,
        q: ArrayLike,
        qd: ArrayLike,
        qdd: ArrayLike,
        link_index: int,
    ):
        """
        Calculate v_G, omega, a_G, alpha for one link.

        Using:

            v_tilde = eta qdot

            a_tilde = eta qdd + eta_dot qdot

        This is exactly the useful Jacobian relationship established in
        Appendix B of the paper.
        """
        q = self._check_state(q, "q")
        qd = self._check_state(qd, "qd")
        qdd = self._check_state(qdd, "qdd")

        eta = self.link_jacobian(q, link_index)
        eta_dot = self.eta_derivative(q, qd, link_index)

        v_tilde = eta @ qd
        a_tilde = eta @ qdd + eta_dot @ qd

        v_G = v_tilde[:3]
        omega = v_tilde[3:]

        a_G = a_tilde[:3]
        alpha = a_tilde[3:]

        return v_G, omega, a_G, alpha, eta, eta_dot

    # ---------------------------------------------------------------------
    # U_i
    # ---------------------------------------------------------------------

    def compute_U_link(
        self,
        q: ArrayLike,
        qd: ArrayLike,
        qdd: ArrayLike,
        link_index: int,
    ):
        """
        Calculate the 6-vector U_i from Eq. (21):

            U_i =
            [ m a_G
              I alpha + omega x (I omega) ]

        Inertia is expressed in the link frame, so angular quantities are
        transformed into that frame before evaluating the rotational term.
        """
        link = self.links[link_index]

        v_G, omega_0, a_G, alpha_0, eta, eta_dot = self.link_kinematics(
            q, qd, qdd, link_index
        )

        T, _, _ = self.forward_kinematics(q)
        R = T[link_index][:3, :3]

        # Transform angular velocity and acceleration to link frame.
        omega = R.T @ omega_0
        alpha = R.T @ alpha_0

        # Newton-Euler momentum derivative in link frame.
        force = link.mass * a_G

        moment_body = (
            link.inertia @ alpha
            + np.cross(omega, link.inertia @ omega)
        )

        # The rotational component of eta is defined in the base frame.
        # Transform the moment back to the base frame so eta.T @ U is
        # coordinate-consistent.
        moment = R @ moment_body

        U = np.hstack((force, moment))

        return U

    # ---------------------------------------------------------------------
    # U*
    # ---------------------------------------------------------------------

    def compute_U_star(
        self,
        q: ArrayLike,
        qd: ArrayLike,
        qdd: ArrayLike,
    ):
        """
        Calculate

            U* = sum_i eta_i.T U_i

        corresponding to Eq. (25) / Eq. (81) of the paper.

        Returns
        -------
        U_star:
            n-vector of generalized dynamic forces.
        """
        q = self._check_state(q, "q")
        qd = self._check_state(qd, "qd")
        qdd = self._check_state(qdd, "qdd")

        U_star = np.zeros(self.n)

        for i in range(self.n):
            eta = self.link_jacobian(q, i)
            U_i = self.compute_U_link(q, qd, qdd, i)

            U_star += eta.T @ U_i

        return U_star

    # ---------------------------------------------------------------------
    # Generalized gravity force
    # ---------------------------------------------------------------------

    def gravity_generalized_force(
        self,
        q: ArrayLike,
    ):
        """
        Calculate generalized gravity forces.

        Q_g = sum_i Jv_i.T @ (m_i g)

        The sign is determined by the gravity vector supplied to Robot().
        """
        q = self._check_state(q, "q")

        Q_g = np.zeros(self.n)

        for i, link in enumerate(self.links):
            eta = self.link_jacobian(q, i)
            Jv = eta[:3, :]

            Q_g += Jv.T @ (link.mass * self.gravity)

        return Q_g

    # ---------------------------------------------------------------------
    # General external wrench contribution
    # ---------------------------------------------------------------------

    def external_generalized_force(
        self,
        q: ArrayLike,
        external_wrenches: Optional[Sequence[Optional[ArrayLike]]] = None,
    ):
        """
        Convert externally applied body wrenches to generalized forces.

        external_wrenches[i] is a 6-vector

            [Fx, Fy, Fz, Mx, My, Mz]

        expressed in the BASE frame and applied at the COM of link i.

        Returns:
            Q_external
        """
        q = self._check_state(q, "q")

        Q = np.zeros(self.n)

        if external_wrenches is None:
            return Q

        if len(external_wrenches) != self.n:
            raise ValueError(
                "external_wrenches must contain one entry per link."
            )

        for i, wrench in enumerate(external_wrenches):
            if wrench is None:
                continue

            wrench = np.asarray(wrench, dtype=float)

            if wrench.shape != (6,):
                raise ValueError(
                    f"external wrench {i} must have shape (6,)."
                )

            eta = self.link_jacobian(q, i)

            Q += eta.T @ wrench

        return Q

    # ---------------------------------------------------------------------
    # Inverse dynamics: single sample
    # ---------------------------------------------------------------------

    def inverse_dynamics_sample(
        self,
        q: ArrayLike,
        qd: ArrayLike,
        qdd: ArrayLike,
        include_gravity: bool = True,
        external_wrenches: Optional[
            Sequence[Optional[ArrayLike]]
        ] = None,
    ):
        """
        Calculate actuator torques for one trajectory sample.

        The required actuator generalized force is

            tau = U* - Q_external

        where Q_external includes gravity and any other supplied external
        wrenches.

        This is the form corresponding to U* = Q for an unconstrained system,
        with actuator forces separated from known external forces.
        """
        U_star = self.compute_U_star(q, qd, qdd)

        Q_external = np.zeros(self.n)

        if include_gravity:
            Q_external += self.gravity_generalized_force(q)

        Q_external += self.external_generalized_force(
            q,
            external_wrenches
        )

        tau = U_star - Q_external

        return tau

    # ---------------------------------------------------------------------
    # Batch inverse dynamics
    # ---------------------------------------------------------------------

    def inverse_dynamics(
        self,
        q: ArrayLike,
        qd: ArrayLike,
        qdd: ArrayLike,
        include_gravity: bool = True,
    ):
        """
        Calculate inverse dynamics for an arbitrary number of trajectory
        samples.

        Parameters
        ----------
        q:
            (N,n) positions
        qd:
            (N,n) velocities
        qdd:
            (N,n) accelerations

        Returns
        -------
        tau:
            (N,n)
        """
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        qdd = np.asarray(qdd, dtype=float)

        self._check_trajectory_shapes(q, qd, qdd)

        N = q.shape[0]
        tau = np.zeros((N, self.n))

        for k in range(N):
            tau[k] = self.inverse_dynamics_sample(
                q[k],
                qd[k],
                qdd[k],
                include_gravity=include_gravity,
            )

        return tau

    # ---------------------------------------------------------------------
    # Matrix form U* = M qdd + N
    # ---------------------------------------------------------------------

    def mass_matrix(
        self,
        q: ArrayLike,
    ):
        """
        Compute M(q) directly from the Gibbs-Appell/Jacobian formulation.

        Because U* is linear in qdd:

            M = dU*/dqdd.

        A direct and numerically cleaner construction is:

            M = sum_i eta_i.T H_i eta_i

        where H_i is the 6x6 body spatial inertia expressed in the base
        frame. This is equivalent to the coefficient of qdd in U*.
        """
        q = self._check_state(q, "q")

        M = np.zeros((self.n, self.n))

        T, _, _ = self.forward_kinematics(q)

        for i, link in enumerate(self.links):
            eta = self.link_jacobian(q, i)

            R = T[i][:3, :3]

            I_base = R @ link.inertia @ R.T

            H = np.zeros((6, 6))
            H[:3, :3] = link.mass * np.eye(3)
            H[3:, 3:] = I_base

            M += eta.T @ H @ eta

        # Numerical symmetrization.
        M = 0.5 * (M + M.T)

        return M

    def nonlinear_vector(
        self,
        q: ArrayLike,
        qd: ArrayLike,
    ):
        """
        Compute N(q, qd) from

            U* = M(q) qdd + N(q, qd).

        We evaluate U* with qdd = 0.
        """
        q = self._check_state(q, "q")
        qd = self._check_state(qd, "qd")

        qdd_zero = np.zeros(self.n)

        return self.compute_U_star(
            q,
            qd,
            qdd_zero
        )

    def dynamics_matrices(
        self,
        q: ArrayLike,
        qd: ArrayLike,
    ):
        """
        Return M(q) and N(q,qd).
        """
        M = self.mass_matrix(q)
        N = self.nonlinear_vector(q, qd)

        return M, N

    # ---------------------------------------------------------------------
    # Validation utilities
    # ---------------------------------------------------------------------

    def validate_dynamics_decomposition(
        self,
        q: ArrayLike,
        qd: ArrayLike,
        qdd: ArrayLike,
        tolerance: float = 1e-7,
    ):
        """
        Check:

            U*(q,qd,qdd) ≈ M(q) qdd + N(q,qd)

        Returns a dictionary with the error.
        """
        U_star = self.compute_U_star(q, qd, qdd)

        M = self.mass_matrix(q)
        N = self.nonlinear_vector(q, qd)

        reconstructed = M @ qdd + N

        error = U_star - reconstructed
        max_error = np.max(np.abs(error))

        return {
            "passed": bool(max_error < tolerance),
            "max_absolute_error": float(max_error),
            "error": error,
            "U_star": U_star,
            "M_qdd_plus_N": reconstructed,
        }

    def validate_mass_matrix(
        self,
        q: ArrayLike,
        tolerance: float = 1e-10,
    ):
        """
        Check symmetry and positive eigenvalues of M(q).
        """
        M = self.mass_matrix(q)

        symmetry_error = np.max(np.abs(M - M.T))
        eigenvalues = np.linalg.eigvalsh(M)

        return {
            "symmetric": bool(symmetry_error < tolerance),
            "symmetry_error": float(symmetry_error),
            "eigenvalues": eigenvalues,
            "positive_definite": bool(np.all(eigenvalues > tolerance)),
            "M": M,
        }

    # ---------------------------------------------------------------------
    # Trajectory utilities
    # ---------------------------------------------------------------------

    @staticmethod
    def _check_state(x, name):
        x = np.asarray(x, dtype=float)

        if x.ndim != 1:
            raise ValueError(
                f"{name} must be a 1D array of length n."
            )

        return x

    def _check_trajectory_shapes(self, q, qd, qdd):
        if q.ndim != 2:
            raise ValueError("q must have shape (N,n).")

        if qd.ndim != 2:
            raise ValueError("qd must have shape (N,n).")

        if qdd.ndim != 2:
            raise ValueError("qdd must have shape (N,n).")

        expected = q.shape

        if qd.shape != expected:
            raise ValueError(
                f"qd shape {qd.shape} does not match q shape {expected}."
            )

        if qdd.shape != expected:
            raise ValueError(
                f"qdd shape {qdd.shape} does not match q shape {expected}."
            )

        if q.shape[1] != self.n:
            raise ValueError(
                f"Trajectory contains {q.shape[1]} joints, "
                f"but robot contains {self.n} links."
            )

    @staticmethod
    def load_trajectory_csv(
        filename: Union[str, Path],
        delimiter: str = ",",
        skip_header: int = 0,
    ):
        """
        Load a trajectory CSV.

        Supported formats:

        1. Three blocks of n rows:
               q
               qd
               qdd

           Example for n=2:
               q1 q2
               ...
               qd1 qd2
               ...
               qdd1 qdd2
               ...

        2. Columns:
               q1 ... qn qd1 ... qdn qdd1 ... qddn

        The function automatically recognizes format (2) when the number
        of columns is 3n only if n can be inferred from the file.
        For unambiguous loading, use load_trajectory_from_arrays() below.
        """
        data = np.loadtxt(
            filename,
            delimiter=delimiter,
            skiprows=skip_header,
        )

        if data.ndim == 1:
            data = data.reshape(1, -1)

        return data

    def load_trajectory(
        self,
        filename: Union[str, Path],
        delimiter: str = ",",
        skip_header: int = 0,
        format: str = "columns",
    ):
        """
        Load q, qd, qdd from CSV.

        format="columns":
            columns = [q | qd | qdd]
            shape = (N, 3n)

        format="rows":
            rows = [q; qd; qdd]
            shape = (3n, N)

        format="q_qd_qdd_rows":
            rows = q rows followed by qd rows followed by qdd rows,
            shape = (3n, N)
        """
        data = self.load_trajectory_csv(
            filename,
            delimiter=delimiter,
            skip_header=skip_header,
        )

        n = self.n

        if format == "columns":
            if data.shape[1] != 3 * n:
                raise ValueError(
                    f"Expected {3*n} columns for n={n}, got {data.shape[1]}."
                )

            q = data[:, :n]
            qd = data[:, n:2*n]
            qdd = data[:, 2*n:3*n]

        elif format in ("rows", "q_qd_qdd_rows"):
            if data.shape[0] != 3 * n:
                raise ValueError(
                    f"Expected {3*n} rows for n={n}, got {data.shape[0]}."
                )

            q = data[:n, :].T
            qd = data[n:2*n, :].T
            qdd = data[2*n:3*n, :].T

        else:
            raise ValueError(
                "format must be 'columns' or 'q_qd_qdd_rows'."
            )

        return q, qd, qdd

    # ---------------------------------------------------------------------
    # Save torque trajectory
    # ---------------------------------------------------------------------

    @staticmethod
    def save_torques_csv(
        filename: Union[str, Path],
        tau: ArrayLike,
        delimiter: str = ",",
    ):
        """Save an (N,n) torque matrix to CSV."""
        tau = np.asarray(tau, dtype=float)

        if tau.ndim != 2:
            raise ValueError("tau must have shape (N,n).")

        np.savetxt(
            filename,
            tau,
            delimiter=delimiter,
        )


# ============================================================================
# Convenience function
# ============================================================================

def gibbs_appell_inverse_dynamics(
    robot: Robot,
    q: ArrayLike,
    qd: ArrayLike,
    qdd: ArrayLike,
):
    """
    Convenience wrapper.

    Example:
        tau = gibbs_appell_inverse_dynamics(robot, q, qd, qdd)
    """
    return robot.inverse_dynamics(q, qd, qdd)


# ============================================================================
# Example robot constructors
# ============================================================================

def create_planar_2link_robot(
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.81,
):
    """
    Create a simple 2-link planar revolute robot.

    Both joints rotate about z.
    Link frames are located at the proximal joint.
    COM is at the center of each link.
    """
    I1z = m1 * l1**2 / 12.0
    I2z = m2 * l2**2 / 12.0

    link1 = Link(
        a=l1,
        alpha=0.0,
        d=0.0,
        theta=0.0,
        mass=m1,
        com=np.array([l1 / 2.0, 0.0, 0.0]),
        inertia=np.diag([0.0, 0.0, I1z]),
        joint_type="R",
        name="Link 1",
    )

    link2 = Link(
        a=l2,
        alpha=0.0,
        d=0.0,
        theta=0.0,
        mass=m2,
        com=np.array([l2 / 2.0, 0.0, 0.0]),
        inertia=np.diag([0.0, 0.0, I2z]),
        joint_type="R",
        name="Link 2",
    )

    return Robot(
        [link1, link2],
        gravity=np.array([0.0, -g, 0.0]),
    )


def create_n_link_planar_robot(
    masses: ArrayLike,
    lengths: ArrayLike,
    g: float = 9.81,
):
    """
    Create an arbitrary-N planar revolute robot.

    Parameters
    ----------
    masses:
        (n,) masses
    lengths:
        (n,) link lengths
    """
    masses = np.asarray(masses, dtype=float)
    lengths = np.asarray(lengths, dtype=float)

    if masses.ndim != 1 or lengths.ndim != 1:
        raise ValueError("masses and lengths must be 1D arrays.")

    if len(masses) != len(lengths):
        raise ValueError("masses and lengths must have the same length.")

    links = []

    for i, (m, L) in enumerate(zip(masses, lengths)):

        Iz = m * L**2 / 12.0

        links.append(
            Link(
                a=L,
                alpha=0.0,
                d=0.0,
                theta=0.0,
                mass=m,
                com=np.array([L / 2.0, 0.0, 0.0]),
                inertia=np.diag([0.0, 0.0, Iz]),
                joint_type="R",
                name=f"Link {i + 1}",
            )
        )

    return Robot(
        links,
        gravity=np.array([0.0, -g, 0.0]),
    )


# ============================================================================
# Simple trajectory generator
# ============================================================================

def generate_sinusoidal_trajectory(
    time: ArrayLike,
    amplitudes: ArrayLike,
    frequencies: ArrayLike,
    offsets: Optional[ArrayLike] = None,
):
    """
    Generate q, qd, qdd for:

        q_i(t) = offset_i + A_i sin(omega_i t)

    Returns:
        q, qd, qdd
    """
    t = np.asarray(time, dtype=float).reshape(-1)
    amplitudes = np.asarray(amplitudes, dtype=float).reshape(-1)
    frequencies = np.asarray(frequencies, dtype=float).reshape(-1)

    if offsets is None:
        offsets = np.zeros_like(amplitudes)
    else:
        offsets = np.asarray(offsets, dtype=float).reshape(-1)

    n = len(amplitudes)

    if len(frequencies) != n or len(offsets) != n:
        raise ValueError(
            "amplitudes, frequencies, and offsets must have the same length."
        )

    q = np.zeros((len(t), n))
    qd = np.zeros((len(t), n))
    qdd = np.zeros((len(t), n))

    for i in range(n):
        w = frequencies[i]
        A = amplitudes[i]

        q[:, i] = offsets[i] + A * np.sin(w * t)
        qd[:, i] = A * w * np.cos(w * t)
        qdd[:, i] = -A * w**2 * np.sin(w * t)

    return q, qd, qdd


# ============================================================================
# Self-test / demonstration
# ============================================================================

def example():
    """
    Demonstration with an arbitrary number of links and arbitrary number
    of trajectory samples.
    """

    # ------------------------------------------------------------
    # 1. Create robot
    # ------------------------------------------------------------

    robot = create_n_link_planar_robot(
        masses=[1.0, 1.2, 0.8, 0.6],
        lengths=[0.5, 0.4, 0.3, 0.25],
    )

    # ------------------------------------------------------------
    # 2. Generate 1000 trajectory samples
    # ------------------------------------------------------------

    time = np.linspace(0.0, 10.0, 1000)

    q, qd, qdd = generate_sinusoidal_trajectory(
        time=time,
        amplitudes=[0.5, 0.4, 0.3, 0.2],
        frequencies=[1.0, 0.8, 0.6, 0.5],
    )

    # ------------------------------------------------------------
    # 3. Compute all actuator torques
    # ------------------------------------------------------------

    tau = robot.inverse_dynamics(
        q=q,
        qd=qd,
        qdd=qdd,
    )

    # ------------------------------------------------------------
    # 4. Print result
    # ------------------------------------------------------------

    print("=" * 70)
    print("GIBBS-APPELL INVERSE DYNAMICS")
    print("=" * 70)

    print(f"Number of links       : {robot.n}")
    print(f"Number of data points : {len(time)}")
    print(f"q shape               : {q.shape}")
    print(f"qd shape              : {qd.shape}")
    print(f"qdd shape             : {qdd.shape}")
    print(f"tau shape             : {tau.shape}")

    print("\nFirst torque sample:")
    print(tau[0])

    print("\nLast torque sample:")
    print(tau[-1])

    # ------------------------------------------------------------
    # 5. Validate M*qdd + N = U*
    # ------------------------------------------------------------

    validation = robot.validate_dynamics_decomposition(
        q=q[100],
        qd=qd[100],
        qdd=qdd[100],
    )

    print("\nDynamics decomposition:")
    print(f"Passed: {validation['passed']}")
    print(
        "Maximum absolute error:",
        validation["max_absolute_error"]
    )

    # ------------------------------------------------------------
    # 6. Validate mass matrix
    # ------------------------------------------------------------

    M_validation = robot.validate_mass_matrix(q[100])

    print("\nMass matrix:")
    print(M_validation["M"])

    print("\nMass matrix eigenvalues:")
    print(M_validation["eigenvalues"])

    print(
        "\nMass matrix positive definite:",
        M_validation["positive_definite"]
    )

    return robot, time, q, qd, qdd, tau


if __name__ == "__main__":
    example()
